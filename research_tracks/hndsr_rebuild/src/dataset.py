"""Dataset helpers for the isolated HNDSR rebuild experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split


IMAGE_EXTENSIONS = (
    "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
    "*.PNG", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF",
)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for paired loaders and metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int


class SatellitePairDataset(Dataset):
    """Paired HR/LR dataset with deterministic pairing and optional augmentation."""

    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int, training: bool) -> None:
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size
        self.training = training
        self.pairs = self._collect_pairs()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _collect_pairs(self) -> list[tuple[Path, Path]]:
        hr_images: list[Path] = []
        lr_images: list[Path] = []
        for pattern in IMAGE_EXTENSIONS:
            hr_images.extend(sorted(self.hr_dir.glob(pattern)))
            lr_images.extend(sorted(self.lr_dir.glob(pattern)))
        if not hr_images or not lr_images:
            raise ValueError(f"No paired images found in {self.hr_dir} and {self.lr_dir}")
        hr_map = {path.stem: path for path in hr_images}
        lr_map = {path.stem: path for path in lr_images}
        common = sorted(set(hr_map) & set(lr_map))
        if not common:
            raise ValueError("No filename-aligned LR/HR pairs were found.")
        return [(hr_map[name], lr_map[name]) for name in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        hr_path, lr_path = self.pairs[index]
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        scale = hr_img.size[0] // lr_img.size[0]
        if self.training:
            hr_img, lr_img = self._random_crop_pair(hr_img, lr_img, scale)
        else:
            hr_img = transforms.CenterCrop(self.patch_size)(hr_img)
            lr_img = transforms.CenterCrop(self.patch_size // scale)(lr_img)
        if self.training and random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        return {
            "name": hr_path.stem,
            "scale": scale,
            "lr": self.transform(lr_img),
            "hr": self.transform(hr_img),
        }

    def _random_crop_pair(self, hr_img: Image.Image, lr_img: Image.Image, scale: int) -> tuple[Image.Image, Image.Image]:
        lr_crop = self.patch_size // scale
        lr_w, lr_h = lr_img.size
        if lr_w <= lr_crop or lr_h <= lr_crop:
            return hr_img, lr_img
        x = random.randint(0, lr_w - lr_crop)
        y = random.randint(0, lr_h - lr_crop)
        lr_box = (x, y, x + lr_crop, y + lr_crop)
        hr_box = (x * scale, y * scale, (x + lr_crop) * scale, (y + lr_crop) * scale)
        return hr_img.crop(hr_box), lr_img.crop(lr_box)


def _limited_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_loaders(config: dict, seed: int) -> DatasetBundle:
    """Build deterministic train/val loaders that mirror the production semantics."""
    data = config["data"]
    train_base = SatellitePairDataset(
        hr_dir=config["paths"]["hr_dir"],
        lr_dir=config["paths"]["lr_dir"],
        patch_size=data["patch_size"],
        training=True,
    )
    val_base = SatellitePairDataset(
        hr_dir=config["paths"]["hr_dir"],
        lr_dir=config["paths"]["lr_dir"],
        patch_size=data["patch_size"],
        training=False,
    )
    train_size = int((1 - data["val_split"]) * len(train_base))
    val_size = len(train_base) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(len(train_base)), [train_size, val_size], generator=generator)
    train_ds = Subset(train_base, train_indices.indices)
    val_ds = Subset(val_base, val_indices.indices)
    train_ds = _limited_subset(train_ds, data.get("train_limit"))
    val_ds = _limited_subset(val_ds, data.get("val_limit"))
    train_loader = DataLoader(
        train_ds,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=data["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data["num_workers"],
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_ds),
        val_size=len(val_ds),
    )
