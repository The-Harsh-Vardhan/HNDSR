import torch

from research_tracks.hndsr_rebuild.src.models import SR3Baseline
from research_tracks.hndsr_rebuild.src.tracker import NullTracker
from research_tracks.hndsr_rebuild.src.utils import load_config


def test_base_config_loads():
    config = load_config("research_tracks/hndsr_rebuild/configs/phase1_sr3_smoke.yaml")
    assert config["model"]["kind"] == "sr3"
    assert config["data"]["fixed_scale"] == 4


def test_sr3_forward_loss_contract():
    model = SR3Baseline(model_channels=16, num_timesteps=32, beta_start=1.0e-4, beta_end=0.02)
    lr_upscaled = torch.randn(2, 3, 64, 64)
    hr = torch.randn(2, 3, 64, 64)
    loss, stats = model.training_step(lr_upscaled, hr)
    assert loss.ndim == 0
    assert "timesteps_mean" in stats


def test_null_tracker_accepts_logs():
    tracker = NullTracker(run_dir="research_tracks/hndsr_rebuild/.tmp/test-null-tracker")
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.log_text("status", "ok")
    tracker.finish()
