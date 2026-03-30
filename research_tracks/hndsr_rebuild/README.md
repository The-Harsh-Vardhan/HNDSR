# HNDSR Rebuild Research Track

This track rebuilds HNDSR as an isolated research workflow instead of extending the old notebook lineage.

## Goals

- Keep production code untouched until a new baseline is defensible.
- Reproduce the paper ladder one step at a time: bicubic, SR3-style diffusion, latent baseline, neural-operator baseline, then hybrid merge.
- Track every run in Weights & Biases with stable config, metrics, and exported samples.
- Use paper-reference remote-sensing datasets first, with Kaggle 4× kept as a control lane.

## Layout

- `configs/`: phase configs and ablation defaults
- `src/`: reusable dataset, metrics, tracker, and model code
- `scripts/`: train, evaluate, export, and ablation entrypoints
- `notebooks/`: control notebook for runtime checks and dashboards
- `reports/`: tracked markdown summaries for each phase

## First Commands

```powershell
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase0_bicubic_ucmerced_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase0_bicubic_aid_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase0_bicubic_rsscn7_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase0_bicubic_kaggle_control_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/train_baseline.py --config research_tracks/hndsr_rebuild/configs/phase1_sr3_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase1_sr3_smoke.yaml
```

## Dataset Contract

- `dataset.family`: `paper` or `kaggle`
- `dataset.name`: `ucmerced`, `aid`, `rsscn7`, or `kaggle_4x`
- `dataset.pairing_mode`: `synthetic_4x` for HR-only paper datasets, `paired` for Kaggle

Paper datasets are modeled as HR-only roots with deterministic synthetic LR generation at fixed `4×`.

## W&B Contract

- Project: `hndsr-research-track`
- Groups: `bootstrap`, `sr3`, `latent`, `operator`, `hybrid`
- Tags: `smoke`, `full`, `ablation`, `merge`

Bootstrap configs default to W&B `offline` mode so runs work before login is configured. Switch `tracking.mode` to `online` once the machine is authenticated.

If W&B is unavailable or init fails, the scripts fall back to a no-op tracker and still write local JSON reports.
