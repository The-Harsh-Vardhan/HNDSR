# HNDSR Rebuild Research Track

This track rebuilds HNDSR as an isolated research workflow instead of extending the old notebook lineage.

## Goals

- Keep production code untouched until a new baseline is defensible.
- Reproduce the paper ladder one step at a time: bicubic, SR3-style diffusion, latent baseline, neural-operator baseline, then hybrid merge.
- Track every run in Weights & Biases with stable config, metrics, and exported samples.

## Layout

- `configs/`: phase configs and ablation defaults
- `src/`: reusable dataset, metrics, tracker, and model code
- `scripts/`: train, evaluate, export, and ablation entrypoints
- `notebooks/`: control notebook for runtime checks and dashboards
- `reports/`: tracked markdown summaries for each phase

## First Commands

```powershell
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase0_bicubic_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/train_baseline.py --config research_tracks/hndsr_rebuild/configs/phase1_sr3_smoke.yaml
python research_tracks/hndsr_rebuild/scripts/evaluate_run.py --config research_tracks/hndsr_rebuild/configs/phase1_sr3_smoke.yaml
```

## W&B Contract

- Project: `hndsr-research-track`
- Groups: `bootstrap`, `sr3`, `latent`, `operator`, `hybrid`
- Tags: `smoke`, `full`, `ablation`, `merge`

Bootstrap configs default to W&B `offline` mode so runs work before login is configured. Switch `tracking.mode` to `online` once the machine is authenticated.

If W&B is unavailable or init fails, the scripts fall back to a no-op tracker and still write local JSON reports.
