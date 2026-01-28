## Project Scripts Overview

This file summarizes runnable scripts under `scripts/` (and `scripts/baselines/`) plus the current AntAWS demo pipeline.

### Core pipeline (AntAWS demo)
- `scripts/run_all.py` – One-command pipeline driven by `configs/exp.yaml`. Calls: prepare -> profile -> group -> dataset -> baselines -> TFT+PINN -> imputation compare -> eval -> figures. Writes `reports/<exp_id>/tables/exp_summary.csv`, `reports/<exp_id>/tables/config_used.yaml`, and `reports/<exp_id>/tables/pip_freeze.txt`.
- `scripts/prepare_antaws_station.py` – Ingests raw AntAWS CSV/Excel via `src/data/antaws_loader.py`. Optional resample. Outputs `data/processed/<exp_id>/<station_id>.csv` and `reports/<exp_id>/tables/prepare_log.csv`.
- `scripts/profile_missingness.py` – Computes missingness stats and plots using `src/analysis/missingness_profile.py`. Outputs `reports/<exp_id>/tables/missingness.csv`, `missingness_meta.csv`, and figures `missingness_by_year.png`, `gap_length_ccdf.png`.
- `scripts/assign_station_groups.py` – Labels coastal/inland using `data/metadata/stations_meta.csv` + `src/data/station_groups.py`. Outputs `data/processed/station_groups.csv` and `reports/<exp_id>/tables/station_groups_used.csv`.
- `scripts/build_dataset.py` – Runs imputation (A or B), feature engineering, time split, windowing, and normalization. Outputs: imputed CSV, `data/processed/<exp_id>/<station>_features.csv`, `reports/<exp_id>/tables/feature_list.csv`, `split_summary.csv`, `{train,val,test}.npz`, `scaler.json`.
- `scripts/run_baselines.py` – Pipeline baselines on the NPZ dataset (naive persistence/seasonal, AR, tabular GBRT, TCN, TFT). Writes `reports/<exp_id>/preds/*.csv` and `reports/<exp_id>/tables/*_metrics.csv`. If run without `--config`, falls back to legacy Darts evaluation.
- `scripts/train_tft_pinn.py` – Trains TFT or TFT+PINN (set `--mode tft` or `tft_pinn`). Saves model to `models/<exp_id>/tft(_pinn).pt`, predictions to `reports/<exp_id>/preds/`, and loss curve to `reports/<exp_id>/figures/loss_curve.png`.
- `scripts/train_tcn.py` – Trains a lightweight TCN baseline on the NPZ dataset. Outputs model and preds similar to TFT.
- `scripts/compare_missing_strategies.py` – Compares imputation A vs B using a tabular regressor. Outputs `reports/<exp_id>/tables/missing_strategy_compare.csv`.
- `scripts/evaluate_models.py` – Aggregates `reports/<exp_id>/preds/*.csv` into `metrics_overall.csv`, `metrics_by_horizon.csv`, and `metrics_extremes.csv`.
- `scripts/make_figures.py` – Builds evaluation plots (metrics bar, RMSE-by-horizon, prediction timeline, extremes) in `reports/<exp_id>/figures/`.

### Baseline subcommands (`scripts/baselines/`)
- `run_naive.py` – Naive mean + seasonal baselines (Darts). Supports `--config/--exp-id` to align with `configs/exp.yaml`.
- `run_arima.py` – ARIMA baseline (Darts). Supports `--config/--exp-id`.
- `run_tcn.py` – Darts TCN baseline. Supports `--config/--exp-id`.
- `run_tft.py` – Darts TFT baseline. Supports `--config/--exp-id`.

### Other utilities
- `scripts/evaluate_datasets.py` – Scans CSV datasets for time-series readiness (timestamp inference, dominant freq, missing rates, longest gaps, duplicates).

### Notebooks and legacy experiments
- `train.ipynb` – Currently an empty placeholder (0 bytes).
- `legacy/windblown_snow/train_dummy.ipynb` – Legacy sandbox using wind-blown snow simulation assets.
- `src/HOSTORY/` – Historical notebooks/scripts (older Darts-based experiments).

### Key module links (current pipeline)
- Config + paths: `src/utils/config.py`.
- AntAWS ingestion: `src/data/antaws_loader.py`, `src/data/resample.py`.
- Missingness + impute: `src/analysis/missingness_profile.py`, `src/data/impute.py`.
- Features + windows: `src/features/build_features.py`, `src/data/split.py`, `src/data/window_dataset.py`, `src/data/normalize.py`.
- Models: `src/models/snow_tft.py`, `src/models/tcn.py`.
- Training: `src/train/train_tft_pinn.py`, `src/train/train_tcn.py`.
- Physics loss: `src/losses/physics_losses.py`.
- Evaluation: `src/eval/metrics.py`, `src/eval/extremes.py`.
- Baselines: `src/baselines/simple.py` (pipeline baselines), `src/baselines/common.py`, `src/baselines/darts_baselines.py` (legacy Darts helpers).

### Legacy/obsolete (kept for reference)
- `scripts/prepare_data.py`, `scripts/train_lstm.py`, `scripts/evaluate_lstm.py`, and `src/EMD_LSTM/emd_lstm.py` still reference deleted `src/RAW_LSTM` modules and are not runnable without refactor.
- Wind-blown snow sandbox assets now live under `legacy/windblown_snow/` (not used by current pipeline):\n  - `legacy/windblown_snow/src/` (legacy SnowTFT, dataset, physics helpers)\n  - `legacy/windblown_snow/scripts/generate_fake_snow_data.py`\n  - `legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv`\n  - `legacy/windblown_snow/docs/windblown_snow_data_spec.md`

### Notes
- All plotting paths create parent directories when needed.
- Encoding defaults to `None`; AntAWS CSVs often require `latin1`.
- For faster runs, reduce `models.*.epochs` and set `max_windows` in `configs/exp.yaml`.
