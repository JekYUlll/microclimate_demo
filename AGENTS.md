## Project Scripts Overview

This file summarizes the runnable scripts under `scripts/` (and `scripts/baselines/`) and their key behaviours/dependencies.

### Root scripts
- `scripts/prepare_data.py` – Loads raw Excel files via `src/RAW_LSTM.data.load_all_stations` using `Settings` from `src/RAW_LSTM.config`; outputs a processed CSV (default `data/processed/meteorology.csv`), optional `--freq` and `--output`.
- `scripts/train_lstm.py` – Baseline RAW_LSTM trainer. Uses `Settings` + `WindowDataset`/`feature_matrix` from `src/RAW_LSTM`, trains `LSTMForecaster`, saves checkpoints to `models/checkpoints/lstm_<station>.pt`. CLI overrides: `--station/--epochs/--batch-size/--lr/--freq/--data/--device`.
- `scripts/evaluate_lstm.py` – Loads a saved RAW_LSTM checkpoint, rebuilds the model, normalises/denormalises, computes RMSE/MAE, saves a prediction plot. Depends on `src/RAW_LSTM` modules.
- `scripts/evaluate_datasets.py` – Scans CSV datasets for time-series readiness (timestamp inference, dominant freq, missing rates, longest gaps, duplicates). Supports encoding fallback; CLI: `--root/--pattern/--expected-freq/--target-col/--encoding/--limit`.
- `scripts/emd_lstm.py` – CLI entry for the EMD+LSTM pipeline (`src/EMD_LSTM/emd_lstm.py`). Key args: data path, target/feature cols, encoding, window/horizon, train ratio, epochs/batch/lr/hidden/num-layers/dropout/seed/device; EMD controls (`--max-imfs`, `--emd-max-sift`, `--emd-spline-kind`, `--emd-log-interval`, `--max-samples`); logging/plotting (`--log-path/--plot/--max-points`, verbosity).
- `scripts/run_baselines.py` – Combined baseline runner using Darts models (naive/seasonal/theta/ETS/ARIMA/TFT/NBEATS). Builds configs from `src/baselines.common` and `src/baselines.darts_baselines`. Supports multi-GPU via Lightning `devices`; logs results; optional `--plot` for RMSE/MAE bar chart.

### Baseline subcommands (`scripts/baselines/`)
- `run_naive.py` – Evaluates NaiveMean and NaiveSeasonal. CLI: data/target/encoding/freq/horizon/train_ratio/stride/season_length/max_windows/log_path/quiet/plot. Outputs table + optional bar plot.
- `run_arima.py` – Evaluates ARIMA (p,d,q from `--arima`). Same CLI shape as above; supports `max_windows`, `plot` bar chart.
- `run_tft.py` – Evaluates TFT baseline only. CLI: data/target/encoding/freq/horizon/input-window/train_ratio/stride/max_windows/epochs/hidden/num_heads/dropout/devices/log_path/quiet/plot. Uses `add_relative_index=True` to auto-generate future covariates; Lightning trainer uses GPU devices if provided.

### Key module links (for reference while reading scripts)
- RAW LSTM utilities live in `src/RAW_LSTM/` (`config.py`, `data.py`, `model.py`).
- EMD LSTM core pipeline is in `src/EMD_LSTM/emd_lstm.py` (EMD controls, logging, plotting, optional truncation).
- Baseline helpers in `src/baselines/` (`common.py` for data prep/logging, `darts_baselines.py` for Darts wrappers).

### Notes
- All plotting paths are optional; if provided, scripts create parent dirs.
- Encoding defaults to `None`; many AntAWS CSVs require `latin1`.
- For long EMD runs, use `--max-samples`/`--emd-max-sift` to keep runtime bounded. For baselines, limit `--max-windows` and prefer TFT/NBEATS for GPU usage.
