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
- `run_tcn.py` – Evaluates TCN baseline. CLI: data/target/encoding/freq/horizon/input-window/train_ratio/stride/max_windows/epochs/kernel_size/num_filters/dilation_base/dropout/devices/log_path/quiet/plot.
- `run_tft.py` – Evaluates TFT baseline only. CLI: data/target/encoding/freq/horizon/input-window/train_ratio/stride/max_windows/epochs/hidden/num_heads/dropout/devices/log_path/quiet/plot. Uses `add_relative_index=True` to auto-generate future covariates; Lightning trainer uses GPU devices if provided.

### Notebook (`train.ipynb`)
- Purpose: wind-blown snow TFT sandbox that wires SnowTFT + physics penalty + load post-processing for quick experimentation.
- Data/feats: loads `data/synthetic/windblown_snow_sample.csv`, parses `timestamp`, and z-scores features. Known history columns live in `KNOWN_COLS` (wind/air state + flux proxies), known future in `KNOWN_FUTURE_COLS` (solar, wind direction). Targets: `air_temperature_c`, `wind_speed_ms`, `snow_mass_flux_kg_m2_s`, `snow_surface_temperature_c`. Optional spectra columns (`size_bin_*`, `velocity_bin_*`) join the known set when `use_spectra_as_known=True`; one-hot dummies from `stability_flag`, `quality_flag`, `data_source` are appended.
- Data loaders: uses `src/snow_dataset.build_loaders` with `window_size=24`, `horizon=6`, `train_ratio=0.8`, `batch_size=64`; `future_known_idx` is derived from `KNOWN_FUTURE_COLS`.
- Model: builds `src/tft_model.SnowTFT(input_dim=len(KNOWN), target_dim=len(TARGET_COLS), known_future_dim=len(future_known_idx), d_model=128, nhead=4, num_layers=2)`, trains with Adam (`lr=1e-3`) + `nn.MSELoss` on CUDA if available.
- Physics term: `physics_penalty` from `src/physics_losses.py` unnormalizes drivers (wind, friction velocity, temps, radiation, RH, stability dummies) and enforces threshold/monotonicity heuristics; combined loss is `mse + 0.1 * physics_penalty` in `run_epoch`.
- Extras/plots: validation plots compare true vs predicted horizons (`matplotlib`). A sample block derives load parameters from the first predicted step using `src/physics_calculations.ParticleBin` + `summarize_loads` (mass/momentum/energy flux, impact pressure, density/hardness/viscosity). A final block concatenates validation batches for a longer timeline view.

### Key module links (for reference while reading scripts)
- RAW LSTM utilities live in `src/RAW_LSTM/` (`config.py`, `data.py`, `model.py`).
- EMD LSTM core pipeline is in `src/EMD_LSTM/emd_lstm.py` (EMD controls, logging, plotting, optional truncation).
- Baseline helpers in `src/baselines/` (`common.py` for data prep/logging, `darts_baselines.py` for Darts wrappers).

### Notes
- All plotting paths are optional; if provided, scripts create parent dirs.
- Encoding defaults to `None`; many AntAWS CSVs require `latin1`.
- For long EMD runs, use `--max-samples`/`--emd-max-sift` to keep runtime bounded. For baselines, limit `--max-windows` and prefer TFT/NBEATS for GPU usage.
