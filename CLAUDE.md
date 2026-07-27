# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Microclimate forecasting and sensor scheduling research platform with two main components:

1. **AntAWS Pipeline** — time-series forecasting for Antarctic meteorological data using TFT (Temporal Fusion Transformer) and PINN (Physics-Informed Neural Networks)
2. **RL Sensor Scheduling Framework** (`rl_sensor_scheduling_framework/` submodule) — reinforcement learning for multi-sensor scheduling under power constraints, optimizing sensor activation to balance power consumption against downstream forecast quality

## Environment Setup

```bash
# Recommended
conda env create -f environment.yml
conda activate darts

# Or pip
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.0+, Darts (u8darts 0.36.0), Stable-Baselines3 2.3+, Gymnasium 0.29+, FastAPI 0.95+.

## Commands

### AntAWS Forecasting Pipeline

```bash
# Full pipeline
python scripts/run_all.py --config configs/exp.yaml

# Individual steps
python scripts/prepare_antaws_station.py --config configs/exp.yaml
python scripts/build_dataset.py --config configs/exp.yaml
python scripts/run_baselines.py --config configs/exp.yaml
python scripts/train_tft_pinn.py --config configs/exp.yaml --mode tft_pinn
python scripts/evaluate_models.py --config configs/exp.yaml
python scripts/make_figures.py --config configs/exp.yaml
```

### RL Sensor Scheduling (Route A — current mainline)

```bash
cd rl_sensor_scheduling_framework

# Full experiment via tmux
bash scripts/run_full_experiment_tmux.sh

# Individual steps (run in order)
python scripts/00_generate_business_data.py
python scripts/00b_pretrain_reward_predictor.py
python scripts/01_train_rl_scheduler.py
python scripts/02_evaluate_scheduler.py
python scripts/03_build_forecast_dataset.py
python scripts/04_train_predictors.py
python scripts/05_evaluate_forecasts.py
python scripts/06_posthoc_analysis.py
```

### Tests

```bash
cd rl_sensor_scheduling_framework
pytest tests/
```

## Architecture

### AntAWS Pipeline (`src/`)

- `src/data/` — ingestion (`antaws_loader.py`), resampling, imputation (strategies A & B), train/val/test splitting, windowed dataset creation, normalization
- `src/features/build_features.py` — rolling windows, differencing, time features, missing-aware features
- `src/models/` — `snow_tft.py` (TFT), `tcn.py` (TCN baseline)
- `src/train/` — `train_tft_pinn.py` (TFT + optional PINN physics constraints), `train_tcn.py`
- `src/losses/physics_losses.py` — soft penalty terms: variable bounds, Clausius-Clapeyron humidity, temporal coherence, vapor pressure smoothing
- `src/baselines/` — naive persistence/seasonal, ARIMA, GBRT via Darts
- `src/eval/` — RMSE/MAE/MAPE/Pearson/DTW metrics, extreme event analysis
- `configs/` — YAML configs drive all data paths, hyperparameters, loss weights, split ratios

### RL Sensor Scheduling (`rl_sensor_scheduling_framework/`)

Data flow: truth generation → reward predictor pretrain → scheduler train/eval → forecast dataset build → frozen predictor eval → posthoc analysis

Key components:
- **Environments** — high-frequency truth replay (`windblown_case`), 1 Hz sampling, 14-day sequences
- **Sensors** — multi-sensor definitions with power costs, refresh intervals, noise levels
- **Estimators** — Kalman filter for state estimation under partial observations
- **RL Agents** — DQN (discrete), subset-conditioned DQN (online feasible subsets), PPO baseline
- **Reward** — frozen pretrained forecast-predictor oracle + switching cost + coverage penalty + constraint violations

### Key Patterns

- **Config-driven**: YAML configs are the single source of truth for experiment parameters
- **Frozen oracle evaluation**: RL scheduler training uses a pretrained, frozen reward predictor — not end-to-end optimization
- **Chronological data splits**: predictor_pretrain → rl_train → rl_val → final_test (prevents distribution shift)
- **Feasible action space**: RL actions are sensor subsets that satisfy power/count constraints
- **Physics-informed losses**: meteorological constraints enforced as soft penalties, not hard constraints

### Submodules

- `rl_sensor_scheduling_framework/` — RL scheduling framework (independent git submodule)
- `experiments_scheduling_suite/` — scheduling/imputation/model sweep experiments (independent git submodule)

The legacy state-reward route is preserved on branch `legacy-state-reward-mainline`.
