## Overview

This repository now contains a minimal-yet-extensible pipeline for training an LSTM
time-series model on the meteorological observations delivered as Excel files in
`data/`. The code paths are organised as follows:

- `src/config.py` – reusable dataclasses that describe paths, data schema, and
  training hyper-parameters.
- `src/data.py` – utilities to ingest the Excel files, clean the columns, and turn
  the long series into overlapping windows for model training.
- `src/model.py` – implementation of a stacked LSTM forecaster baseline.
- `scripts/prepare_data.py` – command-line script that converts the raw Excel files
  into a single processed CSV under `data/processed/`.
- `scripts/train_lstm.py` – reference training entry point that handles
  splitting/normalisation, trains the model, and saves checkpoints in
  `models/checkpoints/`.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include `pandas`, `numpy`, `torch`, and `openpyxl` (needed for
reading `.xlsx` files).

## Typical workflow

1. Place the Excel files in `data/` (already present in this repository).
2. Generate a cleaned CSV (resampled every 6 hours by default):
   ```bash
   python scripts/prepare_data.py
   ```
3. Train the baseline LSTM (use `--station <name>` to pick a single station):
   ```bash
   python scripts/train_lstm.py --epochs 30 --station Greatwall-MeteorologicalObservation-1985-2022
   ```

Model checkpoints and normalisation metadata are stored at
`models/checkpoints/lstm_<station>.pt`. You can adjust the resampling frequency,
window size, or other hyper-parameters via the dataclasses in `src/config.py` or
the CLI overrides (see `--help` for each script).

### Evaluate & plot

After training, run:

```bash
python scripts/evaluate_lstm.py \
  --checkpoint models/checkpoints/lstm_all.pt \
  --plot plots/lstm_holdout.png \
  --station Greatwall-MeteorologicalObservation-1985-2022
```

The script reloads the model, computes MAE/RMSE on the hold-out windows, and
produces an actual-vs-predicted plot (first forecast step) saved under `plots/`.
Use `--max-points` to limit the number of points drawn when the hold-out set is
large. If Matplotlib warns about missing Chinese glyphs, supply a font that
contains them, e.g.

```bash
python scripts/evaluate_lstm.py \
  --font-file /usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc
```

or point to any `.ttf/.otf` file available on your system.

## AntAWS demo pipeline (TFT + PINN)

The AntAWS Q2 demo is configured via `configs/exp.yaml` and produces all tables
and figures under `reports/<exp_id>/`.

Run the full pipeline in one command:

```bash
python scripts/run_all.py --config configs/exp.yaml
```

Key intermediate steps if you want to run them manually:

```bash
python scripts/prepare_antaws_station.py --config configs/exp.yaml
python scripts/profile_missingness.py --data data/processed/<exp_id>/<station_id>.csv --config configs/exp.yaml
python scripts/build_dataset.py --config configs/exp.yaml
python scripts/run_baselines.py --config configs/exp.yaml
python scripts/train_tft_pinn.py --config configs/exp.yaml --mode tft_pinn
python scripts/evaluate_models.py --config configs/exp.yaml
python scripts/make_figures.py --config configs/exp.yaml
```

Outputs include:

- `reports/<exp_id>/tables/metrics_overall.csv` and per-horizon tables
- `reports/<exp_id>/preds/*.csv` for unified model predictions
- `reports/<exp_id>/figures/*.png` for missingness and evaluation figures

## Physics constraints (AntAWS, TFT + PINN)

### 中文说明
当前 `src/losses/physics_losses.py` 中包含的物理拟合/约束如下（均为软惩罚项）：
- **数值边界约束**：`wind_speed_ms >= 0`；`relative_humidity_pct` 限制在 `[0, 100]`。
- **Clausius–Clapeyron 湿度关系**：用温度 `T` 计算饱和水汽压 `e_s(T)`，并约束水汽压不超饱和也不为负（若目标包含 `temperature_c/air_temperature_c` 与 `relative_humidity_pct`）。
- **时间一致性约束**：对预测序列的相邻步差分做 Smooth L1，鼓励物理量随时间平滑变化。
- **水汽压平滑**（可选）：对 `e = RH * e_s(T)` 的时间差分做 Smooth L1，减少水汽压剧烈跳变。

对应开关与权重位于 `configs/exp.yaml` 的 `loss.*` 中（如 `lambda_phys`、`use_bounds_term`、`use_coherence_term`、`use_clausius_clapeyron_term`、`use_vapor_pressure_smooth`）。

### English
The current physics constraints in `src/losses/physics_losses.py` (soft penalties) are:
- **Bounds term**: `wind_speed_ms >= 0` and `relative_humidity_pct` within `[0, 100]`.
- **Clausius–Clapeyron humidity relation**: compute saturation vapor pressure `e_s(T)` from temperature and penalize supersaturation/negative vapor pressure (when both temperature and RH are targets).
- **Temporal coherence**: Smooth L1 on adjacent-step differences to encourage smooth dynamics.
- **Vapor-pressure smoothing** (optional): Smooth L1 on temporal differences of `e = RH * e_s(T)`.

Toggles and weights live under `loss.*` in `configs/exp.yaml` (e.g., `lambda_phys`, `use_bounds_term`, `use_coherence_term`, `use_clausius_clapeyron_term`, `use_vapor_pressure_smooth`).

---

# 2026/01/17

更新为AntAWS完整demo实验。
