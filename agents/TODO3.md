
————————————————————————————————————————
TITLE: New Independent Experiment Suite (Wind-blown Snow / Sensor Scheduling / Imputation / Multi-Model Forecasting)

OWNER: Agent (codeX)
DATE: 2026-02-03

GOAL: Build a fully independent experiment suite (no shared modules) that:

1. evaluates multiple forecasting models (at least Informer, Transformer, LSTM + common baselines),
2. evaluates multiple missingness / sensor scheduling simulation algorithms,
3. evaluates multiple interpolation/imputation methods for each missingness strategy,
4. produces required visualizations (pre-train dataset characterization + post-train prediction plots + a large summary figure in the style of the provided reference image).

IMPORTANT:

* This suite must live in a NEW folder.
* Any reused code must be COPIED into the new folder. Do not import from the old project.
* No shared code dependencies except standard pip libs.

————————————————————————————————————————
SECTION 0. Inputs / Data Assumptions

0.1 Base data format
Support input CSV with at least these columns:

* timestamp (ISO-8601, may include timezone)
* meteorology: air_temperature_c, relative_humidity, air_pressure_pa, wind_speed_ms, wind_direction_deg, solar_radiation_wm2
* snow / blowing snow: snow_mass_flux_kg_m2_s, snow_number_flux_m2_s, optionally bin columns (size bins, velocity bins)
* optional quality flags: quality_flag, missing_reason, etc.

0.2 Frequency

* Data may be regular (e.g., 1s) or already “synthetic regular”.
* The suite must resample to a configured base frequency and produce a regular grid.

0.3 Task definition (forecast target & horizons)
Default target (configurable):

* wind_speed_ms OR snow_mass_flux_kg_m2_s
  Forecasting:
* Multi-step forecasting with horizons: 1-step, 2-step, 3-step (required)
* Rolling window supervised learning:

  * input length: lookback (configurable)
  * output length: horizon (1,2,3)

————————————————————————————————————————
SECTION 1. New Repo Folder Layout (NO SHARED CODE)

Create folder at repository root:

experiments_scheduling_suite/
README.txt
requirements.txt
configs/
base.yaml
datasets/
synthetic.yaml
real.yaml
missingness/
mcar.yaml
block.yaml
duty_cycle.yaml
round_robin.yaml
info_priority.yaml
imputation/
none_maskaware.yaml
linear.yaml
spline.yaml
kalman.yaml
gp.yaml
saits.yaml  (optional)
models/
lstm.yaml
transformer.yaml
informer.yaml
tcn.yaml
xgboost.yaml
naive.yaml
data/
raw/
generated/
processed/
src/
utils/
seed.py
io.py
time.py
metrics.py
plots/
style.py
pretrain_viz.py
per_model_forecast_plot.py
summary_figure.py
data/
generator/
synthetic_windblown.py     (copy/modify your generator here; do not reference old)
preprocessing/
resample.py
normalize.py
split.py
windowing.py
missingness/
base.py
mcar.py
block.py
duty_cycle.py
round_robin.py
info_priority.py
imputation/
base.py
linear.py
spline.py
kalman.py
gp.py
maskaware_features.py
models/
base.py
lstm.py
transformer.py
informer.py
tcn.py
baselines.py
xgboost.py
train/
trainer.py
callbacks.py
eval/
evaluate.py
aggregate.py
scripts/
00_generate_data.py
01_prepare_dataset.py
02_visualize_pretrain.py
03_train_models.py
04_evaluate.py
05_plot_predictions.py
06_plot_summary.py
07_run_sweep.py
reports/
RUN_ID/
config_used.yaml
tables/
figures/
preds/
logs/

RULE:

* No imports from outside experiments_scheduling_suite/src except installed packages.

————————————————————————————————————————
SECTION 2. Required Experiment Factors

2.1 Models (minimum required)
Must implement & run:

* Informer
* Transformer (encoder-decoder or encoder-only for forecasting)
* LSTM

Plus common baselines (at least 3):

* Naive persistence
* MLP
* XGBoost

Recommended additional:

* TCN

2.2 Missingness / Sensor scheduling simulation algorithms (minimum required)
Implement at least 4:

1. MCAR random missingness
2. Block missingness (contiguous outages)
3. Duty-cycle (periodic on/off per sensor)
4. Round-robin (budget-k sensors observed per time tick)

Recommended:
5) Info-priority (heuristic based on training correlation / proxy information)

Apply missingness to INPUT variables (sensors), not necessarily target unless configured.

2.3 Imputation / interpolation methods (minimum required)
For each missingness strategy, run:

* None + Mask-aware features (no interpolation; add missing indicators + time-since-last-seen)
* Linear interpolation
* Spline interpolation
* Kalman smoothing/filter

Optional:

* Gaussian Process interpolation
* Deep imputer (SAITS/BRITS) if time permits

————————————————————————————————————————
SECTION 3. Experiment Matrix

3.1 Default matrix (recommended first pass)
Models: {LSTM, Transformer, Informer, TCN, XGBoost, Naive}
Missingness: {MCAR, Block, DutyCycle, RoundRobin, InfoPriority}
Imputation: {MaskAware, Linear, Spline, Kalman}
Total runs: 6 x 5 x 4 = 120 (support --quick)

3.2 Quick mode (debug)
Models: {LSTM, Informer, Naive}
Missingness: {MCAR, RoundRobin}
Imputation: {MaskAware, Linear}
Total: 12

————————————————————————————————————————
SECTION 4. Core Pipeline Steps (modular, explicit)

STEP A. Data generation or ingestion
Script: scripts/00_generate_data.py

* mode=synthetic: generate CSV into data/generated/NAME.csv

  * copy existing synthetic generator code into src/data/generator/synthetic_windblown.py
  * generator must support configurable frequency (1s, 10s, 15s, 60s)
* mode=real: validate data/raw/*.csv and proceed

Outputs:

* data/generated/*.csv OR data/raw/*.csv

STEP B. Dataset preparation
Script: scripts/01_prepare_dataset.py
Order:

1. load + parse timestamp
2. resample to base frequency (regular grid)
3. apply missingness/scheduling to selected sensors
4. apply imputation/interpolation method
5. feature engineering (optional):

   * wind direction sin/cos
   * stability flags one-hot (optional)
6. normalize using train stats only
7. split time-based: train/val/test
8. windowing into supervised samples for horizons 1/2/3

Outputs per RUN_ID:

* data/processed/RUN_ID/dataset.npz:

  * X_train, y_train (h=1/2/3; either multi-output or separate arrays)
  * X_val, y_val
  * X_test, y_test
  * metadata.json (feature list, target, freq, lookback, etc.)
* reports/RUN_ID/tables/missingness_stats.csv
* reports/RUN_ID/tables/imputation_report.csv

STEP C. Pre-training visualization (dataset characterization)
Script: scripts/02_visualize_pretrain.py
For each missingness strategy x imputation, BEFORE training, output to:
reports/RUN_ID/figures/pretrain/

Required plots:

1. missingness heatmap (time x variables)
2. gap-length distribution (CCDF or histogram)
3. example time window overlays:

   * masked series
   * imputed series
   * (optional) original series if available
4. feature distributions (target + key drivers)

STEP D. Training (multiple models)
Script: scripts/03_train_models.py

* Train each model per RUN_ID (RUN_ID encodes dataset+missingness+imputer)
* Save:

  * checkpoints: reports/RUN_ID/models/MODEL.pt (or joblib for XGBoost)
  * predictions: reports/RUN_ID/preds/MODEL_h1.csv, MODEL_h2.csv, MODEL_h3.csv
  * logs: reports/RUN_ID/logs/MODEL.json (loss curves)

STEP E. Evaluation
Script: scripts/04_evaluate.py
Metrics per model and horizon:

* RMSE, MAE, MAPE (required)
* optional R2

Outputs:

* reports/RUN_ID/tables/metrics_overall.csv (model rows; metric@h1/h2/h3 cols)
* reports/RUN_ID/tables/metrics_long.csv (model x horizon x metric)

STEP F. Post-training prediction plots (per model)
Script: scripts/05_plot_predictions.py
For EACH model produce ONE figure containing:

* 1-step prediction vs actual (time segment)
* 2-step prediction vs actual
* 3-step prediction vs actual
* optional zoomed inset

Output:

* reports/RUN_ID/figures/preds/MODEL_pred_h123.png

STEP G. Large summary figure (style reference)
Script: scripts/06_plot_summary.py
For each dataset variant (missingness + imputer), aggregate across models and output a figure like the reference:

* Top: long test segment overlay with Actual + all model curves
* Include a zoom-in inset region (dashed box + zoom panel)
* Bottom: 3 radar charts (or equivalent polar plots):

  * 1-step metrics
  * 2-step metrics
  * 3-step metrics
    Radar metrics: RMSE, MAE, MAPE (normalize to comparable scales)

Output:

* reports/SWEEP_ID/figures/summary_MISSINGNESS_IMPUTER.png

STEP H. Sweep runner
Script: scripts/07_run_sweep.py

* enumerates model x missingness x imputation
* runs: prepare -> pretrain_viz -> train -> eval -> per-model plots -> summary plots
* writes sweep aggregation tables to reports/SWEEP_ID/tables/

————————————————————————————————————————
SECTION 5. Missingness / Scheduling Algorithms (requirements)

All algorithms mask INPUT sensor channels.
Config option:

* target_always_observed: true/false (default true)

5.1 MCAR
Params:

* p_missing global
* optional per-variable p

5.2 Block missingness
Params:

* n_blocks
* min_len_steps, max_len_steps
* per-variable blocks vs shared blocks

5.3 Duty-cycle
Params:

* period_steps
* on_steps
* random_phase true/false
  Fairness:
* optionally cap simultaneously observed sensors to budget_k; report effective_k

5.4 Round-robin (budget-k)
Params:

* budget_k
* min_on_steps
* sensor order list
  Behavior:
* rotate sensors; keep ON for at least min_on_steps

5.5 Info-priority (heuristic)
Params:

* budget_k, min_on_steps
* weight_method: train_corr (required)
* lag_steps: e.g. {0,1,2,4}
  Logic:
* compute weights from TRAIN ONLY (avoid leakage)
* select top-k subject to min_on_steps
  Optional:
* periodic refresh of weights

————————————————————————————————————————
SECTION 6. Imputation Methods (requirements)

Each imputer implements:

* fit(train_df)
* transform(df) -> df_imputed
* report() -> dict

6.1 Mask-aware (no interpolation)

* keep NaNs or fill with 0
* add is_missing_VAR and time_since_last_seen_VAR features

6.2 Linear interpolation

* per feature, time interpolation
* boundary fill by ffill/bfill (configurable)

6.3 Spline interpolation

* cubic spline per feature
* fallback to linear if not enough points

6.4 Kalman smoothing

* per-feature local level state-space
* optional multivariate if feasible

6.5 GP interpolation (optional)

* per-feature GP with RBF
* only for small subsets due to compute

————————————————————————————————————————
SECTION 7. Model Requirements

7.1 Neural models (LSTM/Transformer/Informer/TCN)
Common interface:

* fit(train_loader, val_loader)
* predict(test_loader)
  Must support:
* horizons 1/2/3 (multi-head or separate runs)
* deterministic seeding

7.2 Baselines

* Naive persistence: y_hat(t+h) = y(t)
* MLP: flattened windows
* XGBoost: flattened windows

————————————————————————————————————————
SECTION 8. Config & CLI

8.1 Config merge
base.yaml + dataset.yaml + missingness.yaml + imputation.yaml + model.yaml

Each run must save:

* reports/RUN_ID/config_used.yaml

8.2 RUN_ID naming
Encode: dataset, freq, missingness+params, imputation
Example:
synthetic_f1s_roundrobin_k2_minon2_linear

8.3 SWEEP outputs

* reports/SWEEP_ID/tables/leaderboard.csv (rank models by RMSE@h1/h2/h3)
* reports/SWEEP_ID/figures/summary_MISSINGNESS_IMPUTER.png

————————————————————————————————————————
SECTION 9. Output Artifacts (must exist)

Per RUN_ID:

* tables/missingness_stats.csv
* tables/imputation_report.csv
* tables/metrics_overall.csv
* figures/pretrain/ (4 required plots)
* figures/preds/MODEL_pred_h123.png for each model

Per SWEEP_ID:

* tables/leaderboard.csv
* figures/summary_MISSINGNESS_IMPUTER.png

————————————————————————————————————————
SECTION 10. Acceptance Criteria (Definition of Done)

* New folder experiments_scheduling_suite runs standalone (no old imports).
* At least 6 models: Informer, Transformer, LSTM + 3 baselines.
* At least 4 missingness/scheduling strategies + visibly different missingness heatmaps.
* At least 4 imputers per strategy.
* Pre-train visualizations exist for every dataset variant.
* Post-train plots per model exist (h=1/2/3 in one figure).
* A large summary figure exists per dataset variant:

  * overlay + inset zoom + 3 radar charts, style matches reference conceptually.
* Reproducibility: deterministic seed per RUN_ID + config snapshot.

————————————————————————————————————————
SECTION 11. Practical Notes

* Informer: use a lightweight PyTorch implementation copied into src/models/ OR implement minimal ProbSparse attention.
* Transformer forecasting: encoder-only with causal masking is acceptable.
* Training speed:

  * default small epochs for sweeps (10–20)
  * support max_windows / max_steps to limit runtime
* High-frequency: allow downsampling in debug mode.

END OF SPEC
————————————————————————————————————————
