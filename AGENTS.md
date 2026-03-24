## Project Scripts Overview

This file summarizes runnable scripts under `scripts/` (and `scripts/baselines/`) plus the independent scheduling suite in `experiments_scheduling_suite/`.

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

### Independent experiment suite (`experiments_scheduling_suite/`)
Standalone scheduling/imputation/model sweep with no imports from the root project.

Core scripts:
- `experiments_scheduling_suite/scripts/00_generate_data.py` – Generate synthetic or validate real CSV input.
- `experiments_scheduling_suite/scripts/01_prepare_dataset.py` – Resample, apply missingness, impute, normalize, split, window into `dataset.npz`.
- `experiments_scheduling_suite/scripts/02_visualize_pretrain.py` – Pre-train dataset characterization plots.
- `experiments_scheduling_suite/scripts/03_train_models.py` – Train models and save per-horizon predictions.
- `experiments_scheduling_suite/scripts/04_evaluate.py` – Compute RMSE/MAE/MAPE tables.
- `experiments_scheduling_suite/scripts/05_plot_predictions.py` – Per-model prediction plots (H=1/2/3).
- `experiments_scheduling_suite/scripts/06_plot_summary.py` – Summary figure (overlay + zoom + radar).
- `experiments_scheduling_suite/scripts/07_run_sweep.py` – Sweep runner (prepare -> viz -> train -> eval -> plots).
- `experiments_scheduling_suite/scripts/run_full_sweep.sh` – Shell script for full combinatorial sweep.

Posthoc analysis:
- `experiments_scheduling_suite/scripts/08_collect_results.py` – Collect metrics + metadata into `_aggregate` tables.
- `experiments_scheduling_suite/scripts/09_plot_cross_strategy.py` – Cross-strategy plots.
- `experiments_scheduling_suite/scripts/10_event_based_eval.py` – Event vs non-event error analysis.
- `experiments_scheduling_suite/scripts/11_significance_tests.py` – Statistical tests.
- `experiments_scheduling_suite/scripts/12_make_posthoc_report.py` – Aggregate report generation.
- `experiments_scheduling_suite/scripts/13_plot_strategy_predictions.py` – Strategy-level prediction overlays.

Configs:
- `experiments_scheduling_suite/configs/base.yaml` – Base run/training settings.
- `experiments_scheduling_suite/configs/datasets/*.yaml` – Synthetic vs real datasets.
- `experiments_scheduling_suite/configs/missingness/*.yaml` – MCAR/block/duty_cycle/round_robin/info_priority + variants.
- `experiments_scheduling_suite/configs/imputation/*.yaml` – Mask-aware/linear/spline/kalman/gp.
- `experiments_scheduling_suite/configs/models/*.yaml` – LSTM/Transformer/Informer/TCN/MLP/XGBoost/Naive.

### RL sensor scheduling framework (`rl_sensor_scheduling_framework/`)
Standalone submodule-style experiment for power-constrained multi-sensor scheduling, state estimation, and downstream forecasting. This framework no longer shares the old `experiments_scheduling_suite/` pipeline; it has its own configs, scripts, reports, and data flow.

Experiment objective:
- Generate one shared high-frequency "truth" environment time series.
- Train/evaluate sensor scheduling policies under per-step power and activation constraints.
- Replay each scheduler on the same truth sequence to produce scheduler-specific estimated state datasets.
- Train common forecasting models on those scheduler-specific datasets.
- Compare forecast accuracy retention against the `full_open` baseline under power savings.

Important RL design note:
- The RL scheduler is **value-based**, not policy-gradient / actor-critic.
- Implemented algorithm: **DQN** with replay buffer, target network, epsilon-greedy exploration, and one-step TD targets.
- Files:
  - `rl_sensor_scheduling_framework/src/scheduling/rl/dqn_agent.py` – DQN agent.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/q_network.py` – Q-network.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/replay_buffer.py` – replay buffer.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/epsilon_scheduler.py` – epsilon schedule.
- No PPO / A2C / SAC / actor-critic implementation is currently used.
- Current reward design is **Scheme A**:
  - RL is still trained with a hand-crafted step cost, not end-to-end downstream predictor retraining.
  - Reward now mixes:
    - normalized / relevance-weighted estimation uncertainty
    - target-aligned state error
    - a **frozen auxiliary forecast-reward oracle**
    - power cost
    - switch cost
    - coverage penalty
  - Core files:
    - `rl_sensor_scheduling_framework/src/evaluation/cost_metrics.py`
    - `rl_sensor_scheduling_framework/src/reward/forecast_reward.py`
    - `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py`
- The frozen forecast-reward oracle is trained **before** RL on a disjoint `reward_pretrain` split and then frozen during scheduler training / evaluation. This avoids joint predictor-scheduler bilevel training in the current paper-scale experiment.

Core scripts:
- `rl_sensor_scheduling_framework/scripts/00_generate_business_data.py` – Generate the shared high-frequency truth CSV for the business case (currently windblown snow / meteorology style data). Typical output: `rl_sensor_scheduling_framework/data/generated/windblown_truth.csv`.
- `rl_sensor_scheduling_framework/scripts/00b_pretrain_reward_predictor.py` – Train the frozen auxiliary reward predictor on the `reward_pretrain` split only and save `reward_predictor.pt`.
- `rl_sensor_scheduling_framework/scripts/01_train_rl_scheduler.py` – Train one scheduler on the truth environment. For rule-based schedulers, this computes repeated rollout metrics; for `dqn`, this performs RL training and writes `scheduler_dqn.pt`.
- `rl_sensor_scheduling_framework/scripts/02_evaluate_scheduler.py` – Evaluate one scheduler on the held-out test split of the truth environment and write `metrics_estimation_eval.csv`.
- `rl_sensor_scheduling_framework/scripts/03_build_forecast_dataset.py` – Replay a trained/evaluated scheduler over the full truth sequence and export one scheduler-specific dataset NPZ containing `input_series` (estimated state), `target_series` (truth state), `observed_mask`, `event_flags`, `power`, `trace_p`, and `feature_names`.
- `rl_sensor_scheduling_framework/scripts/04_train_predictors.py` – Split the scheduler-specific dataset into train/val/test windows, normalize with train statistics, train one predictor, and save `forecast_predictions.npz` + `metrics_forecast.csv`.
- `rl_sensor_scheduling_framework/scripts/04_train_predictors_multi_gpu.sh` – Parallel predictor launcher. Distributes learned predictor jobs across available GPUs; `naive` stays on CPU.
- `rl_sensor_scheduling_framework/scripts/05_evaluate_forecasts.py` – Aggregate predictor runs under one `run_tag`, build `metrics_forecast_all_<run_tag>.csv`, and compare every scheduler against `full_open`.
- `rl_sensor_scheduling_framework/scripts/06_posthoc_analysis.py` – Produce cross-scheduler heatmaps, rank correlation, Pareto-style power-vs-error plots, and scheduler summary tables.
- `rl_sensor_scheduling_framework/scripts/07_plot_scheduler_prediction_curves.py` – For a fixed predictor model, draw prediction-vs-truth curves across all schedulers for a chosen target variable and horizon.
- `rl_sensor_scheduling_framework/scripts/run_full_experiment_tmux.sh` – Non-tmux experiment driver despite the historical name. Runs the full pipeline: truth generation -> reward predictor pretrain -> scheduler train/eval -> dataset build -> multi-GPU predictor train -> aggregate eval -> posthoc.

Main configs:
- `rl_sensor_scheduling_framework/configs/base.yaml` – global seed, split ratios, run lengths, cost weights, and sensor budget constraints.
- `rl_sensor_scheduling_framework/configs/env/windblown_case.yaml` – truth environment settings and state columns.
- `rl_sensor_scheduling_framework/configs/sensors/windblown_sensors.yaml` – sensor definitions, observed variables, and power costs.
- `rl_sensor_scheduling_framework/configs/estimator/kalman.yaml` – linear Gaussian estimator settings.
- `rl_sensor_scheduling_framework/configs/reward/lstm_aux.yaml` – frozen auxiliary reward-predictor config used by Scheme A.
- `rl_sensor_scheduling_framework/configs/scheduler/*.yaml` – `full_open`, `random`, `periodic`, `round_robin`, `info_priority`, `dqn`.
- `rl_sensor_scheduling_framework/configs/predictor/*.yaml` – `naive`, `mlp`, `lstm`, `transformer`, `informer`, `tcn`.

Core modules:
- `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py` – main orchestration logic for scheduler training, evaluation, and dataset building.
- `rl_sensor_scheduling_framework/src/envs/truth_replay_env.py` – deterministic replay environment over the shared truth CSV, with train/val/test split ranges.
- `rl_sensor_scheduling_framework/src/sensors/dataset_sensor.py` – sensor wrapper that reads variables from the truth dataset and applies observation noise / availability.
- `rl_sensor_scheduling_framework/src/estimators/kalman_filter.py` – Kalman estimator used to maintain the belief state and uncertainty summary.
- `rl_sensor_scheduling_framework/src/estimators/state_summary.py` – flatten belief-state features into the RL state vector.
- `rl_sensor_scheduling_framework/src/scheduling/action_space.py` – enumerate feasible discrete sensor subsets under `max_active` and `per_step_budget`.
- `rl_sensor_scheduling_framework/src/forecasting/*.py` – downstream forecasting models trained on scheduler-generated datasets.

RL state / action / reward in this framework:
- State includes:
  - current Kalman state estimate `x_hat`
  - covariance diagonal `diag_P`
  - total uncertainty `trace_P`
  - per-sensor freshness
  - per-sensor coverage ratio
  - budget ratio
  - previous action mask
  - event indicator
- Actions are discrete feasible subsets of sensors, generated by `DiscreteActionSpace`.
- Reward is `-cost`, where cost currently combines:
  - normalized / relevance-weighted uncertainty
  - target-aligned state error
  - frozen-forecast-oracle loss
  - power cost
  - switching cost
  - low-coverage penalty

Data flow:
1. Truth generation: build one shared high-frequency latent/observed environment CSV.
2. Reward predictor pretrain:
   - use `reward_pretrain` split only;
   - train one frozen auxiliary predictor and save `reward_predictor.pt`.
3. Scheduler training/eval:
   - rule-based schedulers use greedy rollout metrics;
   - DQN learns on the RL-train split and is evaluated greedily on the RL-test split;
   - both stages may load the frozen reward oracle.
4. Dataset build: replay each scheduler on the same truth sequence and export one NPZ per scheduler.
5. Forecast training: each predictor trains on one scheduler NPZ, using the estimated state as input and the truth state as target.
6. Aggregate evaluation: compare all scheduler-predictor combinations against `full_open`.
7. Posthoc visualization: heatmaps, rank correlation, power-vs-error tradeoff, and per-model scheduler curve overlays.

Typical outputs:
- Truth CSV: `rl_sensor_scheduling_framework/data/generated/*.csv`
- Scheduler run directories: `rl_sensor_scheduling_framework/reports/runs/<run_tag>_<scheduler>/`
- Predictor run directories: `rl_sensor_scheduling_framework/reports/runs/<run_tag>_<scheduler>_pred_<model>/`
- Aggregate tables:
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>.csv`
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>_comparison.csv`
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>_scheduler_summary.csv`
- Posthoc directory:
  - `rl_sensor_scheduling_framework/reports/aggregate/posthoc_<run_tag>/`

Current interpretation caveat:
- Scheme A is still **not** end-to-end joint optimization of scheduler and downstream forecaster.
- In practice, `forecast_reward` is an auxiliary term; if its scale is too small relative to the uncertainty term, DQN can still optimize for low uncertainty while underperforming on downstream forecasting.
- Current `full_schemeA_v2` results indicate:
  - `round_robin` is the strongest robust baseline on learned predictors, with about `45.7%` power saving and only about `1.0%` mean RMSE increase vs `full_open`;
  - `info_priority` is the next strongest baseline, with about `45.7%` power saving and about `4.6%` mean RMSE increase;
  - `dqn` still trails the rule baselines, with about `54.5%` power saving but about `44.4%` mean RMSE increase, despite much lower estimation uncertainty than the rule baselines.
- Therefore scheduler evaluation should always be separated into:
  - estimation-level metrics (`trace_P_mean`, `power_mean`, `coverage_mean`)
  - forecasting-level metrics (`rmse`, `mae`, comparison vs `full_open`)

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
- Wind-blown snow sandbox assets now live under `legacy/windblown_snow/` (not used by current pipeline):
  - `legacy/windblown_snow/src/` (legacy SnowTFT, dataset, physics helpers)
  - `legacy/windblown_snow/scripts/generate_fake_snow_data.py`
  - `legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv`
  - `legacy/windblown_snow/docs/windblown_snow_data_spec.md`

### Notes
- All plotting paths create parent directories when needed.
- Encoding defaults to `None`; AntAWS CSVs often require `latin1`.
- For faster runs, reduce `models.*.epochs` and set `max_windows` in `configs/exp.yaml`.
