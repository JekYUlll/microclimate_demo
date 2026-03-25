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
- Generate one shared high-frequency truth environment time series.
- Train/evaluate sensor scheduling policies under instantaneous power, startup-peak, and long-horizon energy constraints.
- Replay each scheduler on the same truth sequence to produce scheduler-specific estimated state datasets.
- Train common forecasting models on those scheduler-specific datasets.
- Compare forecast accuracy retention against the `full_open` oracle baseline under power savings.

Important RL design note:
- The RL scheduler is still **value-based**, not policy-gradient / actor-critic.
- Two RL styles now coexist:
  - `linear_gaussian`: classic discrete-action DQN over a pre-enumerated `DiscreteActionSpace`.
  - `windblown`: **subset-conditioned DQN / CMDP-DQN** plus an `OnlineSubsetProjector`; the network evaluates feasible sensor subsets online rather than relying on a static action-id table.
- Files:
  - `rl_sensor_scheduling_framework/src/scheduling/rl/dqn_agent.py` – legacy discrete-action DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/score_dqn_agent.py` – windblown subset-conditioned DQN and constrained subset-conditioned DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/constrained_dqn_agent.py` – CMDP dual-variable layer for the legacy discrete-action DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/q_network.py` – Q-networks, including `SubsetQNetwork` used by the current windblown RL path.
  - `rl_sensor_scheduling_framework/src/scheduling/online_projector.py` – online feasible-subset projector.
- No PPO / A2C / SAC / actor-critic implementation is currently used.

Current reward design is **Scheme A infrastructure with direct primary-target reward active by default**:
- RL is not trained end-to-end with downstream predictor retraining.
- The codebase still contains the frozen auxiliary forecast-reward oracle path (`00b_pretrain_reward_predictor.py`, `src/reward/forecast_reward.py`), but the current default windblown config sets `forecast_reward.enabled: false`.
- In the current default windblown runs, reward is driven mainly by:
  - direct primary-target state error (`beta_prediction`)
  - switch cost
  - coverage penalty
- In the current default windblown runs, these terms are effectively disabled or secondary:
  - estimation uncertainty (`alpha_estimation: 0.0`)
  - frozen auxiliary forecast reward (`beta_forecast: 0.0`)
  - direct power penalty in reward (`lambda_power: 0.0`)
- For `cmdp_dqn`, power is modeled mainly as a constraint rather than as a dominant reward term:
  - hard constraints: instantaneous steady-state power, startup peak power, safety margin
  - long-horizon constraints: average power and episode energy via dual variables / cost critic
- Core files:
  - `rl_sensor_scheduling_framework/src/evaluation/cost_metrics.py`
  - `rl_sensor_scheduling_framework/src/evaluation/constraint_metrics.py`
  - `rl_sensor_scheduling_framework/src/reward/forecast_reward.py`
  - `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py`

The frozen forecast-reward oracle is still trained **before** RL on a disjoint `reward_pretrain` split and then frozen during scheduler training / evaluation when enabled. This avoids joint predictor-scheduler bilevel training in the current paper-scale experiment, but note that the experiment driver still launches this pretrain step even when `forecast_reward.enabled: false`.

Core scripts:
- `rl_sensor_scheduling_framework/scripts/00_generate_business_data.py` – Generate the shared high-frequency truth CSV for the business case. Typical output: `rl_sensor_scheduling_framework/data/generated/windblown_truth.csv`.
- `rl_sensor_scheduling_framework/scripts/00b_pretrain_reward_predictor.py` – Train the frozen auxiliary reward predictor on the `reward_pretrain` split only and save `reward_predictor.pt`.
- `rl_sensor_scheduling_framework/scripts/01_train_rl_scheduler.py` – Train one scheduler on the truth environment. For rule-based schedulers, this computes repeated rollout metrics; for RL schedulers (`dqn`, `cmdp_dqn`), this performs value-based training and writes `scheduler_<name>.pt`.
- `rl_sensor_scheduling_framework/scripts/02_evaluate_scheduler.py` – Evaluate one scheduler on the held-out test split of the truth environment and write `metrics_estimation_eval.csv`.
- `rl_sensor_scheduling_framework/scripts/03_build_forecast_dataset.py` – Replay a trained/evaluated scheduler over the full truth sequence and export one scheduler-specific dataset NPZ containing `input_series` (estimated state), `target_series` (truth targets), `observed_mask`, `event_flags`, `power`, `trace_p`, and `feature_names`.
- `rl_sensor_scheduling_framework/scripts/04_train_predictors.py` – Split the scheduler-specific dataset into train/val/test windows, normalize with train statistics, train one predictor, and save `forecast_predictions.npz` + `metrics_forecast.csv`.
- `rl_sensor_scheduling_framework/scripts/04_train_predictors_multi_gpu.sh` – Parallel predictor launcher. Distributes learned predictor jobs across available GPUs; `naive` stays on CPU.
- `rl_sensor_scheduling_framework/scripts/05_evaluate_forecasts.py` – Aggregate predictor runs under one `run_tag`, build `metrics_forecast_all_<run_tag>.csv`, and compare every scheduler against `full_open`. Also backfills `sMAPE`, `Pearson`, and `DTW` from saved prediction artifacts when needed.
- `rl_sensor_scheduling_framework/scripts/06_posthoc_analysis.py` – Produce cross-scheduler heatmaps, rank correlation, Pareto-style power-vs-error plots, and scheduler summary tables.
- `rl_sensor_scheduling_framework/scripts/07_plot_scheduler_prediction_curves.py` – For a fixed predictor model, draw prediction-vs-truth curves across all schedulers for a chosen target variable and horizon.
- `rl_sensor_scheduling_framework/scripts/08_plot_sensor_activation_timelines.py` – Plot per-sensor on/off timelines together with target truth and power.
- `rl_sensor_scheduling_framework/scripts/09_generate_all_plots.py` – Generate the main prediction-curve and sensor-activation figures for either primary-task targets or a specific single target.
- `rl_sensor_scheduling_framework/scripts/10_posthoc_task_focus.py` – Produce task-focused summaries for the primary target set defined in the environment config.
- `rl_sensor_scheduling_framework/scripts/run_full_experiment_tmux.sh` – Non-tmux experiment driver despite the historical name. Runs: truth generation -> reward predictor pretrain -> scheduler train/eval -> dataset build -> multi-GPU predictor train -> aggregate eval -> posthoc -> primary-target plots -> target-specific plots.

Main configs:
- `rl_sensor_scheduling_framework/configs/base.yaml` – global seed, split ratios, run lengths, reward weights, and sensor budget constraints. Note: for windblown, `configs/env/windblown_case.yaml` is the authoritative source for primary reward / forecast targets; `base.yaml` may still contain stale generic reward-target fields.
- `rl_sensor_scheduling_framework/configs/env/windblown_case.yaml` – truth environment settings, state columns, primary reward targets, and forecast targets.
- `rl_sensor_scheduling_framework/configs/sensors/windblown_sensors.yaml` – sensor definitions, observed variables, and power / startup-peak costs.
- `rl_sensor_scheduling_framework/configs/estimator/kalman.yaml` – linear Gaussian estimator settings.
- `rl_sensor_scheduling_framework/configs/reward/lstm_aux.yaml` – frozen auxiliary reward-predictor config used by Scheme A.
- `rl_sensor_scheduling_framework/configs/scheduler/*.yaml` – `full_open`, `random`, `periodic`, `round_robin`, `info_priority`, `dqn`, `cmdp_dqn`.
- `rl_sensor_scheduling_framework/configs/predictor/*.yaml` – `naive`, `mlp`, `lstm`, `transformer`, `informer`, `tcn`, `pinn`, `sert_like`, `s4m_like`.

Core modules:
- `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py` – orchestration logic for scheduler training, evaluation, and dataset building.
- `rl_sensor_scheduling_framework/src/envs/truth_replay_env.py` – deterministic replay environment over the shared truth CSV, with train/val/test split ranges.
- `rl_sensor_scheduling_framework/src/sensors/dataset_sensor.py` – sensor wrapper that reads variables from the truth dataset and applies observation noise / availability.
- `rl_sensor_scheduling_framework/src/estimators/kalman_filter.py` – Kalman estimator used to maintain the belief state and uncertainty summary.
- `rl_sensor_scheduling_framework/src/estimators/state_summary.py` – flatten belief-state features into the RL state vector.
- `rl_sensor_scheduling_framework/src/scheduling/online_projector.py` – online feasible-subset selection under hard power constraints.
- `rl_sensor_scheduling_framework/src/forecasting/input_augmentation.py` – physical feature augmentation of estimator outputs.
- `rl_sensor_scheduling_framework/src/forecasting/series_preparation.py` – optional missing-aware feature enrichment.
- `rl_sensor_scheduling_framework/src/forecasting/*.py` – downstream forecasting models trained on scheduler-generated datasets.

Current windblown predictor inputs:
- Base estimator state columns:
  - `wind_speed_ms`
  - `wind_direction_deg`
  - `air_temperature_c`
  - `relative_humidity`
  - `air_pressure_pa`
  - `solar_radiation_wm2`
  - `snow_surface_temperature_c`
  - `snow_particle_mean_diameter_mm`
  - `snow_particle_mean_velocity_ms`
  - `snow_mass_flux_kg_m2_s`
- Default derived features:
  - `wind_dir_sin`
  - `wind_dir_cos`
  - `wind_u`
  - `wind_v`
  - `surface_air_temp_gap`
  - `particle_kinetic_proxy`
  - `size_velocity_interaction`
  - `transport_exceedance`
- Optional missing-aware extras for selected models:
  - `is_observed_*`
  - `delta_*`

Current task definition:
- Primary reward targets:
  - `air_temperature_c`
  - `snow_surface_temperature_c`
  - `wind_speed_ms`
- Forecast targets:
  - `air_temperature_c`
  - `snow_surface_temperature_c`
  - `wind_speed_ms`
  - `wind_dir_sin`
  - `wind_dir_cos`
  - `snow_mass_flux_kg_m2_s`
  - `snow_particle_mean_velocity_ms`
- `solar_radiation_wm2` remains part of the latent / observed state, but is no longer used as a primary forecast target because the current truth generator produces sparse impulsive radiation that is not forecastable with the present models.

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
- Actions:
  - `linear_gaussian`: discrete feasible subsets from `DiscreteActionSpace`
  - `windblown`: online feasible subsets produced by `OnlineSubsetProjector` from sensor scores / rankings
- Hard constraints are enforced in the action layer:
  - instantaneous steady-state power limit
  - startup / heating peak power limit
  - optional safety margin
- `dqn` uses `reward = -cost`; in the current default windblown setup this is dominated by direct primary-target state error plus switching and coverage, not by forecast-oracle loss or direct power penalty.
- `cmdp_dqn` uses the same value-based backbone, but average power / episode energy are handled by CMDP-style dual variables and a separate cost critic rather than direct reward maximization.

Data flow:
1. Truth generation: build one shared high-frequency latent/observed environment CSV.
2. Reward predictor pretrain:
   - use `reward_pretrain` split only;
   - train one frozen auxiliary predictor and save `reward_predictor.pt`;
   - this step is currently optional in principle, but the main driver still executes it unconditionally.
3. Scheduler training/eval:
   - rule-based schedulers use greedy rollout metrics;
   - `windblown` RL schedulers evaluate projector-feasible subsets online with `SubsetQNetwork`;
   - both stages may load the frozen reward oracle when that path is enabled.
4. Dataset build: replay each scheduler on the same truth sequence and export one NPZ per scheduler.
5. Forecast training: each predictor trains on one scheduler NPZ, using the estimated state as input and the configured forecast targets as output.
6. Aggregate evaluation: compare all scheduler-predictor combinations against `full_open`.
7. Posthoc visualization: heatmaps, rank correlation, RMSE/DTW/Pearson tradeoff plots, per-model scheduler curve overlays, sensor activation timelines, and primary-target summaries.

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
  - `rl_sensor_scheduling_framework/reports/aggregate/posthoc_<run_tag>/task_focus_primary/`

Current interpretation caveat:
- Scheme A is still **not** end-to-end joint optimization of scheduler and downstream forecaster.
- The current primary-task interpretation is "microclimate digital twin": results should be read first on the primary target set, not only on `snow_mass_flux_kg_m2_s`.
- `dRMSE` alone is insufficient; use `RMSE`, `MAE`, `sMAPE`, `Pearson`, and `DTW` together.
- Aggregate scheduler summaries over the full forecast target set and task-focused summaries over the primary target set are different views; do not conflate them.
- Scheduler evaluation should always be separated into:
  - estimation-level metrics (`trace_P_mean`, `power_mean`, `coverage_mean`)
  - forecasting-level metrics (`rmse`, `mae`, `smape`, `pearson_h1_mean`, `dtw_h1_mean`, comparison vs `full_open`)

Known inconsistencies / watchpoints:
- `rl_sensor_scheduling_framework/configs/base.yaml` still contains generic reward-target fields that may disagree with `rl_sensor_scheduling_framework/configs/env/windblown_case.yaml`; for windblown experiments, the environment config is the authoritative source.
- `rl_sensor_scheduling_framework/scripts/run_full_experiment_tmux.sh` still launches `00b_pretrain_reward_predictor.py` even when `forecast_reward.enabled: false`; this adds runtime and can confuse result interpretation.
- The frozen forecast-reward oracle code path remains in active source, but it is currently disabled by default; if re-enabled, it should be revalidated before being treated as a trustworthy training signal.
- `solar_radiation_wm2` remains in the latent / observed state even though it was removed from the forecast target set; if it continues to degrade estimator or predictor behavior, it should be either smoothed at generation time or removed from the predictor input path.

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
