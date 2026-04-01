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

Current research objective:
- Use reinforcement learning to decide which sensors should be active at each time step under power constraints.
- Optimize sensor scheduling for **future forecasting quality**, not for instantaneous estimation quality alone.
- Evaluate the scheduler in a microclimate digital-twin setting where downstream control-relevant variables must remain predictable even when only a subset of sensors can be powered.

Development lineage and branch preservation:
- Early windblown experiments used a reward dominated by Kalman/state-tracking error, switching cost, and coverage penalty. That route learned a scheduler first and only trained the downstream forecasters afterwards.
- That older route still matters as a historical working baseline and was preserved explicitly on branch `legacy-state-reward-mainline` at commit `761cdee`.
- The current mainline changed because the paper objective is downstream forecasting over future horizons. Instantaneous belief-state error was judged misaligned with that objective.
- The current mainline therefore uses **frozen forecast predictors as reward oracles**: pretrain on a disjoint split, freeze them, and use their future prediction loss as the scheduler training signal.
- This is still not end-to-end bilevel optimization. The scheduler does not retrain the predictor online.

Current mainline design (Route A, active as of 2026-04-01):
- Truth data remain high-frequency at `1 Hz`.
- Shared truth length is now `14 days` (`1209600` steps) so the predictors and RL policies see multiple complete diurnal cycles.
- The truth sequence is split into four disjoint chronological stages:
  - `predictor_pretrain`: `423360` steps (`4.9` days)
  - `rl_train`: `604800` steps (`7.0` days)
  - `rl_val`: `90720` steps (`1.05` days)
  - `final_test`: `90720` steps (`1.05` days)
- Predictor-pretrain and RL-train are intentionally much larger than validation and final test.
- Route A also adds explicit diurnal phase features (`time_of_day_sin`, `time_of_day_cos`) to both the forecast-predictor input path and the RL state summary.

Current experiment objective in practical terms:
- Generate one shared truth environment sequence.
- Pretrain a frozen ensemble of reward predictors on the earliest disjoint split.
- Train each scheduler on `rl_train`, select/monitor on `rl_val`, and evaluate on `final_test`.
- Replay each trained scheduler on the same truth sequence to generate scheduler-specific estimated-state datasets.
- Use frozen-oracle evaluation to compare schedulers against `full_open` on common forecast targets.
- Interpret results first on the microclimate control-relevant targets, not only on `snow_mass_flux_kg_m2_s`.

Important RL design note:
- The main RL backbone is still **value-based**, not actor-critic.
- Two RL styles coexist:
  - `linear_gaussian`: classic discrete-action DQN over a pre-enumerated `DiscreteActionSpace`.
  - `windblown`: **subset-conditioned DQN / CMDP-DQN** plus an `OnlineSubsetProjector`; the network scores feasible sensor subsets online rather than relying on a static action-id table.
- Files:
  - `rl_sensor_scheduling_framework/src/scheduling/rl/dqn_agent.py` – legacy discrete-action DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/score_dqn_agent.py` – windblown subset-conditioned DQN and CMDP-style subset-conditioned DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/constrained_dqn_agent.py` – CMDP dual-variable layer for the legacy discrete-action DQN.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/sb3_ppo.py` – Stable-Baselines3 PPO baseline wrapped around the windblown online-subset projector.
  - `rl_sensor_scheduling_framework/src/scheduling/rl/q_network.py` – Q-networks, including `SubsetQNetwork` used by the current windblown RL path.
  - `rl_sensor_scheduling_framework/src/scheduling/online_projector.py` – online feasible-subset projector.
- External RL baseline currently included:
  - `ppo` via Stable-Baselines3; action = continuous per-sensor scores, execution = projector-feasible subset.
- No A2C / SAC / actor-critic implementation beyond the PPO baseline is currently used.

Current reward design (forecast-oracle mainline):
- The default windblown mainline now enables `forecast_reward.enabled: true`.
- The scheduler reward is driven mainly by:
  - frozen future forecast loss (`lambda_forecast`)
  - switching cost (`lambda_switch`)
  - coverage penalty (`lambda_coverage`)
  - hard-constraint violation penalty (`lambda_violation`)
- The direct state-tracking term is still present in code for backward compatibility but is disabled in the current mainline (`lambda_state_tracking: 0.0`).
- For `cmdp_dqn`, power is modeled mainly as a constraint rather than as a reward-maximization target:
  - hard constraints: instantaneous steady-state power, startup peak power, safety margin
  - long-horizon constraints: average power and episode energy via dual variables / cost critic
- Core files:
  - `rl_sensor_scheduling_framework/src/reward/mainline_reward.py`
  - `rl_sensor_scheduling_framework/src/reward/forecast_reward.py`
  - `rl_sensor_scheduling_framework/src/evaluation/cost_metrics.py`
  - `rl_sensor_scheduling_framework/src/evaluation/constraint_metrics.py`
  - `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py`

Frozen forecast-reward oracle details:
- The reward oracle is trained **before** RL on the disjoint `predictor_pretrain` split and then frozen during scheduler training and evaluation.
- The active oracle config is `rl_sensor_scheduling_framework/configs/reward/lstm_aux.yaml`.
- It currently trains a three-model ensemble:
  - `tcn_reward`
  - `lstm_reward`
  - `transformer_reward`
- Current oracle settings of note:
  - `lookback: 20`
  - `horizon: 3`
  - `horizon_weights: [1.0, 0.8, 0.6]`
  - `loss: huber`
  - `pretrain_schedulers: [full_open, periodic, round_robin, info_priority, random]`
  - richer pretrain rollouts via constant feasible subsets plus random subset switching
- The oracle now consumes time indices so the input augmentation path can derive diurnal phase features consistently.

Core scripts:
- `rl_sensor_scheduling_framework/scripts/00_generate_business_data.py` – Generate the shared high-frequency truth CSV for the business case. Current Route A default uses `truth_steps=1209600`.
- `rl_sensor_scheduling_framework/scripts/00b_pretrain_reward_predictor.py` – Train the frozen reward-predictor ensemble on the `predictor_pretrain` split only and save the oracle artifacts.
- `rl_sensor_scheduling_framework/scripts/01_train_rl_scheduler.py` – Train one scheduler on the truth environment. For rule-based schedulers, this computes repeated rollout metrics; for RL schedulers (`dqn`, `cmdp_dqn`, `ppo`), this performs learning and writes the scheduler checkpoint (`.pt` for DQN family, `.zip` for PPO).
- `rl_sensor_scheduling_framework/scripts/02_evaluate_scheduler.py` – Evaluate one scheduler on the held-out truth split and write `metrics_estimation_eval.csv`.
- `rl_sensor_scheduling_framework/scripts/03_build_forecast_dataset.py` – Replay a trained/evaluated scheduler over the truth sequence and export one scheduler-specific dataset NPZ containing `input_series` (estimated state), `target_series` (truth targets), `observed_mask`, `event_flags`, `power`, `trace_p`, `feature_names`, and now also `time_index`.
- `rl_sensor_scheduling_framework/scripts/04_train_predictors.py` – Split the scheduler-specific dataset into train/val/test windows, normalize with train statistics, train one predictor, and save `forecast_predictions.npz` + `metrics_forecast.csv`.
- `rl_sensor_scheduling_framework/scripts/04_eval_frozen_predictors.py` – Evaluate the frozen reward-predictor family directly on scheduler datasets using the shared input-preparation path.
- `rl_sensor_scheduling_framework/scripts/04_eval_frozen_predictors_multi_gpu.sh` – Parallel frozen-predictor launcher; now also supports CPU-only mode when GPUs are occupied by other users.
- `rl_sensor_scheduling_framework/scripts/05_evaluate_forecasts.py` – Aggregate predictor runs under one `run_tag`, build `metrics_forecast_all_<run_tag>.csv`, and compare every scheduler against `full_open`. Also backfills `sMAPE`, `Pearson`, and `DTW` from saved prediction artifacts when needed.
- `rl_sensor_scheduling_framework/scripts/06_posthoc_analysis.py` – Produce cross-scheduler heatmaps, rank correlation, Pareto-style power-vs-error plots, and scheduler summary tables.
- `rl_sensor_scheduling_framework/scripts/07_plot_scheduler_prediction_curves.py` – For a fixed predictor model, draw prediction-vs-truth curves across all schedulers for a chosen target variable and horizon.
- `rl_sensor_scheduling_framework/scripts/08_plot_sensor_activation_timelines.py` – Plot per-sensor on/off timelines together with target truth and power.
- `rl_sensor_scheduling_framework/scripts/09_generate_all_plots.py` – Generate the main prediction-curve and sensor-activation figures for either primary-task targets or a specific single target.
- `rl_sensor_scheduling_framework/scripts/10_posthoc_task_focus.py` – Produce task-focused summaries for the primary target set defined in the environment config.
- `rl_sensor_scheduling_framework/scripts/run_full_experiment_tmux.sh` – Main experiment driver. Despite the historical name, it is often launched inside tmux manually. Current Route A flow: truth generation -> frozen reward predictor pretrain -> scheduler train/eval -> dataset build -> frozen predictor eval / aggregate eval -> posthoc -> task-focused plots.

Main configs:
- `rl_sensor_scheduling_framework/configs/base.yaml` – global seed, Route A split ratios, run lengths, reward weights, and sensor budget constraints.
- `rl_sensor_scheduling_framework/configs/env/windblown_case.yaml` – truth environment settings, state columns, primary reward targets, and forecast targets.
- `rl_sensor_scheduling_framework/configs/sensors/windblown_sensors.yaml` – sensor definitions, observed variables, and power / startup-peak costs.
- `rl_sensor_scheduling_framework/configs/estimator/kalman.yaml` – linear Gaussian estimator settings.
- `rl_sensor_scheduling_framework/configs/reward/lstm_aux.yaml` – frozen reward-predictor ensemble config used by the active mainline.
- `rl_sensor_scheduling_framework/configs/scheduler/*.yaml` – `full_open`, `random`, `periodic`, `round_robin`, `info_priority`, `dqn`, `cmdp_dqn`, `ppo`.
- `rl_sensor_scheduling_framework/configs/predictor/*.yaml` – `naive`, `mlp`, `lstm`, `transformer`, `informer`, `tcn`, `pinn`, `sert_like`, `s4m_like`, plus the `*_reward.yaml` configs used for the frozen oracle.

Core modules:
- `rl_sensor_scheduling_framework/src/pipelines/truth_pipeline.py` – orchestration logic for scheduler training, evaluation, reward-oracle pretraining, and dataset building.
- `rl_sensor_scheduling_framework/src/envs/truth_replay_env.py` – deterministic replay environment over the shared truth CSV, with split-aware ranges and absolute time indices.
- `rl_sensor_scheduling_framework/src/sensors/dataset_sensor.py` – sensor wrapper that reads variables from the truth dataset and applies observation noise / availability.
- `rl_sensor_scheduling_framework/src/estimators/kalman_filter.py` – Kalman estimator used to maintain the belief state and uncertainty summary.
- `rl_sensor_scheduling_framework/src/estimators/state_summary.py` – flatten belief-state features into the RL state vector; now includes diurnal phase features when provided.
- `rl_sensor_scheduling_framework/src/scheduling/online_projector.py` – online feasible-subset selection under hard power constraints.
- `rl_sensor_scheduling_framework/src/forecasting/input_augmentation.py` – physical feature augmentation of estimator outputs, now also handling time-of-day features when time indices are present.
- `rl_sensor_scheduling_framework/src/forecasting/series_preparation.py` – common preparation of input and target arrays for both predictor training and reward-oracle scoring.
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
  - `time_of_day_sin`
  - `time_of_day_cos`
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
  - diurnal phase (`time_of_day_sin`, `time_of_day_cos`)
- Actions:
  - `linear_gaussian`: discrete feasible subsets from `DiscreteActionSpace`
  - `windblown`: online feasible subsets produced by `OnlineSubsetProjector` from sensor scores / rankings
- Hard constraints are enforced in the action layer:
  - instantaneous steady-state power limit
  - startup / heating peak power limit
  - optional safety margin
- The current windblown mainline optimizes reward built from future forecast loss plus switching, coverage, and constraint-violation terms.
- `cmdp_dqn` uses the same value-based backbone, but average power / episode energy are handled by CMDP-style dual variables and a separate cost critic rather than by direct reward maximization.

Data flow:
1. Truth generation: build one shared high-frequency latent/observed environment CSV.
2. Reward predictor pretrain:
   - use `predictor_pretrain` split only;
   - train the frozen reward-predictor ensemble and save the oracle artifacts;
   - pretraining uses a mixture of `full_open`, rule-based schedulers, constant feasible subsets, and random subset-switching rollouts.
3. Scheduler training/eval:
   - rule-based schedulers use greedy rollout metrics;
   - `windblown` RL schedulers evaluate projector-feasible subsets online with `SubsetQNetwork` or PPO;
   - the active mainline uses the frozen reward oracle during this stage.
4. Dataset build: replay each scheduler on the same truth sequence and export one NPZ per scheduler, including the scheduler-specific estimated state and the absolute `time_index`.
5. Forecast / frozen-oracle evaluation: compare scheduler datasets under a common frozen predictor family, using the same temporal feature preparation path.
6. Aggregate evaluation: compare all scheduler-predictor combinations against `full_open`.
7. Posthoc visualization: heatmaps, rank correlation, RMSE/DTW/Pearson tradeoff plots, per-model scheduler curve overlays, sensor activation timelines, and primary-target summaries.

Typical outputs:
- Truth CSV: `rl_sensor_scheduling_framework/data/generated/*.csv`
- Scheduler run directories: `rl_sensor_scheduling_framework/reports/runs/<run_tag>_<scheduler>/`
- Reward model directory: `rl_sensor_scheduling_framework/reports/runs/<run_tag>_reward_model/`
- Predictor run directories: `rl_sensor_scheduling_framework/reports/runs/<run_tag>_<scheduler>_pred_<model>/`
- Aggregate tables:
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>.csv`
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>_comparison.csv`
  - `rl_sensor_scheduling_framework/reports/aggregate/metrics_forecast_all_<run_tag>_scheduler_summary.csv`
- Posthoc directory:
  - `rl_sensor_scheduling_framework/reports/aggregate/posthoc_<run_tag>/`
  - `rl_sensor_scheduling_framework/reports/aggregate/posthoc_<run_tag>/task_focus_primary/`

Current interpretation caveat:
- The current mainline is still **not** end-to-end joint optimization of scheduler and downstream forecaster.
- The current primary-task interpretation is "microclimate digital twin": results should be read first on the primary target set, not only on `snow_mass_flux_kg_m2_s`.
- `dRMSE` alone is insufficient; use `RMSE`, `MAE`, `sMAPE`, `Pearson`, and `DTW` together.
- Aggregate scheduler summaries over the full forecast target set and task-focused summaries over the primary target set are different views; do not conflate them.
- Scheduler evaluation should always be separated into:
  - estimation-level metrics (`trace_P_mean`, `power_mean`, `coverage_mean`)
  - forecasting-level metrics (`rmse`, `mae`, `smape`, `pearson_h1_mean`, `dtw_h1_mean`, comparison vs `full_open`)
- The Route A design fixes the earlier failure mode where the frozen predictors saw only a narrow early-time distribution and therefore generalized poorly to later segments.

Known inconsistencies / watchpoints:
- `rl_sensor_scheduling_framework/configs/base.yaml` still contains some generic or historical fields; for windblown experiments, `rl_sensor_scheduling_framework/configs/env/windblown_case.yaml` remains the authoritative source for target definitions.
- The current truth generator includes diurnal forcing and storm-regime switching, but **does not yet include seasonal forcing**. Multi-month or multi-year data would not be meaningful without extending `src/envs/windblown_env.py` first.
- The active mainline uses frozen-oracle evaluation rather than retraining final forecasters separately for each scheduler. This is intentional, but it means absolute `full_open` quality is highly sensitive to oracle generalization.
- As of 2026-04-01, the large Route A GPU run is still in progress / under verification:
  - tmux session: `routeA_gpu_20260401`
  - run tag: `routeA_14day_gpu_20260401`
  - its outputs should be checked before treating Route A forecast quality as final.
- `solar_radiation_wm2` remains in the latent / observed state even though it was removed from the forecast target set; if it continues to degrade estimator or predictor behavior, it should be either smoothed at generation time or removed from the predictor input path.
- Some older docs and historical discussion still describe the state-tracking reward route. When in doubt, treat the frozen-forecast Route A configuration as the current mainline and the old route as a preserved baseline branch.

Recovery / backup notes (important if context is lost):
- Historical state-reward mainline branch:
  - `legacy-state-reward-mainline` @ `761cdee`
- Recent nested-repo commits that introduced the current Route A machinery:
  - `5a5ace1` – frozen-forecast mainline pipeline
  - `9862b4a` – diurnal-aware Route A data and RL features
  - `721b425` – docs and regression tests for Route A
- Backup artifacts created on 2026-04-01:
  - reports archive: `/home/zhangzhuyu/backups/rl_sensor_scheduling_framework_reports_20260401.tar.zst`
  - Codex conversation snapshot: `/home/zhangzhuyu/backups/codex_conversations_20260401.tar.zst`
  - git bundle backup: `/home/zhangzhuyu/backups/rl_sensor_scheduling_framework_20260401.bundle`
- If the server is interrupted, these files are the fastest recovery path.
- If future work needs seasonal generalization:
  - extend `rl_sensor_scheduling_framework/src/envs/windblown_env.py` with seasonality
  - add `time_of_year` features
  - likely reduce temporal resolution from `1 Hz` to a coarser step for tractability

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
