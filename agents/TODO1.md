<!-- - [x] 对完整的传感器系统的参数进行建模（风吹雪），模拟出数据
- [ ] （后续需要构建为应用，所以直接用Go写后端）

- [x] 构建一个多特征耦合的模型
- [ ] 使用更复杂的信号分解算法，结合更复杂的模型而不是LSTM
- [ ] 将数学表达的耦合表达为算子
- [x] 添加PINN要素
- [ ] 添加应用间通信，接收实时的时序数据，构建为实时预测的应用
- [ ] 进阶：开始缝合模块，构建一个新的模型 -->
- # TODO.md — AntAWS (single-station → coastal/inland) + Missingness + Feature Eng + TFT+PINN Benchmark

> For CodeX implementation. This is an execution plan with concrete, ordered steps and expected artifacts.
> Target: Q2 engineering paper demo. Start from current notebook + existing scripts; make pipeline reproducible.

---

## 0) Decide the “paper demo” scope (freeze ASAP)

**Inputs (from config):**

* `station_id_main`: one AntAWS station with usable coverage
* `freq`: `3H` (native) as default; optionally `1H` (resample) as ablation
* `targets`: pick 1–2 variables for the paper demo (recommend: `temperature_c`, `wind_speed_ms`)
* `horizons`: list of steps ahead (for 3H data, recommend: `[1, 2, 4, 8]` → 3h, 6h, 12h, 24h)
* `window_size`: number of history steps (recommend 24 steps for 3H = 3 days history; or 56 steps for 7 days)
* `split`: strict time split by year OR by ratio (time-ordered), e.g. `train=0.7, val=0.1, test=0.2`

**Deliverables:**

* `configs/exp.yaml` with above values (single source of truth)
* `reports/<exp_id>/tables/exp_summary.csv` (auto-generated summary)

---

## 1) Data ingestion: AntAWS → canonical CSV

### 1.1 Parse raw AntAWS CSV/Excel

**Implement:**

* `src/data/antaws_loader.py`

  * Read raw files with robust encoding fallback (`utf-8`, `latin1`)
  * Parse columns: `Year, Month, Day, Three-hourly observation time(UTC), Temperature, Pressure, Wind Speed, Wind Direction, Relative Humidity`
  * Build `timestamp = datetime(Year, Month, Day, hour)` where `hour = obs_time`
  * Sort by `timestamp`, drop duplicates (keep first, log count)
  * Standardize column names:

    * `temperature_c`, `pressure_hpa`, `wind_speed_ms`, `wind_dir_deg`, `relative_humidity_pct`

**CLI:**

* `scripts/prepare_antaws_station.py --input <path> --output data/processed/<station_id>.csv --freq 3H`

**Artifacts:**

* `data/processed/<station_id>.csv`
* `reports/<exp_id>/tables/prepare_log.csv` (rows read, duplicates removed, first/last timestamp)

### 1.2 Optional: resample frequency

**Implement:**

* `src/data/resample.py::resample_df(df, freq)`

  * `3H`→`1H` optional (time-based resample)
  * Keep observation-aligned values (use forward-fill for step-wise variables if needed) OR use interpolation only for resampling step (document choice)

**Artifacts:**

* `data/processed/<station_id>_<freq>.csv`

---

## 2) Station classification: coastal vs inland (for later multi-station section)

### 2.1 Metadata table

**Implement:**

* `data/metadata/stations_meta.csv` (manual seed file)

  * Columns: `station_id, lat, lon, elevation_m`
  * If elevation is not available: allow `group_label` manual column

### 2.2 Labeling rule

**Implement:**

* `src/data/station_groups.py::assign_group(meta, elevation_threshold=1000)`

  * `inland if elevation_m >= threshold else coastal`
  * If `group_label` exists, prefer it (log override)

**Artifacts:**

* `data/processed/station_groups.csv`
* `reports/<exp_id>/tables/station_groups_used.csv`

---

## 3) Missingness profiling (make this a paper figure/table)

### 3.1 Compute missingness stats per variable

**Implement:**

* `src/analysis/missingness_profile.py::profile(df, cols)`

  * Overall missing ratio
  * Missing ratio by year
  * Longest consecutive missing gap (in steps)
  * Gap length distribution summary: `p50, p90, p95, max`
  * Count of fully-missing rows across all variables

**CLI:**

* `scripts/profile_missingness.py --data data/processed/<station>.csv --out reports/<exp_id>/tables/missingness.csv`

**Artifacts:**

* `reports/<exp_id>/tables/missingness.csv`
* `reports/<exp_id>/figures/missingness_by_year.png`
* `reports/<exp_id>/figures/gap_length_ccdf.png`

---

## 4) Missing-data handling (turn “old years have many NA” into the engineering highlight)

Implement **two strategies** and make them configurable.

### 4.1 Strategy A (mandatory): limited interpolation + explicit missing features

**Implement:**

* `src/data/impute.py::impute_A(df, value_cols, max_gap_steps)`

  * For each value column:

    * Identify consecutive NaN runs (gaps)
    * If `gap_len <= max_gap_steps`: linear interpolate
    * Else: keep NaN (do NOT fabricate)
  * Create missingness-aware features:

    * `<col>_is_obs` (1 if original non-NA else 0)
    * `<col>_tslo` (time-since-last-observation, in steps; reset to 0 on observation; grows during gaps)
    * `<col>_gap_len` (current gap length if inside a gap else 0)
  * Output both imputed value cols + these missingness features

**Artifacts:**

* `data/processed/<exp_id>/station_imputeA.csv`

### 4.2 Strategy B (optional but recommended): statistical smoother

Pick **one** method to keep scope reasonable.
**Option B1: Kalman smoothing**

* `src/data/impute.py::impute_B_kalman(...)`

**Option B2: STL + local regression**

* `src/data/impute.py::impute_B_stl(...)`

**Artifacts:**

* `data/processed/<exp_id>/station_imputeB.csv`

### 4.3 Paper-specific evaluation of imputation impact

**Implement:**

* `scripts/compare_missing_strategies.py`

  * Train the same baseline model (e.g., LightGBM) on A vs B
  * Report delta in MAE/RMSE
  * Keep this small but presentable

**Artifacts:**

* `reports/<exp_id>/tables/missing_strategy_compare.csv`

---

## 5) Feature engineering (lightweight, meteorology-friendly)

### 5.1 Build feature matrix from canonical df

**Implement:**

* `src/features/build_features.py::build(df, cfg)`

  * Time features:

    * `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`, `month`
  * Dynamics features:

    * `d_temperature_c_1`, `d_temperature_c_2`
    * `d_wind_speed_ms_1`, `d_wind_speed_ms_2`
  * Rolling features (use past-only windows):

    * rolling mean/std for selected vars with windows `[2, 8]` steps
  * Include missingness features produced in Step 4
  * Ensure no leakage: rolling uses only past values
  * Drop rows that cannot form a full window later (handled in dataset builder)

**Artifacts:**

* `data/processed/<exp_id>/station_features.csv`
* `reports/<exp_id>/tables/feature_list.csv`

---

## 6) Dataset builder (windowed multi-horizon, strict time split)

### 6.1 Time-ordered split

**Implement:**

* `src/data/split.py::time_split(df, cfg)`

  * Either by year ranges OR by ratios but time-ordered
  * Return index ranges for train/val/test
  * Save split boundaries for reproducibility

**Artifacts:**

* `reports/<exp_id>/tables/split_summary.csv`

### 6.2 Sliding window generation

**Implement:**

* `src/data/window_dataset.py`

  * Inputs:

    * `X_cols` (features), `Y_cols` (targets)
    * `window_size`, `horizons`, `stride`, `max_windows`
  * Output:

    * `X`: shape `[N, window_size, num_features]`
    * `Y`: shape `[N, len(horizons), num_targets]`
    * `t_ref`: timestamp for each sample (prediction start time)
  * Handling remaining NaNs:

    * If any NaN in target horizon → discard sample
    * If NaNs in features: allow if you have missingness masks + TSLO; otherwise discard (cfg flag)

### 6.3 Normalization (train-only stats)

**Implement:**

* `src/data/normalize.py`

  * Fit scaler on train split only
  * Apply to val/test
  * Save `scaler.json` with mean/std per feature/target (needed for physics penalty unnormalize)

**Artifacts:**

* `data/processed/<exp_id>/{train,val,test}.npz`
* `data/processed/<exp_id>/scaler.json`

---

## 7) Baseline models (paper table backbone)

Unify evaluation so all models output comparable files.

### 7.1 Minimal baseline set (must)

Implement runners that read the same `{train,val,test}.npz` or `station_features.csv` and produce:

* `pred.csv` with `timestamp, y_true(<targets>), y_pred(<targets>), horizon_step`

**Baselines:**

* Naive persistence
* Seasonal naive (same hour-of-day climatology or rolling seasonal)
* ARIMA (single target only is OK)
* LightGBM / XGBoost (tabular; use last-step features + rolling stats)
* TCN (your existing Darts runner)
* TFT (your existing Darts runner)

**Artifacts:**

* `reports/<exp_id>/preds/<model_name>.csv`
* `reports/<exp_id>/tables/<model_name>_metrics.csv`

### 7.2 Integrate existing scripts

Reuse:

* `scripts/run_baselines.py`
* `scripts/baselines/run_*`
  But modify/extend to:
* accept `--exp_id` and write in the unified folder layout
* use the same horizons/window_size as config

---

## 8) Main model: TFT + PINN (paper’s “engineering contribution”)

You already have notebook logic; now make it reproducible.

### 8.1 Extract notebook into modules

**Implement:**

* `src/models/snow_tft.py` (model definition)
* `src/train/train_tft_pinn.py` (train loop)
* `scripts/train_tft_pinn.py --config configs/exp.yaml`

### 8.2 Physics penalty: keep it simple & defensible with AntAWS

Since AntAWS lacks many advanced sensors, use “soft physics” that cannot be criticized as fake:

* Bounds:

  * `wind_speed_ms >= 0`
  * `relative_humidity_pct in [0,100]` (if used)
* Temporal coherence:

  * Penalize unrealistic high-frequency spikes in prediction: Huber loss on first difference of predicted series
* Optional threshold heuristic (only if you include snow flux proxy; otherwise skip)

**Implement:**

* `src/losses/physics_losses.py`

  * `physics_penalty(y_pred, x_context, scaler, cfg)` returns scalar
  * Must support toggles:

    * `lambda_phys = 0` (pure TFT)
    * `use_bounds_term`
    * `use_coherence_term`

### 8.3 Outputs

**Artifacts:**

* `models/<exp_id>/tft_pinn.pt`
* `reports/<exp_id>/preds/tft_pinn.csv`
* training curves: `reports/<exp_id>/figures/loss_curve.png`

---

## 9) Evaluation: metrics + extreme-slice + horizon curves

### 9.1 Metrics

**Implement:**

* `src/eval/metrics.py`

  * MAE, RMSE (mandatory)
  * R2 (optional)
    Compute:
* per horizon
* overall average across horizons
* per target

### 9.2 Extreme slice (paper highlight)

**Implement:**

* `src/eval/extremes.py`

  * For each target on test:

    * bottom 10% y_true slice
    * top 10% y_true slice
    * compute MAE/RMSE for each model
      This often shows PINN/coherence helps during extremes.

### 9.3 Unified evaluator

**Implement:**

* `scripts/evaluate_models.py --config ...`

  * Reads all `reports/<exp_id>/preds/*.csv`
  * Writes:

    * `reports/<exp_id>/tables/metrics_overall.csv`
    * `reports/<exp_id>/tables/metrics_by_horizon.csv`
    * `reports/<exp_id>/tables/metrics_extremes.csv`

---

## 10) Figures: minimal set for Q2 paper

**Implement:**

* `scripts/make_figures.py --config ...`

Generate:

* Fig1: missingness-by-year
* Fig2: prediction timeline (test window) — y_true vs (best baseline) vs TFT vs TFT+PINN
* Fig3: MAE/RMSE bar chart across models
* Fig4: error vs horizon curve (MAE/RMSE by horizon)
* Fig5 (optional): extremes slice bar chart

**Artifacts:**

* `reports/<exp_id>/figures/*.png`

---

## 11) Coastal vs inland extension (optional section, low effort)

Only do this if you can get metadata for 4–6 stations.

### 11.1 Select stations

* 2–3 coastal + 2–3 inland
* Run the same pipeline per station (same config, station list)

### 11.2 Grouped results table

**Implement:**

* `scripts/evaluate_groups.py`

  * Aggregate metrics by group label
  * Output grouped mean ± std

**Artifacts:**

* `reports/<exp_id>/tables/metrics_grouped.csv`

---

## 12) Experiment matrix (what to actually run for the paper)

Run these and stop (don’t expand endlessly):

1. **Main demo (single station, Strategy A)**

* Baselines + TFT + TFT+PINN
* horizons `[1,2,4,8]`

2. **Missing strategy study**

* Strategy A vs Strategy B (or only A if B too slow)
* Compare at least: LightGBM, TFT, TFT+PINN

3. **Ablations (only 2)**

* TFT+PINN with `lambda_phys=0` (pure TFT)
* TFT+PINN full

4. **Extremes evaluation**

* top/bottom 10% slices for target(s)

---

## 13) Reproducibility & one-command run

### 13.1 One-command pipeline

**Implement:**

* `scripts/run_all.py --config ...`

  * calls: prepare → profile → impute → featurize → dataset → train models → eval → figs
  * ensures outputs in `reports/<exp_id>/...`

### 13.2 Repro checklist

* fix seeds (numpy/torch/lightning)
* save config snapshot: `reports/<exp_id>/tables/config_used.yaml`
* save env snapshot: `reports/<exp_id>/tables/pip_freeze.txt`

---

## Definition of Done (DoD)

* [ ] `python scripts/run_all.py --config configs/exp.yaml` finishes without manual notebook steps
* [ ] Generates paper-ready tables + figures under `reports/<exp_id>/`
* [ ] Produces a main results table comparing baselines vs TFT vs TFT+PINN
* [ ] Includes missingness profiling + extreme-slice evaluation
* [ ] README contains exact commands to reproduce results
