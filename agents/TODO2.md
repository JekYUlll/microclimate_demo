# AntAWS Sensor Scheduling / Sampling Strategy Experiments (3-hour data) — Agent Task Spec

Owner: Agent (codeX)  
Dataset: AntAWS (default sampling interval = **3 hours**)  
Primary goal: Evaluate **sensor scheduling / sampling policies** (NO RL) for improving **temperature forecasting** under constrained observations.

---

## 0) What we are testing (clear hypothesis)

We simulate a realistic constraint: **temperature is always observed**, but other sensors cannot be observed continuously due to power/heating limits.  
We compare **different scheduling/sampling policies** under the same observation budget, and evaluate how much they preserve **temperature forecast accuracy**.

**Key outputs**:
- Accuracy vs budget curves
- Robustness across missingness mechanisms
- Coastal vs inland differences
- Extreme temperature event performance

---

## 1) Forecasting task definition (what predicts what)

### 1.1 Target (main task)
- **Target variable**: near-surface air temperature (`temp` or your canonical temperature column)
- **Prediction type**: multi-horizon forecasting

### 1.2 Horizons (3h step)
AntAWS step = 3h. Use horizons in steps:
- **H=1** step = 3h
- **H=2** steps = 6h
- **H=8** steps = 24h

> These align with short / medium / daily scales.

### 1.3 Input variables
- Always included: **temp history**
- Candidate sensors for scheduling: `rh` (humidity), `wind_speed`, `wind_dir`, `rad`/`sw` (radiation/light), `pressure` (if exists), etc.
- Keep the candidate set consistent across stations by intersecting available columns.

### 1.4 Models to run
At minimum:
- `TCN`
- `TFT`
Recommended:
- `TFT+PINN`
Plus existing baselines via `scripts/run_baselines.py` (naive/seasonal/AR/GBRT if already in pipeline)

---

## 2) Two-layer experimental design (feasible + publishable)

We run **two layers**, because “scheduling” needs a clean ground-truth segment.

### Layer A — Natural Missingness (real-world difficulty, broad coverage)
Purpose: show performance on the real AntAWS missingness as-is.
- Stations: **as many as possible** (target 150–250)
- No synthetic scheduling mask applied
- Compare: models + missing handling (impute A/B vs mask-aware if implemented)

### Layer B — Counterfactual Scheduling (core scheduling comparison)
Purpose: isolate the effect of **scheduling policy** (causal comparison).
- Use only station-time segments that are relatively complete.
- Apply synthetic scheduling masks (policies) on top of a “clean” segment.

This is the core of the paper.

---

## 3) Station and time selection (concrete rules)

### 3.1 Run missingness profiling first
Use existing script:
- `scripts/profile_missingness.py` on processed station CSVs (resampled to 3h grid).

We will use its outputs to pick stations for Layer B.

### 3.2 Layer B station eligibility criteria
Select station segments meeting ALL:
- Temperature missing rate < **10%** (prefer < 5%)
- Non-temp variables missing rate < **30%** on average (or at least 3 auxiliary vars available)
- Longest continuous gap (for temp) < **7 days** (7 days = 56 steps at 3h)
- Must have at least **24 months** of data in a usable segment

**Target size**:
- 40 stations total: **20 coastal + 20 inland**
- If not enough, relax non-temp missing rate threshold gradually.

### 3.3 Layer B time window per station
Pick a **continuous 24-month segment**:
- Prefer most recent available segment that meets criteria
- If the most recent is too sparse, pick the longest “clean” segment

Store chosen segment metadata:
- `reports/<exp_id>/tables/layerB_station_segments.csv`
  - columns: station_id, start_time, end_time, group, missing stats

---

## 4) Preprocessing and resampling (3h grid)

### 4.1 Enforce a strict 3h grid before anything
In `scripts/prepare_antaws_station.py`:
- Resample/interpolate timestamps to a strict 3h time index
- **Do not impute** yet (leave NaNs)
- Ensure wind direction uses circular handling if you aggregate (if needed)

Outputs:
- `data/processed/<exp_id>/<station_id>.csv`

---

## 5) Scheduling / sampling policies (NO RL) — what we implement

We implement a mask generator `M[t, var]` where 1=observed, 0=not observed (set NaN).
Temperature is always observed by default.

### Shared constraints across policies
- `temp_always_on = true`
- At each time step, max number of **non-temp** sensors observed = `k` (budget)
- Once a sensor turns ON, it stays ON for at least `min_on_steps`
- Optional warm-up: newly turned-on sensor yields NaN for `warmup_steps` (simulates heating stabilization)

All parameters are in **steps** (1 step = 3 hours).

---

## 6) Budget levels (k) and constraints (step-based)

### Budget values (k)
Use these in Layer B:
- `k ∈ {0, 1, 2, 3, All}` (non-temp sensors per step)
Meaning:
- `k=0`: temp-only (lower bound)
- `All`: oracle (upper bound)

### Temporal constraints
- `min_on_steps ∈ {2, 4}` (6h, 12h)
- `warmup_steps ∈ {0, 1}` (0h, 3h)

Start simple:
- Default: `min_on_steps=2`, `warmup_steps=0`

---

## 7) Policies to compare (must be concrete)

### P0 — Oracle-Full (upper bound)
- Observe all sensors (only natural missingness remains)

### P1 — Temp-only (lower bound)
- Only temperature observed (all other vars masked out completely)

### P2 — Round-robin polling (strong baseline)
- For non-temp sensors, rotate in a fixed order
- At each step, observe the next `k` sensors
- Enforce `min_on_steps`

Parameters:
- `k`, `min_on_steps`

### P3 — Duty-cycle periodic sampling (power-saving baseline)
- Each non-temp sensor is observed for `on_steps` out of `period_steps`
- Can use random phase per sensor

Recommended parameter grid (3h steps):
- `period_steps ∈ {8, 16}`  (24h, 48h)
- `on_steps ∈ {1, 2, 4}`    (3h, 6h, 12h)

*Note*: Duty-cycle doesn’t enforce per-step k directly; it controls per-sensor frequency.  
To keep fairness, optionally cap max simultaneously observed sensors to `k` with a tie-break rule (or report “effective k”).

### P4 — Block-off outages (robustness, not “good scheduling”)
- Simulate prolonged outages for non-temp sensors
- Apply `n_blocks` with expected gap length

Parameters (3h steps):
- `expected_gap_steps ∈ {8, 16, 32}`  (24h, 48h, 96h)
- `n_blocks_per_year ∈ {10, 30}`

This policy mainly tests model robustness to realistic long gaps.

### P5 — Info-priority heuristic (main “scheduling” policy, NO RL)
At each step, select top-`k` sensors by a **weight score** reflecting usefulness for predicting temperature.

Two options (implement at least the first):
1) **train_corr** (required):  
   - Compute on training split only:
     - `w_j = max_{lag in L} |corr( x_j(t-lag), temp(t) )|`
   - Lags `L = {0,1,2,4}` steps = {0h,3h,6h,12h}
2) **event-boost** (optional, adds adaptivity without RL):
   - If |Δtemp| is high (e.g., above 90th percentile on train), temporarily boost wind/radiation weights for the next `B` steps

Scheduling logic:
- Maintain ON sensors for `min_on_steps` before switching
- Recompute weights every `update_every_days` (optional)
  - With 3h steps: 30 days = 240 steps

Parameters:
- `k`, `min_on_steps`
- `lag_steps = [0,1,2,4]`
- optional: `update_every_steps = 240`
- optional: `event_threshold_quantile = 0.9`, `event_boost = 1.5`, `event_duration_steps = 2`

---

## 8) Missing handling modes (how models see missingness)

We need two dataset variants:

### M1 — Impute-then-forecast
Use existing imputation strategies:
- `imputation=A`
- `imputation=B`

Mask is applied BEFORE imputation.

### M2 — Mask-aware forecast (recommended if feasible)
Do not rely on heavy imputation. Add features:
- `is_missing_<var>`
- `time_since_last_seen_<var>` (in steps)

Fill NaNs with 0 only if the model cannot accept NaNs, but keep indicators.

---

## 9) Concrete experiment matrix (doable size)

### Core Layer B sweep (recommended minimal publishable)
Stations: 40 (20 coastal, 20 inland)  
Models: TCN, TFT (optional add TFT+PINN)  
Missing handling: impute-B + mask-aware (if mask-aware exists)

Policies: P1, P2, P3, P5, P0  
Budgets: k ∈ {0,1,2,3,All}

If too large, start with:
- Policies: P2, P3, P5 (+ P1 + P0 for bounds)
- Budgets: {0,1,3,All}
- Models: {TCN, TFT}
This is already enough to show a strong story.

### Layer A runs (broad)
No synthetic mask. Use all stations.
Compare:
- TCN vs TFT vs TFT+PINN
- impute A vs impute B vs mask-aware

---

## 10) What to compare (metrics)

Use existing outputs:
- `metrics_overall.csv`
- `metrics_by_horizon.csv`
- `metrics_extremes.csv`

Primary metrics:
- RMSE / MAE for temperature
- By-horizon RMSE at H=1/2/8 steps

Extreme metrics:
- Evaluate on low-temp extremes (e.g., bottom 5% temp on test)
- Evaluate on rapid-change events (|Δtemp| top 5%)

Add one derived metric (must be reported):
- **Information efficiency**: improvement vs temp-only per added sensor
  - `ΔRMSE(k) / k` relative to `k=0`

---

## 11) Figures to generate (paper-ready)

Required plots (Layer B):
1) **RMSE vs Budget k** (main figure)
- x: k (0,1,2,3,All)
- y: temp RMSE
- line: policy (P2/P3/P5)
- facet or separate panels: model (TCN vs TFT)

2) **RMSE by horizon** under fixed k (e.g. k=1 and k=3)
- bar chart or lines
- compare policies × models

3) **Extremes vs Budget**
- x: k
- y: extreme RMSE (or your extreme metric)
- compare policies

4) **Missingness heatmap examples**
- For one station, one month of test:
  - time × variables mask visualization
  - one subplot per policy (P2/P3/P5)
This proves the scheduling masks are meaningfully different.

5) **Coastal vs Inland boxplots**
- distribution of RMSE across stations, split by group, for a fixed k and policy set

6) **Timeline prediction example**
- One representative station, test segment:
  - true temp vs predicted temp
  - annotate when auxiliary sensors were ON (background shading)

Layer A:
- Optional: overall performance summary across all stations (boxplot of RMSE by model)

---

## 12) Implementation tasks (mapping to your repo scripts)

### 12.1 New module
Create:
- `src/data/masking.py`
Functions:
- `generate_mask(df, sampling_cfg, seed) -> mask_df`
- `apply_mask(df, mask_df, sampling_cfg) -> df_masked`
- `mask_stats(mask_df) -> stats_df`

### 12.2 Build dataset integration
Modify `scripts/build_dataset.py`:
- If `sampling.enable`:
  - Apply station segment crop (Layer B only; driven by config or a station list file)
  - Generate mask and apply before imputation
  - Save:
    - `data/processed/<exp_id>/<station>_masked.csv`
    - `reports/<exp_id>/tables/mask_stats_<station>.csv`
  - Continue pipeline (impute -> features -> splits -> windows -> NPZ)

### 12.3 Station segment selection (Layer B)
Add a small utility script:
- `scripts/select_layerB_segments.py`
Inputs:
- processed station CSVs + `missingness.csv` from profiling
Outputs:
- `reports/<exp_id>/tables/layerB_station_segments.csv`
- `data/processed/<exp_id>/layerB_station_list.txt`

### 12.4 Sweeps / aggregation
If sweeps are separate exp_ids:
- Add `scripts/aggregate_sweeps.py`
  - reads multiple `reports/<exp_id>/tables/metrics_*.csv` + `config_used.yaml`
  - outputs:
    - `perf_vs_budget.csv`
    - the main figures above

---

## 13) Config templates (must be created)

Create sweep-ready config templates (examples):
- `configs/layerB_round_robin_k1.yaml`
- `configs/layerB_round_robin_k3.yaml`
- `configs/layerB_duty_cycle_24h.yaml`
- `configs/layerB_info_priority_k1.yaml`
- etc.

Add these keys (minimum):
```yaml
sampling:
  enable: true
  strategy: "info_priority"   # round_robin | duty_cycle | block | oracle | temp_only
  temp_always_on: true
  budget_k: 1
  min_on_steps: 2
  warmup_steps: 0
  info_priority:
    weight_source: "train_corr"
    lag_steps: [0,1,2,4]
    update_every_steps: 240    # optional
