# Windblown Framework Refactor Plan

## Problem Statement

The current windblown implementation has two structural problems:

1. Scheduling is still effectively modeled as a tiny static discrete action problem. After forcing low-power sensors to be always on, the feasible action set collapsed to only a few actions, which made `dqn` and `cmdp_dqn` converge to nearly identical policies.
2. Forecasting is centered on a single target (`snow_mass_flux_kg_m2_s`), which over-favors the FC4 flux sensor and suppresses the value of the rest of the microclimate sensing stack. This is inconsistent with the actual laboratory digital-twin objective, which is to predict the future microclimate state of a single Antarctic point for delayed control.

## Refactor Goals

### Goal 1: Replace static action-id scheduling in the windblown case

For the windblown environment:
- all sensors are schedulable;
- actions are generated online from sensor scores/rankings;
- feasibility is enforced by a constraint projector, not by a precomputed action-id table.

The online projector must enforce:
- instantaneous steady-state power budget;
- startup / heating peak budget;
- optional safety margin;
- max-active constraint.

Implementation rule:
- for small sensor sets, exact feasible-subset search is allowed at runtime;
- for larger sensor sets, the projector must support greedy/approximate fallback;
- no static config-level hardcoding of windblown action ids.

### Goal 2: Unify baseline schedulers around online projection

For the windblown case, the baseline schedulers should output sensor preferences rather than static action ids.

Required baselines:
- `random`
- `periodic`
- `round_robin`
- `info_priority`
- `full_open`

All non-`full_open` baselines must pass through the same online feasibility projector.

### Goal 3: Replace action-id DQN/CMDP-DQN with score-based RL for windblown

For the windblown case:
- RL should score sensors rather than choose from a static discrete action table;
- the final subset should be chosen by the same online constraint projector;
- CMDP should keep hard instantaneous/peak constraints and use dual variables only for long-term average-energy constraints.

Target implementation:
- factorized / branching value function over sensor on/off decisions;
- exact constrained subset selection at runtime for the current small-N setting;
- preserved compatibility with the old discrete-action implementation for `linear_gaussian`.

### Goal 4: Refocus forecasting on microclimate state prediction

The actual scientific target is a laboratory physical twin of a single Antarctic site. The forecasting problem should therefore prioritize microclimate state prediction, not only snow mass flux.

Primary forecast targets:
- `air_temperature_c`
- `snow_surface_temperature_c`
- `wind_speed_ms`

Secondary forecast targets:
- wind direction in transformed form (`sin/cos` or vectorized form)
- `solar_radiation_wm2`

Snow transport variables:
- keep available for evaluation;
- keep available as optional auxiliary targets;
- do not let them define the entire scheduling objective by themselves.

### Goal 5: Improve feature representation

Add derived and structure-aware features for forecasting:
- `wind_dir_sin`
- `wind_dir_cos`
- `wind_u`
- `wind_v`
- `surface_air_temp_gap`
- `particle_kinetic_proxy`
- `size_velocity_interaction`
- `transport_exceedance`
- observed-mask features
- time-since-observed features

This is required so that the `full_open` upper bound is actually usable by predictors.

## Implementation Phases

### Phase A — Scheduling core refactor
1. Add an online sensor subset projector.
2. Remove windblown dependence on static action-id enumeration.
3. Keep the legacy discrete action space only for `linear_gaussian`.
4. Refactor baseline schedulers to output rankings/scores or subsets.

Acceptance criteria:
- windblown no longer relies on `action_space.size()` for decision making;
- action feasibility is computed online;
- baseline schedulers and `full_open` still run end-to-end.

### Phase B — Score-based RL refactor
1. Add factorized / branching Q-network for sensor on/off scoring.
2. Add score-based DQN for windblown.
3. Add score-based CMDP-DQN for windblown with long-term dual constraints.
4. Preserve old DQN/CMDP-DQN for the toy `linear_gaussian` benchmark.

Acceptance criteria:
- windblown `dqn` and `cmdp_dqn` no longer collapse to identical action-id behavior because of a tiny static action table;
- action selection is performed through the online projector;
- hard constraints are still enforced.

### Phase C — Forecast target / feature refactor
1. Separate forecast targets from RL reward targets.
2. Introduce configurable multi-target forecasting for the windblown case.
3. Add derived meteorological features and observation-structure features.
4. Update plotting/evaluation to handle multi-target outputs correctly.

Acceptance criteria:
- predictor training uses configurable forecast target columns;
- target selection is no longer tied to `reward_target_columns` only;
- derived features appear in predictor inputs;
- forecast plots and CSV metrics remain valid.

### Phase D — Validation
1. Run smoke tests.
2. Run one end-to-end windblown experiment.
3. Inspect scheduler summary, per-model comparison CSVs, and prediction curves.
4. Verify that curves are technically reasonable and no scheduler produces trivial flatline datasets.

Acceptance criteria:
- no catastrophic prediction collapse like the previous `+800%~+1200%` RMSE failures;
- `full_open` is restored as a credible upper bound or near-upper-bound reference;
- learned schedulers differ behaviorally from simple baselines.

### Phase E — Evaluation hardening
1. Expand model-quality evaluation beyond RMSE-only summaries.
2. Add sequence-shape metrics for posthoc comparison:
   - `DTW`
   - `Pearson correlation`
   - `sMAPE`
3. Surface these metrics in:
   - aggregate comparison tables;
   - scheduler-vs-model posthoc heatmaps;
   - per-target prediction-curve summary CSVs.

Acceptance criteria:
- posthoc analysis no longer relies on RMSE alone;
- target-level curve summaries include both pointwise-error and sequence-shape metrics;
- the interpretation of a scheduler no longer depends only on dRMSE.

## Current Status

- Phase A: **completed**
  - windblown now uses online feasible-subset projection rather than static action-id scheduling.
- Phase B: **completed, but still behaviorally suboptimal**
  - score-based `dqn` / `cmdp_dqn` exist and run end-to-end;
  - policy quality still trails strong heuristics under some targets.
- Phase C: **partially completed**
  - forecast targets and reward targets are separated;
  - derived features exist;
  - however, the current display focus can still over-emphasize `snow_mass_flux_kg_m2_s` even when the trained objective is broader microclimate prediction.
- Phase D: **completed for technical validity, not for final scientific convergence**
  - catastrophic flatline / collapse bugs were removed;
  - learned policies are now technically reasonable;
  - `full_open` is not yet consistently restored as the clear upper bound on every target/model slice.
- Phase E: **in progress**
  - richer sequence metrics should be treated as part of the core evaluation stack, not as optional extras.

## Explicit Non-Goals for This Refactor

These items are intentionally left out of the current implementation scope:
- multi-rate sensor mode switching (low-rate/high-rate per sensor);
- fully bilevel joint scheduler-predictor training;
- a physically exact blowing-snow PDE model.

They may be added later, but they are not required to complete the present refactor.
