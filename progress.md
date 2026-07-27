# Progress Log

# 2026-06-21 Current Target Correction
- The current target journal is *Expert Systems with Applications* (ESWA).
  Earlier CRST rewrite entries below are retained only as historical provenance.
- The active operational work is the `rl_sensor_scheduling_framework` Phase 9
  static-break/strong-claim exploration and the 24h BO-1 autonomous run on
  `remote-gpu`.
- Root planning files were updated because the planning hook was still injecting
  the older CRST rewrite goal when commands ran from the repository root.

## Session: 2026-05-22

### Phase 1: Evidence Consolidation
- **Status:** complete
- **Started:** 2026-05-22 06:42:03 CST
- Actions taken:
  - Loaded the `planning-with-files` skill instructions from `/home/horeb/.agents/skills/planning-with-files/SKILL.md`.
  - Checked for existing root planning files; none were present.
  - Ran the session catchup script; no recovery output was produced.
  - Read the planning templates for `task_plan.md`, `findings.md`, and `progress.md`.
  - Created project-root planning files for the PD-PPO scheduling-degeneration follow-up.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning file presence | `ls -l task_plan.md findings.md progress.md` | Three files exist | All three files exist in project root | pass |
| Planning file readability | `sed -n` on all three files | Contents readable and coherent | All three files read successfully | pass |
| V3.1 behavior diagnostics | `conda run -n darts python rl_sensor_scheduling_framework/scripts/47_v31_behavior_diagnostics.py` | CSV/Markdown diagnostics generated | Generated under `reports/v31_s2_main/behavior_diagnostics/` | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-22 06:42 CST | None | 1 | N/A |
| 2026-05-22 06:42 CST | `scripts/47_v31_behavior_diagnostics.py` failed because pandas `to_markdown()` requires missing optional dependency `tabulate` | 1 | Patched the script to use a small local Markdown table renderer instead of adding a dependency |
| 2026-05-22 06:42 CST | Ran behavior diagnostics with `windblown_sensors_complex.yaml` into the default `behavior_diagnostics/` directory, temporarily mixing complex sensor config with balanced rollout summaries | 1 | Patched the script to use sensor-specific output directories for non-balanced configs, then reran balanced diagnostics to restore the main directory |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 is complete; next is Phase 2 diagnostics. |
| Where am I going? | Verify mechanisms behind quasi-static scheduling, then design a defensible scenario sweep. |
| What's the goal? | Convert the Figure 8 scheduling-degeneration discovery into a rigorous experimental and paper-narrative plan. |
| What have I learned? | See `findings.md`: static trap is plausible, but several claims in `05-22-plan-1.md` need verification. |
| What have I done? | Created the persistent planning files and captured the first set of decisions. |

### Phase 2: Baseline and Mechanism Diagnostics
- **Status:** in_progress
- **Started:** 2026-05-22 06:42 CST
- Actions taken:
  - Advanced `task_plan.md` current phase from Phase 1 to Phase 2.
  - Confirmed the repository has pre-existing unrelated dirty state; diagnostics will avoid reverting or touching unrelated files.
- Files created/modified:
  - `task_plan.md` (updated phase status)
  - `progress.md` (logged Phase 2 start)
  - `rl_sensor_scheduling_framework/scripts/47_v31_behavior_diagnostics.py` (created)
  - `rl_sensor_scheduling_framework/reports/v31_s2_main/behavior_diagnostics/` (created by script)
  - `rl_sensor_scheduling_framework/docs/05-22-v31-behavior-diagnostics.md` (created)
  - `findings.md` (updated with diagnostic results)
- Additional actions taken:
  - Read V3.1 S2 runner defaults, custom PPO arguments, sensor config, and policy/projector implementation.
  - Ran behavior diagnostics over all local V3.1 S2 rollouts.
  - Confirmed `feasible_static_projected` is fixed-priority by code.
  - Confirmed the static core subset costs `1.46`, which remains feasible for `B=1.50--1.60`.
  - Confirmed V3.1 event frequency is already substantial, so "events too rare" is not the first explanation.

### Phase 3: Scenario Design Gate
- **Status:** complete
- Actions taken:
  - Rejected the balanced-cost `B=1.50--1.60` full sweep as the next main experiment because it does not break the current static core.
  - Deferred event-frequency and reward-function changes until after cost/coverage mechanisms are clarified.
  - Added Phase 3 gate and pilot acceptance criteria to `docs/05-22-v31-behavior-diagnostics.md`.
  - Identified `physical-cost pilot + static-reference clarification` as the next implementation target.
- Files created/modified:
  - `task_plan.md` (Phase 3 completed)
  - `progress.md` (logged Phase 3 decision)
  - `rl_sensor_scheduling_framework/docs/05-22-v31-behavior-diagnostics.md` (added scenario gate)

### Phase 4: Implementation Plan for New Runs
- **Status:** in_progress
- Actions taken:
  - Advanced current phase to Phase 4.
  - Will inspect script/config paths for a physical-cost pilot without launching training yet.
  - Confirmed `25_v2_train_custom_ppo.py` already accepts and records `--sensor-cfg`.
  - Added `--sensor-cfg` passthrough to `41_v31_pilot.py` and `42_v31_s2_full.py`; defaults remain `configs/sensors/windblown_sensors_balanced.yaml`.
  - Added `--antaws-root` passthrough to the same wrappers after the server pilot exposed a path mismatch.
  - Made wrapper-level `--sensor-cfg` resolve to an absolute framework path when a relative config is not valid from the current working directory.
  - Verified local `--help` and dry-run output for both wrappers.
  - Synchronized the two wrapper scripts to the GPU server.
  - Verified server dry-run for the complex-cost pilot using `configs/sensors/windblown_sensors_complex.yaml`.
  - Started remote tmux pilot `v31_complex_pilot_20260522` at 2026-05-22 06:58 CST:
    - `B=1.70`
    - `seed=41`
    - `workers=1`
    - `gpu_ids=1`
    - output directory `reports/v31_complex_pilot`
  - Restarted `v31_complex_pilot_20260522` at 2026-05-22 06:59 CST with explicit `--antaws-root data/AntAWS/3_hourly`.
  - Restarted `v31_complex_pilot_20260522` at 2026-05-22 07:01 CST after resolving sensor config paths to absolute paths.
  - Confirmed the restarted run generated truth, oracle weights, and candidate-prior table.
  - Inspected candidate-prior table:
    - top subset: `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`
    - power mean: `1.30`
    - oracle loss mean: about `0.182`
    - `fc4_flux` absent from the top 15 candidate-prior rows.
  - Updated `47_v31_behavior_diagnostics.py` so it can read both full-grid `raw/budget*_seed*` outputs and pilot `*budget*_seed*` outputs.
  - Regression-tested the diagnostic script on local `reports/v31_s2_main`.
  - Synchronized the updated diagnostic script to the GPU server.
  - Added `StaticMaskPolicy` to `src/v2/policies.py`.
  - Updated `25_v2_train_custom_ppo.py` so future runs with candidate prior also evaluate `oracle_static_projected`, the lowest-oracle-loss fixed candidate mask.
  - Updated `47_v31_behavior_diagnostics.py` policy order to include `oracle_static_projected`.
  - Verified local and remote `py_compile` for the modified policy/training/diagnostic files.
  - Fixed an accidental remote `rsync --relative` nesting by re-syncing to the project root and removing the nested directory.
- Files created/modified:
  - `task_plan.md` (Phase 4 in progress)
  - `progress.md` (logged Phase 4 start)
  - `findings.md` (logged wrapper-level scenario decision)
  - `rl_sensor_scheduling_framework/scripts/41_v31_pilot.py` (added sensor config passthrough)
  - `rl_sensor_scheduling_framework/scripts/42_v31_s2_full.py` (added sensor config passthrough)
  - `rl_sensor_scheduling_framework/scripts/47_v31_behavior_diagnostics.py` (supports pilot directory layout)
  - `rl_sensor_scheduling_framework/src/v2/policies.py` (added exact static-mask policy)
  - `rl_sensor_scheduling_framework/scripts/25_v2_train_custom_ppo.py` (future runs evaluate `oracle_static_projected`)

## Remote Runs
| Timestamp | Session | Command scope | Status |
|-----------|---------|---------------|--------|
| 2026-05-22 06:58 CST | `v31_complex_pilot_20260522` | Complex-cost V3.1 pilot, `B=1.70`, `seed=41` | running |
| 2026-05-22 06:59 CST | `v31_complex_pilot_20260522` | Restarted with `--antaws-root data/AntAWS/3_hourly` | running |
| 2026-05-22 07:01 CST | `v31_complex_pilot_20260522` | Restarted after absolute sensor config path fix | complete |
| 2026-05-22 10:33 CST | `v31_complex_no_prior_20260522` | Complex-cost no-prior/no-AWBC ablation, `B=1.70`, `seed=41`, `50k` steps | complete |

### Phase 6: Paper Narrative Decision
- **Status:** in_progress
- Actions taken:
  - Advanced current phase to Phase 6 after completing the complex-cost pilot and no-prior ablation diagnostics.
  - Located paper statements that are risky after the new diagnostics:
    - `sections/07_discussion.tex`: "adding an adaptive mechanism for event-conditioned warm-up decisions"
    - `sections/08_conclusion.tex`: "warm-up-aware selection of particle and flux channels"
    - introduction/conclusion claims that adaptive scheduling is empirically necessary in the current configuration
  - Added paper-claim triage guidance to `docs/05-22-v31-complex-pilot-results.md`.
- Files created/modified:
  - `task_plan.md` (Phase 5 complete, Phase 6 in progress)
  - `progress.md` (logged Phase 6 start)
  - `rl_sensor_scheduling_framework/docs/05-22-v31-complex-pilot-results.md` (added paper-claim triage)

### Phase 5: Run, Collect, and Verify
- **Status:** in_progress
- Actions taken:
  - Advanced current phase to Phase 5 after completing the wrapper/scenario/static-reference implementation items.
  - Confirmed remote pilot is still running and healthy at update 14 / 49 (`28672` / `100000` timesteps).
  - No evaluation files exist yet; behavior diagnostics should be run after `evaluation/v2_eval_overall.csv` and rollout NPZ files appear.
  - Confirmed the complex-cost pilot completed and synchronized results from the server to local `rl_sensor_scheduling_framework/reports/v31_complex_pilot/`.
  - Ran behavior diagnostics for the completed complex-cost pilot.
  - Compared `custom_ppo` vs `feasible_static_projected` rollout arrays and found them exactly equal across selected masks, modes, power, peaks, oracle losses, observations, and observed masks.
  - Determined that the complex-cost pilot should not be scaled to a full seed/budget grid.
  - Launched no-prior/no-AWBC mechanism ablation in remote tmux `v31_complex_no_prior_20260522`:
    - output directory `reports/v31_complex_no_prior_seed41`
    - same truth CSV as the completed complex-cost pilot
    - `total_timesteps=50000`
    - `awbc_coef=0.0`
    - `prior_kl_coef=0.0`
    - no `--use-oracle-candidate-prior`
  - Confirmed no-prior/no-AWBC ablation completed and synchronized results to local `rl_sensor_scheduling_framework/reports/v31_complex_no_prior_seed41/`.
  - Patched `47_v31_behavior_diagnostics.py` to support single-run directories and reused external truth CSV paths from metadata.
  - Ran behavior diagnostics for the no-prior/no-AWBC ablation.
  - Found that removing prior/AWBC restores dynamic switching but hurts forecast quality and warmup efficiency:
    - `forecast_weighted_mae_overall=0.145684`
    - `switches_per_step=2.049`
    - `warmup_abort_rate=0.0664`
    - `laser_disdrometer` event lift `-0.0206`
  - Created `rl_sensor_scheduling_framework/docs/05-22-v31-complex-pilot-results.md`.
- Files created/modified:
  - `task_plan.md` (Phase 4 complete, Phase 5 in progress)
  - `progress.md` (logged Phase 5 start)
  - `findings.md` (logged completed pilot interpretation and no-prior ablation rationale)
  - `rl_sensor_scheduling_framework/reports/v31_complex_pilot/` (synced completed pilot results)
  - `rl_sensor_scheduling_framework/reports/v31_complex_no_prior_seed41/` (synced no-prior ablation results)
  - `rl_sensor_scheduling_framework/docs/05-22-v31-complex-pilot-results.md` (created decision memo)

## Additional Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-22 06:58 CST | Remote pilot failed because default `../data/AntAWS/3_hourly` did not contain `Panda100_3h.csv` on the server | 1 | Located server data at `data/AntAWS/3_hourly`, added wrapper-level `--antaws-root` passthrough, and restarted the pilot with the correct path |
| 2026-05-22 07:00 CST | Remote pilot failed because `configs/sensors/windblown_sensors_complex.yaml` was interpreted relative to the project root | 2 | Added wrapper-level absolute path resolution for sensor config arguments |
| 2026-05-22 07:08 CST | Sync command created a nested `rl_sensor_scheduling_framework/rl_sensor_scheduling_framework` directory on the server | 1 | Re-synced changed files to `~/_code/microclimate_demo/` with `--relative` and removed the accidental nested directory |

### New Implementation Track: Physical-Event Mainline
- **Status:** in_progress
- **Started:** 2026-05-22 CST
- Actions taken:
  - Loaded `planning-with-files` again for the new implementation request.
  - Read `rl_sensor_scheduling_framework/docs/05-22-chore.md`.
  - Replaced the old static-trap diagnostic plan with a new physical-event implementation plan.
  - Logged the main claim-repair and implementation decisions in `findings.md`.
- Files created/modified:
  - `task_plan.md` (new implementation plan)
  - `findings.md` (new mainline notes)
  - `progress.md` (this session entry)

### Physical-Event Scaffold Implementation
- **Status:** complete
- Actions taken:
  - Added `configs/sensors/windblown_sensors_physical_event_v2.yaml`.
  - Added optional passthroughs to `scripts/41_v31_pilot.py` and `scripts/42_v31_s2_full.py`:
    - `--target-weights`
    - `--target-scales`
    - `--required-sensors`
    - `--disable-coverage-groups`
    - `--max-active`
  - Added `scripts/48_v31_physical_event_preflight.py` for feasible-subset preflight under fixed budgets.
  - Ran local dry-runs for both V3.1 wrappers with:
    - sensor config `configs/sensors/windblown_sensors_physical_event_v2.yaml`
    - `B=1.20`
    - `startup_peak_budget=1.60`
    - `required_sensors=met_station_core`
    - coverage groups disabled
    - target weights `0.8 0.8 1.2 0.4 0.4 0.55 4.0 2.5 2.5`
  - Ran the physical-event preflight for `B=1.00/1.10/1.20`.
  - Applied first-pass paper claim repair:
    - removed "time-varying power budget" from the PD-PPO architecture caption;
    - downgraded unsupported event-conditioned warm-up language around Figure 8/timeline interpretation;
    - changed the abstract wording to "multi-year Antarctic AWS statistics";
    - removed the conclusion wording implying established warm-up-aware particle/flux selection.
    - clarified in the problem formulation that fixed `B` and peak budgets are simplified deployment-cost limits, not a full battery SOC or time-varying energy-harvesting model.
- Key preflight result:
  - `B=1.00`: no laser-feasible subsets.
  - `B=1.10`: one laser-feasible subset, no laser+FC4 subset.
  - `B=1.20`: six laser-feasible subsets and one laser+FC4 subset.
  - `met_station_core + fc4_flux + laser_disdrometer` has steady power `1.13` and cold-start peak `1.54`, so it is feasible under `B=1.20`, `startup_peak_budget=1.60`.
- Validation:
  - `py_compile` passed for `41_v31_pilot.py`, `42_v31_s2_full.py`, and `48_v31_physical_event_preflight.py`.
  - `rg` found no remaining occurrences of the strongest unsupported phrases:
    - `time-varying power budget`
    - `adaptive scheduling necessary`
    - `selectively warms`
    - `event-conditioned warm-up decisions`
    - `warm-up-aware selection`
    - `multi-season`

### Physical-Event Oracle-Lift Diagnostics
- **Status:** complete; gate failed
- Actions taken:
  - Added `scripts/49_v31_physical_event_oracle_lift.py`.
  - Ran local smoke test with a small linear oracle after correcting the local AntAWS path.
  - The first local smoke attempt failed because `rl_sensor_scheduling_framework/data/AntAWS/3_hourly` did not contain `Panda100_3h.csv`.
  - Retried with local `data/AntAWS/3_hourly` and stations `Panda200 Taishan`; the smoke test completed and wrote `reports/physical_event_v2_oracle_lift_smoke/`.
  - Smoke-test warning: small linear oracle produced negative event lift for laser and FC4, so it should be treated as a gate warning rather than final evidence.
  - Checked the server: no old microclimate tmux sessions or Python jobs were running; one unrelated GPU process from another project remains active and was not touched.
  - Synced the new physical-event config and scripts to the server.
  - Started server tmux `physical_event_oracle_lift_20260522` for the formal TCN oracle-lift diagnostic:
    - output directory `rl_sensor_scheduling_framework/reports/physical_event_v2_oracle_lift_tcn_b120_seed41`
    - `B=1.20`
    - `startup_peak_budget=1.60`
    - `sensor_cfg=configs/sensors/windblown_sensors_physical_event_v2.yaml`
    - `required_sensors=met_station_core`
    - coverage groups disabled by script construction
    - TCN oracle, 18 epochs, full V3.1-style truth length
  - Confirmed the formal server TCN oracle-lift diagnostic completed and synchronized outputs locally to `rl_sensor_scheduling_framework/reports/physical_event_v2_oracle_lift_tcn_b120_seed41/`.
  - Stopped the completed tmux session after sync.
  - Interpreted the Phase 2 gate as failed:
    - best event subset: `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - best event laser subset: `met_station_core|surface_temp_ir|laser_disdrometer`;
    - best event FC4 subset: `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`;
    - `laser_event_lift = -0.0342`;
    - `fc4_event_lift = -0.0221`.
  - Decided not to launch the planned PPO pilot from `physical_event_value_v2`.
  - Inspected the V2 sensor/environment implementation and found that observation noise is currently fixed per variable, not event-conditioned.
  - Identified the likely upstream failure mode: `laser_disdrometer` observes the same particle variables as `snow_particle_counter` but with higher cost, longer warmup, and longer refresh; without event-dependent sensor quality or a unique laser-observed state, the oracle rationally prefers the cheaper particle counter.
- Files created/modified:
  - `rl_sensor_scheduling_framework/scripts/49_v31_physical_event_oracle_lift.py`
  - `reports/physical_event_v2_oracle_lift_smoke/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v2_oracle_lift_tcn_b120_seed41/`
  - `task_plan.md` (marked Phase 2 gate failure and added Phase 2b)
  - `progress.md` (this update)
- Errors:
  - Local smoke attempt 1 failed due to wrong local AntAWS path; resolved by using `data/AntAWS/3_hourly` and local-available stations.

### Phase 2b: Event-Sensitive Observation Repair
- **Status:** in_progress
- Actions taken:
  - Converted the failed no-retrain gate into a new repair phase rather than spending compute on PPO.
  - Started code inspection for the least invasive implementation point:
    - `src/v2/sensor_spec.py` currently parses `noise_std` only;
    - `src/v2/env.py` samples fixed Gaussian noise from `spec.noise_std`;
    - therefore event-conditioned low-cost sensor degradation requires extending the V2 sensor spec and one observation-noise call site.
  - Extended `SensorSpecV2` with optional `event_noise_std` and `event_noise_multiplier`.
  - Updated `WarmupSchedulingEnv` so event steps use event-conditioned observation noise when configured; existing sensor YAMLs remain backward compatible.
  - Added `configs/sensors/windblown_sensors_physical_event_v3.yaml`.
  - Added optional target-diagnostic and dynamic schedule diagnostic modes to `49_v31_physical_event_oracle_lift.py`.
  - Added optional event-microstructure terms to `PublicWeatherSynthesisConfig` and the V3.1 truth-generation CLI:
    - lognormal multiplicative flux perturbation during blowing-snow events;
    - coherent particle-diameter and particle-velocity perturbations;
    - defaults remain zero, preserving old truth generation.
  - Local diagnostics:
    - v3 feasibility is unchanged from v2 at the target budget: `B=1.20` has 48 feasible projected subsets, 6 laser subsets, and 1 laser+FC4 subset.
    - v3 fixed-observation smoke without microstructure still failed for laser.
    - snow-heavy linear smoke still failed for laser/FC4, indicating that weights alone do not repair the scenario.
    - dynamic schedule probe with a small TCN and event microstructure gave the first non-degenerate signal: best overall was `dynamic:calm_core__event_laser_surface`, while static laser event lift was approximately tied with static snow-counter.
  - Synchronized the v3 sensor config, generator changes, and oracle-lift script to the GPU server.
  - Added microstructure passthroughs to `25_v2_train_custom_ppo.py`, `41_v31_pilot.py`, and `42_v31_s2_full.py` for possible gated pilots.
  - Validated `41_v31_pilot.py --dry-run` with v3 sensor config, event microstructure, snow-heavy weights, required `met_station_core`, and disabled coverage groups.
  - Synchronized the updated training wrappers to the GPU server.
  - Started server tmux `physical_event_v3_micro_tcn_20260522` for a formal TCN oracle-lift diagnostic:
    - output directory `rl_sensor_scheduling_framework/reports/physical_event_v3_microstructure_tcn_b120_seed41`;
    - `B=1.20`, `startup_peak_budget=1.60`;
    - `event_microstructure_sigma=0.8`;
    - snow-heavy weights `0.2 0.3 0.2 0.1 0.1 0.1 12.0 8.0 8.0`;
    - schedule diagnostics enabled;
    - PPO still not launched.
  - Formal server TCN diagnostic completed and was synchronized locally to `reports/physical_event_v3_microstructure_tcn_b120_seed41/`.
  - Stopped the completed tmux session after syncing outputs.
  - Formal gate failed:
    - best event subset remained `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - best laser event subset was `met_station_core|radiometer_basic|laser_disdrometer`;
    - `laser_event_lift = -0.0047`;
    - `fc4_event_lift = -0.0137`;
    - dynamic schedule probes were worse than the best fixed subset.
  - Decided not to launch PPO from the current v3 microstructure scenario.
  - New diagnosis: microstructure plus event noise narrows the laser gap, but the low-cost snow counter remains too reliable under dense events. The next repair should model event saturation/partial availability for low-cost particle sensing.
  - Added optional `event_observation_probability` to `SensorSpecV2`.
  - Updated `WarmupSchedulingEnv` to skip individual variable observations according to event-time availability probabilities.
  - Added `configs/sensors/windblown_sensors_physical_event_v4.yaml`, where `snow_particle_counter` has event-time partial availability for particle diameter and velocity.
  - Local v4 smoke:
    - feasibility unchanged at `B=1.20`: 48 feasible subsets, 6 laser subsets, 1 laser+FC4 subset;
    - small TCN smoke produced positive laser event lift: `laser_event_lift = +0.0083`;
    - best event laser subset was `met_station_core|surface_temp_ir|laser_disdrometer`;
    - best overall in smoke was `dynamic:calm_core__event_laser_surface`.
  - Synchronized v4 sensor/env/spec changes to the server.
  - Started formal server TCN gate in tmux `physical_event_v4_saturation_tcn_20260522`:
    - output directory `rl_sensor_scheduling_framework/reports/physical_event_v4_saturation_tcn_b120_seed41`;
    - same v3 microstructure and snow-heavy weights;
    - sensor config `configs/sensors/windblown_sensors_physical_event_v4.yaml`;
    - PPO still not launched.
  - Formal v4 server TCN gate completed and was synchronized locally to `reports/physical_event_v4_saturation_tcn_b120_seed41/`.
  - Formal v4 result:
    - `best_overall = met_station_core|surface_temp_ir|laser_disdrometer`;
    - `best_event_laser = met_station_core|surface_temp_ir|laser_disdrometer`;
    - `best_event_no_laser = met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - `laser_event_lift = +0.0187`;
    - `fc4_event_lift = -0.0221`.
  - Dynamic schedule rows were still worse than the best fixed subset, so fixed-budget PPO remains blocked.
  - Ran a local `B=1.10` v4 smoke to test whether a tighter per-step budget creates dynamic value; it did not pass (`laser_event_lift = -0.0040`).
  - Decision: v4 establishes event value for laser, but fixed per-step constraints still favor static subsets. Move next to the energy-account/SOC gate rather than launching PPO.
- Files created/modified:
  - `rl_sensor_scheduling_framework/src/v2/sensor_spec.py`
  - `rl_sensor_scheduling_framework/src/v2/env.py`
  - `rl_sensor_scheduling_framework/src/data_sources/public_weather_synthesis.py`
  - `rl_sensor_scheduling_framework/scripts/20_build_public_weather_truth.py`
  - `rl_sensor_scheduling_framework/scripts/23_v2_train_ppo.py`
  - `rl_sensor_scheduling_framework/scripts/25_v2_train_custom_ppo.py`
  - `rl_sensor_scheduling_framework/scripts/41_v31_pilot.py`
  - `rl_sensor_scheduling_framework/scripts/42_v31_s2_full.py`
  - `rl_sensor_scheduling_framework/scripts/49_v31_physical_event_oracle_lift.py`
  - `rl_sensor_scheduling_framework/configs/sensors/windblown_sensors_physical_event_v3.yaml`
  - `rl_sensor_scheduling_framework/configs/sensors/windblown_sensors_physical_event_v4.yaml`
  - `rl_sensor_scheduling_framework/reports/physical_event_v3_preflight/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v3_microstructure_tcn_smoke/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v3_microstructure_tcn_b120_seed41/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v4_preflight/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v4_saturation_tcn_smoke/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v4_saturation_tcn_b120_seed41/`
  - `rl_sensor_scheduling_framework/reports/physical_event_v4_saturation_b110_tcn_smoke/`
- Validation:
  - `py_compile` passed for the modified V2 sensor/env/generator/oracle-lift files.

### Phase 5: Energy-Account Architecture Gate
- **Status:** in_progress
- Actions taken:
  - Inspected `PowerProjector`, `SensorRuntime`, `WarmupSchedulingEnv`, and the SB3 wrapper state/action path.
  - Determined that the current fixed-budget implementation has only instantaneous steady/peak constraints and no SOC/cumulative energy state.
  - Created `rl_sensor_scheduling_framework/docs/05-22-energy-account-design.md`.
  - Proposed a minimal normalized energy-account model that preserves instantaneous peak safety while adding:
    - `energy_capacity`;
    - `initial_energy`;
    - `harvest_per_step`;
    - `reserve_energy`;
    - SOC appended to agent state;
    - diagnostics for SOC and deficit.
  - Recorded why energy-account is now necessary: v4 makes laser valuable, but fixed per-step constraints still make static laser best at `B=1.20`.
  - Implemented a default-off energy-account path in `WarmupSchedulingEnv`:
    - SOC state is appended only when `energy_account_enabled=True`;
    - actions are guarded against spending below reserve by dropping optional sensors;
    - step info records `soc`, `soc_ratio`, `energy_deficit`, `energy_guard_dropped`, and cumulative deficit.
  - Added energy-account CLI options to `49_v31_physical_event_oracle_lift.py`.
  - Extended oracle-lift candidate and schedule tables with `soc_min`, `energy_deficit_steps`, `energy_deficit_total`, and `energy_guard_dropped`.
  - Local energy smoke results:
    - first parameter set `capacity=24, initial=12, harvest=0.65, reserve=2` was too tight and clipped dynamic laser heavily;
    - second set `capacity=48, initial=24, harvest=0.80, reserve=4` improved dynamic laser but still clipped it;
    - third set `capacity=48, initial=24, harvest=0.95, reserve=4` passed the local gate:
      - best overall: `dynamic:calm_core__event_laser_surface`;
      - `oracle_loss_mean=0.6326`;
      - `laser_event_lift=+0.0060`;
      - `soc_min=13.92`;
      - `energy_guard_dropped=0`.
  - Synchronized the energy-account env and oracle-lift script to the server.
  - Started formal server TCN gate in tmux `physical_event_v4_energy_tcn_20260522`.
  - Formal non-lead energy gate failed to reproduce local smoke:
    - best overall returned to `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - `laser_event_lift = -0.0056`;
    - no PPO was launched.
  - Added lead-time schedule diagnostics to `49_v31_physical_event_oracle_lift.py`.
  - Local lead-aware smoke passed:
    - best overall: `dynamic:calm_core__lead4_laser_surface`;
    - `laser_event_lift=+0.0060`;
    - no SOC guard drops for the leading dynamic laser schedule.
  - Started formal server lead-aware TCN gate in tmux `physical_event_v4_energy_lead_tcn_20260522`.
  - Formal lead-aware `harvest=0.95` gate also failed:
    - best overall remained `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - `laser_event_lift=-0.0012`;
    - lead laser schedule was clipped heavily by the energy guard.
  - Local `harvest=1.05` smoke passed more cleanly:
    - best overall: `dynamic:calm_core__lead4_laser_surface`;
    - `laser_event_lift=+0.0078`;
    - lead laser schedule had only 2 guard drops in local smoke;
    - static laser still had 30 guard drops.
  - Started one final formal server lead-aware energy gate with `harvest_per_step=1.05` in tmux `physical_event_v4_energy_lead_h105_tcn_20260522`.
  - Formal `harvest=1.05` lead-aware gate completed and was synchronized locally to `reports/physical_event_v4_energy_lead_h105_tcn_b120_seed41/`.
  - Formal result:
    - `laser_event_lift = +0.0064`, confirming positive event value for laser under the v4 + energy-account setting;
    - best overall remained `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
    - best fixed laser row was close in overall loss but had many energy guard drops;
    - lead dynamic schedules improved event loss but did not beat the static snow-counter subset overall.
  - Stopped the completed tmux session and confirmed no remaining server tmux sessions.
  - Decision: do not launch PPO yet. The current evidence supports the mechanism direction but not a stable training target that would clearly demonstrate event-conditioned scheduling.
  - Revisited the energy account after the user pointed out that harvest must be calibrated from event-cluster statistics.
  - Added `scripts/50_v31_energy_account_calibrate.py`.
  - Ran calibration on formal truth:
    - event fraction `0.2701`;
    - mean event run length `17.85`;
    - mean calm run length `48.23`;
    - median calm run length `5`;
    - lead4 trigger fraction `0.3307`;
    - lead4 dynamic average cost `0.5978`.
  - Key correction:
    - the user's formula is directionally right, but for the current diagnostic schedule the harvest lower bound must include calm-core and lead-trigger costs, not only `met + laser * event_fraction`.
    - capacity must be sized from event-burst drawdown, not only average harvest.
  - Local calibrated smoke with `harvest=0.62`, `capacity=300`, `initial_energy=300`, `reserve=20` passed:
    - best overall `dynamic:calm_core__lead4_laser_surface`;
    - `laser_event_lift=+0.0083`;
    - no guard drops.
  - Started formal server calibrated gate in tmux `physical_event_v4_energy_cal_h062_cap300_tcn_20260522`.
  - Synchronized and inspected formal calibrated results:
    - `harvest=0.62`, `capacity=300`: static snow-counter core remained best; do not launch PPO.
    - `harvest=0.94`, `capacity=300`: static laser became best overall, confirming this harvest is too high for dynamic-advantage evidence.
  - Added additional schedule diagnostics to `49_v31_physical_event_oracle_lift.py`:
    - `snow_core__event_laser_surface`
    - `snow_core__event_laser_fc4`
    - `snow_core__lead4_snow_laser_surface`
    - `snow_core__lead4_snow_laser_fc4`
  - Ran a smaller local smoke for the new schedules; it did not pass, but the small oracle was only used to check execution and schedule diagnostics.
  - Synchronized the updated oracle-lift script to the server and started formal gate `physical_event_v4_energy_cal_h062_cap300_newsched_tcn_20260522`.
  - Formal `h=0.62`, `capacity=300`, new-schedule gate completed:
    - best overall still static snow-counter core (`oracle_loss_mean=0.3364`);
    - no-lead `snow_core__event_laser_fc4` became the best event policy (`oracle_loss_event=0.3882`) but suffered many SOC guard drops and poor non-event loss;
    - conclusion: the event value is now clear, but this capacity/harvest pair is not the right physical window.
  - Ran drawdown calibration for no-lead snow-core/event-laser:
    - average dynamic cost is about `0.912`;
    - at `harvest=0.92`, dynamic eval drawdown is about `27--91`, while static laser drawdown is about `246`;
    - selected `capacity=120`, `reserve=20` to allow dynamic event bursts while still clipping static laser.
  - Started formal server gate `physical_event_v4_energy_cal_h092_cap120_newsched_tcn_20260522`.
  - Formal `h=0.92`, `capacity=120` all-window gate completed:
    - static snow-counter core still won overall (`oracle_loss_mean=0.3353`);
    - best dynamic rows were close (`snow_core__lead4_snow_laser_surface=0.3413`, `snow_core__event_laser_surface=0.3418`);
    - event-laser value is strong (`laser_event_lift=+0.0230`) but diluted by the 27% full-truth event fraction.
  - Computed the event-fraction break-even point from the formal table:
    - `snow_core__event_laser_surface` needs about `p_event=0.659`;
    - `snow_core__event_laser_fc4` needs about `p_event=0.548`.
  - Added `--eval-start-indices` to the oracle-lift script.
  - Started storm-window formal gate `physical_event_v4_energy_cal_h092_cap120_storm_tcn_20260522` using event-rich 1024-step windows: `22943`, `6826`, `21704`, `8183`, `17193`, `16151`.
  - Formal storm-window gate passed in the limited sense:
    - best overall: `dynamic:snow_core__event_laser_fc4`;
    - `oracle_loss_mean=0.4177` versus static snow core `0.4254`;
    - event loss improved strongly (`0.3208` versus static snow event `0.3529`);
    - the best dynamic row still had 72 SOC guard drops, so the result needs a cleaner capacity check.
  - Started `physical_event_v4_energy_cal_h092_cap180_storm_tcn_20260522` to test whether a slightly larger capacity removes dynamic guard drops while still clipping static laser.
  - Formal `h=0.92`, `capacity=180` storm-window gate completed and was synchronized:
    - best overall remains `dynamic:snow_core__event_laser_fc4`;
    - `oracle_loss_mean=0.4169` versus static snow core `0.4248`;
    - event loss `0.3190` versus static snow event `0.3517`;
    - no guard drops and no warmup aborts for the best dynamic row;
    - static laser rows remain clipped, e.g. `met_station_core|radiometer_basic|laser_disdrometer` has 438 guard drops.
  - Confirmed no remaining remote tmux sessions.
  - Reviewed `docs/05-22-judge.md`:
    - accepted the central conclusion that calibrated energy-account supports storm-window dynamic advantage but not full-distribution superiority;
    - corrected several paper-risk details in `docs/05-22-judge-review.md`, including PD-PPO vs static wording, h=0.94 interpretation, implemented cost arithmetic, and event-flag assumptions.
  - Added paper assets for the storm-window diagnostic:
    - `paper/tables/energy_account_storm_oracle.tex`;
    - new `Energy-Account Storm-Window Diagnostic` subsection in `paper/sections/06_experiments.tex`;
    - limitation paragraph in `paper/sections/07_discussion.tex`.
  - Compiled `paper/main.tex` successfully with `latexmk`; only existing overfull/BibTeX warnings remain.
  - Added energy-account and explicit `--eval-start-indices` support to `scripts/25_v2_train_custom_ppo.py`, and propagated energy-account fields inside `src/v2/custom_ppo.py`.
  - Synchronized the PPO changes to the server and started single-seed storm-window PPO pilot:
    - tmux: `physical_event_v4_energy_ppo_h092_cap180_storm_seed41_20260522`
    - output: `reports/physical_event_v4_energy_ppo_h092_cap180_storm_seed41`
    - parameters: `B=1.20`, `harvest=0.92`, `capacity=180`, `reserve=20`, storm-window eval starts.
  - First server check passed: oracle file exists and Python process is running.
  - PPO pilot completed and was synchronized locally:
    - `custom_ppo` oracle loss on storm windows: `0.4195`;
    - hand-written oracle dynamic reference: `0.4169`;
    - static snow-core oracle reference: `0.4248`;
    - AoI in the same PPO evaluation: `0.4144`;
    - `custom_ppo` warmup aborts: `14`;
    - laser selected in events/non-events: `0.5487 / 0.2983`, ratio `1.84`.
  - Interpretation: PPO pilot partially supports learnability (beats static snow-core reference and uses laser more during events), but does not satisfy the stricter success criteria (`>3:1` event/non-event laser ratio and `<10` warmup aborts), and does not beat AoI in this single seed.
  - Confirmed no remaining remote tmux sessions.
  - Added curriculum start-index support to PPO:
    - `src/v2/custom_ppo.py` now samples training episodes from optional `train_start_indices`;
    - `scripts/25_v2_train_custom_ppo.py` now accepts `--train-start-indices` and records them in metadata.
  - Verified local and remote `py_compile` for the modified PPO files.
  - Launched storm-window curriculum PPO pilot on the server:
    - tmux: `physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed41_20260523`;
    - output: `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed41`;
    - training and evaluation starts: `22943 6826 21704 8183 17193 16151`;
    - parameters: `B=1.20`, `harvest=0.92`, `capacity=180`, `reserve=20`, `100000` PPO timesteps.
  - Curriculum PPO pilot completed and was synchronized locally:
    - `custom_ppo` oracle loss: `0.4106`;
    - AoI oracle loss: `0.4130`;
    - round-robin oracle loss: `0.4720`;
    - random oracle loss: `0.4660`;
    - feasible static projected oracle loss: `0.5032`;
    - warmup aborts: PPO `5`, AoI `10`, round-robin `768`, random `1956`.
  - Behavior diagnosis:
    - PPO beats AoI in this curriculum setting, so learnability improved.
    - The learned strategy is not the ideal event-laser policy: `laser_disdrometer` selected rate is `0.5187` during events and `0.7440` during non-events.
    - Event-conditioned increases are clearer for `radiometer_basic` (`2.26x`), `snow_particle_counter` (`1.93x`), and `fc4_flux` (`1.35x`).
  - Interpretation:
    - the curriculum pilot supports a narrow learned-policy advantage on storm windows;
    - it still does not justify claiming that PPO reliably learns oracle-like event-triggered laser activation.
  - Started two additional curriculum PPO seeds for robustness:
    - tmux `physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed42_20260523`, output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed42`;
    - tmux `physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed43_20260523`, output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed43`.
  - Both additional curriculum PPO seeds completed and were synchronized locally.
  - Cleaned up the completed PPO tmux session; unrelated `ecbit_mcar_noera5*` sessions were left untouched.
  - Three-seed summary:
    - PPO oracle loss: seed 41 `0.4106`, seed 42 `0.4206`, seed 43 `0.4096`;
    - AoI oracle loss: seed 41 `0.4130`, seed 42 `0.4356`, seed 43 `0.4096`;
    - mean PPO `0.4136 ± 0.0061`, mean AoI `0.4194 ± 0.0141`;
    - PPO warmup aborts: `5 / 33 / 19`, mean `19.0`;
    - AoI warmup aborts: `10` in each seed; round-robin `768`; random about `2094`.
  - Behavior summary over seeds:
    - event bias is strongest for `radiometer_basic` (`1.63x`);
    - `laser_disdrometer` is nearly event-neutral on average (`1.03x`);
    - therefore the result is a learned storm-window advantage, not robust event-laser gating.
  - Ran mechanism diagnosis on existing rollout arrays:
    - reconstructed per-step aborts from `mode_ids`, `selected_masks`, and contiguous `step_indices`;
    - reconstruction exactly matched stored abort totals.
  - Diagnosis result:
    - PPO laser aborts across seeds: `41` total, `10` event, `31` non-event;
    - PPO snow-counter aborts: `16` total, `11` event, `5` non-event;
    - PPO event/non-event oracle loss: `0.3274 / 0.5256`;
    - AoI event/non-event oracle loss: `0.3243 / 0.5431`.
  - Conclusion:
    - low laser event bias is not primarily explained by event-time aborts;
    - PPO's advantage mainly comes from non-event loss reduction while keeping event loss close to AoI.
  - Created `rl_sensor_scheduling_framework/docs/05-23-curriculum-ppo-mechanism-diagnosis.md`.
  - Implemented a default-off event-step reward multiplier:
    - `WarmupEnvConfig.event_reward_multiplier`, default `1.0`;
    - training reward multiplies base oracle/error loss on event steps only;
    - `scripts/25_v2_train_custom_ppo.py` exposes `--event-reward-multiplier` and records it in metadata.
  - Validation:
    - local `py_compile` passed for `src/v2/env.py`, `src/v2/custom_ppo.py`, and `scripts/25_v2_train_custom_ppo.py`;
    - remote `py_compile` passed after syncing the same files.
  - Started single-seed mechanism probe:
    - tmux `physical_event_v4_energy_ppo_h092_cap180_stormcurr_evt15_seed41_20260523`;
    - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_evt15_seed41`;
    - same storm-window curriculum setup as seed 41, with `--event-reward-multiplier 1.5`.
  - Event multiplier probe completed and was synchronized locally.
  - Result:
    - PPO oracle loss `0.4088` vs AoI `0.4118`;
    - baseline seed-41 PPO oracle loss was `0.4106`;
    - PPO event loss improved from `0.3307` to `0.3247`;
    - PPO non-event loss worsened slightly from `0.5144` to `0.5180`;
    - warmup aborts `7`, still below AoI `10`.
  - Mechanism result:
    - laser selected rate became `0.740` in event steps and `0.768` in non-event steps;
    - this is not event-triggered laser gating.
    - non-event laser use remains high even far from event steps, so it is not only warmup bridging across short calm gaps.
  - Added rollout instrumentation for future experiments:
    - `warmup_abort_deltas`;
    - `energy_guard_dropped`;
    - `soc`.
  - Patched `scripts/32_v2_condition_eval.py` for no-retrain generalization evaluation:
    - added `all` episode type for random full-distribution windows;
    - propagated energy-account and event-multiplier metadata into evaluation envs;
    - added `--skip-summary-eval` to avoid slow DTW-heavy summary evaluation when only oracle/behavior diagnostics are needed.
  - Ran fast full-distribution evaluation for already trained curriculum PPO seeds `41/42/43`:
    - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast`;
    - six 1024-step random windows per seed;
    - mean event rate about `0.296`.
  - Full-distribution result:
    - PPO mean oracle loss `0.3130 +/- 0.0171`;
    - AoI mean oracle loss `0.3160 +/- 0.0171`;
    - feasible static projected mean oracle loss `0.3309 +/- 0.0077`;
    - round-robin `0.3422 +/- 0.0241`;
    - random `0.3471 +/- 0.0236`.
  - Boundary:
    - PPO beats AoI in all three seeds;
    - PPO does not beat static in every seed, because seed 42 static projected is `0.3220` vs PPO `0.3311`;
    - laser remains near event-neutral (`1.04x`), so the mechanism interpretation is unchanged.
  - Implemented default-off SOC soft penalty:
    - `WarmupEnvConfig.soc_soft_penalty_buffer`, default `0.0`;
    - `WarmupEnvConfig.lambda_soc_soft_penalty`, default `0.0`;
    - when enabled, adds a constant penalty if SOC after the energy update is below `reserve + buffer`.
  - Propagated SOC soft-penalty fields through custom PPO training/evaluation and condition evaluation metadata.
  - Validation:
    - local and remote `py_compile` passed.
  - Launched three parallel training jobs:
    - baseline curriculum seed 44: `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed44`;
    - baseline curriculum seed 45: `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed45`;
    - SOC soft-penalty probe seed 41: `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_soc001_seed41`, with `soc_soft_penalty_buffer=20` and `lambda_soc_soft_penalty=0.01`.
  - Completed and synchronized all three runs.
  - Storm-window five-seed curriculum result:
    - PPO mean oracle loss `0.4153 +/- 0.0051`;
    - AoI mean oracle loss `0.4176 +/- 0.0105`;
    - feasible static projected `0.4742 +/- 0.0236`;
    - round-robin `0.4451 +/- 0.0167`;
    - random `0.4565 +/- 0.0140`.
  - Per-seed storm-window boundary:
    - PPO beats AoI in `3/5` seeds, not `5/5`;
    - PPO beats feasible static projected, round-robin, and random in `5/5` seeds;
    - new seeds 44 and 45 lose narrowly to AoI (`-0.0021`, `-0.0040` margin from AoI to PPO), so the robust AoI-dominance claim is no longer supported.
  - Full-distribution no-retrain evaluation was extended to seeds 44/45:
    - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_seed44_45_fast`;
    - combined five-seed mean event rate about `0.321`.
  - Full-distribution five-seed result:
    - PPO mean oracle loss `0.3155 +/- 0.0133`;
    - AoI `0.3168 +/- 0.0135`;
    - feasible static projected `0.3318 +/- 0.0062`;
    - round-robin `0.3375 +/- 0.0195`;
    - random `0.3431 +/- 0.0188`.
  - Full-distribution boundary:
    - PPO beats AoI in `4/5` seeds;
    - PPO beats feasible static projected in `4/5` seeds, with seed 42 still the exception;
    - PPO beats round-robin and random in `5/5` seeds.
  - SOC soft-penalty probe result, seed 41:
    - baseline PPO storm oracle loss `0.4106`, aborts `5`, power `1.0128`;
    - SOC-penalty PPO storm oracle loss `0.4069`, aborts `9`, power `0.9960`;
    - event loss improved (`0.3307 -> 0.3234`) while non-event loss slightly worsened (`0.5144 -> 0.5155`);
    - SOC minimum stayed at the reserve boundary (`20.0`), but energy guard still dropped `31` sensor selections.
  - Interpretation:
    - n=5 weakens the claim "PPO reliably beats AoI" but strengthens "PPO reliably beats static/round-robin/random in storm windows";
    - full-distribution generalization still exists on average, but the margin over AoI is small;
    - SOC penalty is promising for event loss and power reduction, but it did not reduce aborts in this seed, so it should not be merged without a multi-seed check.
  - Completed 300k-step seed-41 training probe:
    - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_300k_seed41`;
    - same truth, oracle, energy account, train/eval storm windows, and hyperparameters as 100k seed 41, except `total_timesteps=300000`.
  - Storm-window 300k result:
    - PPO oracle loss improved from 100k `0.4106` to `0.4053`;
    - AoI in same 300k run is `0.4125`;
    - event loss improved from `0.3307` to `0.3190`;
    - non-event loss worsened slightly from `0.5144` to `0.5173`;
    - mean power dropped from `1.0128` to `0.9673`;
    - warmup aborts increased from `5` to `66`.
  - 300k mechanism shift:
    - laser event/non-event selected ratio improved from `0.70x` to `1.52x`;
    - radiometer event bias disappeared (`2.26x -> 1.01x`);
    - snow-counter became non-event-biased (`1.93x -> 0.68x`).
  - 300k full-distribution no-retrain evaluation:
    - PPO `0.3122`, AoI `0.3118`, static projected `0.3367`, round-robin `0.3697`, random `0.3413`;
    - PPO loses narrowly to AoI on full-distribution seed-41 evaluation, despite beating static and other heuristics;
    - full-distribution PPO aborts rise to `75`.
  - Interpretation:
    - longer training does improve the storm-window objective and makes laser more event-biased;
    - it does not solve the AoI problem in full distribution, and it increases aborts substantially;
    - training length alone is not enough for a robust "PPO全面优于AoI" claim.
  - Implemented and tested default-off event-gated actor:
    - `CustomPPOConfig.event_gated_actor`, default `False`;
    - `scripts/25_v2_train_custom_ppo.py --event-gated-actor`;
    - actor uses a shared encoder plus an event encoder mixed by a learnable soft gate initialized near non-event/event specialization.
  - Event-gated 200k seed-41 probe completed:
    - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_eventgated_200k_seed41`;
    - storm-window PPO oracle loss `0.4105`, AoI `0.4128`;
    - warmup aborts `38`;
    - laser event/non-event ratio `0.78x`;
    - event/non-event loss `0.3249 / 0.5219`.
  - Full-distribution event-gated 200k evaluation:
    - PPO `0.3128`, AoI `0.3117`, static projected `0.3350`, round-robin `0.3701`, random `0.3412`;
    - PPO loses to AoI and beats static/round-robin/random;
    - full-distribution aborts `35`, laser ratio `1.07x`.
  - Decision:
    - the minimal event-gated actor did not meet success criteria (`abort < 20`, laser ratio `> 1.3x`, and full-distribution PPO > AoI);
    - it reduces aborts compared with 300k (`66 -> 38` storm) but loses the strong laser event bias and does not improve full-distribution AoI competitiveness.
  - Started new implementation phase for a default-off SOC auxiliary critic probe:
    - selected over hierarchical policy because it directly targets long-horizon SOC credit assignment with much lower implementation and interpretation risk;
    - planned probe: `soc_aux_horizon=16`, `soc_aux_coef=0.1`, `200k` seed 41, event-gated actor disabled.
- Files created/modified:
  - `rl_sensor_scheduling_framework/docs/05-22-energy-account-design.md`
  - `rl_sensor_scheduling_framework/src/v2/env.py`
  - `rl_sensor_scheduling_framework/scripts/49_v31_physical_event_oracle_lift.py`
  - `task_plan.md`
  - `progress.md`

## 2026-05-24 SOC auxiliary critic implementation
- Implemented default-off SOC auxiliary critic support in `src/v2/custom_ppo.py`:
  - added rollout collection of per-step `soc_ratio` and episode ids;
  - added masked future-SOC target construction;
  - added critic-side SOC prediction head and auxiliary MSE loss controlled by `soc_aux_horizon` and `soc_aux_coef`;
  - added `soc_aux_loss` to update metrics and printed training diagnostics.
- Exposed `--soc-aux-horizon` and `--soc-aux-coef` in `scripts/25_v2_train_custom_ppo.py`, including metadata, ablation switches, and training-log output.
- Local validation:
  - `python -m py_compile rl_sensor_scheduling_framework/src/v2/custom_ppo.py rl_sensor_scheduling_framework/scripts/25_v2_train_custom_ppo.py` passed;
  - `python scripts/25_v2_train_custom_ppo.py --help` shows SOC auxiliary flags;
  - `conda activate darts; pytest -q rl_sensor_scheduling_framework/tests/v2/test_custom_ppo.py` passed (`5 passed`).
- Synced `custom_ppo.py` and `25_v2_train_custom_ppo.py` to the GPU server and remote `py_compile` passed.
- Launched SOC auxiliary seed-41 probe on server:
  - tmux: `physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_seed41_20260524`;
  - output: `rl_sensor_scheduling_framework/reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_seed41`;
  - setting: `200k`, `soc_aux_horizon=16`, `soc_aux_coef=0.1`, event-gated actor disabled, same h=0.92/cap180 storm curriculum and eval windows as seed-41 baseline.

## 2026-05-24 SOC auxiliary probe results
- Server run completed:
  - `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_seed41`;
  - tmux session exited normally.
- Storm-window result:
  - PPO `oracle_loss_mean=0.410464` vs AoI `0.414366`, feasible static `0.503722`, round-robin `0.469592`, random `0.468802`;
  - PPO warmup aborts `16` vs AoI `10`, round-robin `768`, random `1956`;
  - mean power `0.8949`, lower than AoI `1.0039`.
- Full-distribution no-retrain eval completed:
  - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_full_eval_fast/all/budget1p20_seed41`;
  - PPO `0.313817` vs AoI `0.312702`, static `0.336291`, round-robin `0.366520`, random `0.342811`;
  - PPO warmup aborts `12` vs AoI `6`.
- Interpretation:
  - SOC auxiliary probe partially passes: it keeps storm-window PPO ahead of AoI and controls aborts below the planned `<20` threshold;
  - it fails the full-distribution criterion because PPO remains slightly worse than AoI on seed 41;
  - auxiliary SOC prediction is learnable, but by itself does not solve the full-distribution AoI dominance problem.

## 2026-05-24 results convergence assets
- Added reproducible convergence asset script:
  - `rl_sensor_scheduling_framework/scripts/52_energy_account_convergence_assets.py`.
- Synchronized SOC auxiliary storm/full-eval results from the server so the local convergence assets include the latest probe.
- Generated convergence assets:
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_main_long.csv`;
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_main_summary.csv`;
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_probe_summary.csv`;
  - `rl_sensor_scheduling_framework/docs/05-24-results-convergence.md`;
  - `rl_sensor_scheduling_framework/paper/tables/energy_account_curriculum_results.tex`.
- Updated paper text:
  - `paper/sections/06_experiments.tex` now separates the energy-account oracle schedule gate from the learned 100k curriculum n=5 result and explicitly avoids robust AoI-dominance claims;
  - `paper/sections/07_discussion.tex` now states the energy-account/latent-event limitations and the non-uniform AoI comparison.
- Verification:
  - `python rl_sensor_scheduling_framework/scripts/52_energy_account_convergence_assets.py` regenerated all CSV/LaTeX/MD assets successfully;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`.

## 2026-05-24 front-matter claim alignment
- Updated `paper/main.tex` abstract to include the calibrated energy-account result and explicitly bound the AoI claim:
  - fixed-budget V3.1 still reports 12.1% lower mean error than AoI;
  - energy-account result now states consistent storm-window wins over static/round-robin/random and only competitiveness with strong AoI.
- Updated `paper/sections/01_introduction.tex` contribution bullet to include the energy-account curriculum result and state that the AoI margin is small/non-uniform.
- Updated `paper/sections/08_conclusion.tex` to add the bounded energy-account conclusion: useful calibrated storm-window adaptive scheduler, no robust AoI dominance, no clean event-triggered laser-gating claim.
- Recompiled paper with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; build passed with only existing overfull/BibTeX page warnings.

## 2026-05-24 claim audit and final alignment
- Created `rl_sensor_scheduling_framework/docs/05-24-claim-audit.md` separating:
  - fixed-budget V3.1 S2 evidence;
  - energy-account oracle diagnostic evidence;
  - learned energy-account PD-PPO evidence;
  - unsupported or over-strong claims.
- Corrected `paper/sections/03_problem_formulation.tex` so `feasible_static_projected` is described as a fixed-priority projected baseline, not an exhaustive static optimum solver.
- Updated `scripts/52_energy_account_convergence_assets.py` so regenerated energy-account curriculum tables clarify that full observation is an unconstrained ceiling and excluded from the boldface constrained-policy comparison.
- Regenerated convergence assets with `conda run -n darts python rl_sensor_scheduling_framework/scripts/52_energy_account_convergence_assets.py`.
- Validation:
  - `python -m py_compile rl_sensor_scheduling_framework/scripts/52_energy_account_convergence_assets.py` passed;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`, producing `main.pdf` with 73 pages.
- Remaining warnings are existing overfull boxes and BibTeX empty-page warnings.

## 2026-05-24 full-text consistency pass
- Completed a follow-up consistency pass over abstract, introduction, methodology, experiments, discussion, conclusion, and core table/figure captions.
- Edited `paper/sections/06_experiments.tex`:
  - softened the budget-sensitivity interpretation so AoI degradation is attributed to freshness-objective mismatch rather than an unproven shift toward event-sensitive particle/flux activation;
  - softened V2-to-V3.1 diagnostic wording so V3.1 is described as broader regime coverage, not a complete correction of dynamic adaptation.
- Edited `paper/sections/05_methodology.tex`:
  - softened EventAwareCritic motivation from guaranteed event-sensor payoff to possible event-regime value.
- Edited `paper/sections/08_conclusion.tex`:
  - changed "best learned adaptive scheduler" to "best learned scheduler evaluated" for the final fixed-budget sweep.
- Edited `paper/tables/main_results_v31.tex` and legacy `paper/tables/main_results.tex`:
  - replaced "best learned adaptive policy" with "proposed learned policy" in captions.
- Verification:
  - `rg` no longer finds the targeted over-strong phrases in active paper files, except for explicit negative claims such as "do not claim robust AoI dominance";
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`, producing `main.pdf` with 73 pages.
- Remaining warnings are existing overfull boxes and BibTeX empty-page warnings.

## 2026-05-24 Figure 4 generator-validation redraw
- Diagnosed the old synthetic-statistics figure as unsuitable for the current paper:
  - it contained an internal `Figure 3:` title conflicting with LaTeX numbering;
  - it used the older V2-style `synthetic_validation.csv` grid asset rather than the current V3.1 G1 validation assets;
  - the validation bars lacked acceptance thresholds;
  - the flux marginal was dominated by zero values and did not show the event-conditioned mechanism.
- Added reproducible redraw script:
  - `rl_sensor_scheduling_framework/scripts/53_redraw_generator_validation_figure.py`.
- Regenerated:
  - `rl_sensor_scheduling_framework/paper/figures/figure3_synthetic_statistics.png`;
  - `rl_sensor_scheduling_framework/paper/figures/figure3_synthetic_statistics.svg`.
- New figure design:
  - panels (a,b): AntAWS vs V3.1 synthetic marginals for air temperature and wind speed;
  - panel (c): wind-speed ACF against AntAWS anchor with tolerance band;
  - panel (d): event-fraction distribution for 512-step windows;
  - panel (e): conditional flux-wind coupling during blowing-snow events;
  - panel (f): validation checklist with pass-margin threshold.
- Updated `paper/sections/04_simulation_environment.tex` caption to describe the new panels.
- Validation:
  - `python -m py_compile rl_sensor_scheduling_framework/scripts/53_redraw_generator_validation_figure.py` passed;
  - `conda run -n darts python rl_sensor_scheduling_framework/scripts/53_redraw_generator_validation_figure.py` regenerated PNG/SVG successfully;
  - SVG text check confirmed no internal `Figure 3:`/`Figure 4:` title and no old `Synthetic ACF` label remain;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`, producing `main.pdf` with 73 pages.
- Remaining warnings are existing overfull boxes and BibTeX empty-page warnings.

## 2026-05-24 Figure 2 state-machine redraw
- Reworked `rl_sensor_scheduling_framework/paper/figures/sensor_state_machine_tikz.tex`:
  - removed white-filled transition-label backgrounds that were visually blocking the figure;
  - replaced dense right-side explanatory panels with two lower rule cards;
  - simplified transition labels so the figure carries the state logic while the caption carries the detailed equations;
  - used restrained state colors and sans-serif TikZ text for a cleaner paper-style schematic.
- Updated `paper/sections/03_problem_formulation.tex` Figure 2 caption from "right-hand panels" to "lower panels".
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`;
  - rendered page 15 with `pdftoppm` and visually checked that the white-label blocking issue is removed.
- Remaining warnings are existing overfull boxes and BibTeX empty-page warnings.

## 2026-05-24 Figure 2 layout refinement
- Refined the Figure 2 TikZ layout after visual review:
  - tightened the horizontal spacing between OFF, WARMING, and READY states;
  - attached `power on` and `warm-up complete` labels directly to their transition arrows;
  - moved the READY self-loop outside the state box so it no longer crosses the READY equations;
  - repositioned the abort label and lowered the rule cards to prevent overlaps.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`;
  - rendered page 15 and confirmed the previously reported READY-loop overlap and loose transition-label placement are resolved.

## 2026-05-24 Figure 6 PD-PPO architecture redraw
- Reworked `rl_sensor_scheduling_framework/paper/figures/pdppo_architecture_tikz.tex`:
  - removed white-filled arrow labels and dense micro-card routing;
  - replaced the previous overly wide layout with a taller runtime/core/training-signal structure;
  - represented the PD-PPO core as a readable component list instead of overlapping internal cards;
  - updated the frozen reward path wording from TCN-specific wording to `Frozen oracle`, matching the current frozen-oracle framing.
- Updated `paper/sections/05_methodology.tex` Figure 6 caption to describe:
  - runtime state encoding and feasibility projection;
  - PD-PPO core components;
  - frozen-oracle forecast reward and PPO optimisation feedback.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`;
  - rendered page 36 with `pdftoppm` and visually checked that white label backgrounds and card-overlap failures are removed.

## 2026-05-24 Table 3 main-result regeneration
- Identified Table 3 as `rl_sensor_scheduling_framework/paper/tables/main_results_v31.tex`.
- Checked the table against `reports/v31_s2_main/v31_s2_main_stats.csv`:
  - the existing numerical values matched the locked V3.1 S2 aggregate CSV;
  - the issue was lack of a reproducible generation path and weak provenance in the table file.
- Added `rl_sensor_scheduling_framework/scripts/54_rebuild_table3_main_results.py`:
  - reads `reports/v31_s2_main/v31_s2_main_stats.csv`;
  - validates that each policy-budget entry has `n=10`;
  - regenerates `paper/tables/main_results_v31.tex`.
- Regenerated Table 3 and updated its caption:
  - now explicitly says locked V3.1 S2 fixed-budget result;
  - clarifies that full observation is an unconstrained diagnostic ceiling;
  - clarifies that feasible static projection is a fixed-priority projected baseline.
- Validation:
  - `python -m py_compile rl_sensor_scheduling_framework/scripts/54_rebuild_table3_main_results.py` passed;
  - `python rl_sensor_scheduling_framework/scripts/54_rebuild_table3_main_results.py` regenerated the table;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`;
  - rendered page 44 and visually checked Table 3 layout.

## 2026-05-25 narrative reconstruction start
- Created active goal for rewriting the paper according to `rl_sensor_scheduling_framework/docs/05-25-1-paper.md`.
- Read the new rewrite package plus `planning-with-files` and `ml-paper-writing` skill instructions.
- Restored existing plan/progress/findings context.
- Added Phase 13 to `task_plan.md`.
- Initial reconciliation findings:
  - the new document is directionally correct but uses some placeholders or stale implementation names;
  - fixed-budget Table 3 is locked at `n=10`, so paper edits must not change it to five seeds;
  - current paper entrypoint is `rl_sensor_scheduling_framework/paper/main.tex`;
  - several requested energy-account edits partially exist from 2026-05-24 but need stronger restructuring and notation cleanup.

## 2026-05-25 narrative reconstruction first pass complete
- Rewrote `paper/main.tex` abstract around regime-dependent scheduling value:
  - fixed-budget V3.1 S2 remains `n=10`;
  - energy-account storm-window oracle advantage is stated as 1.9% overall and 9.3% event-period;
  - curriculum PD-PPO is bounded as robust over static/round-robin/random, competitive but non-uniform against AoI.
- Replaced the Introduction contribution list with five aligned claims:
  - forecast-driven CMDP;
  - PD-PPO with warm-up-aware stabilisation;
  - V3.1 simulation environment;
  - fixed-budget benchmark and behaviour diagnostics;
  - energy-account regime-dependent adaptive value.
- Updated problem/method text:
  - added fixed-budget simplification note pointing to the energy-account diagnostic;
  - added deployment caveat for simulator event context `z_t`;
  - changed the active event-context notation from `e_t` to `z_t` / `z_{t,1}` outside the abstract Proposition 1 counterexample.
- Updated experiments:
  - renamed main result subsection to fixed-budget main results;
  - added fixed-budget AoI mismatch explanation and abort-rate evidence;
  - renamed energy-account subsection to `sec:energy_account`;
  - added Figure 8 fixed-budget behaviour interpretation;
  - added ablation note that EventAwareCritic/ActionEmbedding are architectural supports rather than standalone proven drivers.
- Rewrote discussion:
  - replaced proximity-to-static discussion with `Regime-Dependent Value of Adaptive Scheduling`;
  - expanded limitations around energy account, event-context availability, and SOC/multiscale credit assignment;
  - softened conclusion wording away from robust AoI or clean laser-gating claims.
- Redrew Figure 8:
  - added `scripts/55_redraw_figure8_behavior_diagnostic.py`;
  - regenerated `paper/figures/figure5_sensor_timeline.png` and `.svg`;
  - updated the Figure 8 caption;
  - rendered PDF page 54 and visually checked the figure/caption.
- Validation:
  - forbidden-claim scan over active paper files returned no matches for the targeted phrases;
  - bibliography-key check confirmed `GillGMX500`, `SensecaLPS10`, and `ApogeeSI111` are cited in `tables/sensor_specs.tex`;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed in `rl_sensor_scheduling_framework/paper`, producing a 76-page `main.pdf`.
- Remaining warnings are existing overfull boxes and BibTeX empty-page warnings.

## 2026-05-25 length-reduction archive and plan
- User requested archival of the current paper, then a length-reduction/restructuring pass following `rl_sensor_scheduling_framework/docs/05-25-2-paper.md`, with subagent review loops until the paper is near minor-revision quality.
- Attempted to create a new goal via the goal tool, but the tool refused because the completed prior goal still occupies the thread goal slot. Recorded the new objective in `task_plan.md` Phase 14 and proceeded with the same execution discipline.
- Archived the current paper state before modifying content:
  - `rl_sensor_scheduling_framework/paper_archives/paper_pre_0525_length_reduction_20260525_042051.tar.gz`
  - archive includes `paper/`, `docs/05-25-1-paper.md`, and `docs/05-25-2-paper.md`, excluding VCS metadata.
- Read the full `05-25-2-paper.md` cut plan. Priority order is appendices, related work, problem formulation, simulation, methodology, experiments, discussion, introduction.
- Discovered multi-agent tools and will use subagents for independent manuscript reviews after edit batches.

## 2026-05-25 length-reduction first edit pass
- Completed the first major shortening/restructuring pass:
  - removed `\input{appendix/appendix}` from `paper/main.tex` so Appendix A proof and Appendix B hyperparameter table no longer inflate the submitted main PDF;
  - compressed `sections/02_related_work.tex` from four mini-survey subsections to two positioning subsections;
  - rewrote `sections/03_problem_formulation.tex`, merging hardware grounding, constraints, state/reward, and CMDP/proposition material while preserving Figure 2, Table 1, Proposition 1 sketch, Proposition 2, and event-context deployment caveat;
  - rewrote `sections/04_simulation_environment.tex`, deleting duplicated hardware grounding, reducing V2/V3.1 development history, and preserving G1 validation plus Figure 4;
  - shortened `sections/05_methodology.tex`, reducing equal-weight five-component framing and positioning ActionEmbedding/EventAwareCritic as architectural support;
  - rewrote `sections/06_experiments.tex`, merging protocol/baselines, retaining Table 3, condition table, physical-unit table, energy-account oracle/curriculum tables, Figure 8, and A1 ablation while reducing V2 oracle robustness, training diagnostics, A2, and H1 to short text;
  - rewrote `sections/07_discussion.tex`, demoting DQN to a V2 development diagnostic and tightening limitations;
  - tightened `sections/01_introduction.tex` to three contribution bullets and short paper organisation.
- Incorporated subagent claim-review feedback:
  - changed “prove” language to counterexample/non-equivalence language;
  - removed “naturally accommodates” and “unbiased credit assignment” phrasing;
  - changed full observation from “performance ceiling” to unconstrained diagnostic reference;
  - added the locked energy-account laser event/non-event ratio caveat (`~1.03x`);
  - avoided robust AoI dominance wording.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed;
  - output `paper/main.pdf` is now 45 pages, down from 76;
  - `texcount -inc -sum main.tex`: total 7,125; text words 5,947; captions/outside text 799; 15 floats/tables/figures;
  - target overfull issues from the long target-set formula and repository URL were fixed; only a small existing TikZ overfull remains.

## 2026-05-25 length-reduction final review pass
- Second-round subagent review attempts failed because the account hit the Codex usage limit; no additional subagent feedback was available in this turn.
- Performed local review against the first-round subagent checklist and made final repairs:
  - removed misleading bold emphasis from Table 3 values and regenerated `paper/tables/main_results_v31.tex`;
  - updated `scripts/54_rebuild_table3_main_results.py` so future Table 3 regeneration preserves the non-misleading caption and formatting;
  - changed energy-account curriculum caption from "unconstrained ceiling" to "unconstrained diagnostic reference";
  - added a concise claim-evidence boundary paragraph in `sections/06_experiments.tex`;
  - softened the conclusion's oracle robustness wording to a diagnostic statement;
  - removed remaining target scan hit for "dominance".
- Final validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed;
  - final `paper/main.pdf` remains 45 pages;
  - `texcount -inc -sum main.tex`: total 7,145; text words 5,970; captions/outside text 796; 15 floats/tables/figures;
  - targeted over-claim scan returned no matches for the high-risk phrases checked;
  - load-bearing labels/content remain present: Table 3, energy-account oracle/curriculum tables, A1 ablation, Figure 2, Figure 8, Proposition 1/2, Fernandez-Bes anchor, laser ratio `1.03x`, and event-context caveat.
- Local verdict after repairs: close to minor revision. Remaining risks are not immediate manuscript-structure blockers; they are the stated simulator/external-validity limitations.
# Session: 2026-05-25 CRST Full Rewrite Reset

## Work Started
- Created a persistent goal for a full CRST Research Article rewrite with archive,
  evidence verification, PDF validation, and repeated independent review.
- Loaded `academic-paper-strategist`, `academic-paper-composer`,
  `planning-with-files`, `pdf`, and `microclimate-experiment-server` workflows.
- Recovered prior planning context and confirmed the prior manuscript had already
  undergone an incremental 45-page reduction; this is treated as historical input.
- Archived the current paper source/PDF baseline at
  `rl_sensor_scheduling_framework/paper_archives/paper_pre_full_rewrite_20260525_055257.tar.gz`.
- Retrieved the official CRST Guide for Authors and recorded the binding manuscript,
  data, AI-disclosure, highlights, artwork, and CRediT requirements in `findings.md`.
- Spawned two independent first-round subagent reviews: structural/scope and
  evidence/statistical-claim auditing.

## Errors Logged
| Timestamp | Error | Resolution |
|-----------|-------|------------|
| 2026-05-25 | Subagent spawn rejected `fork_context=true` with explicit `agent_type` | Relaunched inherited-context agents without `agent_type`; both accepted. |

## Next Critical Work
- Build a result/provenance ledger from final artifacts and current manuscript claims.
- Integrate subagent review findings into a new CRST-focused manuscript outline.
- Begin new source construction only after evidence boundaries are fixed.

## Evidence and Strategy Pass
- Read the final V3.1 result CSV, energy-account convergence CSV, V3.1-aligned
  ablation CSVs, and their source scripts/configuration references.
- Created `rl_sensor_scheduling_framework/docs/05-25-full-rewrite-evidence-ledger.md`
  to classify usable, supporting, excluded and prohibited claims.
- Created `rl_sensor_scheduling_framework/docs/05-25-crst-rewrite-strategy.md` with
  a new applied cold-regions thesis, section architecture, artifact plan, CRST
  compliance checklist and review gates.
- Decision: do not revise the shortened old manuscript in place as an argument
  structure; write a new manuscript around the evidence ledger and reuse only
  verified assets.
- Used the experiment-server workflow for read-only evidence validation through
  the `remote-gpu` SSH alias. Confirmed all 30 V3.1 S2 completion markers and
  exact hash equality for the fixed-budget summary.
- Confirmed exact local/remote combined hash equality for the locked energy-account
  aggregation inputs (five storm CSV files plus thirty full-distribution rollout NPZ
  files); the aggregate CSV itself is generated locally from these synchronized inputs.
- Received first independent structural reviewer memorandum (`Zeno`). It recommends
  full reconstruction rather than patching and identifies as pre-draft blockers:
  exhaustive static-comparator feasibility, protocol split reconciliation,
  oracle-limited outcome wording, truth-event diagnostic boundaries, and CRST package
  omissions. Accepted these findings into the strategy and evidence ledger.

## Resume: Static Comparator Blocker
- Recovered the active full-rewrite plan and distinguished it from the older
  subproject closure plan.
- Scanned V3.1 S2 raw artifacts: `30` candidate-prior tables exist, but no saved
  oracle-selected static rollout exists.
- Located the current evaluation/export logic in
  `scripts/25_v2_train_custom_ppo.py` and the existing collector restriction in
  `scripts/43_v31_s2_collect.py`.
- Status: R2 remains in progress; before drafting result claims, determine whether
  the stored prior choice can be replayed on recorded evaluation windows without
  leakage and aggregate the resulting comparator.
- Read the S2 launcher defaults, evaluator collector, candidate-prior builder and
  one locked run's metadata/table. Confirmed old metadata does not record prior
  window starts or an evaluated oracle-static policy; current code uses separate
  deterministic RNG seeds for prior scoring and final evaluation.
- Scanned current manuscript protocol text and found unsupported chronological/
  disjoint-split assertions in Sections 4 and 6 plus an unverified non-overlap
  caption in `tables/condition_results_v31.tex`. These are pre-draft correction
  requirements for the fresh CRST manuscript.
- Reconstructed candidate-prior starts using the saved truth sequences, metadata and
  deterministic sampler. The audit found candidate-prior/final-evaluation overlap in
  `21/30` S2 runs and internal evaluation-window overlap in `21/30` runs. The
  candidate static rollout cannot repair submission-level independence by itself.
- Read `CustomPPO._sample_start_idx()`: absent explicit training starts, S2
  optimization samples throughout the saved truth sequence. Established that any
  independent-evaluation repair must use new truth or retrain, rather than carve
  out a posthoc segment of the old truth.
- Added `scripts/56_v31_protocol_audit.py` to persist the window-independence audit
  and `scripts/57_v31_independent_replay.py` as a separate posthoc, new-truth replay
  path; these do not overwrite locked S2 results.
- The protocol audit smoke run reports prior/final overlap in `21/30`, internal
  final-evaluation overlap in `21/30`, and prior internal overlap in `9/30` runs.
- Began enabling faithful posthoc replay by allowing environments to use saved
  training-period normalisation statistics while executing new independent truth.
- Added external-normalisation support to `src/v2/env.py`, propagated it through
  custom-PPO/static evaluation, and taught forecast replay padding to use the
  training-run initial state mean when declared in replay metadata.
- Verification passed in the `darts` conda environment:
  `python -m pytest -q tests/v2/test_v31_protocol_replay.py
  tests/v2/test_warmup_env.py tests/v2/test_custom_ppo.py
  tests/v2/test_forecast_eval.py` (`18` tests passed).
- Generated the persistent audit artifacts at
  `rl_sensor_scheduling_framework/reports/v31_s2_protocol_audit/`.
- Ran a posthoc independent-truth smoke replay for `budget1p70_seed41` with two
  non-overlapping test windows. Execution succeeded, but the old frozen TCN
  ranked full observation worse than static/PPO on the new truth
  (`0.1952` versus `0.1850`/`0.1899` FW-MAE), confirming that replaying old
  checkpoints does not repair the paper's primary evidence.
- Added explicit partition controls to `scripts/25_v2_train_custom_ppo.py` and
  bounded PPO-start support in `src/v2/custom_ppo.py`; added
  `scripts/58_v31_split_protocol_run.py` as a four-part retraining driver.
- Initial split-driver dry-run failed despite sufficient final-test capacity because
  a random greedy non-overlap sampler fragmented the available interval. Replaced it
  with a capacity-checked random-gap construction that always returns windows when
  the partition is feasible.
- Formal-default split-driver dry-run now passes for `90000` truth steps:
  `oracle_pretrain=[0,31500)`, `rl_train=[31500,76500)`,
  `validation=[76500,83250)`, `final_test=[83250,90000)`, with six
  final-test `1024`-step windows generated without overlap or event filtering.
- Targeted verification after the split support edits passes in `darts`
  (`20` tests covering protocol utilities, environment, custom PPO and forecast
  evaluation).

## 2026-05-25 Takeover Verification and Retraining Decision
- Resumed the rewrite/evidence audit using the root planning files as the active
  `planning-with-files` state.
- Rechecked `reports/v31_s2_protocol_audit/v31_s2_protocol_audit_summary.json`:
  the locked S2 result has prior/final overlap in `21/30` runs, internal
  final-evaluation overlap in `21/30`, and prior internal overlap in `9/30`.
- Confirmed the prior independent-truth smoke replay is not an adequate repair:
  on its new truth the old frozen TCN ranks `full_open_unconstrained` worse than
  the selected static mask and PPO (`0.1952` versus `0.1850` and `0.1899` FW-MAE).
- Confirmed the corrected code path is present: `scripts/58_v31_split_protocol_run.py`
  reserves `oracle_pretrain`, `rl_train`, `validation`, and `final_test` partitions;
  `scripts/25_v2_train_custom_ppo.py` consumes explicit partition controls; and
  `src/v2/custom_ppo.py` bounds PPO sampling to the training interval.
- Validation: `conda run -n darts python -m pytest -q tests/v2/test_v31_protocol_replay.py tests/v2/test_warmup_env.py tests/v2/test_custom_ppo.py tests/v2/test_forecast_eval.py`
  passed (`20` tests); `py_compile` passed for the audit/replay/split and modified
  PPO/environment sources; the formal-default split-runner dry-run passed.
- Decision: the existing S2 outputs remain usable only as historical/same-protocol
  diagnostics. Main manuscript evidence for a proportional chronological split
  requires new training and final-test evaluation under the split-aware runner.
- Deployed the minimal split-protocol code path to the GPU server after local
  verification; remote dry-run generated the same four-part manifest, and the
  remote available test suite passed (`18` tests, with the two additional local
  checks absent from the remote suite before sync).
- Checked server resources immediately before launch: GPU `0` remains occupied by
  an unrelated workload; GPUs `1--5` were idle. No existing microclimate `tmux`
  session conflicted with the new gate.
- Launched the split-protocol gate in remote `tmux` session
  `v31_split_gate_20260526`, using physical GPU `1` and output directory
  `reports/v31_split_protocol_gate/budget1p70_seed41` for `B=1.70`, seed `41`.
- Checked the aggregation path while the gate runs: `scripts/43_v31_s2_collect.py`
  hard-coded the historical `feasible_static_projected` comparator. Added a
  backwards-compatible `--static-policy` option so new split runs can aggregate
  against `validation_selected_static` while old S2 defaults do not change.
- Validation for the collector update: `py_compile` passed and a small in-memory
  budget-check invocation selected `validation_selected_static` when requested.
- Remote gate status at `2026-05-26 00:08 CST`: the truth, frozen TCN oracle, and
  `custom_ppo_candidate_prior.csv` exist; GPU `1` is active and training continues.
- Corrected `docs/05-25-full-rewrite-evidence-ledger.md`: the locked S2 table is
  now classified as historical/same-protocol diagnostic evidence; it records the
  `21/30` overlap blocker and the running split-protocol replacement gate. The
  energy-account learned-policy results are also marked pending a matching
  independence audit or corrected rerun before submission-level claims.
- Corrected `docs/05-25-crst-rewrite-strategy.md`: the core thesis is now a
  hypothesis pending corrected final-test outputs, and the planned fixed-budget
  result section requires a validation-selected static comparator under the
  declared chronological protocol.
- Remote gate status at `2026-05-26 00:11 CST`: PPO training reached
  `14336/100000` timesteps with live history output; no failure was observed.
- Started remote postprocessing watcher `v31_split_gate_post_20260526`. After a
  successful training exit it runs `24_v2_evaluate_rollouts.py` on the final-test
  rollouts and `43_v31_s2_collect.py --static-policy validation_selected_static`;
  this prevents the completed gate from being summarized against the historical
  fixed-priority static label.
- The gate and postprocessing completed successfully (`exit_code=0`,
  `postprocess_exit_code=0`), and gate artifacts were synchronized locally under
  `rl_sensor_scheduling_framework/reports/v31_split_protocol_gate/`.
- Gate final-test FW-MAE at `B=1.70`, seed `41`: full observation `0.1114`,
  validation-selected static `0.1195`, PD-PPO `0.1222`, round-robin `0.1243`,
  random `0.1301`, and AoI `0.1319`. PD-PPO is `2.31%` worse than the selected
  static mask but better than each dynamic heuristic in this one run.
- Added `scripts/59_v31_split_protocol_grid.py`, a new resume-capable parallel grid
  driver that invokes only the split-protocol runner and aggregates against
  `validation_selected_static`. Local `py_compile`, dry-run and all `20` targeted
  tests passed; remote `py_compile` and dry-run passed.
- Launched the full corrected grid in remote `tmux` session
  `v31_split_main_20260526`: output `reports/v31_split_protocol_main`, budgets
  `1.65/1.70/1.75`, seeds `41--50`, and five workers on physical GPUs `1--5`.
- Initial grid health check at `2026-05-26 00:45 CST`: the five first-wave
  `B=1.65` tasks (seeds `41--45`) each emitted their first PPO update; GPUs
  `1--5` each show the expected approximately `872 MiB` allocation and no task has
  failed. GPU `0` had become idle, but the launched five-worker allocation was
  intentionally left unchanged for execution consistency.
- The full split-protocol grid completed remotely at `2026-05-26 04:26 CST` with
  `30/30` done markers and driver exit code `0`; all GPUs were idle afterward.
- Synchronized the complete approximately `1.31 GB` run directory locally under
  `rl_sensor_scheduling_framework/reports/v31_split_protocol_main/`, including
  manifests, final-test rollouts, evaluation CSVs, checkpoints and logs.
- Reran the collector locally against `validation_selected_static`; local aggregate
  SHA-256 values are `1d236291...` (`v31_s2_budget_check.csv`),
  `31aeb9d2...` (`v31_s2_main_stats.csv`) and `dd612fb8...`
  (`v31_s2_significance.csv`).
- Audited every synchronized manifest: `30/30` pass chronological partition and
  non-overlapping final-test-window checks. Full observation is the per-run
  minimum in `29/30` runs and has the lowest mean at all budgets.
- Corrected fixed-budget conclusion: PD-PPO significantly beats round-robin, AoI
  and random at all budgets after Bonferroni correction, but does not significantly
  beat validation-selected static (`6/10`, `4/10`, `4/10` seed wins and
  `p_adj=1.0` for `B=1.65/1.70/1.75`).
- Updated the evidence ledger and rewrite strategy so this completed
  split-protocol table becomes the primary fixed-budget evidence, while the old
  S2 output remains historical diagnostic material.
- Next computation: audit the energy-account/curriculum protocol boundary before
  promoting any dynamic-opportunity result into the new manuscript.

## 2026-05-26 Resume: Energy-Account Independence Audit
- Recovered the active CRST rewrite planning state and loaded the energy-account
  convergence collector plus the current claim/mechanism memos.
- Initial record insertion failed because the selected patch anchor belonged to a
  different planning file; appended this audit entry against the actual file
  endings without altering prior content.
- Confirmed from saved metadata for inspected curriculum runs that storm-window
  PPO training and storm evaluation use identical start indices. The primary
  storm learned-policy comparison is therefore diagnostic replay on training
  windows, pending corrected retraining/evaluation.
- Next action: quantify this issue across all five seeds and test whether the
  separately reported full-distribution no-retrain rollouts intersect those same
  training windows.
- Read the custom-PPO and oracle-helper source paths. Without explicit oracle
  partition controls, oracle rollouts are event-conditioned samples from the full
  truth sequence; old energy-account metadata does not declare such controls.
- Attempted direct `.npz` interval extraction with system `python3`; it failed
  because that interpreter lacks `numpy`. Resolution: run the same read-only
  audit in the existing `darts` conda environment rather than installing packages
  or modifying outputs.
- Confirmed `darts` supplies `numpy`, but an attempted shell assignment from
  `conda run ... which python` resolved to an empty executable path and did not
  execute the interval reader. Resolution: use the confirmed direct interpreter
  path `/home/horeb/miniconda3/envs/darts/bin/python` for the audit.
- Source inspection additionally shows old runs without explicit external
  normalisation statistics normalize agent state on the entire supplied truth,
  which is incompatible with training-only preprocessing for final evidence.
- Added `scripts/60_energy_account_protocol_audit.py`, ran it successfully in the
  `darts` environment, and generated
  `reports/energy_account_protocol_audit_20260526/{energy_account_protocol_audit_summary.json,energy_account_protocol_audit_by_seed.csv}`.
- The audit fails the learned energy-account evidence path: `5/5` storm runs
  reuse PPO training starts for evaluation; `5/5` full evaluations overlap
  training/storm support; reconstructed oracle overlap occurs for both evaluation
  types in `5/5`; `2/5` full evaluations overlap internally; `0/5` declare
  isolated normalisation.
- Updated the evidence ledger, rewrite strategy and earlier energy-account
  claim/mechanism memos so old curriculum outputs remain reproducible diagnostics
  but cannot supply manuscript-level learned-policy results.
- Implemented `scripts/61_energy_account_split_protocol_run.py` for a corrected
  energy-account storm-conditional gate. It applies four chronological partitions,
  training-only normalization, validation-selected static comparison and
  non-overlapping event-stratified final-test windows.
- Validation: `py_compile` passed for the new audit/split scripts and affected
  protocol sources; a dry run produced the intended bottom-level command; a
  reduced CPU end-to-end smoke completed successfully in `/tmp`, producing
  `validation_selected_static` and final evaluation artifacts.
- Remote deployment: confirmed all six GPUs idle, copied the two new scripts to
  the server, passed remote `py_compile`, and used formal dry-run preparation to
  generate the 90,000-step split truth and manifest.
- Launched an initial single-seed server gate in `tmux` session
  `energy_split_gate_20260526` on GPU 0. Its partitions were
  `oracle=[0,27000)`, `rl_train=[27000,67500)`,
  `validation=[67500,78750)`, and `final_test=[78750,90000)`.
- Manifest review shows the final conditional windows have mean event rate only
  about `0.276`; this completed run will be interpreted as an independent
  base-distribution gate, not a replacement storm-regime result.
- Regression verification after the new scripts and documentation changes:
  `/home/horeb/miniconda3/envs/darts/bin/python -m pytest -q
  rl_sensor_scheduling_framework/tests/v2/test_v31_protocol_replay.py
  rl_sensor_scheduling_framework/tests/v2/test_warmup_env.py
  rl_sensor_scheduling_framework/tests/v2/test_custom_ppo.py
  rl_sensor_scheduling_framework/tests/v2/test_forecast_eval.py` passed (`20` tests).
- While the gate remains in oracle preparation, reconciled an older ledger label:
  legacy V3.1 fixed-budget tables are now explicitly marked as superseded by the
  already completed E1b split-protocol grid, avoiding contradictory "pending
  corrected rerun" wording.
- Closed a regeneration hazard in `scripts/52_energy_account_convergence_assets.py`:
  rerunning it now marks the energy curriculum table and convergence memo as
  retrospective non-independent diagnostics and removes winner boldface.
  Regeneration completed locally without changing the archived descriptive values.
- Remote gate reached its first PPO update (`2048/100000` timesteps); no runtime
  failure is visible.
- Before any gate result was produced, checked the truth partition distributions
  and found legacy `clustered` coverage bias: with coverage `0.30`, partition event
  rates were about `0.538/0.192/0.205/0.213`, while coverage `0.60` raised only
  early-partition rates and left validation/final unchanged.
- Stopped the invalid clustered gate, confirmed no residual process, and retained
  its intermediate artifacts under
  `reports/energy_account_split_protocol_invalid_clustered_gate/`. It will not be
  interpreted as evidence.
- Changed `scripts/61_energy_account_split_protocol_run.py` to default to the
  V3.1 `semi_markov` event generator; replacement launch is gated on manifest
  verification of validation/final event support.
- Deployed the `semi_markov` default and prepared a fresh remote split truth.
  Preflight passed: whole-partition event rates are about
  `0.321/0.307/0.307/0.300` and final conditional windows average `0.521`.
- Launched corrected seed-41 gate in `tmux` session
  `energy_split_semimarkov_gate_20260526` on GPU 0, output
  `reports/energy_account_split_protocol_gate_semimarkov/budget1p20_seed41`.
- Scanned the active paper source while the gate runs. The current introduction,
  experiments, discussion and conclusion still state the now-invalid learned
  energy curriculum claim, and the fixed-budget main table remains sourced from
  old S2. Logged these as R3 replacement requirements rather than applying
  provisional prose before corrected evidence is available.
- Repaired the fixed-budget table regeneration path:
  `scripts/54_rebuild_table3_main_results.py` now reads
  `reports/v31_split_protocol_main/v31_s2_main_stats.csv`, uses the
  `validation_selected_static` row, and writes a chronological-final-test caption.
  Regenerated `paper/tables/main_results_v31.tex` accordingly.
- First attempt to repoint the physical-unit MAE generator returned an empty
  table because its pre-existing relative-path convention resolved `reports/...`
  against the outer repository when invoked from there. No LaTeX output was
  overwritten. Resolution in progress: anchor its relative inputs/outputs to the
  framework root and remove only the empty generated CSV from this failed attempt.

## 2026-05-26 Resume: Corrected Evidence Integrated Into Draft
- Loaded `planning-with-files`, `microclimate-experiment-server`,
  `academic-paper-composer`, and `pdf` workflows and recovered the root rewrite
  plan as authoritative.
- Confirmed the replacement energy-account gate remains active on the GPU server in
  `tmux` session `energy_split_semimarkov_gate_20260526`; at the latest check it
  had reached `40960/100000` PPO timesteps with no visible failure.
- Completed the previously open physical-unit generation repair:
  `scripts/35_v2_physical_unit_mae_table.py` now resolves inputs/outputs from the
  framework root and regenerated `paper/tables/physical_unit_mae.tex` from
  `reports/v31_split_protocol_main`.
- Extended `scripts/54_rebuild_table3_main_results.py` to regenerate the
  final-test `event`/`non_event` condition table from the corrected split-protocol
  CSV in addition to the main fixed-budget table.
- Repointed `scripts/36_v2_plot_locked_power_error.py` and
  `scripts/55_redraw_figure8_behavior_diagnostic.py` to the corrected
  split-protocol final-test artifacts and the validation-selected static comparator;
  added that comparator to `scripts/47_v31_behavior_diagnostics.py`.
- Regenerated the main/conditional/physical-unit tables, power-error plot,
  final-test behaviour timeline and behavior-diagnostic report. At `B=1.70`,
  corrected PD-PPO FW-MAE is `0.1334 +/- 0.0110`, versus
  `0.1329 +/- 0.0108` for validation-selected static,
  `0.1408 +/- 0.0119` for round-robin, and `0.1432 +/- 0.0100` for AoI.
- Updated `paper/main.tex` and Sections 1, 3, 6, 7 and 8 so the current draft:
  uses the corrected final-test values and comparator definition; reports the
  fixed-budget non-inferiority boundary accurately; and excludes archived
  non-independent energy-account learned-policy outputs from comparative claims.
- Verification:
  - `py_compile` passed for the updated table/plot/diagnostic generators.
  - Source scan found no remaining current-main-text references to the old
    `0.1620` fixed-budget value or the removed energy curriculum comparison.
  - Abstract length is `206` words, below the CRST 250-word cap.
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` passed,
    producing a `44`-page `paper/main.pdf` with no undefined-reference/citation
    diagnostic; non-blocking overfull-box and bibliography-page warnings remain.
  - Rendered PDF inspection of the front page and pages `25--35` confirmed the
    updated tables and figures are readable and the mechanism/learned-policy
    boundary is visible in the results flow.
- Added `rl_sensor_scheduling_framework/paper/highlights.txt`; validation confirms
  `5` bullets with a maximum length of `79` characters, satisfying the CRST
  highlights count/length requirement without importing pending energy-policy claims.
- Confirmed that CRediT, funding and competing-interest statements remain author
  metadata decisions and are not yet present in `paper/main.tex`.

## 2026-05-26 Continue: Energy Split Gate Monitoring and Submission Audit
- Recovered the authoritative root rewrite plan and retained the nested repository
  planning files as historical context.
- Loaded the `planning-with-files`, `microclimate-experiment-server`,
  `academic-paper-composer`, and `pdf` workflows for the resumed unit of work.
- Confirmed the GPU server is reachable through the `remote-gpu` SSH alias. The
  first SSH authentication check accidentally selected the locally documented sudo
  credential rather than the server credential; retry using the server entry
  succeeded and made no remote changes.
- Checked `energy_split_semimarkov_gate_20260526` at `2026-05-26 06:41 CST`.
  The active `25_v2_train_custom_ppo.py` process had advanced to
  `55296/100000` timesteps with no failure in `runner.log`; the split-protocol
  result is therefore still pending and must not yet be cited.
- Runtime projection from the first 55,296 timesteps is approximately another
  `50--70` minutes for training plus final evaluation/artifact generation.
- Confirmed that the current compiled draft already contains Code and Data
  Availability and an AI-assisted-technologies declaration; CRediT, funding and
  competing-interest metadata remain unfilled author-controlled items.
- Ran a mechanical claim/compliance scan over `paper/main.tex`,
  `paper/sections/*.tex` and the corrected tables. The old energy-account
  curriculum table is not included in the active manuscript, and its only active
  discussion explicitly excludes it as non-independent evidence.
- Cross-checked the fixed-budget significance claims against
  `reports/v31_split_protocol_main/v31_s2_significance.csv`: at each tested
  budget the recorded Bonferroni-adjusted comparisons against round-robin, AoI
  and random are significant, while comparisons against
  `validation_selected_static` have adjusted `p=1.0`.
- Fixed a duplicated/incomplete sentence in
  `paper/sections/06_experiments.tex` under Hyperparameter Sensitivity; this was
  a prose defect only and did not change evidence or numbers.
- Recompiled `paper/main.tex` successfully with XeLaTeX (`44` pages), found no
  undefined-reference/citation diagnostic, and rendered/inspected page `34`
  containing the repaired paragraph; layout is readable.
- Initial declaration detection falsely matched the words `credit assignment` as
  a CRediT statement; refined exact matching confirms that CRediT, funding and
  competing-interest declarations are all still absent and require author input.
- A first LaTeX warning-count regular expression was invalid because of shell
  escape handling; reran with fixed-string matching and recorded `17` overfull
  and `2` underfull boxes, with no undefined reference/citation diagnostic.
- Created `rl_sensor_scheduling_framework/docs/05-26-crst-draft-verification.md`
  as the interim audit record and linked it in R4 of the active plan.
- Audited the Code and Data Availability claim against git state. The public
  `rl_sensor_scheduling_framework` remote resolves to `161d0a8`, while corrected
  protocol scripts `58/59/61`, the rebuilt table generator and
  `reports/v31_split_protocol_main/` are not yet tracked in the local repository.
  Revised `paper/main.tex` to describe the repository as an earlier development
  snapshot and to make versioned deposition of the corrected evidence package an
  explicit pre-submission requirement.
- Recompiled `paper/main.tex` after the availability correction and rendered PDF
  page `38`; the revised declaration is legible without clipping or overlap.
- First independent evidence review found one unsupported protocol assertion:
  `paper/sections/05_methodology.tex` described hyperparameters as selected by a
  held-out validation grid search although the ledger supports only local H1
  sensitivity diagnostics. Replaced it with a pre-specified-configuration
  statement and explicitly labelled H1 as post hoc/local and not selected from
  final-test performance.
- Repaired the conditional table caption at its generator source
  (`scripts/54_rebuild_table3_main_results.py`) and regenerated
  `paper/tables/condition_results_v31.tex`: event labels both define strata and
  are simulated PD-PPO context, while deployment would require inferred context.
  This resolves a possible ambiguity identified by the reviewer.
- One follow-up `latexmk main.tex` invocation failed because it was accidentally
  run from the framework root rather than `paper/`; no artifact was changed by
  that failure and compilation is retried from the document directory.
- Regenerated the corrected condition table using the patched source generator,
  verified `scripts/54_rebuild_table3_main_results.py` with `py_compile`, and
  recompiled `paper/main.tex` successfully (`44` pages).
- Rendered and inspected pages `24` and `28`; the revised method and condition
  caption are readable. The latest build has `15` overfull and `2` underfull box
  warnings and no undefined-reference/citation diagnostic.
- Returned the two evidence-review repairs to the original read-only reviewer;
  focused re-review reported both findings resolved.
- A dependency scan initially failed because an `rg` regular expression contained
  unescaped LaTeX backslashes; reran using fixed-string searches and recovered the
  active `main.tex` dependency chain.
- Created `rl_sensor_scheduling_framework/docs/05-26-crst-submission-checklist.md`
  to define the active submission source set, exclude superseded `raw.tex` and
  non-independent curriculum artifacts, and list evidence/declaration/deposit
  blockers that must be cleared before final packaging.

## 2026-05-26 Continue: Energy Gate Completed, n=5 Extension Started
- Recovered the root CRST rewrite plan and checked the current paper build state.
- Confirmed `paper/main.pdf` was present, with no undefined-reference/citation
  diagnostics in `main.log`; remaining warnings are overfull/underfull boxes.
- Rechecked the server gate `energy_split_semimarkov_gate_20260526`. The run
  completed successfully at `2026-05-26 07:35 CST`, wrote all expected rollouts,
  evaluation CSVs and `exit_code=0`, and was synchronized locally to
  `rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov/`.
- Summarized the seed-41 gate: `custom_ppo` oracle loss `0.47455` versus
  `validation_selected_static` `0.47522`, `round_robin` `0.48008`, `aoi`
  `0.48296`, and `random` `0.49603`; `custom_ppo` warm-up abort count is `206`.
- Decision: treat seed 41 as a protocol-valid weak-positive gate, not as
  manuscript-level learned-policy evidence. Launched seed `42--45` tmux jobs on
  the server to obtain a minimum `n=5` energy-account split-protocol check.
- Verified the four new server jobs started and entered `25_v2_train_custom_ppo.py`;
  each has emitted `split_protocol_manifest.json`, `truth_energy_split.csv`, and
  `v2_tcn_oracle.pt`.
- PDF visual audit found a real abstract prose defect on page 1
  (`sensors. automatic-weather-station statistics.`). Fixed `paper/main.tex`,
  recompiled with `latexmk -xelatex`, and re-rendered page 1. The repaired abstract
  is readable; current PDF has `47` pages and the abstract count is `238` words.
- Added `scripts/62_energy_account_split_protocol_collect.py` to aggregate the
  energy-account split-protocol seed directories into long, summary and
  custom-vs-comparator CSVs. `py_compile` passed, and a partial local run correctly
  reports `complete_seeds=1/5`. The script has been synchronized to the server.
- Added a server-side tmux watcher `energy_split_semimarkov_collect_20260526` that
  waits for seed `42--45` sessions to finish, then runs
  `scripts/62_energy_account_split_protocol_collect.py` and writes
  `reports/energy_account_split_protocol_gate_semimarkov/aggregate/collector.log`.
- Follow-up status check at `2026-05-26 09:42 CST`: all seed `42--45` tmux
  sessions and the collector had exited. Synced the full remote
  `reports/energy_account_split_protocol_gate_semimarkov/` directory locally.
- Aggregate result: `complete_seeds=5/5`. `custom_ppo` wins `4/5` seeds against
  AoI, round-robin, random and feasible static projection, but only `2/5` against
  validation-selected static. Mean oracle loss is `0.46411` for `custom_ppo` and
  `0.45110` for validation-selected static.
- Decision: the corrected energy-account learned-policy result is not strong
  enough to promote a robust dynamic-over-static manuscript claim. It can support
  a bounded dynamic-heuristic comparison or remain a diagnostic; the stronger
  energy-account claim should stay at the reference-policy opportunity level.

## 2026-05-26 Continue: SOC Auxiliary + Abort-Control Gate
- User requested adding SOC auxiliary critic and abort control to continue the
  optimized PD-PPO route under the corrected split protocol.
- Inspected `scripts/25_v2_train_custom_ppo.py` and confirmed support already
  exists for `--soc-aux-horizon`, `--soc-aux-coef`, `--lambda-warmup-abort`,
  `--soc-soft-penalty-buffer`, `--lambda-soc-soft-penalty`, and
  `--ppo-max-candidate-warmup`.
- Updated `scripts/61_energy_account_split_protocol_run.py` to expose and record
  these controls in `split_protocol_manifest.json`, and to pass them through to
  `25_v2_train_custom_ppo.py`.
- Updated `scripts/62_energy_account_split_protocol_collect.py` to include the
  SOC/abort-control settings in its long-form aggregate table.
- Validation:
  - `py_compile` passed for scripts `61` and `62`;
  - dry-run command shows `--soc-aux-horizon 16 --soc-aux-coef 0.1
    --lambda-warmup-abort 0.16 --total-timesteps 200000` passed through;
  - dry-run manifest records the same controls.
- Deployed scripts `61` and `62` to the server and launched tmux
  `energy_socaux_abort2x_seed41_20260526` on GPU `0`, output
  `reports/energy_account_split_protocol_socaux_abort2x_200k/budget1p20_seed41`.
  The run uses seed `41`, semi-Markov events, `200k` timesteps, SOC auxiliary
  horizon `16`, SOC auxiliary coefficient `0.1`, and `lambda_warmup_abort=0.16`.
- Verified the launched run's manifest records the intended controls and that the
  live training log reached update `1` with `soc_aux=0.025035`, confirming the
  auxiliary objective is active.
- Added server-side tmux watcher `energy_socaux_abort2x_collect_20260526` to run
  `scripts/62_energy_account_split_protocol_collect.py --base-dir
  reports/energy_account_split_protocol_socaux_abort2x_200k --seeds 41` after
  the gate session exits.

### Follow-up Status and Document Review
- At `2026-05-26 10:35 CST`, server tmux sessions
  `energy_socaux_abort2x_seed41_20260526` and
  `energy_socaux_abort2x_collect_20260526` were both alive.
- The main run had been active for about `45` minutes and had reached
  `40960/200000` PPO timesteps. Only live-training artifacts existed; no final
  evaluation or aggregate CSV was available yet.
- Confirmed from code that SOC support is implemented as a future-SOC auxiliary
  prediction head and loss in `src/v2/custom_ppo.py`; it is not a full separate
  CMDP/Lagrangian critic. The abort-control part is the raised
  `lambda_warmup_abort=0.16` shaping penalty.
- Reviewed `docs/05-26-02.md` and `docs/05-26-03.md`: the diagnosis is
  directionally reasonable, but any paper import must keep the theory language
  conservative and verify/replace questionable references.
- Created `rl_sensor_scheduling_framework/docs/05-26-02-03-review.md` with the
  corrected assessment: current PPO state does not contain explicit TCN oracle
  forecasts; `horizon=8`, generator lead `5`, and laser warm-up `3` do not support
  an `H < tau` failure claim; SOC auxiliary is an auxiliary prediction head, not a
  full CMDP/CPO implementation.
- Remote follow-up at `2026-05-26 11:08 CST`: tmux sessions
  `energy_socaux_abort2x_seed41_20260526` and
  `energy_socaux_abort2x_collect_20260526` remain active. The main run reached
  `73728/200000` timesteps and has not yet produced final evaluation or aggregate
  outputs.
- Remote follow-up at `2026-05-26 11:54 CST`: the same tmux sessions remain
  active. The main run reached `116736/200000` timesteps, with no final metrics,
  rollout files or aggregate outputs yet.
- Remote follow-up at `2026-05-26 12:54 CST`: the run reached
  `174080/200000` timesteps. It is still in PPO training and has not yet emitted
  the final model, rollout, evaluation or aggregate files.
- Remote follow-up at `2026-05-26 13:40 CST`: the SOC+abort-control gate finished
  with `exit_code=0`; both the main tmux session and collector session had exited.
  Synchronized
  `reports/energy_account_split_protocol_socaux_abort2x_200k/` from the server.
- Final same-run comparison: `custom_ppo` oracle loss `0.47950` vs
  `validation_selected_static` `0.47617`, `round_robin` `0.48098`, `aoi`
  `0.48660`, `feasible_static_projected` `0.48710`, and `random` `0.49873`.
  The strict static-comparator condition failed, so the gate should not be scaled.
- Behavior diagnosis from rollout NPZ files: custom PPO aborts decreased from the
  previous strict-protocol seed-41 count `206` to `81`, but laser selection is
  still anti-event (`event/non-event selected ratio 0.52x`). FC4 (`1.32x`) and
  snow-particle-counter (`2.57x`) are event-biased, but this was insufficient to
  beat the validation-selected static policy.
- A first attempt to use `scripts/47_v31_behavior_diagnostics.py` on the
  energy-account run failed because the script expects older `truth_v31.csv`
  naming; scoped NPZ-based diagnostics were used instead and written to
  `reports/energy_account_split_protocol_socaux_abort2x_200k/diagnostics/sensor_event_usage.csv`.
- Created `rl_sensor_scheduling_framework/docs/05-26-socaux-abort-gate-report.md`
  with the same-run ranking, abort/gating interpretation, and decision not to
  scale the setting.

## 2026-06-02 Fork-Branch PD-PPO Paper Rewrite
- User redirected work to a thorough PD-PPO manuscript rewrite in the current fork
  context and explicitly requested backing up the current paper before creating a
  new TeX source.
- Inspected the current paper structure and confirmed `paper/main.tex` is the
  active old entrypoint with old prose under `paper/sections/`; the nested paper
  git repository was already dirty, so no prior changes were reverted.
- Created backup archive:
  `rl_sensor_scheduling_framework/paper_archives/paper_pre_fork_rewrite_20260602_122720.tar.gz`
  (`69M`).
- Added new clean source files:
  `paper/pdppo_crst_rewrite.tex`,
  `paper/rewrite_sections/01_introduction.tex` through
  `paper/rewrite_sections/08_conclusion.tex`, and
  `paper/pdppo_crst_rewrite_highlights.txt`.
- The new source is not wired to the old `sections/*.tex`; it is organized around
  cold-region monitoring motivation, regime-dependent energy conditions,
  chronological split protocol, corrected fixed-budget results, energy-account
  mechanism diagnostic, limitations and future work.
- Validation:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeds and emits `paper/pdppo_crst_rewrite.pdf` (`23` pages).
  - No undefined citation or label warnings remain.
  - Remaining log issues: two small overfull boxes (`2.0pt`, `0.25761pt`) and an
    existing BibTeX warning for empty `pages` in `Pendyala2024`.
  - Abstract word count is about `215`, within the CRST `250`-word limit.
  - Highlights have lengths `70/75/73/73/62`, all within the `85`-character limit.

### Figure/Table Integration Correction
- User correctly questioned why the clean rewrite contained no figure references.
  The first pass cited four tables but deliberately skipped figures pending
  provenance/artwork review; that was too sparse for a manuscript draft.
- Added figure support to the new source:
  - `pdppo_crst_rewrite.tex`: loaded TikZ and required libraries.
  - `rewrite_sections/04_framework_protocol.tex`: added chronological split
    timeline figure and cross-reference.
  - `rewrite_sections/05_simulation_setup.tex`: added generator-statistics figure
    and cross-reference.
  - `rewrite_sections/06_results.tex`: added power-error tradeoff and behavior
    diagnostic figures with conservative captions.
- Recompiled successfully with `latexmk -xelatex -interaction=nonstopmode
  -halt-on-error pdppo_crst_rewrite.tex`; PDF is now `26` pages.
- Rendered text now contains Figure `1--4` and Table `1--4`. Log has no undefined
  references/citations; only two small overfull boxes and the existing empty-pages
  BibTeX warning for `Pendyala2024` remain.

### Original-Asset and Theory Migration
- User requested retaining the original AWS rendering, chronological dataset split
  schematic, and important theoretical derivations rather than discarding all
  legacy paper assets.
- Added the AWS platform rendering to the new introduction as Figure `1` with a
  conservative caption identifying it as a conceptual rendering, not a field
  photograph or validation record.
- Redrew the PD-PPO framework as a clean TikZ diagram
  (`paper/figures/pdppo_framework_rewrite_tikz.tex`) and inserted it as Figure `2`;
  this is manually generated vector artwork, not AI-generated submission art.
- Preserved the chronological split schematic as Figure `3`.
- Migrated the useful theory into the new problem-formulation and appendix:
  prediction-driven reward non-equivalence, feasibility under action projection,
  the strength of static allocation under fixed budgets, and supplementary proof
  details in `rewrite_sections/appendix_theory.tex`.
- Recompiled successfully:
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
  The PDF is now `31` pages with no undefined references/citations. Remaining log
  items are two tiny overfull boxes, one underfull box, and the existing
  empty-pages BibTeX warning for `Pendyala2024`.
- Fixed a cross-reference wording defect that rendered as `Appendix Appendix A.1`;
  the proof sketch now refers to `Section A.1`.
- Open item: before submission packaging, record the AWS rendering as a
  user-produced Blender render rather than stock, photographic, or AI-generated
  artwork.

### PD-PPO Framework Figure Iterative Review
- User confirmed the AWS rendering is a self-produced Blender render, close to the
  real AWS platform used by the project. The earlier provenance risk is therefore
  narrowed to recording this fact cleanly in the submission materials.
- Reworked `paper/figures/pdppo_framework_rewrite_tikz.tex` through three manual
  review rounds, using Figure `3`'s low-saturation TikZ style as the reference:
  - replaced the previous dense colored-band layout with two light panels and
    compact pastel boxes;
  - removed all white-background arrow-label blocks and old label styles;
  - simplified arrows into solid runtime flow and dashed training/update flow;
  - removed the crowded standalone `Train-only prior` box and folded prior
    information into the PD-PPO box;
  - moved panel titles and rerouted arrows to avoid visible text overlap.
- Updated the Figure `2` caption in
  `paper/rewrite_sections/04_framework_protocol.tex` so it no longer refers to an
  oracle-prior block that is not drawn separately.
- Validation:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeds;
  - rendered page `14` was visually inspected after each iteration;
  - source scan confirms no residual `fill=white`, `alabel`, `dotted`,
    `sffamily`, `Train-only prior`, or `oracle-prior block` text remains in the
    new framework figure/caption;
  - final PDF is `30` pages with no undefined references/citations. Remaining log
    items are unchanged non-blocking warnings: two tiny overfull boxes, one
    underfull box, and the `Pendyala2024` empty-pages BibTeX warning.
- Follow-up layout correction after user review:
  - fixed the `Runtime scheduling loop` label being visually crowded/covered by
    the runtime block area;
  - changed the grey group boxes to TikZ `fit` nodes drawn on the background layer,
    so the Scheduler blue block is unambiguously contained by the runtime panel;
  - recompiled and re-rendered page `14`; the two reported Figure `2` layout
    defects are resolved with no new LaTeX warnings.

### Reference Expansion to Around 25 Entries
- User requested increasing the new clean rewrite's reference count by reusing
  vetted references from the archived manuscript.
- Audited current citation usage: the clean rewrite initially cited `14` unique
  keys despite `references.bib` containing `33` entries copied from the archived
  paper.
- Added semantically placed citations rather than uncited bibliography padding:
  - Introduction: Antarctic AWS / polar forecasting context now cites
    `AntAWS2023`, `Wang2021`, `Ding2025`, `Amory2020`, `Lenaerts2023`,
    and `Monrad2026`.
  - Related Work: resource-constrained sensor scheduling now cites
    `FernandezBes2015`, `Qu2022`, `Alali2024`, `AlAhdab2025`, `Jonah2026`,
    and `Tran2026`.
  - Related Work: learning-based sensing/scheduling now cites `Schulman2017`,
    `Murad2020`, `Wei2020`, `Ogbodo2025`, `Liang2024`, `Pendyala2024`,
    and `Ibrahim2024`.
  - Forecast decision signal: multi-horizon neural forecasting now cites
    `Lim2021` and `Liu2024`.
  - Simulation setup: synthetic environmental time-series grounding now cites
    `Aloni2024`.
- Validation:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeds.
  - `pdppo_crst_rewrite.bbl` now contains `28` `\bibitem` entries.
  - No undefined citations/references remain.
  - Remaining non-blocking warnings: four archived conference entries lack page
    fields (`Liu2024`, `Murad2020`, `Pendyala2024`, `Wei2020`), plus the existing
    small overfull/underfull layout warnings.

### CRST Engineering Abstract and Simulator-Construction Revision
- User requested a CRST-style abstract under `250` words, less abbreviation-heavy
  wording, a more engineering-oriented title, new keywords/highlights, less
  repeated self-limiting language, and a reviewable simulator-construction section.
- Updated `paper/pdppo_crst_rewrite.tex`:
  - title now reads `Forecast-driven sensor scheduling for Antarctic blowing-snow
    monitoring under energy and warm-up constraints`;
  - abstract rewritten to `168` words;
  - keywords reordered to `Blowing snow; Antarctic automatic weather station;
    Sensor scheduling; Energy constraint; Forecasting; Reinforcement learning`;
  - front-matter author emails corrected to `yongzheli@seu.edu.cn` and
    `220245154@seu.edu.cn`, with affiliation `School of Mechanical Engineering,
    Southeast University, Nanjing, China`.
- Updated `paper/pdppo_crst_rewrite_highlights.txt` with five CRST-compliant
  highlights; lengths are `68/66/75/67/67` characters.
- Reduced repeated defensive phrasing in the abstract, introduction, results,
  discussion, conclusion, captions, and data-availability statement. The retained
  limitation language is concentrated in the introduction close and Discussion.
- Reworked `rewrite_sections/05_simulation_setup.tex` to include
  `Simulator construction` with six subparts: meteorological backbone,
  blowing-snow event generation, particle and mass-flux variables, sensor
  observation model, normalized cost assignment, and sanity-check criteria.
- Added supplementary editable table
  `paper/tables/simulator_parameters.tex`, included from
  `rewrite_sections/appendix_theory.tex`, summarizing corrected fixed-budget
  generator, split, event, sensor-cost, oracle, training, selection, and final-test
  parameters.
- Validation:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error
    pdppo_crst_rewrite.tex` succeeds.
  - PDF metadata shows the new title, keywords, and authors.
  - `pdftotext` confirms the corrected emails and new `Simulator construction`
    and `Simulator Parameter Summary` sections are rendered.
  - No undefined citations or references remain. Current PDF is `36` pages.
  - Remaining non-blocking log items: one `2.0pt` overfull box and four underfull
    boxes.

### 06-02-02 PD-PPO Format Polish Pass
- User requested full-manuscript polishing according to
  `rl_sensor_scheduling_framework/docs/06-02-02-ppo-format.md`.
- Applied the document's deterministic patches to the clean rewrite source:
  - removed AI-like meta-framing from the abstract, introduction, results,
    discussion and conclusion;
  - standardised active manuscript spelling to British forms such as
    `normalised`, `favour`, `behaviour`, `optimisation`, `artefacts`;
  - replaced self-referential revision labels such as `corrected manuscript` and
    `rewritten manuscript`;
  - rewrote the limitations paragraph from serial `First/Second/...` phrasing into
    a coherent limitations paragraph;
  - clarified Table 3 and Table 4 captions and added the three-hour logical-epoch
    caveat for the energy-account analysis in the Results text;
  - moved the static-comparator proposition and remark from the forecast-reward
    subsection to the instantaneous-budget subsection, as requested by C-5;
  - added a sentence after the behaviour timeline noting that the single-seed
    timeline should not be read as robust event-triggered laser control.
- Validation:
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error
    pdppo_crst_rewrite.tex` succeeds and updates the `36`-page PDF.
  - Pattern scan finds no remaining instances of the main 06-02-02 flagged phrases
    (`This study asks`, `regime map`, `prevents overclaiming`, `corrected
    manuscript`, `rewritten manuscript`, `not inconsistent`, American
    `normalized/favor/behavior`, etc.) in the active source.
  - Abstract is `206` words, within the CRST `250`-word limit.
  - Highlights lengths are `68/67/75/67/67`, all below the `85`-character limit.
  - Final LaTeX log has no undefined references/citations. Remaining non-blocking
    layout warnings are one `2.0pt` overfull box and three underfull boxes.

### CRST Code-Release Repository
- Created a separate local release repository at
  `/home/horeb/_code/microclimate_demo/forecast-driven-sensor-scheduling` rather
  than publishing the current dirty project workspace.
- Curated release contents to the current manuscript's reproducibility surface:
  PD-PPO/V3.1 code, public-weather synthesis code, sensor configs, selected
  split-protocol and energy-account scripts, compact aggregate CSV/JSON result
  artefacts, manuscript source, figures, tables, highlights, and references.
- Added publication-facing metadata and docs:
  `README.md`, `REPRODUCIBILITY.md`, `DATA_AVAILABILITY.md`, `CITATION.cff`,
  MIT `LICENSE`, `docs/SUBMISSION_PACKAGE.md`, and
  `docs/RESULTS_MANIFEST.md`.
- Sanitised release artefacts:
  - excluded large truth CSVs, rollout NPZs, model checkpoints, logs, pycache,
    and LaTeX build outputs;
  - removed stale local absolute paths from aggregate CSV/JSON files;
  - patched collector scripts so future outputs use repository-relative paths.
- Validation completed:
  - `conda run -n darts python -m pytest tests/v2/test_custom_ppo.py
    tests/v2/test_v31_protocol_replay.py tests/test_public_weather_synthesis.py`
    -> `12 passed`;
  - `conda run -n darts python scripts/54_rebuild_table3_main_results.py`;
  - `conda run -n darts python scripts/62_energy_account_split_protocol_collect.py`
    -> `complete_seeds=5/5`;
  - `conda run -n darts python scripts/36_v2_plot_locked_power_error.py`;
  - `conda run -n darts python -m py_compile scripts/*.py src/v2/*.py
    src/data_sources/*.py`;
  - `latexmk -xelatex -interaction=nonstopmode -halt-on-error
    pdppo_crst_rewrite.tex` from the release `paper/` directory.
- Published GitHub repository:
  `https://github.com/JekYUlll/forecast-driven-sensor-scheduling`.
- Created GitHub release:
  `https://github.com/JekYUlll/forecast-driven-sensor-scheduling/releases/tag/v0.1.0`.
- Updated the active manuscript data-availability statement to point to the
  public repository. DOI generation remains a final archive task before
  journal submission.

## 2026-06-21 ESWA Theory Integration and Plan Consistency
- Reconciled the root plan with the current project state: the SCENEBAL-2
  `117--140` strong-claim experiment is complete, and the active task is now
  ESWA manuscript theory integration / packaging rather than continued 24-hour
  experiment monitoring.
- Synced the subproject planning state:
  `rl_sensor_scheduling_framework/.planning/.active_plan` now points to the
  ESWA planning directory, and the subproject plan records `paper/main.tex` plus
  `paper/sections/*.tex` as the canonical manuscript source.
- Applied the approved specialist-bottleneck theory extension in
  `rl_sensor_scheduling_framework/paper/`: added the definition/proposition,
  appendix proof, simple-cycle boundary, and verified citations.
- Rebuilt `rl_sensor_scheduling_framework/paper/main.pdf` successfully with no
  undefined citations or references. Remaining BibTeX warnings are existing
  empty-page warnings only.

## 2026-06-21 ESWA Manuscript Boundary Consistency
- Added the SCENEBAL-2 design-balance explanation, raw macro sensitivity
  boundary, and current data-availability boundary to the canonical manuscript
  under `rl_sensor_scheduling_framework/paper/`.
- The paper now states that the learned-policy raw unnormalised subtype macro
  gate is not supported (`0/24`), while the supported claim remains tied to
  ordinary step and static-normalised event-regime macro evidence.
- Rebuilt `paper/main.pdf` successfully: `37` pages, no undefined
  citations/references, with only the existing empty-page BibTeX warnings.

## 2026-06-21 New-Claim Manuscript Check
- Checked the canonical paper path for whether it has fully moved to the new
  SCENEBAL-2 claim.
- Fixed residual old-claim wording in highlights (`18 seeds`) and in the
  problem-formulation metric-boundary sentence.
- Rebuilt `rl_sensor_scheduling_framework/paper/main.pdf` successfully and
  confirmed the active source/PDF no longer contain old `18 seeds`,
  `SCENEBAL-1`, `V3.1`, `metpair`, `seed45`, `h075`, `CRST`, or `pdppo_crst`
  wording in the submission-facing files checked.

## 2026-06-21 Figure-Count Repair
- Audited the reduced figure count in the active SCENEBAL-2 manuscript.
- Added two new SCENEBAL-2-specific figures to the Results section:
  metric-boundary diagnostics and behaviour-complexity audit diagnostics.
- Rebuilt `rl_sensor_scheduling_framework/paper/main.pdf`; it now has seven
  active figures, `38` pages, and no undefined references/citations.
