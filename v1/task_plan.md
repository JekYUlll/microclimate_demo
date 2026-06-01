# Task Plan: Forecast-Aware Constrained Sensor Scheduling

## Goal
Implement a new, independent forecast-aware constrained scheduling prototype under
`v1/`, treating `rl_sensor_scheduling_framework/` as an archived evidence and
baseline source. The new implementation should directly address the diagnosed
PD-PPO failure modes: missing explicit forecast context, weak credit assignment,
fixed penalty constraint handling, and lack of teacher guidance.

## Current Phase
Phase 6 in progress: strong-claim redesign. The weak value-residual result is
not enough for the original paper claim, so the active mainline now replaces
heuristic forecast context with a split-compliant learned event forecaster and
will continue toward learned rollout-value / online planning.

## Minimum Claim Target
- In strict chronological split protocol, the deployable forecast-aware DAgger
  policy must beat the validation-selected static policy on held-out final-test
  windows in at least `4/5` seeds, with positive mean paired margin.
- The privileged MPC teacher must also beat the same static comparator in at
  least `4/5` seeds, establishing that the planning objective has dynamic value.
- Required ablations for paper use: no-DAgger and oracle-only objective. These
  must be reported as mechanism controls, not as alternate main methods.

## Non-Negotiable Boundaries
- Do not modify `rl_sensor_scheduling_framework` core source for the new method.
- Reuse archived framework components only through imports or copied data
  artifacts, so old evidence remains reproducible.
- Keep split protocol semantics: oracle pretrain, RL/teacher train, validation
  selection, and final test must remain disjoint.
- Treat any truth-future teacher signal as training-only; deployed policy inputs
  must use forecast or causal features.

## Phase 1: Architecture Scaffold
- [x] Read `v1/05-25-01-mcp-teacher.md` and recovered prior findings.
- [x] Create `v1/` planning files for the new implementation track.
- [x] Write architecture note describing reusable components and new modules.
- [x] Implement forecast/event context feature builder.
- [x] Implement short-horizon MPC/beam-search teacher that leaves env state intact.
- [x] Add minimal tests for forecast context and teacher feasibility.
- **Status:** complete

## Phase 2: Teacher Dataset and Baseline Policy
- [x] Add script to build teacher labels on training starts using archived v2 env.
- [x] Add behavior-cloning dataset format with masks, forecast context and action labels.
- [x] Implement a small forecast-aware sensor-subset policy for BC/AWBC.
- [x] Run local CPU smoke on tiny synthetic truth.
- [x] Add BC checkpoint save/load and a training CLI.
- **Status:** complete

## Phase 3: Protocol Runner
- [x] Add a v1 runner reusing the corrected semi-Markov split protocol.
- [x] Use one frozen oracle per run and keep it fixed across controller variants.
- [x] Evaluate validation-selected static, MPC teacher, and BC policy.
- [x] Add optional custom-PPO checkpoint replay path.
- [x] Validate custom-PPO checkpoint replay on a real saved checkpoint.
- [ ] Gate on seed 41 before scaling. Completed medium bootstrap and
  train-prior variants (`candidate_prior_weight=0.5/1.0`); all failed the
  deployable-policy gate against `validation_selected_static`.
- [ ] Active correction: evaluate static-anchor regret-gated teacher and
  task-weighted event-transport oracle objective before scaling.
- [x] Active correction passed seed-41 medium gate under task-composite
  objective with top-k MPC teacher and one DAgger iteration:
  `protocol_gate_energy_socaux_seed41_task_anchor_fast_dagger1`.
- **Status:** complete for seed-41 gate; pending multi-seed scaling

## Phase 4: Constraint Learning
- [x] Decide whether DAgger-BC is sufficient as the submitted deployable method
  or whether to add cost-vector output / dual-variable fine-tuning.
- [ ] Add reward critic / cost critic training loop because multi-seed BC/KNN
  deployment is unstable despite teacher lift.
- [ ] Compare constrained fine-tuning against BC-only.
- **Status:** in_progress

## Phase 5: Server Experiments and Paper Assets
- [x] Launch and pass server seed-41 medium gate for corrected v1 method.
- [x] Implement batch multi-seed claim-suite launcher.
- [x] Implement claim-suite aggregator and pass/fail assessment.
- [x] Run n=5 with task-composite objective and value-residual deployable
  policy. `value_residual_safe` passed: deployable `4/5`, teacher `5/5`,
  mean deployable margin `+0.002213`.
- [x] Run required ablations: no-DAgger and oracle-only objective.
  `value_residual_no_dagger` passed with the same `4/5` pattern as the main
  method; `value_residual_oracle_objective` failed at `2/5` with negative mean
  deployable margin.
- [x] Aggregate results and produce behavior diagnostics.
- [x] Replace unstable deployable action classification with action-cost/value
  imitation and rerun n=5.
- [ ] Evaluate teacher-label action-support deployment guards (`support4/6/8/12`)
  and rerun the best supported deployable method on n=5.
- [ ] Prefer validation-calibrated action-support selection
  (`support_calib_safe`) over final-test posthoc top-k selection if it passes.
- [ ] Evaluate sensor-level mask BC (`mask_safe`/`mask_anchor_safe`) as the
  fallback deployable layer if action-id support guards are insufficient.
- [ ] Evaluate residual anchor-deviation BC (`residual_safe`) after
  `support1_safe`--`support5_safe` all failed at `2/5` deployable wins.
- [x] Evaluate value-residual action-cost policy (`value_residual_safe`) after
  `residual_safe` and privileged-context BC failed.
- [ ] Update manuscript only if strict static comparator is beaten robustly.
- **Status:** in_progress

## Phase 6: Strong-Claim Redesign
- [x] Admit the current result is insufficient for the original strong claim:
  small n=5 margin, no significance, no cross-setting generalization, and
  DAgger not an active mechanism.
- [x] Replace hand-coded causal event-risk features with a train-only learned
  multi-horizon event forecaster. The learned forecaster writes causal
  probability columns into the truth table, and all downstream teacher/cost/
  deployable policies consume those columns through `ForecastContextConfig`.
- [x] Add `learned_value_residual_safe` claim-suite preset.
- [x] Run local real-seed smoke for the learned-forecast path.
- [x] Run server n=5 learned-forecast candidate:
  `v1_claim_learned_forecast_n5_20260601`. It repeated the weak `4/5`
  pattern with a smaller mean margin and does not satisfy the original strong
  claim.
- [x] Add uncertainty-aware action-cost ensemble and online value planner.
- [x] Run server n=5 learned-forecast + ensemble-value candidate:
  `v1_claim_learned_ensemble_n5_20260601`. It failed at `3/5`, confirming that
  uncertainty-aware absolute-cost ensembling is not the right main route.
- [x] Add anchor-advantage residual policy: directly learns the candidate
  advantage relative to the validation-selected static anchor instead of
  subtracting two independently learned absolute costs.
- [ ] Run server n=5 anchor-advantage residual candidate:
  `v1_claim_learned_advantage_n5_20260601`. The first launch exposed two
  non-result failures: seed42 hit server disk exhaustion, and seed45 exposed an
  anchor projection bug in advantage-data collection.
- [x] Add validation-calibrated anchor-advantage support selection:
  `learned_advantage_residual_calib_safe`.
- [x] Fix anchor-default semantics so residual policies and advantage-data
  collection treat the validation-selected static anchor as a mask submitted to
  the environment projector, not as an action that must be exactly feasible at
  every warmup state.
- [x] Fix residual support semantics so a temporarily invalid support set
  falls back to the static anchor rather than reopening the full OOD action
  space.
- [ ] Run corrected calibrated n=5 candidate:
  `v1_claim_learned_advantage_calib_anchorfix_strict_n5_20260601`. Result:
  teacher `5/5`, deployable `0/5`, mean deployable margin `-0.018997`.
  This route is rejected as the main deployable algorithm.
- [ ] Active correction: evaluate a validation-selected hybrid residual suite
  that includes the stable learned value-residual policy and the rejected
  advantage residual policy, selecting deployables on validation rather than
  committing to advantage residual alone.
- [x] Add `learned_hybrid_residual_calib_safe` preset for that hybrid route.
- [x] Add guarded validation deployable selection:
  `learned_hybrid_residual_guarded_safe`. It keeps the existing validation
  selection path split-compliant but prefers deployable policies whose
  validation margins over the static anchor are not driven by a single
  overfitted mean improvement.
- [x] Add an event-threshold residual candidate:
  `learned_hybrid_event_guarded_safe`. It uses the learned event forecaster to
  trigger a validation-calibrated switch from the static anchor to a
  teacher-supported event action, then lets guarded validation choose among
  value-residual, advantage-residual, and event-threshold deployables.
- [ ] Run server n=5 event-threshold hybrid candidate:
  `v1_claim_learned_hybrid_event_guarded_n5_20260601`.
  Paired confirmation result:
  `v1_claim_learned_hybrid_event_guarded_paired_n5_20260601` passed the n=5
  gate: deployable `4/5`, teacher `5/5`, mean margin `+0.003758`.
- [ ] Scale the final strong candidate beyond the old n=5 single-setting gate:
  more seeds, at least two budgets, and at least one event-regime perturbation.
  Budget matrix completed and failed as a cross-budget robustness claim:
  `B=1.05` deployable `1/5`, `B=1.20` deployable `4/5`, and `B=1.35`
  deployable `1/5`. The sparse-event perturbation (`event_coverage=0.20`)
  passed at `4/5` with positive mean margin.
- [ ] Active correction for the `B=1.35` failure: test an
  event-support-cycle deployable that rotates over teacher-supported event
  actions instead of switching to one fixed event mask. The initial
  time-cycle variant failed (`1/5`, mean margin `-0.008706`) while teacher
  stayed `5/5`; the active follow-up is the freshness-selection grid over the
  same teacher-supported action set.
- **Status:** in_progress

## Error Log
| Time | Error | Resolution |
|---|---|---|
| 2026-05-26 | None yet | N/A |
| 2026-05-27 | Medium teacher collapsed to all-off under saturated oracle loss | Added saturated-loss coverage bootstrap and regression test |
| 2026-05-27 | Candidate-prior teacher improved but still did not beat validation-selected static | Treat current teacher objective as misaligned; do not scale until teacher/reference is redesigned or the static comparator target is changed |
| 2026-05-27 | Absolute-loss MPC can drift away from a strong static solution | Added static-anchor regret guard and task-target oracle reweighting options |
| 2026-05-27 | Oracle-only objective does not value event transport enough for dynamic sensing | Added task-composite objective and event-transport task error |
| 2026-05-27 | BC fitted teacher labels but failed rollout gate | Added one-iteration DAgger; medium seed-41 gate passed |
| 2026-05-27 | Action-cost policy minimized over OOD feasible masks and dropped core context sensors | Added teacher-label action-support guard plus static-anchor inclusion for deployable policies |
| 2026-06-01 | Anchor-advantage fixed-support run produced non-result failures: seed42 disk full and seed45 empty advantage rows | Cleaned server caches, stopped old runs, fixed anchor projection semantics, added regression tests, and relaunched calibrated anchorfix suite |
| 2026-06-01 | New v1 input-prep smoke initially could not find AntAWS data because the default path inherited the old script's working-directory assumption | Changed the default to `data/AntAWS/3_hourly` and added path resolution against the project root and archived framework root |
| 2026-06-01 | `aggregate_budget_matrix.py` parsed the root directory name `budget_matrix_*` as a budget tag | Tightened parsing to match only concrete tags such as `budget1p20` and added a regression test |
| 2026-06-01 | First launch of `v1_budget1p35_event_cycle_20260601` exited immediately because stdout was redirected into a directory that did not exist | Relaunched after `mkdir -p`; no partial experiment outputs were produced by the failed launch |
| 2026-06-01 | Event-support-cycle calibration grid gained `selection_mode`, but the stable sort tie-break still referenced the old tuple index | Fixed the sort key to use the combo id and added freshness-selection regression coverage |
