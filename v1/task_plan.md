# Task Plan: Forecast-Aware Constrained Sensor Scheduling

## Goal
Implement a new, independent forecast-aware constrained scheduling prototype under
`v1/`, treating `rl_sensor_scheduling_framework/` as an archived evidence and
baseline source. The new implementation should directly address the diagnosed
PD-PPO failure modes: missing explicit forecast context, weak credit assignment,
fixed penalty constraint handling, and lack of teacher guidance.

## Current Phase
Phase 5f in progress: value-residual main claim and required ablations are
complete; consolidating paper-ready tables and claims.

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
