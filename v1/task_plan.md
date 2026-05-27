# Task Plan: Forecast-Aware Constrained Sensor Scheduling

## Goal
Implement a new, independent forecast-aware constrained scheduling prototype under
`v1/`, treating `rl_sensor_scheduling_framework/` as an archived evidence and
baseline source. The new implementation should directly address the diagnosed
PD-PPO failure modes: missing explicit forecast context, weak credit assignment,
fixed penalty constraint handling, and lack of teacher guidance.

## Current Phase
Phase 3b complete; ready for Phase 4/5 scaling decision.

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
- [ ] Decide whether DAgger-BC is sufficient as the submitted deployable method
  or whether to add cost-vector output / dual-variable fine-tuning.
- [ ] Add reward critic / cost critic training loop only if multi-seed BC is
  unstable or constraint metrics require it.
- [ ] Compare constrained fine-tuning against BC-only.
- **Status:** pending

## Phase 5: Server Experiments and Paper Assets
- [x] Launch and pass server seed-41 medium gate for corrected v1 method.
- [ ] Run n=5/n=10 with task-composite objective and DAgger-BC.
- [ ] Aggregate results and produce behavior diagnostics.
- [ ] Update manuscript only if strict static comparator is beaten robustly.
- **Status:** pending

## Error Log
| Time | Error | Resolution |
|---|---|---|
| 2026-05-26 | None yet | N/A |
| 2026-05-27 | Medium teacher collapsed to all-off under saturated oracle loss | Added saturated-loss coverage bootstrap and regression test |
| 2026-05-27 | Candidate-prior teacher improved but still did not beat validation-selected static | Treat current teacher objective as misaligned; do not scale until teacher/reference is redesigned or the static comparator target is changed |
| 2026-05-27 | Absolute-loss MPC can drift away from a strong static solution | Added static-anchor regret guard and task-target oracle reweighting options |
| 2026-05-27 | Oracle-only objective does not value event transport enough for dynamic sensing | Added task-composite objective and event-transport task error |
| 2026-05-27 | BC fitted teacher labels but failed rollout gate | Added one-iteration DAgger; medium seed-41 gate passed |
