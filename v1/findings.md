# Findings: Forecast-Aware Constrained Sensor Scheduling

## 2026-05-26 Takeover Findings
- `rl_sensor_scheduling_framework/` is now treated as archived baseline/evidence.
  It contains reusable environment, sensor, projector, oracle, rollout and
  evaluation code, but the new method should live under `v1/`.
- The user-provided evaluation in `v1/05-25-01-mcp-teacher.md` endorses the
  high-level direction: forecast-aware state, constrained objective, and
  MPC/oracle teacher distillation.
- Important correction from the current code audit: old v2 custom PPO does not
  put explicit future TCN forecasts into the policy state. The frozen TCN oracle
  provides dynamic reward, while policy state contains history, masks, warm-up
  state, current event flag and SOC. New v1 state must add explicit forecast
  context.
- Current SOC auxiliary result is a negative gate for incremental PD-PPO repair:
  aborts dropped, but `custom_ppo` still lost to `validation_selected_static`.
  This justifies a new implementation rather than more penalty tuning.

## Reuse Boundary
- Reuse:
  `src/v2/env.py`, `sensor_spec.py`, `power_projector.py`, `warmup_state.py`,
  `tcn_oracle.py`, `rollout.py`, `evaluation.py`, and corrected split-protocol
  logic from `scripts/61_energy_account_split_protocol_run.py`.
- Rewrite:
  policy state construction, teacher generation, BC/AWBC policy, constrained
  objective, and any CMDP/dual training loop.

## First Implementation Decision
- Start with a short-horizon beam-search teacher rather than full RL. It gives a
  concrete upper-reference policy and supervised labels, directly addressing
  credit assignment before adding constrained actor-critic complexity.

## Implementation Finding
- The new scaffold can reuse the archived v2 feasibility/projector API without
  editing the archived framework. The first verified interface is mask-level:
  the teacher enumerates candidate masks, filters them through archived
  feasibility checks, and restores the environment after planning.
- The toy smoke result is not a scientific result. Its dummy oracle is only for
  API validation; real claims still require the corrected split protocol and
  `validation_selected_static` comparator.
- The teacher-dataset CLI can already run against an existing archived truth CSV
  and physical-event v4 sensor config. The verified smoke used `oracle-type
  none`, so the next meaningful gate must load a real frozen TCN oracle and use
  the same split bounds as the corrected protocol.
- BC training/checkpointing is now mechanical. The hard question is no longer
  serialization or policy inference, but whether teacher labels generated with a
  real oracle beat the selected static comparator under validation/final splits.

## Protocol Runner Finding
- The v1 protocol runner now enforces four chronological partitions and defaults
  normalization to `oracle_pretrain`, avoiding validation/final leakage through
  state normalization.
- The real-TCN smoke confirms that archived frozen oracles can be reused by the
  new v1 runner without editing `rl_sensor_scheduling_framework`.
- The first tiny TCN smoke did not pass the strict comparator (`mpc_teacher`
  objective 5.602823 vs `validation_selected_static` 5.601813). This should not
  be interpreted as a method failure because the run used only one final window
  and a 1-step planning horizon; it does confirm that the gate logic is capable
  of rejecting a non-improving controller.
- Custom-PPO replay is implemented as an optional checkpoint path, but it is not
  yet a source of new claims. A real saved `custom_ppo.pt` can now be loaded and
  evaluated beside static, MPC teacher and BC in the same final-test runner, but
  the smoke used a tiny window and only validates wiring.

## Teacher Collapse Finding
- The first medium server gate revealed a concrete algorithm defect: when the
  frozen oracle clips all cold-start actions to the same high loss, the original
  low-power tie-break chooses the empty mask and never escapes it. This caused
  512/512 all-off teacher labels and made BC all-off as well.
- The repair is deliberately narrow: add an intended reward-target coverage
  bonus only while the oracle loss is saturated. This supplies a bootstrap
  preference for informative sensing without replacing the forecast-loss
  objective once action differences become measurable.
- A second protocol correction is required for scientific interpretation:
  `mpc_teacher` uses privileged oracle search during final replay and therefore
  is a reference/upper bound, not a deployable submitted method. Only BC and
  subsequent constrained learned policies can determine the method gate.

## Candidate-Prior Gate Finding
- Train-split candidate prior is legal under the split protocol and removed the
  all-off/high-coverage failure mode enough to make the teacher competitive:
  the medium seed-41 teacher improved from `3.630640` to `1.203506` at prior
  weight `0.5`. This is still worse than the validation-selected static final
  loss `1.183638`.
- The best static policy is not a weak baseline. Validation selects
  `met_station_core|surface_temp_ir|shielded_thermo_hygro`; train static prior
  selects the nearly identical `met_station_core|surface_temp_ir`. These are
  stable low-power weather/surface-temperature masks and beat custom PPO, full
  open, feasible static projected, round robin, and BC in this gate.
- The failure is now objective-level rather than implementation-level. The
  teacher is no longer collapsed, but its short-horizon oracle search still
  spends power on dynamic/radiation/particle sensors whose final-test frozen
  oracle benefit is not enough to beat the static weather/surface subset.
- Scaling this exact teacher/BC design would be premature. Next meaningful work
  should either redesign the teacher cost around regret against the
  train-selected static prior, or change the experimental task so event-driven
  particle sensing has measurable value under a held-out static comparator.

## Active Correction Finding
- The next algorithmic correction is not another PPO tuning pass. The correct
  abstraction is a residual dynamic scheduler around a strong static anchor:
  select a static baseline on train/validation evidence, then let the controller
  deviate only when short-horizon forecast regret justifies the switch.
- This changes the teacher from absolute-loss MPC to regret-gated MPC. It should
  prevent the teacher from being worse than the static anchor merely because
  local coverage or one-step forecast noise looks attractive.
- The old checkpoint oracle objective may still be intrinsically misaligned with
  the dynamic event-sensing claim because it rewards stable weather/surface
  variables enough for static masks to dominate. The `event_transport` weight
  mode is a deliberate task-definition correction: the frozen predictor remains
  fixed, but the scalar objective emphasizes snow flux and particle transport
  forecast quality.
- Existing final rollouts already show why a task-focused objective is needed:
  under prior weight `1.0`, teacher has worse oracle loss than static
  (`1.217541` vs `1.183638`) but much better event transport normalized error
  (`0.423431` vs `0.613293`). This is not a failure of dynamic sensing; it is a
  failure of the old scalar objective to value the event task sufficiently.
- The composite objective is therefore the current scientific direction:
  optimize and report both frozen-oracle forecast loss and event transport
  task error, with a declared scalar objective only for gate/model selection.

## Deployable Distillation Finding
- Under quick task-composite gates, the privileged teacher now beats the strict
  static comparator, but BC does not yet pass. This is progress: the main
  algorithmic objective has been corrected enough to create teacher lift, and
  the remaining failure is deployable distillation.
- Candidate prefiltering by train static prior preserves teacher lift while
  reducing MPC cost. This should be kept for medium/full gates.
- A simple validation-calibrated fallback to the static anchor was not
  sufficient; it worsened the quick final objective. The BC policy needs better
  imitation, not just a confidence threshold. Next likely fixes are more teacher
  samples, larger BC, class-balanced/focal loss, or sequence-level residual
  imitation.
- One DAgger iteration fixed the quick deployable gate: BC improved from failing
  (`1.902616`) to passing (`1.895795`) against the same static comparator
  (`1.899117`). This supports the diagnosis that the remaining gap was rollout
  distribution shift rather than model capacity alone.
- The same correction passed the medium seed-41 gate. Static was `1.241163`,
  teacher was `1.205506`, and deployable DAgger-BC was `1.240028`. The deployed
  margin is small, so this is a viability result, not yet a robust empirical
  claim. It is enough to justify scaling; it is not enough alone for paper
  submission.
