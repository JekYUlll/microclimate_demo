# Task Plan: Forecast-Aware Constrained Sensor Scheduling

## Goal
Implement a new, independent forecast-aware constrained scheduling prototype under
`v1/`, treating `rl_sensor_scheduling_framework/` as an archived evidence and
baseline source. The new implementation should directly address the diagnosed
PD-PPO failure modes: missing explicit forecast context, weak credit assignment,
fixed penalty constraint handling, and lack of teacher guidance.

## Current Phase
Phase 13 active: v7 regime-causal scenario calibration. The first local smoke
gate now passes after moving from pure instantaneous-budget tuning to
average-power constrained regime complementarity. The v7 setting makes heavy
direct sensors instantaneously feasible but not sustainable as static anchors;
masked-predictor audits pass on seed41/42 12k and seed43 30k. This is a scene
precondition, not a scheduler claim. Next action is a server-scale claim-input
generation/calibration run under the locked v7 smoke setting, storing large
artifacts under `~/data`.

Historical Phase 11 summary: deployable student-interface redesign. The accepted
v6/`event_transport_rich` scene removes the continuous `core+laser` shortcut:
static laser duty is `0`, and the privileged teacher beats static through
SPC/fc4/context/transport complementarity. Window-level audit now confirms MPC
teacher value in `60/60` train/validation/final windows for seeds `41/42/44`.
The blockage is deployable transfer: contextual-duty, teacher-rate/cycle,
sequence-mask imitation, recurrent value, rank-aware recurrent value,
recurrent cost-DAgger, option-planner, runtime-risk, cost-KNN, macro-option
snippet retrieval, Branch-F teacher-improvement gate, and dense always-dynamic
macro retrieval all failed the seed41/42/44 deployable gate while teacher
stayed positive. Sequence/window-value retrieval, absolute-cost rollout-value
planning, and one-step static-anchor advantage scoring have now also failed
even under privileged oracle-regime context. Active correction must move to a
true window-level dynamic-eligibility model or a constraint-aware self-rollout
planner trained under its own rollout distribution. The first window-level
eligibility implementation completed at deployable `0/3` while teacher
remained `3/3`; replacing its option-planner inner executor with a macro
teacher-snippet executor also completed at deployable `0/3`, teacher `3/3`.
The active conclusion is now stronger: similarity-based snippet replay behind
a scalar window gate is insufficient. A rollout-value self-distribution
diagnostic also completed at deployable `0/3`, teacher `3/3`: it reduced
training losses but produced uniformly negative validation margins. The active
conclusion is now that both similarity-based snippet replay and absolute-cost
plus feature-delta rollout planning are misaligned interfaces. The deployable
forecast interface has now been extended with split-compliant learned
continuous forecasts, so future students can condition on predicted transport
and task intensity instead of learned event probability alone. The next
correction must redesign the deployable optimization interface itself: direct
sequence/window outcome verification with richer candidate generation, or a
learned digital-twin objective that predicts static-anchor margins rather than
one-step absolute action costs.
The first branch has now completed negatively: the formal augmented
sequence-value verifier reached deployable `0/3` while the privileged teacher
remained `3/3`. Learned event plus learned continuous forecast context and
richer sequence candidates did not solve validation-tail fragility. The active
correction then moved to learned digital-twin / static-anchor margin
interfaces. The first executed-step learned-digital-twin diagnostic also
completed negatively: deployable `0/3`, teacher `3/3`; executed-outcome data
collection and model training worked, but every validation row was negative or
tail-unsafe. The follow-up multi-candidate window-margin verifier also
completed negatively under local `16/32`-step calibration: deployable `0/3`,
teacher `3/3`. Its useful finding was a calibration-unit mismatch. The
full-rollout calibration correction then completed formally and also failed:
deployable `0/3`, teacher `3/3`; no candidate row passed the static-margin
risk guard, although several had small positive mean validation margins with
negative q25 tails. This closes KNN window-candidate gating and threshold-only
calibration as a main route. The active correction is now a causal
forecast-rollout planner: use the learned event and continuous forecasts as
the rollout substrate for teacher-style short-horizon scoring, instead of
choosing among hand-coded executors by nearest-neighbor window memories. The
first `learned_utility_planner_riskband_safe` smoke completed positively on
a tiny single-start server run, but this is only plumbing plus weak evidence.
The formal utility-planner diagnostic has now completed negatively:
deployable `0/3`, teacher `3/3`; every utility row was worse than the static
anchor on validation or had unsafe lower tails, so the guard correctly disabled
the deployable in all seeds. This closes hand-scored causal utility over
teacher-support masks as a main route. The next correction must stop adding
scalar scoring wrappers and instead use a stronger static-aware constrained
planner interface. The first implementation is a task-only proxy-MPC policy:
it short-horizon plans over teacher-supported feasible masks using learned
transport forecasts, sensor coverage, and column-age freshness, then calibrates
against the static anchor on validation. The all-column proxy smoke was
negative; the task-only proxy smoke was positive (`10.122692` vs static
`10.138463`, teacher `10.115600`) and selected the deployable. This is still
only plumbing plus weak evidence. The formal 3-seed
v6/`event_transport_rich` diagnostic has now completed negatively:
deployable `0/3`, teacher `3/3`. Proxy-MPC achieved positive mean validation
margin in seeds42/44, unlike the scalar utility planner, but all seeds retained
negative q25 tails and the strict guard disabled deployment. The current
hand-specified proxy score is therefore closed as a claim-ready route. Phase 12
is now specified in
`v1/docs/06-06-02-branch-h-revised-execution-plan.md`: train-only, paired
256-step outcomes over a train-selected anchor bank; grouped mean/q25/negative
risk learning; one-sided calibration; and a window-level outer controller that
selects a dynamic proxy-MPC controller or static fallback. This replaces the
leaky/pseudo-replicated draft specification in `06-06-01-v1md`. The active
implementation has since moved to a causal rollout-world robust planner after
the direct risk-controller route failed. The rollout world models pass
train-only gates across seeds `41/42/43/44/45`, and conservative robust
planning gives strict final passes for seeds `41/42`. However, the event-heavy
support restriction sweep only recovers validation candidates for seeds
`43/45`; both fail the held-out final strict gate, while seed44 has no
validation-safe dynamic candidate. Current five-seed status is validation
selected `4/5`, final strict pass `2/5`, final positive mean `3/5`; therefore
the route remains not claim-ready. The follow-up 12-start validation diagnostic
rules out a simple validation-sample-count fix: seed42 would be rejected by
12-start validation despite final success, while seed43 still passes despite
final failure. Run-level support/margin/mean/q25 threshold tuning is therefore
closed as the main correction. Next action: diagnose regime/feature mismatch
between accepted validation windows and final failures, or redesign the
planner around online per-window risk eligibility rather than per-run
selection. The per-window trace diagnosis has now identified the concrete
failure mode: action-effect overestimation for short dynamic deviations from
strong static anchors. Seed43's final failure worsens flux error despite
predicted dynamic advantage; seed45's final failure leaves task errors
unchanged while oracle loss worsens. The next implementation should be an
online action-effect / break-even verifier, or an intervention-outcome model
that predicts dynamic-vs-anchor effect directly. The component trace result
narrows the source: failed dynamic decisions are dominated by predicted
`event_weighted_oracle` gains, while explicit `task_error` support is negative
or zero, so the next correction must verify raw dynamic deviations online
rather than add another run-level threshold. A first non-negative
task-component verifier improves final strict pass from `2/5` to `3/5`
without harming seeds `41/42`, but seed44 remains validation-tail unsafe and
seed45 remains a final q25 failure. The active blocker is now zero-task-signal
oracle-effect overestimation, not ordinary support/margin calibration. A
hold-effect component guard was tested on seed45 and made validation worse, so
simple planned/hold component thresholds are closed; the next serious route is
direct paired intervention-effect learning/calibration. The first seed45
train/validation effect audit confirms scalar predicted-advantage thresholds
are insufficient. Richer multi-seed train effect data have now been collected:
they show substantial positive opportunity but negative aggregate lower tail,
so the active implementation step is a seed/group-aware learned effect
verifier calibrated for accepted-row mean/q25 safety. That row-level verifier
has now failed across causal, compact, and guard-aware feature modes. A
window-level ceiling audit shows the deeper issue: `selected_dynamic` and
`all_raw` expose complementary but incompatible opportunity regimes. The next
implementation should therefore redesign the planner interface around
window-level action-source / candidate-neighborhood selection, not another
row-level threshold. The source-oracle ceiling is optimistic enough to justify
that prototype: per-window choice among `anchor`, `selected_dynamic`, and
`raw_bypass` yields at least one safe train window in every seed. However, the
first replan-level source-selector models fail leave-one-seed safety (`0/5`
safe seeds). Selector stacking is therefore closed for now. The next active
implementation should modify the planner objective/action search directly,
especially by reducing `event_weighted_oracle` dominance and requiring
task-level improvement support. The first objective sweep partially validates
this: seed44 becomes final-safe under a task-only objective, but seed45 still
fails/collapses to static. The active next step is seed45 candidate-support or
anchor-neighborhood redesign under the task-dominated objective. That direct
seed45 support/margin sweep has now failed. The active next step is a formal
validation-selected objective-family protocol: original component-guarded
robust planner plus task-only robust planner, selected by validation without
using final outcomes. That aggregation now reaches the minimum target:
validation pass `5/5`, final strict pass `4/5`, final positive mean `5/5`.
Next work should harden this result into a reproducible suite and add
mechanism diagnostics; do not overclaim single-objective dominance.

## Minimum Claim Target
- In strict chronological split protocol, the deployable forecast-aware DAgger
  policy must beat the validation-selected static policy on held-out final-test
  windows in at least `4/5` seeds, with positive mean paired margin.
- The privileged MPC teacher must also beat the same static comparator in at
  least `4/5` seeds, establishing that the planning objective has dynamic value.
- Required ablations for paper use: no-DAgger and oracle-only objective. These
  must be reported as mechanism controls, not as alternate main methods.

## Phase 13: V7 Regime-Causal Scenario Calibration
- [x] Define v7 success gates in code, not only in prose:
  regime-specific best masks must differ; best static must lose meaningful
  margin to a regime-conditioned oracle; `core+SPC+FC4` must no longer be a
  full-window low-risk solution.
- [x] Add a v1-local truth builder/postprocessor that creates onset, active,
  and decay/thermal phases without modifying archived framework source.
- [x] Add a v7 sensor configuration that creates real complementarity among
  high-precision wind, thermal/radiation, particle, and flux sensors.
- [x] Add a lightweight static-dominance audit based on train-fitted masked
  predictors and final-split regime-conditioned evaluation.
- [x] Run local small-scale v7 smoke audits before any server-scale rerun.
- [ ] Run server-scale v7 claim-input generation and calibration gate.
- **Status:** in_progress

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
- [ ] Next correction: test a teacher-rate deployable for `B=1.35`. The
  freshness event-support grid also failed (`1/5`), because validation did not
  select event-support-cycle in most seeds. The teacher's successful behavior
  is closer to duty-cycle power saving over teacher-supported masks than to a
  single event action.
- [ ] Follow-up correction: test teacher-label sequence cycling. The
  teacher-rate run also failed (`1/5`, mean margin `-0.007056`), so the next
  candidate preserves the teacher's high-switch temporal sequence rather than
  only matching average sensor active rates.
- [ ] Boundary conclusion: `B=1.35` remains a negative deployable boundary.
  Teacher-sequence and BC/KNN guarded hybrids both failed (`1/5` deployable
  wins) while teacher stayed `5/5`; further low-rank teacher-compression tweaks
  are not the current priority.
- [ ] Active scaling pivot: expand the supported `B=1.20` operating point to
  additional seeds using v1-generated inputs, then aggregate old and new runs
  to determine whether the main forecast-triggered claim survives beyond n=5.
- [ ] Scaling result: the event/value guarded route does not survive the
  stricter n=15 seed-win criterion. Combined B=1.20 result is deployable
  `10/15`, teacher `14/15`, positive mean margin, but below the required
  `12/15` deployable wins.
- [ ] Active correction: evaluate the B=1.20 BC/KNN guarded hybrid extension.
  Early seeds `46--47` pass, but the full n=10 corrective result is still
  running.
- [ ] Follow-up correction: because BC/KNN guarded reached only `5/8` before
  becoming mathematically unable to hit `8/10`, run the teacher-rate guarded
  extension as a distinct compression mechanism rather than another action
  classifier.
- [ ] Additional check: run BC/KNN guarded on the original seeds `41--45`.
  The extension result is `7/10`; if old seeds are `5/5`, the same BC/KNN
  method can still reach the combined `12/15` threshold.
- [x] Close the BC/KNN narrow path. Old-seed BC/KNN failed seeds `41` and `42`,
  so it cannot reach the combined `12/15` threshold.
- [x] Implement the next algorithmic tier: learned rollout-value planning.
  This trains a train-split feature-transition surrogate and deploys a
  short-depth learned planner over the action-cost model, instead of treating
  scheduling as one-step action classification.
- [ ] Run server n=5 planner candidate:
  `v1_claim_b1p20_n5_planner_20260601`, root
  `v1/artifacts/claim_suite_b1p20_n5_learned_hybrid_planner_guarded`.
  The first normalized-cost planner failed at deployable `2/5`, teacher
  `5/5`, mean deployable margin about `+0.000346`, so it is rejected as
  evidence but retained as a diagnostic.
- [ ] Active correction: rerun the same n=5 planner after separating rollout
  planning cost learning from one-step residual learning. The rollout planner
  now trains on raw additive candidate costs rather than per-state normalized
  ranking targets. Active server session:
  `v1_claim_b1p20_n5_planner_raw_20260601`, root
  `v1/artifacts/claim_suite_b1p20_n5_learned_hybrid_planner_raw_guarded`.
  If it improves the original seed set to at least `4/5`, scale to seeds
  `46--55`; if it fails, inspect transition-model error, raw-cost calibration,
  validation selection, and candidate support before adding another shallow
  policy head.
- [ ] Parallel correction after raw planner partial failure: test a combined
  teacher-mix candidate suite that lets validation choose among teacher active
  rate matching, teacher sequence cycling, event threshold, and value residual
  under the same split. Active server session:
  `v1_claim_b1p20_n5_teacher_mix_20260601`, root
  `v1/artifacts/claim_suite_b1p20_n5_teacher_mix_guarded`.
  Rationale: completed raw seeds show the teacher's advantage is mainly
  lower-power high-switch temporal mixing, while selected students remain close
  to high-power static anchors.
- [x] Close the raw planner and teacher-mix candidates. Both completed at
  deployable `2/5`, teacher `5/5`; teacher-mix reproduced the same selected
  event/value policies because rate/cycle candidates failed validation.
- [x] Closed correction: evaluate contextual-duty compression. This trains a
  sensor-level teacher probability model, then deploys a closed-loop
  duty-deficit/freshness/power controller over teacher-supported masks. It is
  intended to recover the teacher's low-power, high-switch temporal mixture
  without using privileged future rollouts at deployment. Guard-aware
  contextual-duty completed at deployable `3/5`, teacher `5/5`; rejected as
  the main claim route.
- [x] Closed correction: evaluate a recurrent sequence-mask student. This
  trains on ordered teacher/DAgger trajectories with previous-mask memory and
  deploys a causal GRU policy constrained to teacher-supported feasible masks.
  Result: deployable `3/5`, teacher `5/5`, mean deployable margin
  `+0.002161`; failed the `4/5` n=5 gate. The policy fit teacher masks
  almost perfectly but was never selected as the final deployable.
- [ ] Next algorithmic tier: evaluate recurrent objective-aware value
  student. This keeps the causal GRU state from sequence-mask, but trains on
  per-state teacher rollout costs over feasible candidate masks rather than
  sensor-level BCE. Active server session:
  `v1_claim_b1p20_n5_recurrent_value_20260602`, root
  `v1/artifacts/claim_suite_b1p20_n5_recurrent_value_guarded`.
- [ ] Active correction: recurrent-value exposed a no-op selection gap where
  a policy identical to the static anchor can pass a zero-margin guard and be
  selected. Added positive-margin recurrent presets and launched rank-aware
  posguard n=5: session
  `v1_claim_b1p20_n5_recurrent_rank_posguard_20260602`, root
  `v1/artifacts/claim_suite_b1p20_n5_recurrent_rank_posguard`.
- [x] Closed counterfactual: event-threshold-only B=1.20 n=15 ran in
  `v1_claim_b1p20_event_threshold_only_seq_20260602`. Partial status is
  superseded by final early-stop status: deployable `7/11`, teacher `10/11`,
  mean deployable margin `+0.000016`; it could no longer reach `12/15`.
- [x] Closed validation-transfer correction:
  `learned_event_threshold_valguard_safe`, which keeps the same event-threshold
  deployable policy but calibrates its action/threshold/aggregation by paired
  validation static-margin guard with positive mean margin `0.001`. Final n=5
  result: deployable `3/5`, teacher `5/5`, mean margin `+0.000050`; failed
  the `4/5` gate and kept a negative validation-to-final transfer gap.
- [x] Added strict deployable fallback semantics for validation selection:
  `--deployable-selection-require-guard-pass`. Historical presets keep their
  old behavior; the new `learned_event_threshold_strict_valguard_safe` preset
  falls back to the static anchor if no deployable candidate passes the paired
  validation guard.
- [ ] Prepared next fallback tier: recurrent anchor-advantage student. This
  directly learns candidate improvement relative to the static anchor under a
  recurrent state, rather than learning absolute candidate costs and
  subtracting anchor cost at deployment. Preset:
  `learned_hybrid_recurrent_advantage_posguard_safe`. Active server session:
  `v1_claim_b1p20_n5_recurrent_advantage_posguard_20260602`, root
  `v1/artifacts/claim_suite_b1p20_n5_recurrent_advantage_posguard`.
- [ ] Audit and complete the recurrent-advantage run after fixing result
  accounting. Validation and final gate now share `DEPLOYABLE_POLICY_NAMES`,
  including `forecast_aware_recurrent_advantage`. Early seeds `41--43`
  already disabled recurrent-advantage under the positive paired static-margin
  guard, so this branch is trending negative as an algorithmic route.
- [x] Recurrent-advantage finished below the n=5 gate. Stop adding
  supervised policy heads and move to validation-transfer correction: explicit
  candidate-policy audit, start-conditioned risk modeling, or a deployable
  selector trained to predict final-like transfer from validation rollouts.
- [x] Closed validation-transfer correction: run the event-threshold-only
  counterfactual. Transfer audit showed event-threshold transfers better than
  value-residual on the combined B=1.20 evidence, so the active remote session
  `v1_claim_b1p20_event_threshold_only_seq_20260602` runs original seeds
  `41--45`, extension seeds `46--55`, then aggregates and audits the combined
  result. It failed the strong gate, so the next selector would need to model
  start-conditioned transfer risk rather than choosing by mean validation
  objective.
- [ ] Active dense-validation correction: run
  `v1_claim_b1p20_n5_event_threshold_valguard_dense12_20260602`, root
  `v1/artifacts/claim_suite_b1p20_n5_event_threshold_valguard_dense12`.
  This keeps the event-threshold valguard policy class but increases
  validation starts from `4` to `12` via `--static-selection-rollouts 12`,
  testing whether the transfer failure is driven by undersampled validation
  starts rather than by the event-threshold policy class itself.
- [x] Dense-validation n=5 completed and passed: deployable `4/5`, teacher
  `5/5`, mean margin `+0.007063`; start-level `15/20` wins. This is the
  strongest current positive result but remains only an n=5 gate.
- [ ] Active scaling: launch the same dense12 event-threshold valguard setting
  on extension seeds `46--55`, then aggregate with original seeds `41--45` to
  test whether the route reaches the stronger n=15 evidence bar.
  Launched session:
  `v1_claim_b1p20_ext_event_threshold_valguard_dense12_46_55_20260602`, root
  `v1/artifacts/claim_suite_b1p20_ext_event_threshold_valguard_dense12_46_55`,
  using GPU `0/1` and `max_parallel=2`. Current partial status after seeds
  `46--49`: deployable `2/4`, teacher `3/4`, combined deployable `6/9`.
  The n=15 deployable gate remains possible only if every remaining seed
  `50--55` passes; otherwise this scaling route fails and the next step is a
  transfer-risk/regime-compatibility selector.
- [x] Close dense12 valguard scaling. Early-stopped after seed50 made the
  strong deployable gate impossible. Combined early-stop result:
  deployable `7/11`, teacher `10/11`, mean deployable margin `+0.001250`;
  aggregate and audits saved under
  `v1/artifacts/claim_suite_b1p20_dense12_combined_early_stop/aggregate`.
- [x] Implement first transfer-risk selector branch:
  `static_margin_risk` and preset `learned_event_threshold_riskcalib_safe`.
  Local and remote core tests both pass (`42 passed` after the positive-center
  extension). This branch ranks
  guard-failing candidates by validation margin distribution risk rather than
  absolute objective and records event-threshold calibration rows in the
  manifest.
- [ ] Launch risk-calibrated dense12 n=5 once server load allows:
  root `v1/artifacts/claim_suite_b1p20_n5_event_threshold_riskcalib_dense12`,
  preset `learned_event_threshold_riskcalib_safe`, seeds `41--45`. If n=5 is
  positive, scale to extension seeds; if n=5 is negative, move to a stronger
  regime-compatibility selector rather than adding more policy heads. Active
  session:
  `v1_claim_b1p20_n5_event_threshold_riskcalib_dense12_20260602`. Startup
  check confirmed seed41/42 running with `static_margin_risk`. Final result:
  deployable `4/5`, teacher `5/5`, mean deployable margin `+0.004409`,
  median `+0.008416`, sign-test `p=0.375`; n=5 gate passes. Seed44 failed
  badly because the selected calibration row had negative center validation
  margins, so the stronger positive-center branch remains necessary before
  scaling.
- [x] Implement positive-center fallback for `static_margin_risk`: reject a
  selected deployable if it has mean margin below the configured minimum or
  negative median margin. This should preserve seed43-style guard-fail wins and
  reject seed44-style negative-center failures. Added
  `--deployable-selection-require-positive-center` and preset
  `learned_event_threshold_riskcenter_safe`; local and remote tests pass.
- [ ] Run positive-center risk selector n=5:
  `v1_claim_b1p20_n5_event_threshold_riskcenter_dense12_20260602`, root
  `v1/artifacts/claim_suite_b1p20_n5_event_threshold_riskcenter_dense12`,
  preset `learned_event_threshold_riskcenter_safe`, seeds `41--45`, GPUs
  `3/5`, `max_parallel=2`. Seed41/42 are valid and pass. The first seed43/44
  continuation exposed a dispatch bug where final deployable selection did not
  pass `require_positive_center`; invalid artifacts were quarantined under
  `_invalid_positive_center_bug_20260602`. Fixed run
  `v1_claim_b1p20_n5_event_threshold_riskcenter_fixed_43_45_20260602`
  completed and passed the n=5 gate: deployable `4/5`, teacher `5/5`,
  conservative zero-fallback deployable margin mean `+0.011105`.
- [ ] Next scaling action: run the same fixed positive-center risk selector on
  extension seeds `46--55`, aggregate with original seeds `41--45`, and test
  whether it reaches the stronger `12/15` deployable-win bar. If it fails,
  move from validation-center filtering to a start/regime-conditioned transfer
  selector rather than another supervised policy head. Early-stopped after
  extension seed50 made `12/15` mathematically impossible. Combined result at
  stop: deployable `7/11`, teacher `10/11`, conservative deployable margin
  mean `+0.006174`. Selected rows are reliable (`7/7` wins), but deployable
  coverage is insufficient because negative-center regimes fall back to static.
- [ ] Active correction after riskcenter scaling failure: design a
  regime/start-conditioned fallback for validation-negative regimes. The next
  experiment should not merely add another supervised policy head; it should
  either (a) estimate transfer risk at the start/regime level and choose among
  multiple low-risk event actions, or (b) learn a conservative fallback policy
  that can beat static when the event-threshold row has negative center
  validation support. First diagnostic launched:
  `learned_hybrid_rate_riskcenter_safe` on extension seeds `48--51`, root
  `v1/artifacts/claim_suite_b1p20_ext_rate_riskcenter_diagnostic_48_51`.
  Result: deployable `1/4`, teacher `4/4`; teacher-rate is rejected on the
  negative-center seeds and does not solve coverage. Next branch should test a
  state-conditioned duty/sequence fallback or a learned transfer selector, not
  average duty-rate matching. Implemented and launched the first
  state-conditioned duty fallback:
  `learned_hybrid_contextual_duty_riskcenter_safe`, session
  `v1_claim_b1p20_ext_contextual_riskcenter_diag_48_51_20260603`, root
  `v1/artifacts/claim_suite_b1p20_ext_contextual_duty_riskcenter_diagnostic_48_51`.
  This riskcenter variant calibrates contextual-duty by paired static-margin
  risk and requires positive-center validation support. Result: deployable
  `2/4`, teacher `4/4`, mean deployable margin `-0.000735`; contextual-duty
  itself transfers only `1/2` when selected. This branch is rejected as a
  reliable fallback.
- [ ] Next correction: implement an explicit transfer-risk/regime selector.
  It should not add another policy head. It should score candidate deployment
  rows using features that explain seed47/49-style guard-fail wins vs.
  seed48/50-style failures: validation margin mean/median/q25/min,
  negative-start count, selected static anchor, candidate policy type/action,
  event statistics, start-level margin dispersion, and simple rollout behavior
  summaries. The selector can first be trained/evaluated as a leave-one-seed-out
  diagnostic over existing completed runs before any new expensive policy run.
  First audit completed: only `9` de-duplicated selected rows and one final
  loss exist, so learned LOO selection is underdetermined. Proceed with an
  opt-in fixed risk-band selector (`positive_center`, q25 floor, and
  negative-start cap) as a prospective diagnostic on unused seeds `52--55`.
  Implemented `learned_hybrid_contextual_duty_riskband_safe` and launched
  remote session `v1_claim_b1p20_ext_contextual_riskband_52_55_20260603`,
  root `v1/artifacts/claim_suite_b1p20_ext_contextual_duty_riskband_52_55`.
  The run uses a predeclared q25 floor `-0.005` and negative-start cap `4`.
  Partial seeds `52--53` are already negative: deployable `0/2`, teacher
  `2/2`, deployable margin mean `-0.003640`. Both selected event-threshold
  rows had acceptable validation centers and q25 margins but transferred
  negatively. Completed seeds `52--55` close the branch: deployable `1/4`,
  teacher `4/4`; only seed55 wins via contextual-duty. Fixed validation-margin
  risk-band selection is rejected. Next work should diagnose regime/start
  transfer directly and test online regime-conditioned action choice, not
  another global validation-summary threshold.
- [x] Add calibration-transfer audit helper:
  `v1/scripts/audit_calibration_transfer.py`. It reads
  `event_threshold_policy.calibration_row` from manifests and compares the
  validation margin distribution to final-test margin.
- [x] Add start-level transfer audit:
  `v1/scripts/audit_start_transfer.py`. Initial diagnostic on the completed
  valguard run shows `11/20` final-start wins, with seed44 `0/4` and seed45
  `4/4`; event coverage alone does not explain transfer.
- [ ] Contextual-duty first result failed at deployable `3/5`, teacher `5/5`.
  The policy class did not become the selected deployable because its
  non-paired mean-objective hyperparameter calibration did not satisfy the
  paired static-margin guard. Active correction: rerun with guard-aware
  contextual-duty calibration under
  `learned_hybrid_contextual_duty_guardcalib_safe`.
- **Status:** in_progress

## Phase 7: Transfer-Structure Redesign
- [x] Read `v1/docs/06-03-01.md` and `v1/docs/06-03-02.md`.
- [x] Ignore Paper 1 / major-revision content in those documents; Paper work is
  handled on the user's fork branch.
- [x] Close the following as low-value mainline work: fixed global risk-band
  selectors, teacher-rate fallback, contextual-duty fallback, recurrent/mask
  supervised heads, and immediate PPO retraining.
- [x] Add `v1/scripts/audit_transfer_structure.py` to diagnose whether failures
  are protocol/regime shift or dynamic-policy instability using existing
  manifest/rollout artifacts.
- [x] Run transfer-structure audit on riskcenter/riskcalib/riskband families.
  Key result: unique static validation-vs-final Spearman is only `0.204`, and
  unique deployable validation-margin-vs-final-margin Spearman is only `0.280`.
  Validation aggregation is therefore not a reliable deployment selector.
- [x] Seed44 diagnosis: final event rate is normal (`0.7666`, comparable to
  seed41 `0.7813`), and event-threshold does not fail because events are absent
  or SOC is worse. The failure is action/regime/temporal-mixing compatibility.
- [x] Next zero-training gate: compute a conditional static/dynamic upper bound
  from existing final rollouts. If even per-start oracle mixing cannot reach
  the scaled evidence target, stop algorithmic tuning and downgrade the claim.
  Result on available dynamic final rollouts: direct best dynamic `9/13`,
  per-start oracle static fallback `13/13`, mean fallback margin `+0.008638`.
  Conditional deployment is worth implementing.
- [ ] If the upper bound is positive, implement a deployable online switch:
  static anchor as conservative policy, event/contextual dynamic policy as
  candidate, with causal regime features plus cost/SOC and objective-risk
  filters. A cost-only CAPS-style filter is insufficient because seed44 has
  acceptable power/SOC but still loses.
- [ ] Re-run only the smallest decisive gate first: original seeds plus the
  known transfer-failure extension seeds. Scale only if the online switch
  improves the failure seeds without damaging the clean positive seeds.
- **Status:** in_progress

## Phase 8: Scenario Calibration
- [x] Promote "successful scene calibration" to the active target. No new
  claim-suite or deployable-policy rerun should be launched before the scenario
  passes calibration gates.
- [x] Define structural success gates:
  `core+laser+fc4` and `laser+fc4` must be infeasible under instantaneous and
  startup constraints; `core+laser` and `core+SPC+fc4` must remain feasible.
- [x] Define energy success gate:
  constant `core+laser` must fail to last one evaluation window, the proxy
  stack must last, and the max laser duty over proxy must sit in a useful
  interior range rather than near 0 or 1.
- [x] Add candidate sensor config
  `v1/configs/sensors/windblown_sensors_physical_event_v5_constraint_active.yaml`.
  It separates mass flux from laser, raises fc4 from nearly-free to non-free,
  and makes `snow_particle_counter` a lower-cost saturated proxy.
- [x] Add structural audit script `v1/scripts/audit_scenario_calibration.py`.
- [x] Run structural audit. Result:
  `v5_constraint_active_b1p20_e70` passes both structural and energy gates;
  current v4 fails both; v5 e90 passes structure but fails energy activation.
- [x] Expose scenario parameters in future launch scripts so calibrated values
  (`sensor_cfg`, `selection`, `max_active`, `initial_energy`, `reserve_energy`)
  cannot be overwritten by old hard-coded defaults.
- [x] Add static/teacher-only calibration runner
  `v1/scripts/run_static_teacher_calibration_gate.py`. This deliberately
  avoids BC/DAgger/deployable training and evaluates only
  validation-selected static vs. privileged MPC teacher.
- [x] Run seed41 mini smoke checks with the calibrated v5/e70 scenario.
  Uniform 128-step smoke: teacher beats static by `+0.037300`; event-rich
  128-step smoke: teacher beats static by `+0.105578`. In both, static no
  longer selects the old continuous direct stack and instead selects
  `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`.
- [x] Next calibration gate after structural and mini-smoke success:
  static/teacher-only seed41/42/44 check with longer windows and multiple
  rollouts, not deployable training. Success requires static no longer
  selecting a continuous full-direct stack, MPC teacher positive margin, and
  teacher using nontrivial laser/proxy temporal switching in event-rich
  windows.
- [x] Multi-seed event-rich static/teacher-only calibration passed for
  v5/e70. Results saved under
  `v1/artifacts/static_teacher_calibration_v5_multiseed_20260603`: teacher
  wins `3/3`, mean margin `+0.031648`, minimum margin `+0.028920`; executed
  static direct `laser+fc4` duty is `0` in all three seeds; teacher uses
  `16--17` unique masks and selective laser in `2/3` seeds.
- [x] Scenario v5/e70 is accepted as the next algorithm-development scene.
  This does not prove the final deployable claim; it only clears the
  precondition for resuming small deployable-policy gates.
- **Status:** complete

## Phase 9: Calibrated-Scene Deployable Gate
- [x] Choose the smallest deployable-policy gate under v5/e70. It should reuse
  the existing best deployable route first, with calibrated scenario flags:
  `sensor_cfg=v5_constraint_active`, `selection=event_rich`,
  `initial_energy=70`, `reserve_energy=20`, `max_active=4`, and no immediate
  n=15 scaling.
- [x] Reject simple event-threshold riskcenter on seed41. It preserves teacher
  value but deploys almost exclusively `met_station_core|surface_temp_ir`, does
  not use laser/fc4, and loses slightly to static.
- [x] Test contextual-duty/riskcenter on seed41. At task-error weight `0.2` it
  loses the objective but recovers a useful teacher-like fc4/laser/context
  mixture and improves physical forecast metrics.
- [x] Add `--task-error-weight` to `v1/scripts/run_claim_suite.py` and test
  the calibrated scene at `w=0.30`.
- [x] Close the global `w=0.30` small gate. Seed41 passes, but the combined
  seed41/42/44 result fails: deployable `1/3`, teacher `3/3`, mean deployable
  margin `-0.005130`. The break-even task weights conflict by static-anchor
  regime: seed41 needs `w>0.185`, while laser-anchor seeds42/44 need
  `w<~0.142`.
- [ ] Active correction: implement an anchor/mechanism-conditioned deployable
  decision rather than another global scalar weight. Proxy/fc4 static anchors
  need selective laser/task-error emphasis; laser static anchors need a
  different teacher-compression mechanism that preserves snow-task accuracy
  while lowering power/oracle loss. If this cannot clear seed41/42/44, write a
  deeper redesign report instead of launching more broad reruns.
- [x] Test the existing teacher-mix temporal-compression shortcut under v5/e70
  before deeper scene changes. Result: deployable `1/3`, teacher `3/3`, mean
  margin `-0.013113`; rejected.
- [ ] Pivot: design a complex scene where no single static direct stack is
  sufficient. Required properties: `core+laser` must not dominate the task
  alone; `core+laser+fc4` remains infeasible; proxy/context stacks help in
  distinct regimes; and teacher/static calibration must show teacher wins by a
  larger non-marginal margin before deployable training resumes.
- **Status:** blocked for v5/e70 deployable scaling; new scene calibration
  required

## Phase 10: Complex Scene Recalibration
- [x] Audit why selected static remains strong under v5/e70 using sensor power,
  task target observability, and static anchor composition.
- [x] Propose v6 scene changes with explicit success gates: reduce continuous
  `core+laser` dominance, increase complementarity between laser/fc4/proxy
  sensors, and preserve feasibility of multiple dynamic alternatives.
- [x] Run structural-only and first event-rich static/teacher-only calibration
  before any new deployable-policy rerun.
- [x] Add and run an `event_transport_rich` static/teacher-only calibration
  because the event-rate-only v6 gate still leaves seed44 with a small
  teacher margin and worse task-error than static.
- [x] Accept a scene only if validation-selected static is not a near-complete
  task sensor and teacher wins with a non-marginal margin across seed41/42/44.
- [x] Run the first deployable small gate on accepted v6/event-transport using
  a teacher-mixture deployable, starting with contextual-duty before any
  n=5/n=15 scaling.
- [x] Contextual-duty deployable smoke failed (`1/3`, mean margin `-0.005401`);
  validation selected event-threshold in all seeds, so this did not test a
  successful teacher-mixture compression.
- [x] Run teacher-rate/cycle small gate on the same accepted scene before
  deciding whether the next correction is sequence-style mask imitation.
  Teacher-mix also failed (`1/3`, mean margin `-0.005401`) and validation
  again selected `forecast_aware_event_threshold` in all seeds. Teacher-rate
  and teacher-cycle were not viable validation-selected compressions.
- [x] Next deployable correction: test policies that preserve temporal state
  or objective values directly under v6/event-transport, starting with
  `learned_hybrid_sequence_mask_guarded_safe` and
  `learned_hybrid_recurrent_value_guarded_safe` on seeds `41/42/44`.
- [x] Sequence-mask small gate completed and failed (`1/3`, mean margin
  `-0.005401`). It fit teacher masks almost perfectly, but validation still
  selected `forecast_aware_event_threshold` in all seeds; teacher-label
  imitation alone is not the bottleneck.
- [x] Run recurrent objective-aware value student on the same accepted scene
  before concluding that the current student family cannot compress the
  teacher.
- [x] Recurrent objective-aware value student completed and failed (`1/3`,
  mean margin `-0.002967`). Seed44 selected recurrent value, but final was
  exactly equal to the static anchor, so this is a no-op fallback rather than
  a dynamic win. Best-action accuracy was low (`0.074/0.369/0.205`).
- [x] Targeted correction before declaring this student tier blocked: run a
  rank-aware recurrent value variant with positive-margin guard and denser
  train starts, so the recurrent cost model is not trained on only 512 rows
  and zero-margin no-op recurrent selections are rejected.
  Active tmux session:
  `v1_claim_v6_transport_recurrent_rank_posguard_dense_20260603`, output root
  `v1/artifacts/claim_suite_v6_transport_recurrent_rank_posguard_dense_smoke_20260603`.
- [x] Rank-aware recurrent dense/positive-guard smoke completed and failed:
  deployable `1/3`, teacher `3/3`, mean deployable margin `-0.005538`.
  Recurrent rows increased to `1536` and accuracy improved to `0.37--0.48`,
  but recurrent-value failed its own positive static-margin guard in all seeds
  and was disabled.
- [x] Active correction: change the student interface rather than retuning the
  same recurrent-value head. Implement a recurrent cost-DAgger path that
  collects candidate rollout costs on deployable-policy visited states, then
  rerun only a small seed41/42/44 gate if local/remote tests pass.
- [x] Recurrent cost-DAgger completed and failed: deployable `1/3`, teacher
  `3/3`, mean deployable margin `-0.005538`. It doubled recurrent rows to
  `3072` and improved best-action accuracy to `0.47--0.59`, but
  recurrent-value still failed the positive static-margin guard in all seeds
  and was disabled. This closes the single recurrent scorer interface.
- **Status:** complete for scene calibration and recurrent-student tier;
  blocked for deployable claim without a new student interface

## Phase 11: Online Option/Planner Student Interface
- [x] Design a deployable causal online interface that does not compress the
  teacher into one mask classifier/scorer. Static anchor remains the safe
  default; dynamic behavior is expressed as a small set of causal options or
  short online planning choices over teacher-supported masks.
- [x] Audit existing rollouts to extract option candidates and features:
  static anchor, teacher-supported masks, event/transport forecast features,
  freshness/SOC/power state, and start-level objective transfer indicators.
- [x] Implement the first option-planner policy with validation calibration:
  enter/exit/dwell guards, static fallback, and objective-risk filters. The
  policy must use only causal features and no final-test outcomes.
- [x] Validate locally with unit tests and a seed41 CPU/small smoke before any
  remote gate.
- [x] Run only the accepted v6/event-transport seed41/42/44 gate first.
  Result: formal `FAIL`, deployable `1/3`, teacher `3/3`, mean deployable
  margin `-0.000289`. Seed44 is a real option-planner win, seed42 passes
  validation guard but loses final, and seed41 has no deployable passing the
  positive static-margin guard.
- [ ] Diagnose option validation-to-final transfer before any scaling:
  compare option activation timelines, duty rates, event-risk distribution,
  start-level margins, and risk-band guard behavior for seed42 versus seed44.
- [ ] Implement the next option-risk correction only after that diagnostic;
  likely candidates are stricter q25/risk-band validation, final-like
  transport-diverse validation starts, or a conservative option controller
  with per-option regret guards.
- [x] Implement first option-risk correction: option selection now includes a
  validation-calibrated rate-balance penalty against teacher-duty targets.
- [x] Run the seed41/42/44 balanced option-planner smoke before any scaling.
  Result: formal `FAIL`, deployable `0/3`, teacher `3/3`, mean deployable
  margin `-0.003681`. Seed42 selected `rate_balance=3.0` but final transfer
  worsened; seed44 failed the strict guard after rate balancing.
- [ ] Next correction: stop average-duty heuristics and implement a
  start/window-level transfer-risk or causal option-value selector. It should
  decide whether an option policy is safe for the current validation/final-like
  regime, not merely match teacher duty rates.
- [x] Fix and rerun start/window transfer audit so multi-root rows retain
  their source root. Result: balanced option-planner has only `1/4` seed42
  final-start wins and worsens the old option-planner selected-row distribution.
- [x] Add a pure rollout-value positive-guard preset so the next diagnostic is
  not confounded by older value-residual or event-threshold deployable heads.
- [x] Run the accepted v6/event-transport seed41/42/44 pure rollout-value
  gate. Success requires rollout-value itself to pass validation guard and
  improve over the static anchor on final; otherwise this closes the current
  short-horizon learned planner branch. Result: formal `FAIL`, deployable
  `0/3`, teacher `3/3`; rollout-value failed validation margins in all seeds
  and no deployable was selected.
- [ ] Implement the next correction on the option-planner branch: a
  start/window-level transfer-risk selector that keeps the static anchor unless
  validation evidence indicates the option policy is safe for the current
  regime. Do not retune average duty/rate or the pure rollout-value planner.
- [x] Add and launch the first conservative start-guard option-planner preset:
  pure option-planner, no old deployable heads, no rate-balance correction,
  and zero allowed negative validation starts.
- [x] Aggregate and diagnose the start-guard option-planner smoke before
  deciding whether a true runtime per-window selector is worth implementing.
- [x] Implement and test a runtime/window-level risk guard. Paired replay
  failed at deployable `0/3`; dense-validation/risk-band also failed by
  rejecting all runtime-risk deployables before final deployment.
- [x] Implement and test the first stronger teacher-cost interface. The
  cost-KNN risk-band gate failed at deployable `0/3` while teacher stayed
  `3/3`; every non-static cost-memory candidate had negative validation
  margins. Close one-step teacher cost-vector retrieval as a main route.
- [x] Implement teacher trajectory / macro-option sequence interface. The new
  `ForecastAwareMacroOptionPolicy` preserves short train-split teacher label
  snippets, selects snippets by causal feature nearest-neighbor matching, and
  uses learned event-risk thresholds plus static fallback.
- [x] Run accepted v6/event-transport seed41/42/44 macro-option risk-band gate.
  Result: deployable `0/3`, teacher `3/3`, mean deployable margin `0.000000`,
  mean teacher margin `+0.110656`. All final runs fell back to static; dynamic
  macro-option rows were negative on validation.
- [x] Run broader objective/forecast-transfer diagnostic. The audit confirms
  teacher dynamic value in `3/3` seeds and `12/12` final windows; about `60.4%`
  of objective lift comes from weighted task-error reduction and `39.6%` from
  frozen-oracle loss.
- [x] Implement auditable deployable context persistence. Future learned-event
  runs now save augmented truth probability columns and `TeacherDataset`
  feature names, enabling direct forecast/regret-window audits.
- [x] Run teacher-improvement alignment audit on the server. Result:
  weak-positive learned-event alignment (`AUC=0.606/0.586/0.517` for
  seeds `41/42/44`), enough for a guarded Branch F smoke but not enough for
  scaling.
- [x] Implement Branch F guarded smoke: replace generic event-risk entry with
  a learned teacher-improvement probability gate around the dynamic
  macro-option policy. Added server-validated preset
  `learned_teacher_improvement_gate_smoke`.
- [x] Evaluate Branch F guarded smoke on accepted v6/event-transport
  seeds `41/42/44` on `remote-gpu`, tmux
  `v1_teacher_gate_v6_20260604`, output root
  `v1/artifacts/claim_suite_v6_transport_teacher_improvement_gate_smoke_20260604/`.
- [x] Audit the window/sequence-level teacher value before implementing the
  next student. Result: `v1/artifacts/window_teacher_value_audit_v6_20260605/`
  shows MPC teacher beats validation-selected static in `60/60` declared
  train/validation/final windows for seeds `41/42/44`; validation mean
  margins are `+0.079045`, `+0.069905`, and `+0.096935`.
- [ ] Active implementation: stop scalar first-action teacher-improvement
  gates and implement a window/sequence-level teacher-value student or a
  deployable learned-world-model MPC.
- [x] Evaluate dense always-dynamic macro-option sequence retrieval as the
  first low-cost post-audit implementation branch. Server tmux:
  `v1_dense_macro_20260605`; output root:
  `v1/artifacts/claim_suite_v6_transport_macro_option_dense_always_20260605/`.
  Result: deployable `0/3`, teacher `3/3`, mean deployable margin `0.000000`.
  No dynamic row had positive validation mean margin; close similarity-only
  teacher-snippet retrieval.
- [x] Active next: run the accepted v6/event-transport seed41/42/44 gate for
  `learned_sequence_value_riskband_safe` on `remote-gpu`. Success requires the
  sequence-value deployable to pass validation-risk selection and beat the
  static anchor on final, not merely to train a low-loss value model. Active
  tmux: `v1_sequence_value_20260605`; output root:
  `v1/artifacts/claim_suite_v6_transport_sequence_value_riskband_20260605/`.
  Result: deployable `0/3`, teacher `3/3`. The train sequence-value dataset
  has positive labels, but validation calibration is negative/unstable and no
  deployable is selected.
- [x] Active targeted diagnostic: add and run a full-bank/high-threshold
  sequence-value variant. It should score the full `369--380` sequence bank
  and extend the advantage threshold grid above `0.1`. If this also fails,
  close the sequence-value route and move to a larger deployable learned
  dynamics/planning redesign. Preset implemented and remote dry-run validated:
  `learned_sequence_value_fullbank_riskband_safe`. Result: deployable `0/3`,
  teacher `3/3`; close the current sequence-value route.
- [x] Audit and redesign the deployable causal forecast context.
  The immediate diagnostic is to test whether oracle/privileged context or
  richer learned multi-target forecast context can separate teacher-positive
  windows from static-safe windows. Do not add more sequence threshold/snippet
  variants unless this diagnostic identifies a concrete missing feature.
  Implemented diagnostic preset:
  `learned_sequence_value_oracle_context_fullbank_safe`, which uses truth-future
  event context with the fullbank sequence-value policy and no learned-event
  forecaster. Result: deployable `1/3`, teacher `3/3`, mean deployable margin
  `-0.004845`. Perfect future event flags are not sufficient; event-context-only
  is closed as the missing-signal hypothesis.
- [ ] Active implementation: add richer continuous regime / task-forecast
  context and a per-window dynamic-eligibility layer, or replace sequence
  compression with a deployable learned-world-model planner. The next branch
  must be designed around seed44-style validation-fragile regimes rather than
  another global sequence threshold or snippet-bank variant. First diagnostic
  completed with privileged continuous future summaries:
  `learned_sequence_value_oracle_regime_fullbank_safe` reached deployable
  `1/3`, teacher `3/3`, mean deployable margin `+0.001014`. This improves
  safety but not dynamic coverage; close sequence-value retrieval/compression
  as a main route.
- [x] Implement the first learned-world-model planner diagnostic:
  `learned_rollout_value_oracle_regime_posguard_safe`, using raw action-cost
  learning, feature-transition learning, depth-2 beam planning, privileged
  event/continuous regime context, and strict validation risk-band selection.
  Result: deployable `0/3`, teacher `3/3`, mean deployable margin `0.000000`.
  Every rollout-value threshold had negative mean validation margin in all
  seeds, so the candidate was correctly disabled before final deployment.
- [x] Implement and run the direct static-anchor advantage diagnostic:
  `learned_advantage_oracle_regime_posguard_safe`. Result: deployable `0/3`,
  teacher `3/3`; all best validation rows were negative
  (seed41 `-0.006727`, seed42 `-0.015483`, seed44 `-0.013301`) and the
  strict guard disabled the candidate in every seed.
- [x] Implement and formally evaluate task-only proxy-MPC as the first
  constraint-aware multi-step planner. Result: deployable `0/3`, teacher
  `3/3`; seeds42/44 have positive mean validation margin but all seeds have
  negative q25 tails, so the strict guard disables deployment.
- [ ] Active next: implement a learned full-window static-anchor margin and
  downside-risk model from paired candidate self-rollouts. Use it inside
  constrained receding-horizon selection; do not continue manual proxy-weight
  tuning.
- **Status:** in_progress; current option-planner and rate-balance branches
  closed as non-scaling; pure rollout-value, oracle-regime absolute-cost
  rollout-value, oracle-regime one-step anchor-advantage, pure startguard, runtime-risk,
  one-step cost-KNN, macro-option snippet, Branch F teacher-improvement gate,
  and dense always-dynamic macro retrieval branches closed as non-scaling;
  teacher value now confirmed clean on train/validation/final windows

- [x] Run remote dry-run to verify the calibrated scenario flags propagate into
  `run_protocol_gate.py`.
- [x] Launch at most the original seed41/42/44 gate first. Success requires
  teacher remains positive and deployable does not simply collapse to the
  static anchor under the easier calibrated scene.
- [x] First seed41 event-threshold riskcenter smoke completed and failed the
  deployable gate. Teacher beats static, but `forecast_aware_event_threshold`
  is slightly worse than static (`1.307823` vs `1.306918`). Do not scale this
  route.
- [x] Active correction: test a teacher-mixture deployable under v5/e70
  because the teacher's seed41 advantage comes from fc4/laser/context temporal
  mixing, while event-threshold collapsed to a low-power `core+surface` action
  and did not reduce task error.
- [x] Contextual-duty recovers the teacher-like mixture and improves physical
  metrics, but fails the current `task_composite` objective because
  `task_error_weight=0.2` underweights the task-error improvement relative to
  frozen-oracle loss. This is now an objective-calibration issue.
- [x] Add a clean task-error-weight override to `run_claim_suite.py` and rerun
  the seed41 contextual-duty gate at `w=0.30` before any scaling. Seed41
  passes under the task-targeted objective: static `1.373655`, teacher
  `1.314480`, deployable `1.344876`.
- [ ] Run the remaining small calibration-scene gate on seeds42/44 at
  `w=0.30`. If either fails, inspect before any scaling; if both pass, then
  consider a controlled n=5 gate.
- [x] Aggregate and append CHANGELOG before any scaling decision.
- **Status:** in_progress

## Phase 12: Mean-Risk Window Controller
- [x] Review and correct the Branch H draft protocol.
- [x] Lock the corrected execution plan in
  `v1/docs/06-06-02-branch-h-revised-execution-plan.md`.
- [x] H0: implement train-only non-overlap fit/calibration starts, train
  anchor bank, margin-sign tests, and feature-availability audit.
- [x] H1: implement resumable paired 256-step window outcome collection.
  The standalone source-run adapter reuses frozen forecasts/oracle, recomputes
  the train-only static bank, and collects risk-fit/risk-calibration outcomes.
- [x] H2: implement grouped mean/q25/negative-risk models and one-sided
  calibration against constant baselines.
- [x] H3: implement the window-level mean-risk controller with static fallback.
- [x] H4: pass local engineering tests, then run a remote seed41 data/model
  pilot before any formal 3-seed experiment.
  Formal seed41 `32/12` paired collection completed. The data gate passed,
  but both the original and causal-history H2 models failed chronological
  calibration transfer, so H3 remained locked.
- [x] H5: replace one-shot 256-step outer selection with a receding 64-step
  macro-risk controller. The 64-step paired data remain dynamically valuable,
  but fit-only and chronological H2 prediction both fail; the shorter label
  horizon does not solve regime transfer, so receding H3 remains locked.
- [x] H6: close or validate the proxy-controller action interface before more
  risk-model work. First finish the fit-selected anchor-score threshold
  diagnostic. If it fails, test only a small static-anchor Hamming-neighborhood
  action family; do not launch another full risk-model experiment until that
  interface shows non-negative calibration q25.
  Both diagnostics failed chronological selection. Hamming-1 reduces downside,
  but no fixed base controller is safe on fit and transferable to calibration.
- [ ] Run seeds41/42/44 only if the data/model gate passes. Require at least
  `2/3` final wins before any n=5 expansion.
- **Status:** complete; fixed proxy-controller selection is closed

## Phase 13: Static-Anchor Residual Action Controller
- [x] I0: specify a deployable residual action space around the static anchor:
  no-op plus one-sensor add/drop, mandatory core retained, projector-feasible,
  with dwell/warmup state preserved.
- [x] I1: collect common-random-number paired 64-step outcomes for each
  residual action at blocked risk-fit/calibration starts. Labels are direct
  static-anchor objective improvements, not proxy scores or controller IDs.
  Formal seed41 `32/12` collection completed with 192/74 dynamic rows.
- [x] I2: train action-conditioned mean/lower-tail/negative-risk models using
  only causal forecast/history/state features and grouped chronological
  calibration.
  Corrected-CRN XGBoost passes the full train-only data/model gate:
  fit-CV q25 improvement `+16.82%`, chronological q25 improvement `+20.99%`,
  positive Brier improvement, and monotonic risk bins.
  H3 then exposed that residual collection still inherited teacher top-k
  action support. Recollect with all projector-feasible, core-preserving
  Hamming-1 actions. Corrected full-support XGBoost also passes, with stronger
  fit-CV rank and chronological Brier improvement.
- [x] I3: implement a receding 64-step controller that applies a residual only
  when its calibrated lower bound is non-negative; otherwise keep the static
  anchor.
- [ ] I4: run seed41 train-only data/model gate. Use validation/final only if
  chronological risk calibration passes, then require `2/3` seed wins before
  expansion.
  The first validation anchor sweep failed `0/8`; final remains locked.
  Full-support train-only gate passes, but its first validation replay used
  zero dynamic blocks in 12 starts. Train-only threshold selection has been
  corrected to prioritize independent-start dynamic coverage among risk-safe
  candidates. The resulting replay activated 3/12 starts with positive mean
  and zero q25, but had 2 negative starts and failed the risk gate.
- [x] I5: audit activated validation trajectories without changing thresholds.
  Separate action-ranking error from 64-step persistence/state error, then
  specify the next architecture before any new formal rollout.
  Result: all dynamic windows selected action 42; negative windows used only
  one block. The deployed anchor transition was outside the top-8 train anchor
  bank, making anchor-conditioned residual semantics OOD.
- [x] I6: audit label/deployment state alignment before scaling.
  Found a critical mismatch: labels compare cold-start constant masks, while
  deployment executes a residual after an anchor-conditioned prefix. Stopped
  the expanded run before fit-row generation.
- [x] I7: implement prefix-conditioned counterfactual collection. Run the
  anchor for one 64-step block, snapshot complete environment/RNG state, build
  causal features at the boundary, and branch anchor continuation versus all
  valid one-hop residuals from the identical snapshot.
  Server smoke passes all phase/previous-mask/non-overlap audits; local and
  remote tests report `100 passed`.
- [ ] I8: after regression tests, recollect with broad anchor coverage and
  rerun train-only model gates. Validation and final remain locked.
  Collection and grouped/chronological modeling completed. Exact-boundary
  filtering retains 6,848/1,712 rows and strong q25/Brier improvements. The
  corrected tail-bin gate passes, but all `392` global train-only deployment
  threshold combinations fail. No validation/final replay is permitted yet.
- [ ] I9: audit per-anchor train-only calibratability using one fixed procedure
  and the existing threshold grid. Quantify passing-anchor coverage, sample
  sufficiency, threshold stability, and failure modes. If support is broad,
  implement precomputed anchor-conditioned calibration with an explicit
  minimum-sample/fallback rule; otherwise improve the model or anchor-class
  representation before any downstream replay.
  Completed: 10/42 anchors pass leave-one-start-out; action97 fails. Do not
  implement unrestricted per-anchor deployment.
- [ ] I10: form a train-prequalified risk-supported anchor bank from the 10
  leave-one-start-out passing anchors. First quantify each anchor's static
  objective gap to unrestricted static action97 and compare that gap with its
  held-out dynamic margin. If plausible, persist full-calibration thresholds
  for every prequalified anchor and define one locked validation-selection
  rule. The unrestricted static baseline must not be weakened.
  Correcting for the 1:1 anchor/residual duty cycle leaves only action116 with
  positive train-only net feasibility (`+0.009535`).
- [ ] I11: add HistGBDT to the formal chronological trainer using the
  hyperparameters already selected by fit-only grouped CV. Repeat the model
  gate, global calibration, leave-one-start-out anchor audit, and duty-cycle
  feasibility calculation. Lock the model/anchor only after this comparison.
  Completed: HistGBDT passes, global calibration has 50 valid combinations,
  and action116 is the sole duty-cycle-feasible LOSO anchor.
- [ ] I12: persist the action116-specific full-calibration threshold profile.
  Audit the validation evaluator so the locked controller is compared against
  unrestricted validation-selected static action97. Permit exactly one
  validation replay only if no threshold/model/anchor choice remains open.
  Completed and failed: mean/q25 `-0.004009/-0.045261`, 7/12 negative
  starts; final test was not run.
- [ ] I13: decompose the fixed action116 validation trajectories into static
  anchor gap, residual-pulse gain, selected actions, and predicted risk.
  Decide whether the failure is anchor transfer or action-ranking transfer
  before changing architecture. No validation threshold search is allowed.
  Completed: dynamic improves action116 by `+0.006028` mean, but action116
  trails action97 by `+0.010036`; anchor quality is the primary failure.
- [ ] I14: densify action97 calibration on all 32 existing chronological
  calibration starts using prefix-conditioned common-random-number branching.
  Keep the HistGBDT model fixed initially, calibrate action97 on the denser
  rows, and require a start-level tail gate before any further validation.
  Completed with `384/96` rows and positive opportunity on `64.8%/62.5%` of
  starts. Because fit support also increased fourfold, rerun fit-only model
  comparison rather than applying the old broad-anchor model unchanged.
- [ ] I15: compare GBDT/HistGBDT/XGBoost by grouped fit-only CV on dense
  action97, train the selected family chronologically, and calibrate over all
  32 later starts. Validation remains locked until every gate passes.
  Fit-only selection completed: HistGBDT wins with Spearman `0.4159` and q25
  improvement `+12.99%`.
  Chronological gate failed: q25 `+5.41%`, Brier `-18.995%`.
- [ ] I16: audit dense action97 chronological drift by residual action,
  negative prevalence, forecast/event regime, SOC/runtime state, and time.
  Use fit-only blocked backtests to decide between rolling calibration,
  regime-conditioned risk, or a different deployment abstraction.
  Audit completed: event/particle/weather covariates shift and feature-margin
  associations reverse; the last fit quartile has strongest opportunity.
- [ ] I17: run at least two fit-only chronological backtests comparing
  expanding-history HistGBDT with recent-window HistGBDT. Predeclare tail
  improvement and Brier/rank stability as the selection criteria. Apply a
  rolling protocol to actual calibration only if it wins consistently.
  Completed and failed: Q3 is mixed and Q4 worsens q25/Brier.
- [ ] I18: diagnose decision-level calibration on dense action97. Evaluate the
  top-ranked residual plus static fallback over all 32 calibration starts and
  compare its safety with the failed row-level Brier/q25 gates. This is
  diagnostic only; no validation replay is authorized by this result.
  Completed: safe but only 3/32 dynamic starts, all action42.
- [ ] I19: build an action42-specific binary intervention model. Compare model
  families using grouped fit-only CV, then run one chronological calibration
  gate. Runtime support must be restricted to action42 if the gate passes.
  Fit-only gate failed across all tree families.
- [ ] I20: test one domain-predeclared compact feature profile on action42
  using fit-only grouped CV and strongly regularized models. If it cannot
  recover stable q25/risk signal, close residual-risk regression and specify a
  stronger model-based teacher/world-model architecture.
  Completed and failed; direct residual-risk regression is closed.
- [ ] I21: verify dynamic headroom from existing accepted-scenario
  teacher/oracle evidence. If headroom remains, specify and begin a
  probabilistic world-model + robust receding-horizon planner that evaluates
  counterfactual sensor schedules through estimator and frozen-oracle dynamics.
  Headroom verified on seed41: teacher margin `+0.103367`.
- [ ] I22: audit MPC teacher runtime causality: state features, event flags,
  future forecasts, reward-oracle calls, and action feasibility. If causal,
  promote it into the deployable planner architecture; otherwise replace the
  leaking components with learned probabilistic world-model inputs.
  Completed: teacher is clairvoyant through real-environment branch stepping.
- [ ] I23: design and implement the causal planning core:
  probabilistic multi-horizon world model, scenario truth adapter, robust
  expected/CVaR beam objective, and receding-horizon policy. Reuse physical
  estimator/runtime/energy/projector code; prohibit access to real future truth
  in planner branches by construction and test invariance to hidden-future
  mutations.
  Scenario adapter, expected/CVaR beam search, receding policy, and hidden
  future invariance are implemented and pass the `108`-test core suite.
  The trainable five-member ensemble now passes its train-only chronological
  audit with `+28.40%` normalized-RMSE skill over persistence and `83.22%`
  coverage for its nominal 80% residual interval. The first validation-gated
  planner replay failed (`mean -0.03096`, q25 `-0.07755`, `2/4` negative);
  final stayed locked. Active I23 correction is a train-only rollout-history
  audit, because the current model gate used full truth histories while the
  deployed model receives stale/partial scheduler histories.
  Rollout-history shift audit completed: scheduler-history RMSE is `0.73949`
  versus full-truth-history `0.65976` on the same train-only audit segment
  (`+12.1%`), with largest degradation on snow particle diameter/velocity.
  Active correction is now a mask-aware rollout-trained world model, not CVaR
  or threshold tuning. The first fixed rollout-world-model smoke passed at
  horizon `6`: normalized RMSE `0.62344`, persistence `0.89615`, skill
  `+30.43%`, and interval coverage `84.22%`.
  Horizon-12 rollout-world models now pass for seeds `41/42/43/44/45`.
  Conservative robust planning passes final for seeds `41/42`, but scalar
  anchor-margin calibration fails seeds `43/44/45`. Event-heavy
  support-top-k restriction recovers validation pass for seeds `43/45` but
  final transfer fails: seed43 final mean `-0.000592`; seed45 final mean
  `+0.001137` but q25 `-0.002297`, below the strict gate. Seed44 remains
  validation unsafe. Support/margin tuning is closed as a main path until a
  validation-sampling/transfer diagnostic explains whether four validation
  windows are under-sampling tail risk. The 12-start diagnostic completed:
  seed41/43 pass, seed42/44/45 fail. This catches more tail risk but does not
  predict final transfer cleanly, because seed42 is a final-pass false
  rejection and seed43 is a final-fail false acceptance. Run-level validation
  thresholds are now exhausted for this planner family. Final trace diagnosis
  shows the planner overestimates the effect of brief dynamic deviations:
  seed43 increases flux task error, and seed45 worsens oracle loss while task
  errors stay unchanged. Active correction is now an action-effect /
  break-even verifier, not more support/margin tuning.
- **Status:** in_progress; causal robust planner implementation active

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
| 2026-06-01 | First rollout-value planner used per-state normalized action-cost targets, which are valid for one-step residual ranking but not additive across simulated planner steps | Kept normalized costs for value-residual, added separate raw-cost dataset/model for the rollout planner, verified locally and remotely, and launched raw-cost n=5 rerun |
| 2026-06-02 | Sequence-mask fitted teacher masks well but early completed seeds were still selected as old event/value policies | Added recurrent objective-aware action-cost student that combines GRU memory with candidate rollout-cost scoring, then launched a guarded n=5 suite |
| 2026-06-02 | `audit_start_transfer.py` used `DataFrame.to_markdown`, which failed on the server because `tabulate` was not installed | Replaced it with an internal markdown table renderer and reran the audit successfully |
| 2026-06-02 | First risk-calibrated n=5 launch used nonexistent input root `v1/artifacts/claim_inputs_semimarkov_b1p20` | Relaunched with the correct original-seed root `rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov`; no seed result was produced by the failed attempt |
| 2026-06-02 | `static_margin_risk` deployable selection did not populate static-start margin fields in `deployable_selection.validation_rows`, causing policy-transfer audit margins to be `NaN` | Changed final deployable selection to compute static-start margins for both `static_margin_guard` and `static_margin_risk`; current selection behavior unchanged because the risk preset has one deployable candidate |
| 2026-06-02 | Aggregate mean deployable margin skipped no-deployable strict fallbacks as `NaN`, inflating the mean for positive-center runs | Count completed no-deployable fallbacks as static fallback with `0` deployable margin; added regression coverage and reran the final aggregate |
| 2026-06-02 | Initial rsync of the aggregate-script fix targeted remote `v1/` instead of `v1/scripts/` and `v1/tests/` | Removed the two mistaken remote copies, synced to the correct subdirectories, and reran remote core tests successfully |
| 2026-06-02 | First riskcenter extension launch reused the original-seed input root, which lacks `budget1p20_seed46` | Found the extension input root `v1/artifacts/claim_inputs_semimarkov_ext_b1p20`, archived the failed log, and relaunched with the correct root |
| 2026-06-03 | Local dry-run for extension diagnostic failed because extension input artifacts are only on the server | Re-ran dry-run on `remote-gpu` after syncing the updated script |
| 2026-06-03 | First remote dry-run of `learned_hybrid_rate_riskcenter_safe` failed because the new preset had not been synced and then because it was missing from one base-command preset set | Synced the script, added the preset to the objective set, and re-ran dry-run |
| 2026-06-03 | New rate-riskcenter preset initially inherited `value_residual` and missed `bc-preserve-warming` / risk event-threshold calibration | Removed value-residual from the preset and added preserve-warming plus `static_margin_risk` event-threshold calibration; dry-run now confirms intended flags |
| 2026-06-03 | Risk-band sync initially used `rsync --relative` with a `v1/` destination and created a mistaken remote `v1/v1` nesting | Verified the nested directory contained only the five mistaken sync copies, removed it, re-synced each file to its correct target path, and reran remote tests |
| 2026-06-03 | First larger local static/teacher calibration smoke failed after rollout compute because `save_rollout` was called with the old positional `sensor_ids` signature | Changed the call to keyword arguments `sensor_ids=...` and `state_columns=...`; reran mini static/teacher calibration successfully |
| 2026-06-03 | First remote tmux calibration launch expanded `$seed` before tmux execution, producing an empty `--seed` argument and only `seed_event_rich.log` | Relaunched with a remote Python wrapper that constructs the tmux command and keeps the seed loop inside the tmux shell |
| 2026-06-03 | First attempt at the remote Python wrapper embedded a here-doc inside a quoted SSH command, so the local shell misparsed part of the script | Switched to the standard `ssh remote-gpu python3 - <<'PY'` form; tmux now runs seed41/42/44 with correct arguments |
| 2026-06-04 | Rate-balance sync initially repeated the known `rsync --relative` into remote `v1/` mistake and created `v1/v1/...` | Verified the nested path contained only the mistaken copied files, removed `v1/v1`, and re-synced each file to its exact target path |
| 2026-06-04 | First remote dry-run of the balanced option-planner smoke used obsolete `run_claim_suite.py` argument names (`--config`, `--output-root`) | Re-ran with the actual interface (`--sensor-cfg`, `--out-root`) and confirmed the accepted v6/event-transport flags plus `--option-planner-rate-balance-grid 0.0 1.0 3.0` |
| 2026-06-04 | First log polling command for the balanced option-planner tmux used double-quoted SSH shell text, so local shell expansion erased the remote loop variable `$d` | Re-ran the poll with single-quoted remote shell text and verified all three seed logs |
| 2026-06-04 | First local old-vs-balanced aggregate comparison assumed the new `option_planner_rate_balance_weight` column existed in the old aggregate | Re-ran with column-presence checks so old and new aggregate schemas can be compared safely |
| 2026-06-04 | First remote launch command for the pure rollout-value smoke joined `run_claim_suite.py` arguments with `&&`, which would have run the default suite instead of the intended preset | Immediately killed the mistaken tmux session before any seed output was produced, confirmed no child `run_protocol_gate.py` remained, and relaunched with `shlex.join()` for the Python command |
| 2026-06-04 | First macro-option rsync repeated the known `rsync --relative` into remote `v1/` mistake, creating remote `v1/v1/...` copies | Verified the nested path contained only the six just-synced files, removed `v1/v1`, re-synced each file to its exact destination, and reran remote py_compile/pytest/dry-run successfully |
| 2026-06-05 | First Branch F smoke launch used nonexistent sensor cfg `configs/sensors/windblown_sensors_physical_event_v6.yaml` | Relaunched with the manifest-confirmed v6 config `v1/configs/sensors/windblown_sensors_physical_event_v6_complex_static_break.yaml`; no completed seed result was produced by the failed launch |
| 2026-06-05 | First window-level audit launch produced no live logs because `conda run` captured stdout despite `python -u`; the first script version also wrote no partial output until completion | Stopped the tmux run, added per-start progress logging and per-run partial CSV writes, then relaunched with `conda run --no-capture-output` |
| 2026-06-05 | Sequence-value sync repeated the known `rsync --relative` into remote `v1/` mistake and created `v1/v1/...` copies | Verified the nested path contained only the five just-synced files, removed `v1/v1`, re-synced each file to its exact destination, then passed remote `py_compile`, core pytest, and dry-run validation |
| 2026-06-05 | One dense-macro polling SSH connection closed with exit 255 after the local sleep interval | Followed the server skill protocol: ping showed 0% packet loss, a retry SSH succeeded, and the tmux experiment was still running normally |
| 2026-06-05 | Dense-macro artifact sync timed out once during result transfer | Confirmed server reachability, reran `rsync` with `--partial`, completed sync, then aggregated and logged the result |
| 2026-06-05 | First oracle-context sequence-value preset patch accidentally removed the preset from the objective/choices sets and left it in the learned-event set | Repaired the preset membership: objective/choices/preserve/no-BC/sequence-value sets include it, learned-event set excludes it; local and remote dry-runs now show `--forecast-truth-future` without `--learned-event-forecast` |
| 2026-06-05 | Sequence-value calibration returned the best invalid dynamic row when risk-band/positive-center selection found no passing validation row | Stopped the first oracle-regime launch before completion; changed `calibrate_sequence_value_policy` to return no row and disable the sequence-value candidate, forcing static fallback when no validation row passes |
| 2026-06-05 | Learned-twin sync repeated the known `rsync --relative` into remote `v1/` mistake and created remote `v1/v1/...` shadow copies | Verified the shadow directory contained only the just-synced files, removed it, re-synced with `rsync -azR` to the project root, and reran remote compile/tests |
| 2026-06-05 | Smoke cleanup `pkill -f smoke_twin_rollout_20260605_seed41` matched the SSH command itself and returned exit `255` | Followed the server protocol: ping had `0%` packet loss, retry SSH succeeded, and no old smoke child process remained |
| 2026-06-05 | First learned-twin tmux launch wrapper embedded an internal single quote inside the outer single-quoted SSH here-doc command, producing `NameError: name 'run_claim_suite' is not defined` before launch | Relaunched with all-double-quoted remote Python strings; tmux `v1_twin_rollout_20260605` started normally |
| 2026-06-05 | First formal learned-twin aggregate command used nonexistent `--root`; second aggregate used default `main` preset and therefore found no completed main runs | Reran `aggregate_claim_suite.py` with positional suite root and `--main-preset learned_twin_rollout_posguard_safe` |
| 2026-06-06 | First dry-run artifact sync failed because the local destination directory did not exist | Created the exact destination directory and re-ran the single-file rsync successfully |
| 2026-06-06 | First local H2 smoke used `scripts/train_window_risk_model.py` from the repository root, but the script lives under `v1/scripts/` | Re-ran with `v1/scripts/train_window_risk_model.py`; model training and artifact persistence completed |
| 2026-06-06 | First H3 engineering smoke allowed an all-static validation result with zero margin to pass the non-strict risk gate | Required at least one dynamic validation window and strictly positive mean margin; added regression coverage and verified final deployment remains disabled |
| 2026-06-06 | First automatic follow-up tmux command omitted `set -e`, so a failed H2 gate could have fallen through to validation | Replaced the watcher before data completion with a `set -e` command chain; validation now cannot run after a nonzero gate check |
| 2026-06-06 | A mixed planning-file rsync flattened the revised Branch H document into the remote `v1/` root | Removed the extra root-level copy, synced the document to `v1/docs/`, and verified both paths |
| 2026-06-06 | First full Branch H collector reused source forecasts trained through the end of `rl_train`, creating in-sample train versus out-of-sample validation forecast mismatch | Stopped at partial static collection, archived it with `_invalid_source_forecast_scope`, added oracle-pretrain-only forecast preparation, and restarted the formal seed41 chain |
| 2026-06-06 | The first oracle-pretrain-only restart still used current task-sensor truth as event/continuous forecast input, leaking the information that deployment must acquire through scheduling | Stopped and archived it with `_invalid_latent_forecast_inputs`; restricted formal forecast inputs to `met_station_core`, forced that sensor into all anchors/support masks, and restricted continuous outputs to the three proxy-MPC task variables |
| 2026-06-06 | Core-only forecaster inputs did not by themselves make runtime planner features causal: continuous context still read current task truth and `_state()` included the simulator truth event flag | Stopped and archived the partial collector with `_invalid_current_task_truth`; use learned h1 as the task-current proxy, remove the truth event dimension from the Branch H window state, and add invariance/missing-prediction regression tests |
| 2026-06-06 | Proxy-MPC support and target rates were initially computed from all source teacher labels, including the internal risk-calibration time block | Archived the affected fit rows, subset teacher labels by absolute `step_indices` to risk-fit windows only, verified 832 eligible rows, and restarted without recomputing frozen forecasts/static bank |
| 2026-06-06 | First fit summary diagnostic read `feature_audit` from the collection manifest instead of the feature-schema file | Re-read the correct `window_risk_feature_schema.json`; experiment artifacts were unaffected |
| 2026-06-06 | A long polling SSH connection closed with exit 255 | Ping showed 0% loss, immediate SSH retry succeeded, and both tmux sessions were still healthy |
| 2026-06-07 | First rollout-world-model smoke failed before training because `train_rollout_world_model.py` used `replace` without importing it | Added `from dataclasses import asdict, replace`; no model result was produced by the failed smoke |
| 2026-06-07 | First completed rollout-world-model smoke reported impossible normalized RMSE (`1800`) because dataset prediction audit compared normalized network outputs directly to physical truth | Unnormalized `_predict_dataset_members` before residual/audit calculations; the smoke model result is invalid and must be rerun |
| 2026-06-07 | Event-heavy support sweep wrapper printed selected validation rows but crashed with `NameError: support_top_k`, so selected finals were not launched automatically | Parsed the completed `60/60` validation gate artifacts, manually reran seed43/45 selected finals with the same split-locked gate script, and logged the wrapper bug in progress/changelog |
