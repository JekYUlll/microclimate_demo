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

## Claim-Suite Requirement
- The new mainline should be judged only against the strict split-protocol
  evidence it produces under `v1/`.
- Minimum paper-usable claim: on the declared final-test split, deployable
  DAgger-BC beats the validation-selected static comparator in at least `4/5`
  seeds and has positive mean paired margin. Anything below that remains a
  viability/prototype result.
- Required mechanism controls are no-DAgger and oracle-only objective. They
  test, respectively, whether rollout distribution correction and task-composite
  objective are necessary for the final claim.

## Action-Support Deployment Finding
- The `cost_safe` n=5 run did not rescue the deployable claim. Teacher still
  beat static in `5/5`, but the best deployable policy beat static in only
  `2/5`, with mean margin `-0.014865`.
- The learned action-cost policy failed for a clear implementation reason: at
  runtime it minimized over the full feasible mask set, including OOD masks that
  were not teacher-supported. It often dropped `met_station_core`, causing the
  frozen oracle loss to saturate near `10.0`.
- BC has a weaker version of the same support problem: it over-selects some
  low-frequency teacher-label actions during rollout. The next correction is to
  constrain deployable action selection to high-frequency teacher-label support
  plus the validation-selected static anchor.
- A second deployable correction is sensor-level mask imitation. Instead of
  treating 163 candidate masks as unrelated classes, it predicts per-sensor
  logits from the same forecast-aware state and lets the existing power
  projector form a feasible subset. This should generalize better across
  similar masks and gives a cleaner deployment story if it passes validation.
- The local seed41 probe supports this correction: plain mask BC nearly tied the
  static comparator, and mask BC with a small validation-static anchor bias beat
  it. This does not establish a claim, but it identifies `mask_anchor_safe` as
  the next n=5 candidate to run on the server.
- Server `support1_safe` confirms the expected lower bound: the privileged
  teacher beats static in all five seeds, but top-1 action support is too
  restrictive for deployment (`2/5` deployable wins, mean margin slightly
  negative). In seeds where the only allowed action is the validation-static
  anchor, the deployable policy cannot produce a strict improvement. Wider
  support sets (`support2_safe`--`support5_safe`) are therefore the meaningful
  candidates.
- Server `support2_safe` improves the mean deployable margin but not the
  win-rate criterion: seeds 41 and 42 pass, while seeds 43--45 lose. This
  indicates that simply adding one more frequent teacher action is not a robust
  deployable-policy repair. The teacher/reference side remains strong; the
  unresolved issue is choosing when to deviate from the static anchor.
- Server `support3_safe` also wins only `2/5` and has negative mean margin.
  The fixed top-k support route is therefore unlikely to be the final
  deployable algorithm. If support4/5 do not reverse the trend, the next
  correction should learn a validation-calibrated residual deviation rule
  around the static anchor rather than widening action support blindly.
- Server `support4_safe` remains `2/5`, with the same failure block
  (seeds 43--45). This makes the diagnosis stronger: the missing component is
  not action availability but temporal decision quality. Wider support gives BC
  more ways to deviate incorrectly from the anchor.
- The complete support-small grid (`support1_safe`--`support5_safe`) failed
  uniformly at `2/5` deployable wins while the teacher stayed at `5/5`.
  Therefore the current paper-usable core is the teacher/objective result, not
  the deployable policy. The next implementation should stop treating the
  problem as multiclass action imitation and instead learn a residual
  anchor-deviation decision with validation-calibrated risk control.
- The residual deployable repair implements that diagnosis directly: static
  anchor is the default action, a binary gate predicts whether to deviate, and
  validation selects the gate threshold. This is a cleaner deployable story than
  fixed top-k support because it makes the risk-control mechanism explicit and
  split-compliant.
- Server `residual_safe` failed more clearly (`1/5` deployable wins). The
  failure is therefore deeper than action support or a scalar deviation
  threshold. Current deployed features use a causal wind/event heuristic, while
  the teacher optimizes with privileged future rollouts. The next diagnostic
  should test a privileged future-event context for BC; if that passes, the
  missing component is a learned forecast-context module, not the scheduler
  policy head.
- `oracle_context_safe` is a diagnostic, not a candidate method: it gives BC
  future event context directly. Its purpose is to separate two hypotheses:
  policy-head/distillation failure versus missing forecast-context information.
- `oracle_context_safe` failed at `1/5`, so the dominant bottleneck is not only
  missing future-event context. The multiclass BC action head is the wrong
  deployable abstraction for this teacher: it can fit labels, but its rollout
  choices do not preserve the teacher's sequence-level cost advantage. The next
  serious implementation should learn deployable action values / rollout costs
  with a causal forecast model, or use an online planner over learned dynamics,
  rather than selecting an action id from teacher-label frequencies.
- The value-residual repair is the next serious test of that diagnosis: it
  keeps the static anchor as the default, scores only a small supported action
  set with a learned teacher-cost model, and uses validation to decide how much
  predicted advantage is required before deployment deviates from the anchor.
- Server `value_residual_safe` is the first deployable route to satisfy the
  strict n=5 gate. It beats the validation-selected static comparator in
  `4/5` seeds with mean paired margin `+0.002213`; the privileged teacher beats
  static in `5/5` seeds with mean margin `+0.030599`. The improvement is small
  and not statistically significant at n=5 (`sign_test_two_sided_p=0.375`), so
  the correct paper wording is "consistent controlled improvement under the
  pre-registered gate", not a large-effect or significance claim.
- The result also clarifies the mechanism story. Pure action-id BC, top-k
  support BC, residual binary gating, and privileged-context BC all failed, but
  anchor-default action-value residual selection passed. The useful innovation
  is therefore split-compliant forecast-aware residual value selection around a
  validation-chosen static anchor, not generic imitation learning.
- Behavior diagnostics for `value_residual_safe`: deployable mean power
  `1.1732`, static mean power `1.1779`, teacher mean power `0.9370`; deployable
  switch rate `0.2088` versus static `0.0041` and teacher `1.3771`; all final
  policies had zero warmup aborts and zero steady/peak constraint violations.
- Value-residual ablations clarify which components can be claimed. Removing
  DAgger did not change the pass pattern (`4/5`, mean margin `+0.002213`),
  because the current value-residual policy is driven mainly by the action-cost
  dataset and validation-calibrated residual threshold. Therefore DAgger should
  not be presented as a necessary mechanism for this version.
- The oracle-only objective ablation failed (`2/5`, mean margin `-0.006959`)
  while its privileged teacher still beat static in all five seeds. This is the
  strongest mechanism evidence so far: dynamic planning value exists under the
  frozen oracle, but the deployable scheduler needs the task-composite
  event/transport objective to translate that value into static-comparator
  improvements on final test.

## Strong-Claim Redesign Findings
- The original strong claim requires a scheduler whose forecast awareness is a
  learned causal component, not the previous wind-speed heuristic. Otherwise the
  method is better described as a value-residual static-anchor repair, not a
  general forecast-aware constrained scheduler.
- The next necessary architecture step is therefore to make the forecast context
  learned and split-compliant. The implemented learned event forecaster trains
  only on pre-validation data and produces multi-horizon probability columns
  consumed uniformly by teacher data collection, action-cost learning,
  validation calibration, and final testing.
- A local tiny-window smoke is not evidence for or against the claim, but it
  verified the new forecast path is operational. Formal evidence must come from
  the server n=5 `learned_value_residual_safe` run and then a wider multi-setting
  matrix.
- Server learned-forecast n=5 finished at the same qualitative level as the
  old weak result: deployable `4/5`, mean margin `+0.001856`, sign-test
  `p=0.375`; teacher remained `5/5` with mean margin `+0.030599`. This proves
  the learned causal event forecast path works, but it does not by itself
  strengthen the claim.
- Server learned-ensemble n=5 failed (`3/5`, mean margin `-0.003409`) despite
  teacher `5/5`. The seed43 failure was large enough to rule out
  uncertainty-penalized absolute-cost ensembling as the main deployable route.
- The next correction should target the decision variable directly. The current
  value-residual policy learns absolute short-horizon action costs and then
  subtracts the anchor cost at runtime. For very small final margins, independent
  absolute-cost errors dominate the residual. The new anchor-advantage residual
  path instead trains on `cost(anchor) - cost(candidate)` directly and uses
  validation only to calibrate the required predicted advantage before deviating
  from the validation-selected static anchor.
- The first anchor-advantage server launch found an implementation-level
  semantic mismatch rather than a valid negative result. Static baselines submit
  a desired mask to the environment and allow the online projector / warmup
  logic to execute the feasible projection. The first advantage dataset instead
  required the exact validation-static anchor action to appear with finite
  beam-search first-action cost at every training state. When the startup peak
  or warmup state projected that anchor, no rows were collected for seed45.
- Corrected anchor semantics: advantage labels now use the cost of repeatedly
  submitting the validation-static anchor mask, even when the current execution
  is projected, and deployable residual policies can always fall back to that
  anchor mask when predicted advantage is insufficient. This is a cleaner match
  to the comparator and should prevent non-result crashes from being confused
  with scheduler failure.
- A second support-safety issue was fixed before accepting new evidence: if all
  teacher-supported action IDs are temporarily infeasible under warmup/startup
  projection, a residual policy must not reopen the full feasible action space.
  Reopening the full space reintroduces the earlier OOD action-cost failure
  mode. The corrected residual policies return an empty supported set in that
  case, causing an explicit fallback to the validation-static anchor.
- Server `learned_advantage_residual_calib_safe` with the anchor projection and
  strict-support fixes failed decisively: deployable `0/5`, mean margin
  `-0.018997`, while the teacher stayed `5/5` with mean margin `+0.030599`.
  This is no longer an implementation crash; it is a negative result for the
  direct advantage-regression deployable route.
- The failure pattern is informative. Advantage residual often improves the
  task-event error but increases frozen-oracle / weighted forecast error enough
  to lose the task-composite objective. Validation calibration overfits this
  tradeoff and does not transfer to final windows. Therefore the next route
  should not replace the stable value-residual policy with direct advantage
  regression; it should treat advantage residual as an optional deployable
  candidate selected against the value-residual baseline on validation.

## Guarded Deployable Selection Finding
- The failed anchor-advantage route shows a specific validation risk: a
  deployable policy can improve a task-event term or win the validation mean
  while still making large harmful deviations from the static anchor on some
  windows. A split-compliant fix is to keep final-test untouched but make
  validation selection margin-aware: prefer policies that beat the static
  anchor on mean validation objective and do not lose badly on individual
  validation windows.
- This is not a posthoc final-test filter. The new `static_margin_guard`
  criterion is a deployable-selection rule applied before final replay. Its
  purpose is to choose between value-residual and advantage-residual candidates
  more robustly, not to change the teacher or the objective.
- The completed hybrid run exposed a stricter issue: in three seeds the
  value-residual policy was not even better than the static anchor on
  validation, but the selector still deployed it because it selected only among
  deployable policies. A static-aware guard prevents harmful deployment, but by
  itself it cannot create wins. The next deployable mechanism needs a simpler
  dynamic trigger that can reproduce the teacher's event-timing advantage.

## Event-Threshold Residual Finding
- The current teacher label distribution is dominated by a small set of
  event-sensing masks such as `met_station_core|radiometer_basic|laser_disdrometer|fc4_flux`
  and `met_station_core|surface_temp_ir|laser_disdrometer`. This suggests a
  lower-variance deployable controller: keep the selected static anchor by
  default, and switch to a high-frequency teacher event action only when the
  learned causal event forecast crosses a calibrated threshold.
- This route is closer to the original claim than another action-value model:
  the deployed decision explicitly depends on learned future event probability,
  the switched action is teacher-supported, and all action/threshold choices are
  validation-only. It may still fail, but it directly tests whether the dynamic
  sensing value can be recovered with a lower-variance policy class.
- The paired confirmation run passed the strict n=5 gate:
  deployable `4/5`, teacher `5/5`, mean deployable margin `+0.003758`.
  This is the strongest deployable result so far because the selected policy is
  explicitly forecast-triggered in three seeds and the validation guard uses
  paired start-level comparisons under the same sensor-noise seed.
- The limitation remains clear: seed44 still fails, and the sign test is not
  significant at n=5 (`p=0.375`). The result supports the core feasibility
  claim and mechanism direction, but the next evidence must test robustness
  across budgets and event-regime perturbations before making a full paper
  claim.

## Strong-Claim Scaling Finding
- Cross-budget evidence should be interpreted as operating-regime support, not
  as a monotonic guarantee. The first budget-matrix result shows `B=1.05`
  fails clearly: deployable `1/5`, mean margin `-0.011709`, and even the
  privileged teacher loses in seed41. This makes tight-budget robustness a
  negative boundary condition for the current method.
- The same matrix reproduces `B=1.20` exactly: deployable `4/5`, teacher `5/5`,
  mean margin `+0.003758`. This confirms the main result is not an artifact of
  the previous output directory, but it is still one calibrated operating
  budget.
- The completed `B=1.35` result also fails deployable robustness: deployable
  `1/5`, teacher `5/5`, mean margin `-0.009496`. This is not a teacher/objective
  failure; it is a deployable selection/trigger failure under a looser power
  operating point where the validation-selected static anchor is already strong.
- For event-regime scaling, the clean route is to prepare v1 inputs directly:
  generate the truth sequence and frozen oracle only, then run the v1
  claim-suite. Re-running archived custom PPO just to obtain an oracle would
  mix historical machinery into the new method and waste runtime.
- The sparse-event perturbation (`event_coverage=0.20`) passes the current n=5
  gate: deployable `4/5`, teacher `5/5`, mean margin `+0.013532`. This supports
  an event-regime robustness claim for a sparser event distribution, but it
  does not rescue the cross-budget robustness claim.
- The `B=1.35` teacher label distribution shifts toward several high-frequency
  four-sensor event masks rather than the smaller event masks that worked at
  `B=1.20`. A single event-threshold action is therefore too low-rank for this
  regime. The next deployable candidate should preserve the forecast-triggered
  structure but rotate through teacher-supported event masks to recover the
  teacher's coverage diversity.
- A purely time-indexed rotation is a cheap first test, but it may be
  misaligned with the semi-Markov observation process: the correct event-time
  action is often the subset containing the stalest event-relevant sensors, not
  simply the next subset in a cycle. The added `freshness` selector is a
  low-cost correction that keeps the deployed policy causal and validation
  calibrated while using online sensor age to choose among teacher-supported
  event actions.
- The completed time-cycle run validates that concern. It failed at `1/5`
  deployable wins with negative mean margin, while the teacher remained `5/5`.
  The event-support set itself is not enough; the deployed policy needs a
  state-dependent choice within that support. Online freshness is the next
  lowest-cost causal state signal to test before introducing a larger planner.
- The freshness-grid result is also negative despite selecting freshness in
  all five event-support-cycle calibrations. The stronger diagnosis is that
  B=1.35 teacher value comes from duty-cycle power saving across several
  teacher-supported masks, not from event-triggered selection of one or a few
  high-power event masks. A deployable policy should therefore target teacher
  sensor active rates and adjust online by duty deficit/freshness, rather than
  only thresholding event probability.
- The first teacher-rate attempt only slightly improved mean margin and was
  never selected by validation, so average duty targets are still too weak.
  The next plausible compression is sequence-level rather than rate-level:
  replay the teacher-label sequence with feasibility lookahead, preserving the
  high switching structure that appears to generate the teacher's lower-power
  forecast benefit.
- Teacher-sequence replay did not improve the B=1.35 boundary: it repeated the
  teacher-rate pattern (`1/5`, mean margin `-0.007056`) and the sequence policy
  lost the validation static-margin guard in every seed. Restoring BC/KNN to
  the guarded hybrid candidate set also failed (`1/5`, mean margin
  `-0.008513`), with only seed44 passing.
- The actionable conclusion is no longer "try another small B=1.35
  compression." Across six B=1.35 deployable attempts, teacher wins remain
  `5/5` but deployable wins remain `1/5`. This is a real operating-regime
  mismatch: looser budget makes the validation-selected static anchor strong
  enough that deployable forecast-triggered deviations do not transfer
  reliably. The paper path should state this as a boundary and strengthen the
  supported `B=1.20` claim with more seeds and perturbation evidence.
- The first B=1.20 seed-extension is also not strong enough for the original
  robustness claim. On seeds `46--55`, the event/value guarded route wins
  `6/10`; combined with the original seeds `41--45`, it wins `10/15`, below
  the required `12/15` for an 80% seed-win criterion. The mean deployable
  margin remains positive, and the privileged teacher wins `14/15`, so the
  dynamic forecasting objective is still valid. The failure is deployable
  transfer and validation selection, not absence of dynamic value.
- The failures are split across the two selected deployable families:
  event-threshold wins `3/6` on the combined set of selected event-threshold
  seeds in the extension, and value-residual wins `3/5` in the extension.
  Therefore simply preferring one of these two heads is not enough. The active
  correction is to expand the guarded candidate set with direct BC/KNN teacher
  imitation and let validation choose; early completed seeds `46--47` both
  pass under that route.
- Adding BC/KNN to the guarded candidate set is also insufficient on the
  extension seeds. It reaches only `5/8` before the `8/10` target becomes
  impossible. This matters because BC/KNN is not a fundamentally different
  temporal model; it is another action classifier over the same rollout
  distribution. The next correction should test a different compression of the
  teacher behavior, such as active-rate/freshness targets, before investing in
  heavier learned online planning.
- The completed BC/KNN extension ends at `7/10`, so it is an improvement but
  not a standalone extension pass. However, because the original event/value
  n=5 result lost only seed44, the BC/KNN method still has one narrow path to a
  combined `12/15`: it must run `5/5` on the original seeds. That check is
  necessary before rejecting BC/KNN as the combined B=1.20 main method.
- The old-seed BC/KNN check failed immediately on seeds 41 and 42, closing
  that narrow path. This strengthens the conclusion that the current
  deployable layer is the bottleneck; adding more guarded action classifiers
  does not reliably transfer teacher value to final-test rollouts.
- The active algorithmic correction is now a learned rollout-value planner,
  not another teacher-label compression. It adds a train-split
  action-conditioned transition surrogate and uses learned beam planning over
  causal policy features. This directly targets the missing temporal component
  identified by the n=15 failure: event thresholds and action classifiers can
  decide whether to deviate from the static anchor, but they do not model how a
  deviation changes subsequent warmup, freshness, observation history, and
  forecast context.
- The planner remains a deployable student rather than a privileged teacher:
  transition labels are learned from train-split simulator rollouts, but final
  decisions use only the fitted transition model, fitted action-cost model,
  current env state, and learned event probability columns. This is a stronger
  paper-level algorithmic story than BC/KNN/rate/cycle because it introduces a
  causal model-predictive mechanism that can be ablated.
- The first rollout planner exposed a target-scale issue: the action-cost
  dataset originally normalized costs within each state so the model could rank
  candidates for one-step residual decisions. A beam planner sums predicted
  costs across simulated steps, so those relative per-state normalized scores
  are not an additive objective. The corrected raw-cost planner uses a separate
  raw candidate-cost model for planning while preserving normalized targets for
  the one-step value-residual baseline.
- Interpreting the raw-cost planner result should focus on whether the planner
  itself is selected by validation and whether its rollout behavior changes
  power/switching relative to the static anchor. If validation still selects
  value-residual or event-threshold in most seeds, the bottleneck remains
  validation transfer/candidate support; if rollout-value is selected but fails,
  inspect transition error and raw-cost calibration before adding another
  deployable head.
- Partial raw-cost planner evidence points to a stronger mechanism diagnosis:
  the raw planner's cost and transition models train successfully, but
  validation still selects value-residual or event-threshold in completed seeds.
  The selected students remain near the high-power static anchor, while the
  privileged teacher gains margin by using lower-power, high-switch temporal
  mixtures. The next deployable test should therefore compress teacher temporal
  mixing directly, not merely add another one-step action-value head.
- Full raw-cost planner and teacher-mix evidence closes two routes. The raw
  planner and the teacher-mix suite both finished at deployable `2/5`, teacher
  `5/5`, with identical final selected policies and margins. Teacher-rate and
  teacher-cycle were not selected because their validation objectives/margins
  were weaker than value-residual or event-threshold.
- The failure of global teacher-rate is not evidence against duty-cycle
  compression in general; it only shows that a single global target active-rate
  vector is too low-rank. A better deployable abstraction is contextual duty:
  learn teacher sensor active probabilities from causal forecast-aware state,
  then use online duty-deficit/freshness/power feedback to choose among
  teacher-supported feasible masks. This directly targets the teacher's
  low-power high-switch behavior while remaining split-compliant and
  deployable.
- The first contextual-duty implementation is not enough. The formal n=5 run
  reached only deployable `3/5` while teacher stayed `5/5`; contextual-duty
  itself was never selected by the final validation guard. The important
  mechanism is calibration mismatch: contextual-duty hyperparameters were
  chosen by a separate mean validation objective, then re-evaluated under a
  paired static-margin guard. Seed44 illustrates this cleanly: the calibration
  mean looked best, but paired validation had two negative starts, so the guard
  rejected it and selected value-residual instead.
- The next correction is not another policy head. It is to calibrate
  contextual-duty with the same paired static-margin criterion used for final
  deployable selection. This keeps the split protocol intact and directly
  tests whether the contextual-duty policy class has robust validation support
  once its hyperparameters are selected under the right criterion.
- Guard-aware contextual-duty calibration is a necessary consistency fix but
  does not change the policy class: the policy still maps current causal state
  to teacher sensor probabilities and then uses global duty deficit/freshness
  feedback greedily. Early guardcalib evidence shows it still loses selection
  to event/value policies in completed seeds. This points to a sequence-memory
  gap rather than another scalar calibration gap.
- The next deployable abstraction is therefore a recurrent sequence-mask
  student. It uses the existing split-compliant teacher labels but preserves
  their rollout order: a GRU receives current forecast-aware state and the
  previous selected mask, then outputs sensor logits that are constrained to
  teacher-supported feasible masks. This is still deployable and causal, but it
  directly targets the teacher's high-switch temporal mixture rather than
  compressing it into iid action labels or global active rates.
- The formal guard-aware contextual-duty n=5 result closes that route as a
  main claim candidate: deployable wins are `3/5` while the teacher remains
  `5/5`. Contextual-duty calibration passed the guard in seeds 41 and 44, but
  final validation still selected value-residual because it had lower
  validation objective; in seeds 42, 43 and 45 contextual-duty failed the guard.
  Therefore the failure is not merely hyperparameter-selection mismatch. The
  policy class itself is too weak or too unstable for the original claim.
- Early sequence-mask evidence isolates a sharper problem: the GRU student can
  fit ordered teacher masks almost perfectly, but validation still prefers the
  old event-threshold/value-residual heads in completed seeds. This means
  temporal memory alone is not enough if the loss is still sensor-level
  imitation. The next necessary student should learn the teacher's objective
  surface over candidate masks, not only the mask sequence.
- The recurrent-value student is designed for that gap. It uses the same causal
  inputs as sequence-mask plus previous executed mask memory, but trains on
  beam-search first-action rollout costs for each feasible teacher-supported
  candidate. Its deployment rule is therefore a recurrent action-value
  decision with validation-calibrated anchor deviation, not another static
  action classifier.
- The completed sequence-mask n=5 run confirms the diagnosis. It failed at
  deployable `3/5` even though teacher stayed `5/5`, and the sequence-mask
  student was never selected for final replay. High sequence imitation accuracy
  (`0.94--1.00` exact match) did not translate into robust validation margins.
  Therefore the bottleneck is not simply remembering the teacher's previous
  masks; it is learning when a deviation has enough forecast-objective value to
  beat the static anchor on held-out starts.
- Early recurrent-value training shows the next difficulty: direct cost
  regression can run and can be selected by validation in at least one seed,
  but its top-1 candidate accuracy is unstable. A rank-aware loss is therefore
  a justified next variant, because deployment depends mainly on masked
  candidate ordering and anchor advantage, not calibrated absolute cost.
- Recurrent-value also exposed a validation-selection artifact: with
  `min_mean_margin=0.0`, a deployable that falls back to the validation static
  anchor on every step obtains exactly zero paired margin, passes the
  static-margin guard, and can displace non-identical candidates. Seeds 42 and
  43 in the first recurrent-value n=5 run selected this no-op recurrent policy
  and then exactly matched the static objective on final test. This is not a
  useful positive result; corrected recurrent presets must require a positive
  paired validation margin before a recurrent policy can be treated as
  guard-passing.
- The next recurrent objective should be anchor-relative, not just
  absolute-cost ranking. The deployable decision is "deviate from static only
  if a candidate has enough positive advantage"; therefore a recurrent
  anchor-advantage student is better aligned than predicting candidate costs
  and subtracting an inferred anchor cost. This does not change the split
  protocol: all advantages are collected on train starts using teacher rollout
  costs, and validation still gates any deployable use.
- Recurrent-advantage also exposed a result-accounting hazard: newly added
  deployable policy names must be counted consistently in both validation
  selection and final gate summaries. The missing
  `forecast_aware_recurrent_advantage` entry in the final deployable filter
  has been fixed by centralizing deployable names in
  `DEPLOYABLE_POLICY_NAMES`; otherwise a genuinely winning recurrent-advantage
  final rollout could have been ignored in `gate_summary.json`.
- Early recurrent-advantage evidence is negative at the algorithm level, not
  just at the accounting layer. On the first three original seeds, the
  recurrent-advantage candidate fails its own positive paired static-margin
  calibration and is disabled before final selection. The active final
  candidates therefore remain the older value-residual/event-threshold heads.
  If the completed n=5 result follows this pattern, the next correction should
  stop adding supervised policy heads and instead attack the validation-transfer
  problem directly.
- The recurrent-advantage branch is now closed as a main route after three
  completed seeds: only seed43 passes, while seeds41 and 42 fail after
  recurrent-advantage is disabled and value-residual is selected. This makes a
  `4/5` original-seed pass impossible. The negative result reinforces that
  more recurrent supervised heads are not the right next move.
- Transfer audit identifies a simpler counterfactual before heavier redesign:
  in the combined B=1.20 event/value guarded evidence, selected
  event-threshold policies win `6/8`, while selected value-residual policies
  win `4/7` and show a systematic negative validation-to-final transfer gap.
  The immediate test is therefore not another model class but an
  event-threshold-only deployable route to determine whether value-residual is
  actively hurting the n=15 claim.
- The event-threshold-only counterfactual is partially negative but highly
  diagnostic. At `8/15` completed seeds it has deployable `5/8`, teacher `7/8`,
  and mean deployable margin `-0.000389`; passing the strict `12/15` gate now
  requires all seven remaining seeds to pass. The selected event-threshold
  policy has positive validation margin on average but slightly negative final
  margin, so the immediate bottleneck is validation-to-final calibration rather
  than another missing supervised policy head.
- The next low-cost correction is validation-guarded event-threshold
  calibration. Instead of choosing the event action, threshold and probability
  aggregation by validation mean objective, the new local preset chooses them
  with the same paired static-margin guard used for deployable selection. This
  keeps the split protocol intact and isolates whether the event-threshold
  route fails because its threshold is overfit to too few validation starts.
- Validation-guarded event-threshold calibration is also negative as a strong
  claim route. On original seeds `41--45`, `learned_event_threshold_valguard_safe`
  reaches deployable `3/5` while the teacher remains `5/5`; mean deployable
  margin is only `+0.000050`. The transfer audit shows validation margin mean
  `+0.003035` but final margin mean `+0.000050`, with mean transfer gap
  `-0.002985`. This means paired-margin threshold calibration reduces neither
  the core validation-transfer risk nor the deployable instability enough for
  the required claim.
- The earlier pause note is superseded by the user's clarification that v1
  should continue while PD-PPO cleanup proceeds on a forked branch. The
  substantive conclusion still holds: the next v1 step should be a larger
  validation-transfer design change, not another small supervised head.
- User clarified that v1 should continue because PD-PPO cleanup is on a forked
  branch. The immediate v1 direction is therefore validation-transfer redesign,
  not another supervised policy head.
- A concrete protocol weakness was found in validation deployable selection:
  `static_margin_guard` can still select a failing deployable when no candidate
  passes the guard, because the selector only ranks guard-passing rows first
  and then falls back to the best failing row. This behavior is useful for
  diagnostics but too permissive for deployable-claim semantics.
- Added strict guard fallback as an opt-in protocol mode. It does not improve
  win count by itself, but it prevents unsupported validation candidates from
  being counted as deployable evidence and gives a cleaner baseline for future
  transfer-risk selectors.
- The next active hypothesis is that validation selection is undersampled: the
  old validation guard used only four validation starts. Dense-validation
  selection with twelve starts is now running to test whether threshold/action
  calibration stabilizes before adding a more complex start-conditioned
  transfer-risk model.
- Start-level transfer audit on the completed valguard run shows the final
  failure is structured by seed/regime rather than by event coverage alone.
  Across 20 final starts, event-threshold wins `11/20` with mean margin
  `+0.000055`, but seed44 is `0/4` and seed45 is `4/4` while event rates are
  all high and similar. Future selectors should therefore include regime/static
  anchor compatibility features, not only event probability thresholds.
- Action-mask inspection supports that interpretation. The same event switch
  `107 -> 46` loses seed41 but wins seed42, while seed44 uses `107 -> 41` and
  loses all starts. Therefore the failure cannot be explained by a single bad
  event mask; it depends on the interaction between seed/regime, selected
  static anchor, and event action.
- Dense-validation partial evidence is currently positive. With twelve
  validation starts, completed seeds `41--43` all beat static on final test and
  the start-level audit gives `11/12` wins. This supports the hypothesis that
  the earlier valguard failure was at least partly caused by undersampled
  validation calibration. The unresolved test is whether seed44, previously a
  structured failure case, is rescued by dense validation.
- Dense validation does not rescue seed44, but it does diagnose the failure:
  seed44's validation margin is negative on `11/12` validation starts. This
  means the remaining protocol issue is not calibration blindness for seed44;
  it is the non-strict historical selector that still deploys a candidate after
  the guard fails. The strict fallback preset is therefore not just defensive
  bookkeeping; it is the correct deployable semantics for unsupported regimes.
- Dense-validation n=5 is now the strongest positive v1 result: deployable
  `4/5`, teacher `5/5`, mean margin `+0.007063`, and `15/20` final-start wins.
  Compared with the 4-start valguard run, denser validation materially improves
  transfer, but not perfectly: seed44 remains a clear negative regime. The next
  scientific question is scaling, not another local tweak: run extension seeds
  `46--55` under the same dense12 setting and aggregate against the original
  seeds.
- Extension seeds `46--47` start positively for the deployable route
  (`2/2` wins), but they complicate the selector story. Seed47 fails the paired
  validation guard yet wins on final test, while seed44 fails the guard and
  loses. Therefore a binary strict guard is too blunt: the eventual deployable
  selector should estimate transfer risk or regime compatibility rather than
  treating all guard failures as no-deploy cases.
- Extension seeds `48--49` turn dense12 scaling negative at the partial
  extension level: deployable becomes `2/4` with mean margin `-0.003784`, while
  teacher is `3/4`. The validation-to-final transfer gap is again negative
  (`-0.007192` mean), and start-level wins drop to `8/16`. This means the
  current route is no longer just an undersampled-validation problem. The
  needed correction is a transfer-risk or regime-compatibility selector that
  can distinguish seed47-style guard-fail wins from seed48/49-style guard-fail
  losses.
- The first low-cost transfer-risk correction is now implemented as
  `static_margin_risk`. The rationale is empirical and narrow: on completed
  dense12 seeds, final margin has high Spearman association with validation
  margin mean/median and negative-start count, while raw objective can select
  a guard-failing row with worse transfer risk. This is not a final solution to
  regime compatibility, but it directly addresses the current calibration
  failure without changing the teacher, oracle, or policy class.
- Dense12 valguard is now closed as a scalable claim route. After seed50 failed
  and seed51 passed, the combined evidence is deployable `7/11` and teacher
  `10/11`; the remaining four extension seeds cannot reach the required
  `12/15` deployable wins. The important signal is not that the teacher failed:
  it mostly did not. The bottleneck is deployable transfer, especially in
  extension regimes where validation margins are weak or unstable.
- Risk-calibrated dense12 has an early positive mechanism result on seeds
  `41--42`: it changes event-threshold calibration to robust validation
  rows with zero negative validation starts and improves both final margins
  relative to the dense12 valguard run. This is not yet a claim result, but it
  directly supports the diagnosis that raw-objective fallback during calibration
  was selecting unnecessarily risky thresholds.
- Risk-calibrated seed43/44 splits the failure mode cleanly. A guard-failing
  candidate with positive center validation margins can still transfer
  (seed43), but a negative-center candidate should be rejected before final
  deployment (seed44). Therefore the next selector should require
  positive-center validation support for `static_margin_risk`, while still not
  requiring the full strict guard.
- Positive-center fallback is now implemented and tested. This is a selector
  correction, not a new policy class: it changes deployability semantics so
  validation rows with negative mean or median static margins cannot be treated
  as acceptable deployment evidence. The expected next experiment is
  `learned_event_threshold_riskcenter_safe`, especially if the current
  risk-calibrated run either fails seed45 or passes n=5 but remains vulnerable
  to the seed44 negative-center failure pattern.
- Risk-calibrated dense12 passes the original n=5 gate but does not remove the
  underlying transfer-risk flaw. It reaches deployable `4/5`, teacher `5/5`,
  and mean deployable margin `+0.004409`; however, seed44 remains a large
  negative (`-0.033481`) tied to a negative-center calibration row. Therefore
  the result is a valid small-n positive gate, but scaling it directly would
  repeat the known failure mode. The positive-center risk selector remains the
  right next branch before any extension-seed scaling.
- The first `riskcenter` seed44 result was invalid as evidence because of a
  dispatch bug, not because positive-center semantics failed conceptually.
  The selector function itself returned `None` for a negative-center row, but
  final deployable selection forgot to pass the `require_positive_center`
  flag. This is now fixed and regression-tested; seed43--45 must be rerun
  before interpreting `riskcenter`.
- Corrected `riskcenter` seed44 confirms that positive-center fallback changes
  the evidence semantics in the intended way. The same negative-center
  validation row is still diagnosed by calibration, but it is no longer
  promoted into final deployable evidence. This means the selector can cleanly
  separate "teacher has dynamic value in this regime" from "the deployable
  event-threshold policy is validation-supported in this regime."
- Corrected `riskcenter` seed43 confirms the complementary case: a candidate
  can fail the full paired validation guard but still have positive center
  support and transfer positively. This supports using positive-center as a
  middle ground between permissive risk ranking and overly strict guard
  fallback.
- Fixed `riskcenter` passes the original n=5 gate with cleaner semantics:
  deployable `4/5`, teacher `5/5`, conservative mean deployable margin
  `+0.011105` when the unsupported seed44 fallback is counted as zero. The
  selected event-threshold rows transfer on all selected seeds (`4/4`), while
  seed44 is explicitly excluded. This is a useful positive result, but it is
  still a small single-setting gate with sign-test `p=0.375`; it supports
  scaling the selector, not stopping the experimental program.
- Aggregate summaries should count no-deployable fallback as zero margin, not
  missing data. The earlier `NaN` behavior inflated mean margins whenever a
  strict selector rejected all deployable candidates, even though the win count
  remained correct.
- Early extension seeds46--47 are encouraging for the deployable selector but
  expose a teacher-evidence caveat. Deployable wins both seeds, while the MPC
  teacher fails seed47. This means the extension run may support a deployable
  validation-transfer claim even if the "teacher always has dynamic value"
  statement has to be weakened or localized.
- Extension seeds48--49 show the cost of conservative positive-center
  semantics. The selector correctly rejects negative-center validation rows,
  but those rejections count as deployable failures and push the combined
  scaling test to a fragile `6/9` state. This suggests the current selector is
  useful for evidence cleanliness but may be too conservative to recover the
  full `12/15` strong claim unless all remaining seeds transfer.
- The fixed positive-center selector fails the strong scaled seed-count claim
  by early stop: combined evidence reaches only `7/11`, making `12/15`
  impossible. The important nuance is that selected deployable rows are
  reliable (`7/7` final wins); the failure is coverage, not selected-policy
  transfer. The next algorithmic target should be a fallback or transfer model
  for negative-center regimes, because simply tightening selection improves
  honesty while reducing deployable coverage.
- In extension seeds48--50, the MPC teacher's advantage is mostly a
  duty-rate/temporal-mixing behavior over `fc4_flux`, not a different
  event-threshold mask. Event-threshold deployments in the same dense12
  setting lose on those seeds, while the teacher lowers `fc4_flux` duty cycle
  to about `0.50--0.68` and keeps the rest of the core stack active. This
  motivates testing teacher-rate or contextual duty fallback specifically in
  negative-center regimes.
- The teacher-rate fallback diagnostic is negative. Matching average teacher
  sensor duty rates is not enough to recover the teacher advantage in
  negative-center extension regimes; teacher-rate fails positive-center
  validation on seeds48--50 and is rejected on seed51. The remaining
  plausible fallback has to preserve temporal/state-conditioned sequencing
  rather than only matching marginal duty rates.
- Contextual-duty is also insufficient as the negative-center fallback. It
  introduces state-conditioned teacher-mask probabilities and duty-deficit
  feedback, but the riskcenter diagnostic reaches only deployable `2/4` while
  the teacher remains `4/4`. The key failure is not just coverage: seed50 has
  positive-center validation support for contextual-duty but loses on final by
  `-0.009780`. This means the positive-center heuristic is too weak for
  state-conditioned policies; the next correction should explicitly estimate
  validation-to-final transfer risk using start/regime features, validation
  margin distribution, selected static anchor, selected dynamic policy, and
  perhaps rollout behavior summaries. Adding another duty-style or supervised
  policy head is unlikely to close the remaining claim gap.
- The first transfer-risk audit shows why this is hard. Across de-duplicated
  selected deployable rows from current riskcenter-style evidence, there are
  only `9` rows and only one final-loss row. A posthoc fixed rule
  (`positive_center` plus validation negative-start count `<=4`) avoids that
  one loss, but leave-one-seed-out cannot learn it when seed50 is held out
  because the training fold has no negative example. Therefore a learned
  final-outcome selector is underdetermined at the current data volume. The
  next defensible step is prospective testing of a predeclared risk-band rule
  on unused extension seeds, while acknowledging that this is still a selector
  heuristic rather than a mature learned transfer model.
- The prospective risk-band partial result on unused seeds52--53 is negative
  and more damaging than the earlier coverage failures. The predeclared
  selector deployed event-threshold in both seeds because validation support
  looked acceptable: seed52 even passed the full guard, and both seeds had
  positive mean/median/q25 static margins. Both transferred negatively on
  final test. Therefore fixed validation-margin lower-tail rules are not
  enough to model transfer risk. The remaining bottleneck is regime/start
  distribution shift between validation and final windows, not merely an
  overly permissive risk threshold or an insufficient duty-style fallback.
- The completed risk-band run on seeds52--55 closes the fixed-selector branch.
  Deployable wins are only `1/4` while the teacher remains `4/4`. The one
  useful positive is seed55, where contextual-duty wins every final start,
  but this is not enough to support the claim and does not explain the
  event-threshold inversions in seeds52--53. Validation margin statistics can
  certify some obviously bad rows, but they cannot predict the regime shift
  that makes an apparently well-supported event-threshold policy lose on final
  windows. The next method needs either online regime-conditioned action
  choice or an explicitly learned start/regime transfer model trained from
  validation subwindows, not another global validation-summary threshold.
- The 2026-06-03 direction documents should be applied only to the v1
  experiment branch; Paper 1 references are ignored because the user is handling
  paper work on a fork branch.
- A new transfer-structure audit confirms that the failure is not just a weak
  deployable selector. Across the current riskcenter/riskband/riskcalib family,
  unique static validation-vs-final objective Spearman is only `0.204`, so even
  fixed static anchors experience protocol/regime shift. Unique deployable
  validation-margin-vs-final-margin Spearman is only `0.280`, so aggregate
  validation margins cannot reliably decide dynamic deployment either.
- Seed44 is not an event-sparse boundary case. Its final event rate is
  `0.7666`, close to seed41's `0.7813`, and event-threshold has slightly better
  SOC/power than the static anchor while still losing badly (`-0.0335` in the
  riskcalib run). The teacher wins by using lower-power multi-mask temporal
  mixing. Therefore a cost-only online filter is insufficient; any CAPS-style
  switch must include objective/regime compatibility, not only budget/SOC.
- Literature check supports policy-switching and real-time budget framing, but
  the direct v1 lesson is narrower: use CAPS/TREBI ideas as deployment-time
  conditional switching and upper-bound diagnostics, not as justification for
  immediate PPO retraining. Verified primary sources: CAPS
  `https://arxiv.org/abs/2412.18946`, TREBI
  `https://proceedings.mlr.press/v202/lin23h.html`, MPC-PPO
  `https://arxiv.org/abs/2504.20815`, and BAFS-CMDP
  `https://openreview.net/forum?id=rYhK13RkA5`.
- The conditional-deployment upper bound is positive enough to continue.
  Combining available final start-transfer rows for seeds
  `41--47,49--53,55`, the best direct dynamic policy wins `9/13` seeds with
  mean margin `+0.002593`. A per-start oracle fallback that uses dynamic only
  on positive final starts wins `13/13` with mean margin `+0.008638`. This is
  not deployable evidence because it uses final outcomes, but it proves that
  conditional fallback can in principle recover the known failure seeds
  (`44`, `50`, `52`, `53`) without harming positive seeds. Therefore the next
  implementation should approximate this start/online decision causally.
- The user's correction on scenario design is accepted. Because v1 is a
  simulation/virtual deployment line, scenario calibration should deliberately
  create a non-trivial dynamic scheduling problem before gradually relaxing
  toward harder realism. This is not a posthoc data edit; it is part of method
  development.
- Static is currently strong because the B=1.20 scenario allows the direct
  measurement stack to stay on. A power/static audit over 15 seeds shows
  selected static masks include `met_station_core`, `laser_disdrometer`, and
  `fc4_flux` in `100%` of seeds; top-10 static candidates include
  `laser_disdrometer` in `97.3%`, while `snow_particle_counter` is never used.
  The selected static power mean is `1.1619` under budget `1.20`, so the static
  baseline operates close to the constraint but still keeps all critical direct
  sensors active.
- The current dynamic deployable is therefore solving a marginal replacement
  problem, not the intended hard scheduling problem. Event-threshold often
  swaps cheap context/proxy sensors while laser remains continuously active.
  Recalibration should target three properties: static cannot continuously
  operate the full direct stack; energy depletes within an evaluation horizon
  if laser is held on; cheap proxy sensors remain useful but cannot directly
  solve the primary targets.
- The v5 constraint-active sensor calibration is the first candidate that
  satisfies the structural scenario gate. Under `B=1.20`, energy capacity
  `70`, and reserve `20`, `core+laser+fc4` and `laser+fc4` are infeasible,
  while `core+laser` and `core+SPC+fc4` remain feasible. The same v5 sensors
  with capacity `90` are still too loose because max laser duty over proxy is
  about `0.986`, effectively re-opening continuous laser use.
- The v5/e70 energy gate is active but not degenerate: constant `core+laser`
  lasts about `178.6` steps of a 256-step evaluation horizon, proxy
  `core+SPC+fc4` can last continuously, and the max laser-over-proxy duty is
  about `0.816`. This places the scenario in a useful interior regime rather
  than forcing laser always off or always on.
- Static/teacher mini-smoke checks support the calibrated scenario direction
  but are not yet final evidence. In seed41 uniform 128-step smoke,
  validation-selected static chooses
  `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`, not the old
  direct laser stack, and the MPC teacher beats it by `+0.037300`. In seed41
  event-rich 128-step smoke, the same static anchor is selected and the teacher
  beats it by `+0.105578` while using `15` unique masks including
  `met_station_core|laser_disdrometer`.
- The event-rich mini smoke is the most useful positive signal so far:
  final event rate is `0.828125`, static power is `0.56`, teacher power is
  `0.56875`, and teacher uses laser for only part of the rollout. This matches
  the desired problem structure: static relies on a cheap context/flux stack,
  while dynamic scheduling can spend laser budget selectively during events.
- The formal event-rich static/teacher calibration gate passes for v5/e70 on
  seeds `41`, `42`, and `44`. Teacher wins `3/3` with mean margin
  `+0.031648` and minimum margin `+0.028920`. Executed static rollouts never
  contain the old direct `laser+fc4` stack, while teacher uses `16--17` unique
  masks. This is sufficient to accept v5/e70 as the next algorithm-development
  scene.
- Static candidate labels are not the same as executed static schedules. Seed44
  selected raw candidate `met_station_core|laser_disdrometer|fc4_flux`, but
  the rollout executed `met_station_core|laser_disdrometer` for `936/1024`
  steps and `met_station_core` for `88/1024` steps, with actual `fc4` duty
  `0`. Therefore all future "static direct stack" diagnostics must use
  `selected_masks` from rollout artifacts rather than `selected_static_sensor_ids`
  alone.
- The calibrated scene activates a useful energy asymmetry: static laser duty
  remains high in seeds42/44 (`0.914`) but cannot combine laser with fc4,
  while teacher uses much lower laser duty on average (`0.1126`) and shifts
  work to fc4/context masks. This confirms that the dynamic value is mainly
  selective high-power sensing plus low-power temporal mixing, not brute-force
  continuous direct sensing.
- The `snow_particle_counter` is not yet a central mechanism in this gate.
  Teacher uses it only in seed42 (`0.1787` duty), while fc4/context switching
  dominates. This is acceptable for scenario calibration, but if the later
  paper claim emphasizes proxy-particle sensing specifically, the sensor
  role calibration must be tightened further.
- The first calibrated-scene deployable smoke rejects the simple
  event-threshold route as the immediate mainline. On v5/e70 seed41,
  teacher remains positive (`1.291953` vs static `1.306918`), but
  `forecast_aware_event_threshold` is slightly worse than static (`1.307823`).
  The failure is mechanistic, not just noise: deployable duty is essentially
  `met_station_core|surface_temp_ir` (`1014/1024` steps) with no fc4/laser,
  while the teacher uses fc4 duty `0.625`, laser duty `0.042`, and `15`
  unique masks.
- In that seed41 smoke, the deployable and static have identical
  task-composite task error (`0.607612`), while the teacher reduces it to
  `0.456508`. Therefore the deployable is optimizing a low-power variant of
  the static context stack rather than recovering the teacher's forecast-value
  mechanism. The next deployable candidate should preserve teacher temporal
  mixture or sensor-duty structure, not another single event-threshold action.
- Contextual-duty under v5/e70 seed41 is a negative gate under the current
  objective but a positive mechanism result under physical forecast metrics.
  It loses the current `task_composite` objective (`1.311998` vs static
  `1.306918`) because its frozen-oracle loss is higher, but it improves MAE
  (`3.584` vs `5.528`), RMSE (`18.475` vs `20.404`), DTW (`2.741` vs
  `5.518`), and task error (`0.5163` vs `0.6076`). Unlike event-threshold, it
  actually recovers teacher-like fc4/laser/context mixing.
- Zero-cost objective sensitivity on the completed contextual rollout shows
  the decision boundary is around task-error weight `0.256`: at `w=0.25`
  contextual-duty still barely loses by `-0.000513`, while at `w=0.30` it wins
  by `+0.004054`. Event-threshold remains negative for all tested weights
  because its task error is identical to static. Therefore the next correction
  should be objective-weight calibration toward physical task error, not
  another deployable architecture.
- A clean `w=0.30` seed41 rerun passes, but the mechanism changes. The
  validation-selected static anchor becomes
  `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`, and validation
  selects event-threshold rather than contextual-duty. The final deployable
  alternates between that static anchor and `met_station_core|laser_disdrometer`
  (`laser` duty `0.582`), beating the static anchor on the task-composite
  objective (`1.344876` vs `1.373655`) while the teacher remains better
  (`1.314480`).
- The `w=0.30` pass is not a general "all forecast errors are lower" result.
  The deployable improves the configured snow/task error (`0.2816` vs static
  `0.5315`) but has worse broad MAE/RMSE than the static context/flux stack.
  Therefore any eventual claim under this route must be explicitly
  task-targeted: adaptive scheduling improves control-relevant snow/event
  forecast objective under a calibrated energy-constrained scene.
- The v5/e70 `w=0.30` seed41/42/44 small gate fails as a deployable claim:
  deployable `1/3`, teacher `3/3`, mean deployable margin `-0.005130`.
  This does not invalidate scenario calibration because the privileged teacher
  still beats static in all three seeds; the blockage is deployable
  compression/selection.
- The failure is anchor-regime specific. Seed41's static anchor is a
  proxy/fc4 stack, so adding selective `core+laser` improves the configured
  task error. Seeds42/44 already have laser in the static anchor, so the
  selected deployables lower power and improve oracle/broad metrics but worsen
  the configured snow-task error. A single global scalar objective cannot
  resolve this: seed41 breaks even at task weight `0.1848`, while seeds42/44
  break even only below about `0.142`.
- The next useful correction is not another global task-error weight or a
  broader rerun. It must be conditional on the static anchor / sensing regime:
  proxy/fc4 anchors need task-weighted selective laser; laser anchors need a
  teacher-compression mechanism that preserves task accuracy while recovering
  the teacher's lower-power temporal mixture.
- The v5/e70 teacher-mix diagnostic closes the "use existing temporal teacher
  compression" shortcut. It also fails at deployable `1/3` while teacher stays
  `3/3`. Seed42 did select `forecast_aware_teacher_cycle`, but final objective
  still lost to static (`1.201180` vs `1.185582`). Seed44 again selected
  event-threshold and lost (`1.164705` vs `1.138113`). Therefore the current
  scene still gives static too much direct measurement strength in laser-anchor
  regimes.
- The remaining scene-design issue is sharper than before: v5/e70 prevents
  continuous laser+fc4 co-activation, but it still allows a static
  `core+laser` anchor to cover the snow-task targets well enough. Dynamic
  policies can save power and improve frozen-oracle/broad errors, yet lose the
  configured task error because static laser is a strong direct observation of
  the target. A truly complex scheduling scene must make every single static
  direct stack incomplete: either laser alone is insufficient, or its continuous
  duty is infeasible, or task value shifts among sensors/regimes fast enough
  that one fixed mask cannot cover all regimes.
- Partial v6 static-only evidence is positive before the teacher summaries are
  available. In the seed41/42/44 event-rich calibration run, validation-selected
  static execution has `laser_disdrometer` duty `0.0` in all three seeds and
  `core+laser` duty `0.0` in all three seeds. Static anchors have moved to
  fc4/context/proxy stacks: seed41 and seed44 select
  `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`, while seed42
  selects `met_station_core|surface_temp_ir|snow_particle_counter|fc4_flux`.
  This means v6 has structurally removed the old continuous `core+laser`
  snow-task shortcut; the remaining acceptance condition is whether the MPC
  teacher beats these new static anchors with non-marginal margins.
- The completed v6 event-rich static/teacher gate passes formally but should
  not yet unlock deployable training. Teacher wins `3/3` and the static
  direct-stack execution count is `0`, so the `core+laser` blockage is fixed.
  However seed44 has only `+0.0141` teacher margin and teacher task-error is
  worse than static (`0.4193` vs `0.3630`), with the objective win coming from
  oracle loss. A window diagnostic explains the weakness: event-rate-only
  selection gives seed44 lower particle/transport variability than the other
  seeds. A transport-aware event selection raises seed44 diameter std from
  `0.049` to `0.088` and velocity std from `2.56` to `3.89` while retaining
  high event rate (`0.742`). The next calibration should use this harder
  predefined selection before accepting the scene.
- The `event_transport_rich` v6 gate resolves the seed44 caveat and is accepted
  as the current development scene. Teacher beats static in all three seeds
  with mean margin `+0.0981` and minimum margin `+0.0484`; static execution
  never uses the old direct stacks and static laser duty is `0.0`. More
  importantly, teacher task-error is lower than static in all three seeds:
  margins `+0.1287`, `+0.2380`, and `+0.2148`. The mechanism is not selective
  laser; teacher laser duty is `0.0`. The dynamic value now comes from
  switching among SPC, fc4, radiometer/surface/context, and wind/thermo
  support masks. This is a cleaner complex scheduling scene for the algorithm,
  but the eventual claim must be framed as proxy/context/flux complementarity
  under transport-rich event windows.
- The first deployable smoke on the accepted v6/event-transport scene fails,
  despite the teacher remaining strong. The contextual-duty preset is
  validation-selected down to `forecast_aware_event_threshold` in all three
  seeds; deployable wins only seed42 and has mean margin `-0.0054`. This is not
  a scene failure: teacher margins are `+0.0933`, `+0.0968`, and `+0.1287`.
  The failure is student-compression specific. A two-mask event-threshold
  student overuses coarse SPC/fc4/context swaps and cannot reproduce the
  teacher's 15--18-mask mixture. Contextual-duty itself also has worse
  validation objective than event-threshold in all seeds. The next mechanism to
  test should preserve teacher duty rates or mask sequence structure directly.
- The teacher-mix follow-up closes the average-duty/cycle shortcut under the
  accepted v6/event-transport scene. It also fails at deployable `1/3` with the
  same mean margin `-0.0054`, and validation again selects
  `forecast_aware_event_threshold` in every seed. Teacher-rate has negative
  validation static-margin means in all seeds, while teacher-cycle is unstable
  and catastrophically worse on seed42. Therefore the remaining student
  bottleneck is not marginal teacher duty matching; it is preserving
  stateful/objective-aware multi-mask sequencing.
- Sequence-mask imitation closes the pure teacher-label imitation shortcut
  under the same scene. The model reaches near-perfect teacher-mask accuracy,
  but validation still selects event-threshold in all seeds and the aggregate
  remains deployable `1/3`. This separates "can imitate teacher labels on the
  collected states" from "can deploy a better closed-loop scheduler"; the
  remaining plausible branch must train on objective/cost structure, not only
  mask identity.
- The first recurrent objective-aware value student is also insufficient. It
  trains on candidate rollout costs instead of mask identity, but each seed
  has only `512` recurrent cost rows and best-action accuracy stays low. The
  only recurrent-selected final case, seed44, exactly matches the static
  objective, revealing a no-op fallback rather than a useful dynamic schedule.
  A fair final check for this tier needs rank-aware training, positive-margin
  guards, and denser train-start coverage; if that fails, the current student
  interface is the blockage, not just a hyperparameter choice.
- The rank-aware recurrent correction has a narrow interpretation. Increasing
  `train_rollouts` from `4` to `12` should raise recurrent cost rows from
  roughly `512` to `1536` per seed, so a continued failure would not be
  attributable to the earlier tiny recurrent dataset alone. The positive
  margin guard disables recurrent-value as a candidate if its own validation
  calibration fails, but it does not force recurrent-value to be selected over
  event-threshold. Therefore a repeat of event-threshold selection would mean
  the current objective-aware student still cannot produce a validation-clean
  dynamic improvement; the next correction should change the teacher/student
  interface rather than retune this preset again.
- The active rank-aware recurrent dense run has already answered the main
  recurrent question before final replay completes: with `1536` recurrent rows
  per seed, rank loss improves best-action accuracy into the `0.37--0.48`
  range, but recurrent-value still fails its own positive paired
  static-margin guard in all three seeds and is disabled. This closes
  data-scarcity as the primary explanation. If the final aggregate fails, the
  next useful work is not another recurrent-value hyperparameter sweep; it is
  a different teacher/student interface, most likely one that exposes
  candidate rollout costs or deploys an online objective-conditioned planner
  instead of compressing the teacher into a single recurrent scorer.
- Recurrent cost-DAgger improves supervised ranking accuracy but still does
  not solve deployment. Adding one cost-DAgger pass doubles recurrent rows to
  `3072` and raises best-action accuracy to roughly `0.47--0.59`, yet the
  recurrent-value policy still fails the positive paired static-margin guard
  in all three checked seeds. This separates "learning teacher-cost rankings
  on more on-policy states" from "achieving validation-clean improvement over
  the static anchor". Unless final replay unexpectedly changes the aggregate
  through older fallback heads, the next algorithmic correction should move
  away from single-step recurrent scoring toward explicit online planning or
  a teacher that emits deployable causal subgoals/options.
- Final cost-DAgger aggregate confirms that conclusion: deployable remains
  `1/3` while teacher is `3/3`, with mean deployable margin `-0.005538` and
  teacher margin mean `+0.110397`. The accepted scene is no longer the main
  blockage: continuous `core+laser` is gone, static laser duty is `0`, and the
  teacher's dynamic value is robust. The unresolved problem is the deployable
  interface. A useful next implementation should expose a small causal option
  set or online objective-conditioned planner with static fallback, dwell/entry
  guards, freshness/SOC state, and transport-risk features, instead of
  compressing the teacher into one recurrent action scorer.
- The first concrete version of that interface is now implemented as an
  online option-planner student. It is deliberately not another supervised
  mask/value head: teacher labels define a compact option support and target
  duty rates, but deployment chooses among options using only causal learned
  event forecasts, current freshness, SOC, sensor power, switch cost, and
  transport-role priors. Validation still owns all parameter selection. The
  key test is not whether this policy imitates the teacher better, but whether
  it can pass the positive validation static-margin guard and improve the
  seed41/42/44 final gate where recurrent cost learners failed.
- The first option-planner gate is a partial but insufficient improvement.
  It fails formally at deployable `1/3`, teacher `3/3`, but the mean
  deployable margin is only `-0.000289` rather than the recurrent
  cost-DAgger tier's `-0.005538`. Seed44 is a real final win
  (`+0.007593`), seed41 is correctly blocked by the positive validation guard
  and falls back to static, and seed42 is the important failure: option-planner
  passes validation guard but loses final by `-0.008462`. This changes the
  bottleneck from "student cannot produce a dynamic policy at all" to
  "validation-clean causal option policies do not transfer reliably to final
  windows." The accepted scene remains effective because teacher margins are
  still `3/3` with mean `+0.110397`.
- The next useful diagnostic should focus on option-risk transfer, not on
  scaling the same preset. Specifically: compare seed42 validation and final
  option activation timelines, option duty rates versus teacher rates, event
  probability distribution shift, start-level margin patterns, and whether the
  static-margin guard should incorporate q25/risk-band constraints or final-like
  transport diversity in validation selection. A broader n=5/n=15 run of the
  current option-planner would mostly measure this known transfer defect.
- The rate-balanced option-planner smoke closes the simplest duty-matching
  correction. Validation selected `rate_balance_weight=3.0` for seed42, the
  previous final-transfer failure, but final performance worsened
  (`-0.011043` margin versus `-0.008462` before). Seed42's radiometer duty did
  move toward the teacher (`0.799 -> 0.659`), but event-time surface duty moved
  further away (`0.075 -> 0.018`) and final oracle loss stayed worse than the
  static anchor in all four final windows. Seed44 also moved from a previous
  final win to a strict-guard fallback because the balanced option had a
  negative validation start below the guard floor. Therefore the remaining
  issue is not average teacher-duty mismatch. It is state/window-conditioned
  option value and target-channel allocation under validation-to-final regime
  shift. The next useful algorithmic correction should estimate option
  transfer risk or causal rollout value at the start/window level; another
  duty/rate heuristic is unlikely to move the claim.
- Root-aware start-transfer audit reinforces this conclusion. Balanced
  option-planner only wins `1/4` seed42 final starts with mean margin
  `-0.011188`, whereas the old option-planner selected rows had `4/8` starts
  and near-zero mean (`-0.000425`) because seed44 contributed three positive
  starts. The balanced branch therefore did not merely fail the aggregate
  threshold; it degraded the window-level distribution of final margins.
- A pure learned rollout-value planner is not the missing transfer-risk
  mechanism. Under the accepted v6/event-transport scene, the strict
  rollout-value preset fails validation in all three original seeds before
  final transfer is relevant: validation mean margins are `-0.056379`,
  `-0.032013`, and `-0.072909`, with guard pass false in every seed. Teacher
  still beats static `3/3`, so the scene remains valid; the failure is the
  learned short-horizon cost/transition planner interface. The next useful
  correction should reuse the option-planner branch, where seed44 had a real
  final win, and add start/window-level risk selection to decide when the
  option policy is safe enough to leave the static anchor.
- A pure option-planner startguard also fails. Tightening the run-level guard
  to require zero negative validation starts does not preserve the earlier
  seed44 win; under a pure option preset, the calibrated option policy has two
  negative validation starts in all three seeds. This reveals that the earlier
  seed44 win is not a robust property of the option-planner family alone, but
  depends on the older hybrid calibration/support path. Another run-level
  threshold tweak is unlikely to solve the deployable claim. The next useful
  mechanism should make the guard operational at runtime/window level or
  expose more of the MPC teacher's decision structure.
- Switching-pattern audit clarifies the current v6/event-transport behavior.
  Static masks have no switching and keep three or four sensors always on, but
  with laser duty `0`. Deployable students still look like a compressed fixed
  core: event-threshold averages `3.33` always-on sensors and switches on only
  `7.16%` of steps; option-planner averages `2.67` always-on sensors and
  switches on `23.92%`, with three-or-more simultaneous sensor toggles only
  `0.59%`. The MPC teacher is qualitatively different: only
  `met_station_core` is always on, while fc4/SPC/context/wind/thermo channels
  rotate; any switch occurs on `70.45%` of steps, two-or-more switches on
  `51.85%`, and three-or-more on `15.97%`. Therefore the accepted scene does
  create a genuinely dynamic target schedule; the unresolved issue is that the
  deployable policy interface still collapses it into mostly static support.
- The first full validation-selected cyclic/dwell baseline result is negative.
  On v6/event-transport seed41, validation selected dwell `16` over masks
  `97|106|116|107`, but final objective was `1.251438` versus static
  `1.214565`; teacher remained strong at `1.121306`. The heuristic had zero
  warmup aborts but used more power than static (`0.751250` vs `0.620000`) and
  worse task error. This suggests that merely cycling among good static masks
  with dwell constraints is not enough; useful dynamics require
  state/window-conditioned selection, not blind schedule diversity.
- Seed42 repeats the same conclusion with a subtler tradeoff. The selected
  cyclic/dwell baseline used dwell `8` and achieved lower task error than
  static (`0.465526` vs `0.481737`) with lower power (`0.808125` vs
  `0.840000`), but its oracle loss rose and the final task-composite objective
  still lost (`1.314925` vs static `1.301378`). Therefore the heuristic is not
  merely "too conservative"; it optimizes the wrong temporal allocation under
  the frozen forecast objective.
- The completed 3-seed cyclic/dwell baseline closes blind dwell cycling as a
  serious competitor. It loses to static in all checked seeds: margins
  `-0.036873`, `-0.013547`, and `-0.023658`, with mean `-0.024692`. In the
  same runs, the MPC teacher wins all three with mean margin `+0.106255`.
  Mean cyclic power is also higher than static (`0.790833` vs `0.740000`) and
  mean task error is worse (`0.454651` vs `0.426085`). This strengthens rather
  than weakens the current diagnosis: the dynamic value is real, but it is not
  captured by a simple dwell-constrained round-robin over good static masks.
- The first runtime/window-level risk-guard student smoke is not a clean
  algorithmic verdict. Aggregate result was formally negative
  (`1/3` deployable wins, mean margin `+0.000349`, teacher `3/3`), but it
  exposed a validation replay inconsistency: runtime-risk calibration used
  different static/candidate `seed_offset` ranges from final deployable
  selection. Seed42 is the clear symptom: the calibrated runtime-risk row had
  positive margins on all validation starts (`mean +0.008975`, `min
  +0.001625`, `0` negative starts), but the later validation-selection replay
  rejected the same policy after margins flipped negative. The immediate fix is
  paired replay semantics, not another threshold sweep: static and candidate
  rollouts for the same validation start must share the same deterministic
  seed offset, aligned with deployable selection.
- The paired-replay runtime-risk rerun gives the clean verdict for this
  mechanism: it fails under the accepted v6/event-transport scene. Aggregate
  result is deployable `0/3`, mean margin `-0.003977`, teacher `3/3`.
  Seed42 and seed44 both pass validation selection and deploy runtime-risk, but
  both lose on final. Start-level final audit over those selected runs is
  `2/8` wins with mean margin `-0.006127` and median `-0.014430`. Therefore
  the remaining issue is not replay-noise or absence of a runtime guard; it is
  validation-to-final transfer of the guard's causal risk signal. A dense
  validation/risk-band check is worth one run to test sample scarcity, but a
  second failure should stop runtime-risk threshold tuning.
- Read-only audit of the current student interface shows why further
  threshold-only runtime guards have low expected value. `ForecastAwareRuntimeRiskGuardPolicy`
  only gates an existing option planner on a scalar causal risk score, while
  the teacher code can already compute per-candidate first-action rollout
  costs through `beam_search_first_action_costs`. If dense validation also
  fails, the next mechanism should use those teacher cost distributions or
  teacher short-horizon option sequences as the training signal, rather than
  reducing the teacher to labels, duty rates, or a scalar event-risk gate.
- Dense-validation runtime-risk fails more strongly than the paired rerun: it
  rejects every runtime-risk deployable before final deployment. With twelve
  validation starts and q25/risk-band checks, the only retained rows are
  static-equivalent zero-margin policies; non-static runtime-risk rows have
  negative validation margins. Teacher still wins all three seeds with mean
  margin `+0.110656`, so the scene is still valid. The deployable blockage is
  the student interface, not validation sample scarcity.
- The one-step teacher-cost memory interface is also closed. The cost-KNN
  risk-band gate stores teacher first-action cost vectors and uses causal
  nearest-neighbor retrieval, but no non-static candidate survives validation
  in seeds `41/42/44`. Best validation mean margins are negative in every seed
  (`-0.033282`, `-0.020071`, `-0.045767`), and each best candidate still has
  many negative validation starts. This separates "having teacher first-action
  cost distributions" from "deploying a useful dynamic scheduler." The next
  interface must preserve longer teacher temporal structure, such as
  trajectory snippets or macro-option sequences, not only one-step labels,
  cost vectors, scalar risk scores, or average duty rates.
- The macro-option sequence student is the first test of that longer temporal
  interface. It deliberately does not learn another action classifier: it
  stores teacher label snippets from train-split rollouts and uses causal
  nearest-neighbor state matching plus learned event-risk gating to decide
  which snippet to replay. This should be interpreted as a test of whether
  preserving teacher temporal mixing improves deployable transfer. If it
  fails under the same risk-band validation guard, the next correction should
  step back to objective/forecast-oracle transfer diagnostics rather than
  adding another shallow supervised head.
- Macro-option did fail under that guard. The best validation rows were
  static-equivalent (`event_threshold=1.0`) in all three seeds, while non-static
  macro-option rows were negative before final replay. This closes the current
  "expose more teacher structure" tier: labels, duty rates, scalar risk,
  recurrent one-step values, one-step cost memory, and short teacher snippets
  have all failed to produce validation-clean deployable improvement. Since
  the privileged teacher remains strongly positive, the useful next analysis is
  not another compression interface, but why deployable causal policies cannot
  transfer the teacher's dynamic objective into the frozen-forecast validation
  objective. Candidate root causes are learned event forecast misalignment,
  frozen-oracle loss sensitivity, static-anchor selection bias, and the fact
  that the teacher uses privileged state transitions while students only see
  weak causal risk features.
- Objective-transfer audit over the three latest v6 roots confirms the scene is
  not the blocker. The MPC teacher beats static in `3/3` seeds and `12/12`
  final windows with mean objective margin `+0.110656`. The improvement is
  composed of mean oracle-loss margin `+0.043868` and weighted task component
  `+0.066789` (`task_error_weight=0.3`), so about `60.4%` of the objective lift
  comes from event-transport task-error reduction and about `39.6%` from the
  frozen oracle itself.
- The audit also exposes an artifact/design gap: learned-event forecasting is
  enabled, but augmented `learned_event_p_h*` columns are not saved into the
  truth artifacts, and `teacher_dataset.npz` lacks feature names. Therefore the
  current artifacts cannot directly test whether learned event probabilities
  align with teacher-improvement windows. Future runs should persist auditable
  deployable context before another policy-interface tier is attempted.
- The artifact gap is now patched for future runs. `run_protocol_gate.py` saves
  `truth_with_learned_event_forecast.csv` after learned-event augmentation, and
  `TeacherDataset` persists feature names including the appended event forecast
  probability/time/confidence slice. This is not a behavioral improvement by
  itself; it makes the next objective/forecast-transfer diagnostic falsifiable.
- Teacher-improvement alignment audit was run on the server after reconstructing
  learned-event forecasts for the old v6 macro-option root. The signal is weak
  but nonzero: step-level AUC is `0.606/0.586/0.517` for seeds `41/42/44`, with
  positive event-probability gaps in all seeds. This rejects the idea that
  learned-event probabilities are useless, but it also rejects immediate
  scaling. Branch F should therefore be a guarded smoke: train a direct
  teacher-improvement entry gate and let it decide when to open the dynamic
  option/macro policy.
- Branch F is now closed. The teacher-improvement gate smoke failed at
  deployable `0/3` while the MPC teacher stayed `3/3`. The failure is
  diagnostic: per-step first-action cost-margin labels are not equivalent to
  the MPC teacher's sequence-level advantage. Seed44 is the clearest case:
  the training labels were all negative (`positive_rate=0.0`) even though the
  final MPC teacher beat static by `+0.100440`; the selected gate was a
  near-static/warmup variant and lost final by `-0.007250`. Seeds 41/42 had
  many positive train labels (`0.9316/0.7832`) but every validation-calibration
  row had negative mean margin, so validation rejected the deployable. Further
  scalar event/first-action threshold wrappers are low value. The next
  correction must expose window/sequence-level teacher value or replace the
  privileged teacher with a deployable learned-world-model planner.
- The new window-level teacher-value audit gives the first direct evidence
  that the Branch-F failure is not caused by absent teacher value on the
  validation split. For seed41, paired replay of validation-selected static
  versus MPC teacher is positive in `20/20` declared windows: train `4/4`
  with mean margin `+0.156649`, validation `12/12` with mean `+0.079045`
  and minimum `+0.034804`, and final `4/4` with mean `+0.076549`. Therefore
  the deployable failure is not "validation has no dynamic value"; it is that
  the student candidate family produces negative validation margins despite a
  clean positive teacher target. The audit is still running for seeds `42/44`.
- The same window-level audit has now completed seed42 with the same result:
  `20/20` declared windows are teacher-positive. Seed42 mean margins are
  train `+0.058332`, validation `+0.069905` with minimum `+0.041044`, and
  final `+0.072852`. Across completed seeds `41/42`, the static-vs-teacher
  target is positive in `40/40` train/validation/final windows. This makes the
  current bottleneck sharper: validation is rich enough to identify teacher
  value, but existing deployable students still produce validation-negative
  schedules. The next model should learn/predict teacher sequence value itself
  or plan with a deployable learned dynamics model, not gate a weak student by
  a scalar event or first-action proxy.
- The completed window-level audit extends this to seed44 and closes the
  teacher-value question: MPC teacher beats validation-selected static in
  `60/60` declared windows across seeds `41/42/44` and train/validation/final.
  Validation mean margins are `+0.079045`, `+0.069905`, and `+0.096935`; the
  worst validation window is still positive (`+0.034804`). Oracle-loss margins
  and task-error margins are both positive in every seed/split summary. This
  strongly supports the current scenario and teacher objective. The failure
  mode is now specifically deployable causal control: our students can see
  training labels/proxies but do not model the sequence-level value that the
  teacher obtains through closed-loop planning.
- Interface audit during the dense macro smoke sharpens the next direction:
  `ForecastAwareMacroOptionPolicy` selects a train-split teacher snippet by
  nearest-neighbor feature similarity and then replays the first feasible labels.
  That preserves short teacher temporal structure, but it still has no model
  of the sequence's expected objective margin relative to the static anchor.
  Therefore a failure of the dense always-dynamic macro branch should be read
  as evidence against similarity-only snippet transfer, not against the broader
  idea of sequence-level deployable control. The next meaningful mechanism
  should train on sequence/window value itself: feature plus candidate action
  sequence -> predicted static-anchor margin, with validation-calibrated
  execution thresholds.
- The implemented sequence-value head operationalizes that diagnosis without
  changing the accepted scene or privileged teacher. Training data include the
  teacher future snippet, the static anchor sequence, and sampled teacher
  snippets evaluated from the same causal state. This is a stronger interface
  than first-action cost or macro retrieval because the model's target is the
  actual sequence-level advantage that the deployable policy needs for
  validation transfer.
- Dense always-dynamic macro retrieval failed cleanly and should not be
  retuned further. Even after removing event entry gating and broadening
  snippet support, all `36` validation-calibration rows per seed had negative
  mean margin over the static anchor (`0` positive rows in seeds `41/42/44`).
  The best rows were still negative (`-0.018479`, `-0.008884`, `-0.005069`),
  while the MPC teacher remained positive in all seeds. This closes
  similarity-only teacher-snippet transfer; the next route must learn
  candidate sequence value or use a deployable learned dynamics/planning model.
- The first sequence-value student also fails the deployable gate, but its
  failure is more informative than macro retrieval. The train dataset contains
  many positive static-anchor advantage rows (`34--48%`), and the value model
  reaches low SmoothL1 losses around `0.004`. However, validation calibration
  remains negative or unstable: seed41/42 have no positive-mean threshold row,
  and seed44's best positive-mean row still has negative q25 and many negative
  starts. This indicates a train-to-validation selection transfer failure,
  not absence of sequence-level labels. Before closing the route entirely,
  rule out two implementation-level bottlenecks: scoring only the first
  `128` snippets of a `369--380` row sequence bank, and using an advantage
  threshold grid capped at `0.1`.
- The full-bank/high-threshold sequence-value diagnostic rules out those two
  implementation-level bottlenecks. Scoring the full sequence bank and
  extending thresholds to `0.5` still yields deployable `0/3`. The selected
  risk rows mostly become static-equivalent fallback; the only nonzero selected
  validation margin is seed42's `+0.000346`, too weak for positive-center
  semantics. Sequence-value should be closed for now. The next useful question
  is not "which threshold/snippet length", but whether the deployable causal
  context is too weak: the teacher's dynamic value is present, yet students
  cannot identify safe dynamic windows from learned-event features.
- Read-only implementation audit during the oracle-context run clarifies what
  this diagnostic can and cannot prove. The current `ForecastContextConfig`
  exposes only event probabilities, time-to-event, and confidence; even
  `truth_future=True` gives perfect future event flags, not continuous future
  transport intensity, particle microstructure, wind-regime trajectory, or
  task-error proxy forecasts. `ForecastAwareSequenceValuePolicy` scores
  `current causal feature + flattened candidate mask sequence` and starts the
  highest predicted sequence whenever its advantage exceeds a scalar threshold.
  Calibration chooses one global threshold over validation starts; it does not
  learn a separate start/window-level "dynamic allowed" selector. Therefore:
  if oracle-context improves materially, the next route is richer deployable
  forecast context; if it still fails, the bottleneck is broader than event
  timing and likely requires a causal learned-world-model / per-window planner
  or a regime-conditioned dynamic-eligibility layer rather than more
  sequence-threshold tuning.
- The oracle-context result confirmed the latter branch. With perfect future
  event flags, sequence-value still fails at deployable `1/3` and negative
  mean margin. Seed42's small win is not enough evidence (`+0.000896` final
  margin, validation q25 `-0.001807`), and seed44 is the decisive negative:
  validation mean margin `+0.001359` selected dynamic threshold `0.0`, but
  final margin fell to `-0.015432`. Therefore the deployable missing signal is
  not merely event timing. The next useful target is richer continuous
  regime/task context and/or an online per-window eligibility model that can
  refuse dynamic execution in seed44-style validation-fragile regimes.
- Follow-up code audit found that the sequence-value calibrator was too
  permissive: when the risk-band/positive-center selector returned no passing
  row, `calibrate_sequence_value_policy` still fell back to the best invalid
  row and left the candidate enabled. This explains why oracle-context seed44
  executed a dynamic sequence with validation q25 `-0.017638` and `6`
  negative validation starts. The fix is conservative and split-compliant:
  no passing validation row disables the sequence-value deployable, forcing
  static fallback. Any future sequence-value result must use the fixed
  semantics before being interpreted.
- The fixed oracle-regime diagnostic closes sequence-value retrieval as a main
  route. Privileged continuous summaries improved safety and produced one real
  small win (seed41, `+0.003042`), but did not recover the teacher's
  `+0.080286` mean margin: seed42 was disabled by the risk band, and seed44
  was final static-equivalent. This means the failure is not just missing
  event or continuous context; the deployable interface is too weak. It
  retrieves/replays train teacher snippets behind a global validation
  threshold instead of doing per-window causal verification or planning.
  Further context additions to the same sequence-value retriever should be
  deprioritized.
- Read-only audit of the active rollout-value planner found no obvious
  calibration-layer bug: it evaluates thresholds per validation start, writes
  paired static margins, and disables the candidate if no risk-band row passes.
  The likely algorithmic weak points are instead inside the learned planner:
  it learns raw absolute action costs along the teacher-state distribution,
  learns one-step feature deltas, then expands future actions only from the
  teacher-label support set without rechecking predicted future feasibility
  under energy/warmup constraints. If the current run fails by invalid or
  fragile validation rows, the next correction should be constraint-aware /
  self-rollout data collection or static-anchor margin modeling, not another
  threshold-only change.
- The rollout-value oracle-regime run confirms that diagnosis strongly enough
  to reject the current absolute-cost planner. Even with privileged event and
  continuous future context, all validation threshold rows are negative in
  all seeds (`best` mean margins: seed41 `-0.043041`, seed42 `-0.022456`,
  seed44 `-0.012741`). This is not a final-test transfer issue: validation
  already says the deployable planner is worse than the static anchor, and
  the guard correctly disables it. The remaining teacher gap is therefore not
  solved by "raw cost + learned feature delta + short beam"; the next useful
  interface must model static-anchor advantage / dynamic eligibility directly
  or train the planner under its own rollout distribution with explicit
  future feasibility constraints.
- The oracle-regime direct anchor-advantage result rules out the simplest
  "just learn static-anchor margin per action" correction. Best validation
  margins are still negative in all seeds (`-0.006727`, `-0.015483`,
  `-0.013301`) with negative q25 margins and `7--9` negative starts. Because
  this diagnostic used privileged future event/continuous regime context and
  low train losses for seed41/42, the missing interface is not merely richer
  causal context or a different scalar action target. The teacher advantage is
  window/trajectory-level; one-step learned action scoring, whether absolute
  cost or anchor advantage, does not preserve it.
- The new window-eligibility route explicitly changes the target from
  first-action/one-step value to deployable student window advantage. Its
  training memory is built by paired replay of the same dynamic student and
  the validation-selected static anchor over training windows; validation then
  chooses a KNN margin threshold under the same risk-band semantics. This
  should be interpreted as a test of whether the existing deployable
  option-planner has identifiable safe windows, not as a final learned-world
  model. If its calibration is uniformly negative, the blockage is the dynamic
  option executor itself, not the gate.
- The completed window-eligibility smoke shows the existing option-planner
  executor has some identifiable positive windows but not enough tail safety.
  Seed41/42 achieve positive mean validation margins, yet fail risk-band
  gating because negative starts remain above the allowed count; seed44 is
  weaker and keeps negative q25. This moves the bottleneck one level down:
  the next useful change should replace the dynamic candidate/executor with
  teacher-sequence/window candidates or self-rollout planning, rather than
  adding another scalar gate on top of the same option-planner.
- Replacing the window-gate inner executor with `ForecastAwareMacroOptionPolicy`
  does not solve the deployable interface. The formal 3-seed macro-window
  diagnostic completed at deployable `0/3`, teacher `3/3`, with mean
  deployable margin `0.0` and teacher margin `+0.080286`. Best validation
  rows had positive mean margins (`+0.007506`, `+0.020843`, `+0.005176`) but
  failed risk-band selection: seed41/44 retained negative q25 and `4`
  negative starts, and seed42 still had `2` negative starts despite positive
  q25. This closes similarity-based teacher-snippet execution under a scalar
  KNN window gate. The next route must perform per-window causal verification
  or self-rollout planning with constraints under the executed distribution,
  not just retrieve/replay teacher snippets.
- The rollout-value self-distribution diagnostic rules out the simplest
  covariate-shift explanation for the learned planner failure. Self-rollout
  collection doubled the transition rows and nearly doubled action-cost rows,
  and retraining improved nominal losses, but validation margins became
  uniformly negative in all seeds. Best mean margins were seed41 `-0.019960`,
  seed42 `-0.013485`, and seed44 `-0.036263`, with negative q25 margins and
  `7--10` negative starts. The problem is therefore not just that the planner
  was trained on teacher-state distribution; the absolute action-cost plus
  one-step feature-delta planning interface is itself misaligned with the
  static-anchor margin objective. More self-iterations should be deprioritized
  until the objective/interface changes.
- Feature-interface audit found a real deployable information limitation:
  before the new infrastructure, deployable policies could consume learned
  event probabilities but continuous future context was either oracle
  `continuous_truth_future=True` or a causal persistence fallback. The new
  learned continuous forecaster closes that infrastructure gap in a
  split-compliant way by training on oracle-pretrain plus RL-train bounds and
  writing causal prediction columns consumed by the same
  `ForecastContextConfig`. This does not by itself fix the rejected policy
  interfaces; it makes the next direct window/sequence outcome verifier a
  fairer deployable test because it can condition on forecasted transport and
  task intensity rather than event timing alone.
- The augmented sequence-value interface is the first new post-blockage
  candidate that changes both the input signal and candidate family: it trains
  directly on `(causal forecast state, candidate action sequence) ->
  static-anchor window margin`, while exposing teacher snippets, static
  constant sequences, and teacher-support cycle/dwell sequences to the same
  verifier. This is still not a learned digital twin, but it is a stronger
  test than scalar window gates because the sequence itself is scored as part
  of the outcome model. The tiny smoke only validates plumbing; the formal
  3-seed diagnostic is needed to decide whether this interface has real
  transfer value.
- The formal augmented sequence-value diagnostic is negative and closes this
  branch as the main deployable route. Learned event plus learned continuous
  context, richer candidate banks, and direct sequence scoring still produced
  deployable `0/3`, while the privileged teacher stayed `3/3` with mean margin
  `+0.085283`. The important detail is not model underfitting: the sequence
  model trained on `7.6k` rows per seed with low loss and nontrivial positive
  rates, but validation-tail safety failed. Seed42/44 had tiny positive mean
  validation rows, yet q25 margins were negative and `5--8` starts lost to
  static. This points away from more snippet generation or threshold tuning
  and toward a learned digital-twin / static-anchor margin objective that
  models executed rollout outcomes and tail risk directly.
- The first executed-step learned-digital-twin smoke validates the new data
  path but not the algorithm claim. It successfully trained rollout
  cost/transition models from actually projected static-anchor, MPC-teacher,
  and random step outcomes. The rollout-value calibration row passed on the
  single smoke validation start (`+0.04538` margin), yet the subsequent unified
  deployable-selection replay fell back to static because the same calibrated
  policy became static-equivalent under a different validation seed offset.
  This is expected noise for a one-start smoke and reinforces that formal
  interpretation must use multi-start risk-band validation; it does not by
  itself prove the learned-twin interface succeeds.
- The formal executed-step learned-twin diagnostic is negative and closes the
  current absolute-cost planner correction. Each seed collected a balanced
  executed-outcome dataset (`4608` cost rows and `4608` transition rows) and
  trained low-loss cost/transition models, but validation rows remained
  negative or tail-unsafe in all seeds. Best rows were seed41 mean
  `-0.020325`, q25 `-0.045759`, `8` negative starts; seed42 mean
  `-0.002241`, q25 `-0.011505`, `7` negative starts; seed44 mean
  `-0.001792`, q25 `-0.011399`, `6` negative starts. Since the privileged
  teacher remains positive in all seeds, the blockage is not lack of dynamic
  value or teacher quality. It is that the deployable planner still optimizes
  learned one-step absolute step cost plus feature deltas, while the claim
  needs static-anchor window margin and tail-risk optimization.
- Implementation audit after that result identified the next non-redundant
  interface. The existing window-eligibility policy is close but too narrow:
  it builds a KNN memory of true train-window margins for one selected dynamic
  executor and then gates that executor. This cannot choose among candidate
  mechanisms per runtime window and therefore collapses when any single
  executor has poor validation tails. The next useful deployable policy should
  keep the true paired window-margin target, but lift the action space from
  "open/close one executor" to "select among multiple deployable candidate
  families using predicted lower-tail margin; otherwise anchor." This directly
  tests whether safe dynamic windows are identifiable when executor choice is
  part of the window-level decision.
- The first tiny smoke for the multi-candidate window-margin verifier confirms
  the plumbing but not the algorithm. It produced one positive train-window
  macro row and negative option rows, which is exactly the type of candidate
  heterogeneity the new interface is meant to exploit. However, the single
  validation start was negative for both threshold rows, so the risk gate
  disabled deployable execution. This is not formal evidence because the smoke
  used only one train and one validation start with very short windows. The
  meaningful test is the formal seed41/42/44 diagnostic with the full v6
  window grids.
- The formal local-window candidate-margin diagnostic is negative, but it
  reveals a more specific failure than "candidate generation is useless."
  Teacher remains strongly positive in all three seeds (`+0.103`, `+0.099`,
  `+0.139`), while the deployable is disabled in all three. Seed42 is the
  decisive implementation-level diagnostic: the local `16`-step calibration
  row has positive mean and q25 margin, but the unified full validation replay
  has q25 `-0.009918` and `5` negative starts. Therefore local window
  calibration can select policies that improve the first short segment but
  damage the full validation trajectory. Further local threshold tuning would
  be wasted; the calibration unit must match the full static-anchor validation
  horizon used by the claim.
- The full-rollout window-candidate patch closes that calibration mismatch
  without changing the deployed runtime policy. It keeps the same per-window
  candidate selection memory, but selects hyperparameters through full
  validation replay and writes q25/pass diagnostics to CSV. The smoke confirms
  plumbing only: full-rollout calibration can select and evaluate the
  candidate, but a one-start final replay still lost to static. The formal
  seed41/42/44 run is needed to decide whether the previous failure was mostly
  calibration-unit mismatch or whether the candidate family itself remains
  too weak.
- The formal full-rollout window-candidate diagnostic answers that question:
  the candidate family itself remains too weak. Full validation replay removed
  the short-window calibration artifact, and no row passed risk guard in any
  seed. The best mean validation rows were small positive in all seeds, but
  their lower tails stayed negative (`q25=-0.021212`, `-0.009918`,
  `-0.012833`). Since the privileged teacher again beats static in all seeds,
  the remaining gap is not scene validity or teacher value. It is that the
  deployable interface only gates a few hand-coded dynamic executors using
  nearest-neighbor window memories. The next non-redundant interface should
  use learned causal forecasts as the rollout substrate itself, approximating
  teacher-style short-horizon scoring without future truth.
- The first utility-planner smoke is positive but must not be overread. It
  proves the causal forecast utility interface is wired through the strict
  protocol: learned event/continuous forecasts are trained split-compliantly,
  validation risk-band calibration can select the planner, and final replay
  can beat the validation-selected static anchor on a tiny single-start case.
  The reason it is non-redundant is that it scores masks from forecasted
  variable intensity and sensor coverage directly, rather than selecting among
  teacher snippets or KNN window margins. The formal 3-seed run will decide
  whether this signal has stable validation tails or whether the smoke is just
  a short-window artifact.
- Read-only audit during the formal utility-planner run found no obvious
  calibration-unit mismatch: `calibrate_utility_planner_policy()` first
  evaluates the static anchor over all validation starts, then replays each
  utility hyperparameter row over the same validation starts and applies the
  same static-margin risk-band selector. This differs from the rejected local
  window-candidate route, whose first version selected on a shorter local
  window. If the utility planner fails, the likely cause is policy-score
  misspecification or insufficient action/state dynamics, not another
  short-window calibration artifact.
- The formal utility-planner diagnostic confirms that diagnosis. Hand-scored
  causal utility over teacher-supported masks is worse than the static anchor
  on validation in all three seeds. Seed44 is closest to neutral, but still
  has negative q25 and a large negative minimum start; seed41/42 are clearly
  negative. Meanwhile the privileged teacher again wins `3/3` with large
  final margins and much lower event-task error. This closes scalar utility
  scoring, KNN window gating, teacher-snippet replay, and one-step raw
  cost/advantage scoring as main routes. The remaining useful direction is a
  stronger planner/world-model interface whose explicit target is multi-step
  static-anchor improvement under the deployable policy's own projected
  rollout distribution.
- The task-only proxy-MPC smoke is the first post-utility positive signal,
  but only at smoke scale. The useful correction was not the beam search alone:
  the all-column proxy version still lost because wind/surface context diluted
  the transport-task objective. Restricting the deployed proxy context to
  transport task columns let validation select a dynamic policy and final smoke
  beat the static anchor. This supports a narrow hypothesis for the formal run:
  causal forecast planning may be viable only when the planner's proxy state is
  aligned to the task-composite transport objective, not to generic
  microclimate coverage.
- The formal task-only proxy-MPC result preserves only part of that hypothesis.
  It is materially closer than scalar utility scoring: seeds42/44 achieve
  positive best mean validation margins (`+0.005049`, `+0.008540`) while
  seed41 is nearly neutral (`-0.001605`). However, all three rows have negative
  q25 margins and multiple negative starts, so strict selection disables the
  deployable in every seed. The remaining error is now specifically downside
  risk across windows, not uniformly wrong average scoring. A useful next
  model should therefore predict paired full-window margin distributions
  against the static anchor, including a lower-quantile or negative-start
  target, and collect those labels under candidate self-rollouts. Further
  manual adjustment of event/freshness/power weights would repeat the same
  proxy misspecification.
- Review of `v1/docs/06-06-01-v1md` accepts Branch H's high-level objective but
  rejects the implementation specification as written. The project margin
  convention is `static_objective - candidate_objective` (positive is better),
  while the document reverses it. Its proposed `192 * 256 = 49152` training
  samples are only `192` window outcomes; repeating one window label across
  steps would be pseudo-replication. A q25 value is also not an observed
  per-row target: quantile regression must train directly on scalar paired
  margins across many independent windows, and negative-risk calibration needs
  enough repeated/regime-diverse outcomes. Using a validation-selected static
  anchor to generate train labels would leak validation selection into model
  training unless the model is trained over a train-selected/static-anchor
  bank and the validation anchor is only instantiated afterward. Candidate
  perturbations must operate on contiguous macro blocks aligned with the
  receding horizon, not replace `1--3` isolated steps in a `256`-step sequence.
  Finally, the proposed q25 error threshold of `0.05` is dimensionally invalid;
  calibration should use quantile coverage, pinball loss, and held-out-train
  risk metrics. Branch H remains the right direction only after these
  corrections.
- The two proposed pre-diagnostics also need rescoping. Teacher-improvement
  alignment is already complete: step-level learned-event AUC is only
  `0.606/0.586/0.517` for seeds `41/42/44`, so event probability is a weak
  feature and cannot be treated as sufficient. The formal proxy-MPC roots save
  learned event/continuous truth tables plus static/teacher rollouts, but no
  rollout for rejected proxy candidates. Therefore negative-start feature
  analysis is not artifact-only: it requires deterministic replay of the
  selected best calibration rows. To avoid validation-driven feature
  engineering, that analysis should be performed on train or a reserved
  train-calibration split, not used to redesign features and then evaluate on
  the same validation starts.
- Branch H's first true 256-step paired smoke confirms candidate heterogeneity
  at the correct decision horizon. For seed41, both sampled controllers beat
  the train anchor on two fit windows, while one controller won and one lost on
  the held-out train-calibration window. This is the label structure the
  mean/q25 model needs: dynamic value exists, but controller identity and
  regime determine downside. The sample is only `4+2` rows and is not evidence
  of generalization.
- Sharing one seed offset across all controllers for a fixed `(start, anchor)`
  is preferable to assigning each controller a different offset. It preserves
  exact static-candidate common-random-number pairing, makes cross-controller
  comparisons less noisy, and allows one cached static rollout per anchor
  rather than one per controller.
- A validation gate that accepts `margin_mean >= 0` and `q25 >= 0` is unsafe
  when the outer policy can choose static fallback per window. It allows an
  all-static policy to manufacture a zero-margin "pass" without demonstrating
  dynamic value. The corrected gate requires at least one dynamic validation
  window and strictly positive aggregate mean margin before final dynamic
  deployment can be enabled.
- Reusing source forecasts trained on `oracle_pretrain + rl_train` is not
  acceptable for Branch H's transfer test. Risk-fit/calibration features would
  be in-sample for the forecaster, while validation/final features would be
  out-of-sample, confounding scheduler transfer with forecast-model scope.
  Branch H now retrains both forecast families on `oracle_pretrain` only,
  freezes them, and uses the same prepared truth for train, validation, and
  final controller features.
- Restricting training time alone is insufficient if the forecast input
  contains the current truth of sensors whose acquisition is being scheduled.
  Such task-sensor inputs make the controller feature causally unavailable
  before the action. Formal Branch H therefore uses only variables supplied by
  the always-required `met_station_core`; all eligible static anchors and
  dynamic support masks must include that sensor.
- The continuous forecast target must match the planner objective rather than
  reproduce every historical forecaster output. Under `core_exogenous`, the
  frozen model predicts only `snow_mass_flux_kg_m2_s`,
  `snow_particle_mean_diameter_mm`, and
  `snow_particle_mean_velocity_ms`. This yields 24 horizon outputs and avoids
  spending model capacity on wind/surface targets unused by proxy-MPC.
- Causal model inputs do not guarantee causal controller features. The shared
  forecast-context code historically combined learned future predictions with
  the current truth value of each target, and the archived environment state
  exposed the current simulator event label. Both are unavailable before the
  sensing decision. Formal Branch H now uses the learned h1 task prediction as
  the current proxy and explicitly removes the event-label state dimension.
- Feature-name token audits are necessary but insufficient when generic fields
  such as `agent_state_####` can hide simulator-only values. Causal invariance
  tests that perturb unavailable truth while holding learned predictions fixed
  are required for this interface.
- Internal risk calibration must also be isolated from policy-family
  preprocessing. Proxy-MPC support masks and teacher target rates are learned
  quantities, not immutable sensor metadata. They must be computed from
  teacher rows whose absolute `step_indices` fall inside risk-fit windows,
  excluding the later risk-calibration block.
- Recomputing all 163 static masks was unnecessary after
  `met_station_core` became mandatory. Only 64 masks satisfy that deployment
  requirement; prefiltering them preserves the ranking semantics and cuts
  static-bank wall time without changing outcomes.
- The 256-step paired data contain real dynamic value but do not support a
  deployable one-shot risk predictor. Fit and later calibration have nearly
  identical mean margin, yet the feature-to-margin relationship does not
  transfer chronologically. Random grouped CV is optimistic.
- Adding strict-past 64/256/1024-step core/forecast history improves fit-only
  grouped CV, especially with XGBoost, but does not repair later transfer.
  More model capacity is therefore not the main fix.
- The structural cause is decision-horizon mismatch: an outer policy sees an
  8-step frozen forecast and past regime summaries but must commit one
  controller for 256 stochastic steps. A high-confidence threshold safely
  selects only one of 12 calibration windows.
- The next architecture is receding macro-risk control: estimate risk over
  64-step blocks, preserve dwell/warmup state, reselect at block boundaries,
  and evaluate the composed controller over the original 256-step windows.
- The 64-step label horizon does not improve risk learnability. Although its
  mean dynamic margin is positive on both fit and chronological calibration,
  the downside tail is worse than at 256 steps and both grouped-CV and later
  calibration risk metrics fail. Horizon mismatch was therefore only part of
  the problem.
- The remaining proximal hypothesis is action-interface misspecification:
  proxy-MPC can choose masks far from a strong static anchor using a score
  whose numerical improvement is not calibrated to true paired objective
  improvement. Before another learned risk model, test whether restricting
  candidate masks to a small Hamming neighborhood of the anchor improves the
  chronological lower tail.
- Increasing the proxy planner's anchor-improvement score threshold does not
  solve this misspecification. Fit selection chose threshold `0.02` with a
  positive q25, but its independent calibration q25 was `-0.0157` and half
  of calibration starts had negative mean margin. Proxy score magnitude is
  therefore not a transferable estimate of true static-anchor improvement.
- Restricting proxy-MPC to Hamming-1 neighborhoods sharply reduces downside,
  but fixed controller identity still does not transfer. The only fit-safe
  Hamming-1 controller loses on calibration, while the calibration-positive
  Hamming-1 controller was unsafe on fit. Wider neighborhoods recover mean
  gain but reintroduce negative q25 tails.
- The useful signal is therefore the residual action, not the hand-coded
  controller. The next learner should predict paired advantage for explicit
  no-op/add/drop actions over a short macro block. This removes controller-ID
  compression, produces denser action-specific supervision, and preserves the
  static anchor as an actual executable fallback.
- The first direct residual labels validate that reformulation. Roughly half
  of anchor/windows contain a beneficial single-sensor change, and an oracle
  selector with static fallback has positive mean value on both fit and later
  calibration. No fixed residual has a stable non-negative tail, so context
  conditioning remains necessary rather than optional.
- Full residual collection strengthens the opportunity result but rejects the
  first representation. Later calibration has more and larger positive oracle
  residuals than fit, so action scarcity is not the blocker. A 356-dimensional
  feature vector for only 192 fit rows cannot beat constant baselines even
  under grouped cross-validation. The immediate correction is structured
  dimensionality reduction, not more rollouts or a larger model.
- That modeling diagnosis was superseded by a lower-level protocol error.
  The environment's single sequential RNG is consumed only by active sensors,
  so changing a mask changes all later availability/noise draws. Equal seeds
  were therefore not common random numbers. Short 64-step residual margins are
  especially vulnerable to this action-dependent noise. Paired outcomes must
  be recollected after fixed-order pre-drawing before judging learnability.
- Correct common-random-number coupling materially changes the scientific
  conclusion. With the same architecture and compact features, all fit-only
  model families improve lower-tail prediction, and XGBoost transfers to the
  later train-calibration block with positive Brier improvement and perfectly
  ordered risk bins. The residual learner was not intrinsically unlearnable;
  its labels were previously corrupted by action-dependent stochastic draws.
- The first H3 failure exposed a second inherited proxy assumption:
  restricting residual actions to teacher top-k masks defeats anchor-local
  generalization. Residual actions are physically defined by one feasible
  sensor add/drop, so support must come from the projector-feasible mask set.
  Teacher support can be a prior or feature, but cannot delete valid local
  actions from the direct residual action space.
- Correcting residual support strengthens both opportunity coverage and model
  transfer. The model now learns sensor-level add/drop effects across anchors,
  so it can score feasible neighbors of a validation-selected static fallback
  without requiring that exact action ID in the train anchor bank.
- Full feasible support alone did not activate the controller on validation.
  The first validation replay used static fallback for all 12 starts because
  every candidate failed the selected lower-bound threshold, even though some
  passed the predicted-mean and negative-probability filters.
- Deployment calibration must align with the downstream gate without using
  downstream labels. Requiring dynamic use but selecting train thresholds
  primarily for conservatism can choose a vacuous policy. Among already
  risk-safe train-calibration candidates, independent-start dynamic coverage is
  therefore a legitimate first tie-break; it tests whether activation itself
  transfers while preserving the mean/q25/negative-start constraints.
- Coverage-aware calibration transfers activation: 3/12 validation starts use
  a residual and aggregate mean remains slightly positive. It does not transfer
  the required tail guarantee: two of those starts lose slightly. This rules
  out another scalar threshold search as the next step. The useful distinction
  is now whether errors arise from selecting the wrong residual action or from
  holding a locally reasonable action for an entire 64-step block.
- The remaining error is primarily anchor-transition OOD, not macro persistence.
  All activated validation windows choose the same target action, but the
  semantic residual depends on the anchor mask. Top-8 anchor training observes
  action 42 as removal of `snow_particle_counter` from anchor 106, whereas
  validation uses it as removal of `surface_temp_ir` from anchor 97. Sensor
  delta features permit extrapolation but 26 fit examples across three other
  anchors are too weak for a tail guarantee. Direct residual learning should
  cover the feasible anchor domain explicitly rather than assume cross-anchor
  compositional generalization from a top-k bank.
- A deeper audit supersedes that deployment interpretation: the collected
  target was not an anchor transition at all. It was the objective difference
  between two constant masks initialized from separate reset states. A true
  residual label must condition on the anchor-induced estimator, runtime,
  freshness, previous-action, energy, and RNG state at the decision boundary.
  Without that shared prefix snapshot, anchor/delta features describe a
  transition that never occurred in the labeled rollout.
- Common random numbers fix stochastic coupling only after initial states are
  equal. It cannot repair counterfactual labels whose environment states differ
  because each policy starts from reset. Prefix-conditioned snapshot branching
  is therefore mandatory before further model or threshold work.
- A residual policy also needs execution support matching its label support.
  Alternating anchor-conditioning and residual-pulse blocks makes every learned
  action an explicit one-block intervention from the declared anchor rather
  than an uncontrolled sequence of residual states. This retains warmup and
  switching effects while providing a static recovery block.
- Broad anchor coverage should be uniform for action-conditioned learning.
  Always including the train-best anchor in every start overweights one mask
  and leaves the validation-selected anchor underrepresented. Uniform rotation
  gives exact domain coverage without using validation identity.
- Prefix-conditioned labels reveal a meaningful energy-feasibility domain.
  Some instantly feasible masks cannot remain the intended anchor for a full
  64-step conditioning block because the energy guard projects them. These
  rows are not valid single-sensor residuals and must be excluded from model
  fitting; runtime must likewise decline dynamic action when the boundary mask
  differs from the declared anchor.
- On exact boundaries, direct residual risk is substantially more learnable
  than the earlier constant-mask proxy. All model families improve q25 and
  negative-risk prediction under grouped CV, and chronological GBDT/XGBoost
  improve both by roughly 15--19%.
- A lower-quantile model should be audited against lower-tail ordering. Mean
  margin need not increase monotonically across q25 bins when rare large
  positive outliers change the mean. Realized q25 and negative-rate ordering
  are the dimensionally aligned diagnostics.
- The corrected prefix-conditioned XGBoost passes its prediction gate, but a
  single global deployment threshold fails across heterogeneous anchors:
  none of 392 train-only candidates simultaneously achieves dynamic use,
  positive mean, non-negative q25, and the negative-start bound.
- Pooling all anchor transitions into one start-level calibration objective
  averages actions with different sensor removals/additions and different
  energy/runtime states. The next question is whether each anchor has a
  train-supported risk-safe operating point under one symmetric procedure.
  This must be answered for every anchor before introducing anchor-specific
  deployment; inspecting only the validation-selected anchor would create an
  avoidable selection bias.
- Symmetric leave-one-start-out calibration confirms that anchor identity is a
  first-class deployment variable, but also exposes severe small-sample
  optimism: 36/42 anchors pass on the same eight starts used for threshold
  selection, while only 10/42 pass when every start is held out once.
- The validation-selected static anchor 97 is not in the risk-supported set.
  Its held-out-start aggregate has negative mean and q25 with two negative
  starts. Therefore attaching a per-anchor threshold to action 97 would not be
  justified by train-only evidence.
- Several other anchors have strong and stable dynamic opportunity, especially
  action 47, whose held-out margins are positive on all eight starts. A
  defensible next architecture may select only among anchors prequalified by
  train-only residual safety while retaining the unrestricted strongest static
  mask as the comparator. Whether this can succeed depends on the static
  objective penalty of moving away from anchor 97.
- Residual opportunity must be discounted by the execution duty cycle. Under
  the implemented alternating anchor/residual policy, only half the blocks earn
  residual margin, while the anchor's static quality gap applies throughout.
  After this correction, only action116 has a positive train-only feasibility
  estimate. Large per-pulse margins at actions 47/15/51/54 are insufficient
  because those anchors begin too far behind the best train static mask.
- Fit-only model-family selection matters at deployment, not only prediction.
  HistGBDT improves chronological tail modeling enough to produce 50 globally
  valid threshold combinations where XGBoost produced none, while retaining a
  valid q25 coverage and risk-bin ordering.
- The global HistGBDT calibration is encouraging but does not authorize an
  arbitrary anchor: it averages outcomes over all anchor groups. The stricter
  anchor-specific LOSO and duty-cycle feasibility filters still identify only
  action116. This separation prevents a favorable pooled metric from masking
  an unsafe deployed anchor.
- The fully train-locked HistGBDT/action116 controller does not transfer to
  validation against the unrestricted static comparator. It activates in all
  12 windows without physical failures, yet loses on 7 and has strongly
  negative q25. Therefore the remaining issue is objective transfer, not
  vacuous fallback, hard-constraint enforcement, or warmup abort behavior.
- Paired decomposition narrows that transfer failure to anchor quality. The
  controller has positive mean and q25 improvement relative to action116, but
  action116's average deficit to action97 is larger. This argues against
  abandoning direct residual prediction: it argues for calibrating residuals
  around the strongest static anchor with enough chronological starts.
- Action97 currently has only eight calibration boundary states because the
  broad collection rotates 16 of 64 anchors per start. Its failure under
  leave-one-start-out therefore does not yet distinguish intrinsic unsafe
  residuals from undersampled anchor-specific calibration.
- Dense action97 data remove that ambiguity at the opportunity level. Positive
  one-hop residuals exist on roughly two thirds of both fit and later
  calibration starts, and oracle fallback has positive mean on both. However,
  no fixed residual transfers safely; action42 changes from slightly positive
  fit mean to negative calibration mean. Context conditioning is still
  necessary.
- Dense action97 fit-only CV remains learnable but is harder than the broad
  anchor task: all three families improve q25 by about 12--13%, while negative
  Brier gains are only 1.6--4.4%. HistGBDT has the strongest rank and mean
  prediction and is selected without calibration/validation exposure.
- Dense action97 chronological evaluation rejects the current representation.
  The model retains moderate ranking but loses lower-tail calibration and makes
  negative-risk probability worse than a constant prevalence baseline. More
  rows at the same temporal support are therefore insufficient; the model
  needs either explicit regime-shift handling, rolling calibration, or a
  representation whose risk semantics transfer over time.
- The shift is not merely a higher negative prevalence. Later windows have
  substantially higher learned event probability and particle forecasts, and
  some feature-margin associations reverse. The final fit quartile is also the
  most opportunity-rich, so uniform full-history training may dilute the
  relevant recent regime. Rolling training is now a testable protocol change,
  not an arbitrary recency heuristic.
- Simple rolling training is not robust across fit-only backtests. Its Q3
  improvement does not repeat on Q4, so the drift is not a monotonic process
  that can be repaired by discarding older data. The next distinction is
  whether row-level risk calibration is unnecessarily strict for a policy that
  only executes one top-ranked residual with static fallback.
- Start-level thresholding can isolate a small safe subset despite poor
  row-level probability calibration, but coverage is only 3/32. All selected
  rows use action42, consistent with its fit-only status as the sole residual
  with positive mean. A binary action42/no-op intervention is a cleaner
  statistical problem than learning three heterogeneous residual risks.
- The cleaner binary problem is still not learnable with the current
  high-dimensional tree representation. Action42-only grouped CV cannot beat
  constant baselines on mean or negative risk and improves q25 by only 5.5%.
  Either the relevant signal is obscured by dimensionality or the causal
  forecasts do not contain enough information about 64-step intervention
  value.
- Domain-predeclared compact features rule out dimensionality as the primary
  explanation: regularized low-dimensional models also fail on fit-only CV.
  The one-shot boundary representation does not contain stable information
  about realized 64-step intervention value. The next model must predict
  future trajectories/distributions and evaluate actions through the actual
  estimator/oracle dynamics rather than directly regress a noisy scalar margin.
- The accepted scenario still has large dynamic value: the existing MPC teacher
  beats the strongest static comparator by roughly 8.5% on seed41 while using
  learned forecasts and no configured truth-future input. The failure belongs
  to policy compression/risk transfer, not to scene headroom. A causal teacher
  should be considered an algorithm candidate rather than only a label source.
- The teacher is not causal despite its forecast flags. Its beam search obtains
  future outcomes by stepping the real truth-replay environment under each
  candidate. Explicit forecast leakage tests did not cover this planner-level
  truth access. Future protocol audits must distinguish feature leakage from
  simulator-branch leakage.
- A deployable replacement can reuse the teacher's action search and physical
  state machinery only after branch truth is replaced with sampled learned
  trajectories. This is the central architectural requirement, not an optional
  robustness enhancement.
- The causal planner can preserve the archived simulator without copying its
  leakage: instantiate a compact shadow environment containing no source future
  rows, restore only current dynamic state, and let the frozen oracle score
  sampled scenario futures. This keeps physical semantics while making future
  truth access structurally impossible.
- Causal planning requires split-locked normalization as well as causal
  forecasts. `WarmupSchedulingEnv` defaults to mean/std over its entire truth
  table, and reset initializes history from those values. A hidden-future
  mutation therefore changes the initial agent state unless normalization
  statistics are supplied from an allowed training prefix.
- Expected plus upper-tail CVaR is now an explicit sequence objective rather
  than a posthoc deployment guard. This lets uncertainty affect action search
  directly while retaining hard feasibility in every scenario branch.
- A simple bootstrap MLP ensemble plus chronological residual resampling is
  already materially better than persistence on the accepted v6 training
  distribution: `28.4%` normalized-RMSE skill with near-nominal uncertainty
  coverage (`83.2%` for an 80% interval). The world-model route is therefore
  empirically viable enough to test as a controller, not merely an
  architectural placeholder.
- Parameter uncertainty alone is small (`0.0749` normalized member spread);
  calibrated residual uncertainty is doing most of the scenario widening.
  Robust planning should therefore retain the residual bank rather than using
  ensemble members as the only scenarios.
- Good latent-truth forecasting does not imply good closed-loop world-model
  forecasting. The first robust planner wins on two validation windows but
  loses strongly on two, while remaining physically feasible. Its model was
  trained and audited on complete truth histories, then queried on stale
  scheduler observation histories. This unresolved covariate shift is a more
  fundamental issue than planner risk-weight tuning.
- The causal planner is not merely falling back: it differs from the static
  anchor on `64.1%` of steps. Therefore the negative mean is evidence of
  incorrect counterfactual ranking under deployment histories, not a vacuous
  static result.
- The rollout-history shift is measured, not speculative. On the world model's
  own train-only audit segment, replacing complete truth history with
  static-anchor scheduler history increases normalized RMSE by `12.1%`.
  The degradation is concentrated exactly on the event-transport task
  variables that drive dynamic scheduling value. This makes a mask-aware
  rollout-trained world model the next necessary step.
- The mask-aware rollout-trained model fixes the measured failure mode at
  engineering-smoke scale. Even with only two members, six-step horizon, and
  two epochs, it reaches normalized RMSE `0.6234` on rollout histories versus
  `0.8961` persistence, with task variables back near or below the latent
  model's truth-history error. This justifies a horizon-12 model gate before
  any further planner validation.
- The horizon-12 rollout-trained model remains strong enough for planner use.
  Its normalized RMSE is `0.6409` versus `0.9673` persistence, improving skill
  to `+33.7%` while retaining near-nominal 80% interval coverage (`83.0%`).
  Event-transport variables do not collapse at the longer horizon
  (`0.846/0.741/0.723` normalized RMSE for flux/diameter/velocity). The next
  uncertainty is therefore no longer world-model gate quality, but whether
  robust planner action ranking transfers under this rollout-history model on
  validation windows.
- The first validation replay with the horizon-12 rollout-history model
  improves the old robust-planner mean margin but still fails the gate
  (`-0.0152` mean, `-0.0453` q25, `2/4` negative starts). Because dynamic use
  remains high (`64.1%`) and there are no hard violations or warmup aborts,
  the failure is not fallback vacuity or physical infeasibility. The active
  bottleneck is negative-tail sequence ranking: the planner must learn when
  the static anchor is already safer than a dynamic rollout, not merely how to
  generate causal scenarios.
- Trace diagnostics identify the concrete ranking error. The robust planner
  mostly swaps anchor action 97 (`surface_temp_ir`) for action 106
  (`snow_particle_counter`) and predicts robust improvement over the repeated
  anchor almost everywhere. Realized outcomes show the tradeoff is asymmetric:
  particle velocity improves, but particle diameter degrades enough to dominate
  negative windows. The current objective/model interface is therefore
  under-penalizing diameter risk from SPC-heavy dynamic sequences.
- The validation gate also used a mismatched temporal contract:
  `planning_horizon=3` and `replan_interval=4`. Since the receding policy holds
  each selected mask for four steps, one executed step was outside the scored
  sequence. Before changing risk weights, the next fair correction is to align
  planning depth to the hold interval (`planning_horizon=4`) using the same
  locked validation-only protocol.
- The horizon-hold mismatch is not the root cause. Setting
  `planning_horizon=4` worsens the validation gate (`-0.0318` mean,
  `-0.0801` q25) by increasing the same SPC-heavy dynamic preference. The
  planner's failure is therefore target-specific risk ranking: it treats SPC
  swaps as robustly beneficial because velocity improves, while realized
  particle-diameter error grows enough to dominate the task objective.
- A strong anchor-improvement margin is the first validation-level fix that
  controls the negative tail. Margin `0.25` passes validation with mean
  `+0.0327`, q25 `0`, and zero negative starts by allowing dynamics only when
  the predicted improvement over repeated anchor is large. This is not a
  strong dynamic-use result yet (`4.7%` average dynamic rate), but it provides
  a split-compliant final-test candidate and supports the interpretation that
  the world-model planner needs calibrated static-anchor conservatism.
- The seed41 final result confirms the conservative guard transfers at least
  on the first held-out final block: mean margin `+0.0179`, q25 `0`, and no
  negative starts. The useful claim is not yet "frequent adaptive scheduling";
  it is narrower: a causal world-model planner can selectively activate
  dynamic masks only when predicted advantage is large enough, preserving the
  static anchor on unsafe windows. This needs multi-seed replication before it
  can support a paper-level claim.
- The first multi-seed replication is genuinely useful but incomplete. Seed42
  validates and transfers strongly, with final mean `+0.0292`, q25 `+0.0103`,
  and dynamic rate above `50%`. Seed44 fails validation for every tested
  anchor-improvement margin even though its world-model audit passes. This
  separates model forecasting quality from deployment selection quality:
  seed44 likely has a different static anchor/support geometry or target-risk
  tradeoff that the scalar margin grid cannot handle. Current evidence is
  `2/3` available seeds with deployable dynamic selection and `2/2` positive
  finals among selected seeds, not a robust all-seed claim.
- Seed44 confirms the static-anchor geometry hypothesis. Its selected static
  anchor already contains SPC, laser, and fc4, so dynamic planning is no longer
  "add event sensing when needed"; it is mostly "remove or reshuffle event
  sensors around an already strong event stack." The best validation row uses
  dynamic actions on every step and loses in two windows by worsening both
  particle diameter and velocity. A scalar global guard cannot reliably repair
  that geometry; the next evidence step is either more source seeds or a
  different anchor-neighborhood controller that can decide when the event-heavy
  anchor should simply be preserved.
- Adding seeds43/45 confirms that scalar anchor-margin calibration alone does
  not solve event-heavy anchors. Both new seeds have teacher headroom and
  passing rollout-world models, but no tested margin passes validation. The
  issue is not lack of dynamic value or model audit failure; it is choosing a
  safe action neighborhood around anchors that already include SPC/fc4. A
  support-restriction or anchor-neighborhood controller is now a more plausible
  correction than another global margin value.
- Event-heavy support restriction is useful but insufficient. It recovers
  validation-passing dynamic candidates for seeds43 and 45, showing that action
  neighborhood size matters, but both selected candidates fail the held-out
  final strict gate. Seed43 turns slightly negative on final; seed45 keeps a
  positive mean but misses the q25 gate by a small margin. Seed44 still has no
  validation-safe dynamic candidate. The current blockage is therefore no
  longer just event-heavy support geometry; it is validation-to-final transfer
  under sparse dynamic activation. The next diagnostic should test whether the
  four-window validation gate is under-sampling tail risk before adding another
  planner scoring wrapper.
- The validation-sampling diagnostic rules out a simple fix by increasing the
  number of validation starts. Twelve-start validation catches additional tail
  risk in seed42 and seed45, but it is not aligned with final transfer: seed42
  would be rejected despite final success, while seed43 still passes despite
  final failure. The blocker is therefore a regime-transfer / online eligibility
  problem, not merely under-sampled run-level validation. Run-level
  mean/q25/negative thresholds should be treated as exhausted for this planner
  family.
- Coarse regime summaries are also insufficient. Event-rate and transport
  ranking shifts show real validation/final differences, but they do not form
  a reliable selector in the current evidence: seed42 and seed43 both move to
  lower final event/ranking regimes, yet one succeeds and one fails. The next
  useful diagnostic must inspect planner trace-level counterfactual ranking
  errors inside individual windows.
- Per-window traces identify the current robust planner failure mode. In
  seed43 final failure, the planner predicts large dynamic advantage but the
  realized effect is a flux-error increase; in seed45 final failure, it
  predicts dynamic advantage but task errors are unchanged and oracle loss
  worsens. The failure is therefore action-effect overestimation for short
  deviations from strong static anchors. A viable next algorithm needs an
  online effect/break-even verifier or a model trained to predict intervention
  outcomes relative to the anchor, not another scalar run-level selector.
- Predicted component tracing narrows the overestimation source. The bad
  dynamic rows are not driven by candidate prior, switching, power, warmup, or
  energy terms; they are driven almost entirely by predicted
  `event_weighted_oracle` improvement. Explicit `task_error` is negative in the
  seed43 failure and zero in the seed45 failure. This means the next guard
  should not be framed as another run-level margin threshold. It should verify
  whether a specific dynamic deviation clears an online break-even test against
  the static anchor, preferably with features from the planned raw-vs-anchor
  sequence and a split-compliant calibration step.
- The first online component guard confirms that diagnosis only partially.
  Requiring non-negative predicted task-component margin repairs seed43's
  final failure, because the bad rows had negative task support. The same guard
  leaves seed45 unchanged, because the failing rows have zero task-component
  signal rather than an explicitly negative one. A stricter positive threshold
  collapses to all-static validation. Therefore a component-sign guard is a
  useful safety layer but not the final verifier; the remaining failure needs
  direct prediction/calibration of whether oracle-component gains will realize
  as paired static-anchor margin.
- Applying the non-negative task-component guard across all five selected/best
  configurations improves robust-planner evidence from `2/5` to `3/5` strict
  final pass without damaging seeds41/42. This confirms the guard is not just
  posthoc overfit to seed43. However, seed44 remains a validation-tail failure
  and seed45 remains a held-out q25 failure. The remaining research blocker is
  now narrower: reject or correct zero-task-component dynamic deviations whose
  predicted oracle advantage does not realize as paired objective margin.
- Direct hold-effect component guarding is not the missing piece. On seed45,
  comparing the raw first action held for the replan interval against anchor
  hold makes validation worse and blocks final evaluation across all tested
  total/task threshold variants. This suggests the planner's problematic
  dynamic decisions are not separable by a hand threshold over planned or
  hold-level component margins. The next serious route is to collect paired
  intervention outcomes and learn/calibrate dynamic-vs-anchor effect directly.
- The first direct intervention-effect audit validates the new target but
  rejects scalar thresholding. Seed45 train/validation effect rows have
  negative lower tails, and neither predicted robust advantage nor total
  component margin yields a train-safe one-dimensional threshold. The useful
  next abstraction is therefore a learned effect verifier with richer
  context/runtime features, trained on multiple train splits and calibrated
  before validation/final replay.
- Multi-seed train intervention-effect data sharpen the blocker. Across seeds
  `41--45`, raw dynamic deviations contain substantial opportunity
  (`350/763` rows positive) but the aggregate mean and q25 are still negative
  (`-0.004770` and `-0.017111`). The planner-selected subset is better than
  the full raw set, but not reliably safe across seeds. Therefore the problem
  is not absence of dynamic value; it is lower-tail control for short dynamic
  deviations from strong static anchors. The next verifier must be judged by
  group/seed-held-out accepted-effect mean and q25, not by classifier AUC or
  scalar threshold fit.
- The first causal feature-mode effect verifier does not solve lower-tail
  control. On the deployable `selected_dynamic` boundary, the best method is
  still the scalar predicted-advantage score, and it reaches only `2/5`
  held-out safe seeds with pooled q25 slightly below zero. Tree and linear
  learned verifiers accept more rows but keep negative held-out lower tails.
  This suggests either feature representation is still missing the relevant
  regime signal, or the paired intervention target is too noisy for row-level
  seed transfer under the current planner interface.
- Compact and with-guard feature variants do not change the row-level
  verifier conclusion. The best deployable boundary remains the scalar
  predicted-advantage threshold with `2/5` safe held-out seeds. Therefore the
  failure is not simply high-dimensional feature dilution or exclusion of
  current guard state. The next plausible abstraction is window/start-level
  dynamic eligibility, where the decision is whether a whole local window
  should allow dynamic deviations at all.
- Window-level opportunity is split across action streams. Under the current
  planner-selected dynamic stream, seed43 has no oracle-safe train windows;
  under the broader all-raw stream, seeds41/42 have no oracle-safe train
  windows. This rules out a single global row-level verifier as the main
  correction. The next planner interface needs a window-level choice of action
  source or candidate neighborhood, so that event-heavy anchors can sometimes
  use broader raw candidates while already-good anchors preserve the current
  conservative selected stream.
- The source-oracle ceiling is the first useful optimistic signal after the
  row-level verifier failure. If a controller may choose among `anchor`,
  `selected_dynamic`, and `raw_bypass` at the window level, every seed has at
  least one safe train window. This does not prove learnability, but it changes
  the next implementation target from "predict row effect" to "predict window
  action-source eligibility."
- The first source-selector model fails despite the source-oracle ceiling.
  Random forest, HistGBDT, and logistic classifiers cannot turn current
  replan-level causal/runtime features into held-out safe source choices.
  This suggests the missing ingredient is not another classifier wrapper but
  the planner objective itself: event-weighted oracle gains are still too easy
  to rank above static-anchor preservation without realized task improvement.
- Objective-dominance sweep partially validates that diagnosis. Seed44, which
  previously failed validation-tail safety, becomes final-safe when the robust
  planner ignores oracle-loss/event weighting and optimizes task-error on all
  steps. This is a real algorithmic signal: the remaining planner should be
  task-improvement dominated. Seed45 does not improve under the same change,
  indicating its blocker is also candidate-support/neighborhood geometry, not
  only objective weighting.
- Seed45 task-only support/margin sweep closes the simple candidate-size
  rescue under task-only objective. The seed either collapses to all-static
  with positive anchor margins or becomes validation-negative with wider
  support/zero margins. The near-term useful route is therefore not "fix
  seed45 at all costs"; it is to make objective-family selection a formal
  validation-stage part of the algorithm, because task-only repairs seed44
  while the original component-guarded robust planner already repairs
  seeds41/42/43.
- Validation-selected objective-family aggregation reaches the current minimum
  claim target: `4/5` strict final pass and `5/5` positive final mean. The
  scientific interpretation should be careful: the result supports a
  conservative validation-selected robust planning family, not a single fixed
  objective that dominates in every seed. Seed45 remains the residual
  lower-tail failure.
- Scenario diagnosis update: v5/v6 did not create enough regime-specific
  sensor value. It mostly changed the constraint surface: laser shortcuts were
  blocked, but the scene still lets a low-risk static stack
  (`core+SPC+FC4`) cover onset/active/decay task variation too well. A useful
  v7 scene must make sensor value phase-dependent: onset should need precise
  wind/thermo context, active transport should need particle/flux sensing, and
  decay should need radiation/surface-temperature context. The gate should
  measure static dominance directly, not only feasible-mask structure.
- V7 exploration finding: merely changing instantaneous feasibility is
  insufficient. FC4 and SPC each become universal static shortcuts when they
  are cheap and when the audit ignores long-run duty. The first useful scene
  signal appears only after treating direct particle/flux sensors as heavy
  phase resources: instantaneously selectable, but not sustainable under the
  average power budget. Under that framing, a phase-conditioned selector beats
  the best sustainable static mask in seed41/42/43 smoke audits, while using
  three distinct phase masks and staying below average power `0.62`.
