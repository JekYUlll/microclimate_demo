# Strong-Claim Redesign Plan

Date: 2026-06-01

## Why The Previous Result Is Insufficient

The value-residual result is useful but not enough for the original paper
claim. It passes a minimal n=5 gate, but the margin is small, the sign test is
not significant, DAgger is not an active mechanism, and the result is proven in
only one budget/scenario setting.

The original claim requires a genuinely forecast-aware constrained scheduler,
not a static-anchor repair that only occasionally deviates from a validation
static mask.

## Revised Success Criteria

A candidate can support the original claim only if it satisfies all of these:

- Beats the validation-selected static comparator with a materially positive
  margin across more than the old n=5 single-setting gate.
- Keeps the privileged teacher ahead of static, so the dynamic objective remains
  valid.
- Works across at least two power budgets and one event-regime perturbation.
- Uses causal deployable forecast features, not future truth.
- Shows an ablation in which removing the learned forecast / planning component
  degrades the deployable result.
- Has zero or near-zero hard-constraint violations and no warmup-abort failure
  mode.

## Current Mainline Architecture

### 1. Learned Causal Forecast Context

The old policy used a hand-coded wind-speed event-risk heuristic. The new path
trains a multi-horizon event forecaster using only data before validation. Its
predicted probabilities are injected into the truth table as
`learned_event_p_h*` columns and consumed through `ForecastContextConfig`.

This makes forecast awareness a learned module that can be ablated and improved.

### 2. Action-Conditioned Value and Planner

The scheduler should not imitate action IDs directly. It should learn a
state-action estimate of short-horizon rollout cost under the teacher objective.
The value-residual policy was the first version of this idea, but n=15 scaling
showed that one-step residual selection is still too brittle.

Server evidence shows that a plain learned event forecast is not enough: it
repeats the old weak `4/5` result with a small mean margin. An uncertainty-aware
absolute-cost ensemble was also insufficient (`3/5`, negative mean margin), and
direct anchor-advantage regression failed decisively (`0/5`). Adding BC/KNN,
rate matching, and teacher-sequence compression also failed to produce robust
deployment.

The active route is therefore a learned rollout-value planner:

```text
state/action -> learned short-horizon cost
state/action -> learned next causal policy feature
planner(state) -> short-depth beam search over supported feasible masks
```

The feature-transition model is trained only on the train split. During final
deployment, the planner uses only current causal state, learned event forecast
columns, the fitted transition model, the fitted action-cost model, and the
hard feasibility layer.

### 3. Online Planner

The current planner candidate is `forecast_aware_rollout_value`. It chooses
actions by online optimization over feasible subsets using:

- learned event forecast;
- action-conditioned cost/value estimates;
- learned causal feature transitions;
- hard power and startup feasibility;
- SOC and warmup state;
- validation-calibrated risk control.

It retains the validation-selected static anchor as a safety fallback, but the
claim should come from learned forecast/value planning being selected and
improving final-test objective, not from the anchor alone.

## Immediate Experiments

1. Completed: `learned_value_residual_safe`, n=5, budget 1.20.
   Result: weak pass (`4/5`, mean margin `+0.001856`), not enough for the
   original claim.
2. Completed: `learned_ensemble_value_safe`, n=5, budget 1.20.
   Result: fail (`3/5`, mean margin `-0.003409`).
3. Completed: `learned_advantage_residual_calib_safe`, n=5, budget 1.20.
   Result: fail (`0/5`, mean margin `-0.018997`), so direct advantage
   regression is rejected as the main deployable route.
4. Completed: B=1.20 extension of event/value and BC/KNN guarded hybrids.
   Result: event/value combined `10/15`; BC/KNN extension `7/10` and old-seed
   check immediately failed seeds `41--42`. Shallow classifier expansion is
   not enough.
5. Running: `learned_hybrid_planner_guarded_safe`, n=5, budget 1.20.
   Purpose: test whether learned feature-transition planning can recover the
   teacher's temporal value while remaining deployable and validation-guarded.
6. If planner n=5 improves the original seed set, scale it to seeds `46--55`
   and then retest sparse-event perturbation. If it fails, inspect transition
   error and candidate support before adding another policy head.

## Claims That Remain Forbidden Until Proven

- Generalization across budgets or event regimes.
- Statistical significance.
- End-to-end scheduler-forecaster joint optimization.
- Necessity of DAgger.
