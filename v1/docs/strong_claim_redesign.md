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

### 2. Action-Conditioned Rollout Value

The scheduler should not imitate action IDs directly. It should learn a
state-action estimate of short-horizon rollout cost under the teacher objective.
The current value-residual policy is the first version of this idea.

Server evidence shows that a plain learned event forecast is not enough: it
repeats the old weak `4/5` result with a small mean margin. An uncertainty-aware
absolute-cost ensemble was also insufficient (`3/5`, negative mean margin).
The next active route is therefore not a larger ensemble, but a different
target: learn the candidate advantage relative to the validation-selected static
anchor directly.

The anchor-advantage residual policy trains on:

```text
advantage(state, action) = cost(validation_static_anchor) - cost(action)
```

and deploys only when the predicted advantage clears a validation-calibrated
threshold. This targets the real decision boundary instead of subtracting two
independently learned absolute costs at runtime.

### 3. Online Planner

The final deployable method should choose actions by online optimization over
feasible subsets using:

- learned event forecast;
- action-conditioned cost/value estimates;
- hard power and startup feasibility;
- SOC and warmup state;
- validation-calibrated risk control.

The planner may retain a static anchor as a safety fallback, but the scientific
claim should come from learned forecast/value planning, not from the anchor.

## Immediate Experiments

1. Completed: `learned_value_residual_safe`, n=5, budget 1.20.
   Result: weak pass (`4/5`, mean margin `+0.001856`), not enough for the
   original claim.
2. Completed: `learned_ensemble_value_safe`, n=5, budget 1.20.
   Result: fail (`3/5`, mean margin `-0.003409`).
3. Running: `learned_advantage_residual_safe`, n=5, budget 1.20.
   Purpose: test whether direct anchor-relative advantage learning improves the
   deployable margin and fixes the seed44/low-margin failure mode.
4. Prepared fallback: `learned_advantage_residual_calib_safe`, n=5, budget
   1.20. This lets validation select both support size and advantage threshold,
   reducing sensitivity to a manually chosen teacher-label top-k.
5. If the advantage route passes with materially better margin, scale it to
   more seeds and budget matrix. If it fails, the next change must be a larger
   online planner or environment/task redesign, not another threshold-only
   variant.

## Claims That Remain Forbidden Until Proven

- Generalization across budgets or event regimes.
- Statistical significance.
- End-to-end scheduler-forecaster joint optimization.
- Necessity of DAgger.
