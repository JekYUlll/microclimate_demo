# 2026-06-04 Deployable Interface Blockage Report

## Status

- Accepted scene: v6 / `event_transport_rich`.
- Current evidence: privileged teacher consistently beats validation-selected
  static, but every deployable interface tested in this tier is rejected by
  validation or falls back to static.
- Latest completed gate:
  `v1/artifacts/claim_suite_v6_transport_macro_option_riskband_20260604/`.

## Locked Result

| route | deployable | teacher | mean deployable margin | mean teacher margin | verdict |
|---|---:|---:|---:|---:|---|
| runtime-risk dense | 0/3 | 3/3 | 0.000000 | +0.110656 | rejected |
| cost-KNN memory | 0/3 | 3/3 | 0.000000 | +0.110656 | rejected |
| macro-option snippets | 0/3 | 3/3 | 0.000000 | +0.110656 | rejected |

The macro-option selected rows all used `event_threshold=1.0`, which is
static-equivalent fallback. Non-static macro rows were already negative on
validation:

| seed | best dynamic mean margin | best dynamic min margin | min negative starts |
|---:|---:|---:|---:|
| 41 | -0.027508 | -0.058914 | 9/12 |
| 42 | -0.002222 | -0.026749 | 5/12 |
| 44 | -0.040022 | -0.111339 | 9/12 |

## What Is Closed

- More validation starts alone: closed by dense runtime-risk.
- Scalar event/risk guard: closed by paired and dense runtime-risk.
- One-step teacher cost exposure: closed by cost-KNN.
- Short teacher trajectory replay: closed by macro-option snippets.
- Blind dynamic baselines: cyclic/dwell already lost 0/3.

## Current Diagnosis

The scene is not the immediate blocker. Teacher margins remain large and
consistent, so dynamic value exists in the privileged planning objective.

The blocker is the deployable causal transfer layer. Student policies can see
learned event probabilities, freshness, SOC, and current estimator state, but
those signals are not sufficient to identify when dynamic teacher behavior
improves the frozen forecast objective over a strong static anchor. Preserving
more teacher structure did not solve this.

## Likely Root Causes

- Learned event probabilities are too weak as a regime variable for choosing
  dynamic schedules.
- Frozen oracle loss remains highly static-anchor sensitive; dynamic schedules
  may reduce task error while worsening oracle loss enough to lose the
  composite objective.
- Validation-selected static is not just a weak baseline. It often encodes a
  strong target-specific sensor subset and can absorb much of the task value.
- Teacher uses privileged rollout transitions; deployed students only use
  causal compressed features. The gap is now structural, not just supervised
  accuracy.

## Required Next Direction

Do not launch another teacher-compression variant immediately.

Next useful work should be zero-retrain diagnostics and objective correction:

1. Decompose static vs teacher vs failed dynamic candidates by oracle loss,
   task error, power, warmup aborts, and event/non-event windows.
2. Audit learned event forecast calibration against windows where teacher
   actually improves over static.
3. Test whether validation selection should target teacher-improvement
   windows rather than generic event windows.
4. If objective conflict is confirmed, redesign the frozen reward/evaluation
   objective before any new deployable policy.

