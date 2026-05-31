# Claim Results: Value-Residual Forecast-Aware Scheduling

Date: 2026-06-01

## Protocol

- Split: strict chronological protocol with disjoint train, validation, and final-test windows.
- Comparator: validation-selected static feasible mask.
- Main deployable policy: `forecast_aware_value_residual`.
- Gate: deployable policy must beat the static comparator in at least `4/5`
  seeds and have positive mean paired margin; privileged MPC teacher must also
  beat static in at least `4/5` seeds.

## Main Result

Artifact:
`v1/artifacts/claim_suite_semimarkov_n5_value_residual/aggregate/claim_summary.csv`

| preset | seeds | deployable wins | deployable mean margin | teacher wins | teacher mean margin | sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `value_residual_safe` | 5 | 4 | +0.002213 | 5 | +0.030599 | 0.375 |

Per-seed deployable margins:

| seed | margin vs static | pass |
| ---: | ---: | --- |
| 41 | +0.003631 | yes |
| 42 | +0.007828 | yes |
| 43 | +0.000795 | yes |
| 44 | -0.001840 | no |
| 45 | +0.000650 | yes |

Interpretation: the minimum deployable claim passes, but the effect size is
small and the n=5 sign test is not significant. The manuscript should claim a
controlled, split-compliant improvement under the pre-defined gate, not a large
or statistically significant effect.

## Ablations

Artifact:
`v1/artifacts/claim_suite_semimarkov_n5_value_residual_ablate/aggregate/claim_summary.csv`

| preset | seeds | deployable wins | deployable mean margin | teacher wins | teacher mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| `value_residual_no_dagger` | 5 | 4 | +0.002213 | 5 | +0.030599 |
| `value_residual_oracle_objective` | 5 | 2 | -0.006959 | 5 | +0.020893 |

Mechanism interpretation:

- Removing DAgger did not change the pass pattern. In this implementation,
  DAgger is not the active mechanism because the value-residual policy is driven
  mainly by the action-cost dataset, the supported action set, and the
  validation-calibrated residual threshold.
- The oracle-only objective failed while the privileged teacher still beat
  static in every seed. This supports the task-composite objective as a
  necessary component for the deployable scheduler.

## Constraint And Behavior Summary

Artifact:
`v1/artifacts/claim_suite_semimarkov_n5_value_residual_ablate/aggregate/behavior_policy_summary.csv`

| preset | policy | mean power | min SOC mean | aborts | switch rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `value_residual_no_dagger` | value residual | 1.1732 | 113.5080 | 0.0 | 0.2088 |
| `value_residual_no_dagger` | static | 1.1779 | 113.9820 | 0.0 | 0.0041 |
| `value_residual_no_dagger` | teacher | 0.9370 | 160.2320 | 0.0 | 1.3771 |
| `value_residual_oracle_objective` | value residual | 0.7720 | 155.8440 | 0.0 | 0.4205 |
| `value_residual_oracle_objective` | static | 0.6620 | 155.4240 | 0.0 | 0.0000 |
| `value_residual_oracle_objective` | teacher | 0.5662 | 174.7980 | 0.0 | 1.3688 |

Final policy metrics report zero warmup aborts and zero steady/peak constraint
violations for the value-residual main run.

## Paper-Safe Claims

- A forecast-aware value-residual scheduler can outperform a
  validation-selected static feasible schedule under a strict chronological
  split in `4/5` seeds.
- A privileged MPC teacher consistently beats static (`5/5`), establishing that
  the scheduling objective has dynamic value.
- The deployable improvement depends on the task-composite objective; the
  oracle-only objective does not support the same deployable claim.
- The implemented value-residual method is constraint-compliant in the reported
  final evaluations.

## Claims To Avoid

- Do not claim statistical significance at n=5.
- Do not claim DAgger is necessary for the current passing method.
- Do not claim a large absolute performance gain.
- Do not claim end-to-end joint optimization of scheduler and forecaster.
