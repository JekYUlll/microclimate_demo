# 2026-06-04 Objective Transfer Audit

## Scope

This audit uses only completed v6 / `event_transport_rich` artifacts. It does
not retrain policies or rerun final evaluation.

Inputs:

- `v1/artifacts/claim_suite_v6_transport_runtime_risk_denseval_20260604/`
- `v1/artifacts/claim_suite_v6_transport_cost_knn_riskband_20260604/`
- `v1/artifacts/claim_suite_v6_transport_macro_option_riskband_20260604/`

Generated outputs:

- `v1/scripts/audit_objective_transfer.py`
- `v1/artifacts/objective_transfer_audit_v6_20260604/objective_transfer_audit.md`
- `v1/artifacts/objective_transfer_audit_v6_20260604/objective_seed_summary.csv`
- `v1/artifacts/objective_transfer_audit_v6_20260604/objective_pair_rows.csv`

## Main Result

The accepted scene is still valid as a dynamic scheduling scenario:

- MPC teacher beats validation-selected static in `3/3` seeds.
- MPC teacher beats static in all `12/12` final windows.
- Mean teacher objective margin is `+0.110656`.

The teacher advantage is not a single metric artifact:

- Mean oracle-loss margin: `+0.043868`.
- Mean raw task-error margin: `+0.222628`.
- With `task_error_weight=0.3`, task component contributes `+0.066789`.
- Therefore about `60.4%` of teacher objective lift comes from the task-error
  component and about `39.6%` from frozen-oracle loss.

## Interpretation

The static baseline is strong, but not because the scene lacks dynamic value.
The privileged teacher wins every final window, and its win is supported by
both objective terms. The failure is that deployable causal students cannot
identify these windows/actions under validation without falling back to static.

This makes another shallow teacher-compression interface low-value. The closed
tier already includes scalar runtime risk, one-step teacher cost memory, and
short teacher snippet replay.

## Artifact Gap

The runs enabled learned event forecasting, but the augmented learned-event
probability columns were not saved in the truth CSV or rollout NPZ files, and
`teacher_dataset.npz` does not store feature names.

Consequence: current artifacts cannot directly audit whether the learned event
probabilities are calibrated to teacher-improvement windows. Future runs should
save either:

- augmented truth with `learned_event_p_h*` columns; or
- feature names / feature slices for `TeacherDataset.features`.

## Decision

The next correction should target objective/forecast transfer, not another
student interface wrapper. The minimum useful implementation change is to make
the deployable context auditable by saving learned-event forecast columns and
feature names. After that, selection should be redesigned around
teacher-improvement/regret windows rather than generic event-risk windows.
