# PD-PPO No-Warmup Paper Track

This directory is an isolated project track for the no-warmup PD-PPO paper.
It does not own or modify the `v1/` exploration line or the existing
`rl_sensor_scheduling_framework/` scene-recalibration line.

## Purpose

- Organize no-warmup PD-PPO evidence.
- Rerun only the missing server-side experiments needed for a defensible paper.
- Produce a fresh manuscript draft under `paper/`.

## Boundaries

- Experiments run on the server only.
- Local work is limited to planning, wrapper scripts, result aggregation, and
  paper drafting.
- Existing framework code is treated as an experiment backend. New orchestration
  belongs in this directory.
- Claims must distinguish:
  - advantage over selected/static allocations;
  - advantage over dynamic heuristics;
  - operationally constrained heuristic comparisons;
  - dynamic-duty validity.

## Main Files

- `task_plan.md` - active execution plan.
- `findings.md` - persistent evidence and decisions.
- `progress.md` - chronological session log.
- `configs/no_warmup_matrix.yaml` - experiment matrix to complete.
- `scripts/collect_no_warmup_results.py` - aggregate existing and new results.
- `scripts/remote_start_no_warmup_advantage_grid.sh` - server tmux launcher.
- `docs/evidence_ledger.md` - claim-by-claim evidence ledger.
- `paper/paper.tex` - fresh manuscript draft.
