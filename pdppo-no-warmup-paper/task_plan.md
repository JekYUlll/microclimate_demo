# Task Plan: No-Warmup PD-PPO Paper Track

## Goal

Build an isolated no-warmup PD-PPO paper line: organize existing no-warmup
results, rerun the missing server-side experiments needed for a defensible
advantage scenario, and produce a fresh paper draft.

## Non-Conflict Rule

This track must not modify:

- `v1/` exploration code, plans, or paper outputs;
- existing PD-PPO scene-recalibration plan files under
  `rl_sensor_scheduling_framework/.planning/`;
- existing PD-PPO paper files under `rl_sensor_scheduling_framework/paper/`.

Allowed writes are limited to this directory, plus server-side experiment
outputs produced by wrapper scripts.

## Current Phase

Phase 3

## Phases

### Phase 1: Isolated Project Bootstrap

- [x] Create a new top-level project directory.
- [x] Create fresh plan, progress, and findings files.
- [x] Add result aggregation and remote-run wrapper skeletons.
- [x] Add a fresh paper draft skeleton.
- **Status:** complete

### Phase 2: Evidence Inventory

- [x] Aggregate current no-warmup base-grid results.
- [x] Aggregate current no-warmup hard-duty reduced results.
- [x] Build a claim-gate summary table.
- [x] Identify exact missing runs and failed gates after server status check.
- **Status:** complete

### Phase 3: Server-Only补跑

- [ ] Complete remaining no-warmup base grid if still running.
- [x] Complete hard-duty no-warmup seed43.
- [x] Add strict split-protocol DQN diagnostic runner.
- [x] Launch DQN diagnostic pilot for B=1.65/1.70, seeds 41--43.
- [x] Aggregate DQN diagnostic results.
- [ ] If needed, run one targeted advantage profile that keeps dynamic duty
      valid while improving over selected/static and operational baselines.
- [x] Sync only light result artifacts: CSV, JSON metadata, logs, summary
      tables.
- **Status:** pending

### Phase 4: Paper Claim Freeze

- [ ] Decide the primary claim boundary from completed evidence.
- [ ] Freeze the table set and figure list.
- [ ] Mark unsupported claims explicitly as rejected.
- **Status:** pending

### Phase 5: Fresh Manuscript Draft

- [ ] Expand `paper/paper.tex` into a complete first draft.
- [ ] Add tables generated from the evidence ledger.
- [ ] Add figures or figure placeholders with exact data sources.
- [ ] Add a focused bibliography.
- **Status:** pending

## Acceptance Gates

| Gate | Requirement |
| --- | --- |
| Static advantage | PD-PPO must beat validation-selected/static allocations in the accepted setting. |
| Dynamic heuristic comparison | Report round-robin/AoI/random honestly; do not hide them. |
| Operational realism | Final promoted setting should avoid multiple always-on/off sensors. |
| Server-only execution | No training/evaluation experiment may run locally. |
| Paper honesty | The paper may claim static-baseline advantage only if dynamic-baseline superiority is not supported. |

## Current Working Hypothesis

No-warmup weakens the static shortcut enough for PD-PPO to beat selected/static
allocations in many seeds. However, without hard duty it often learns
quasi-static schedules; with hard duty it becomes behaviorally valid but the
performance advantage is not reliable. The paper must be built around the
strongest reproducible no-warmup claim that survives these gates, likely a
static-shortcut-break/regime-map claim rather than broad dynamic-policy
dominance.

## Immediate Next Actions

1. Run the aggregator against current framework reports.
2. Check server state for the running hard-duty seed43 and base no-warmup grid.
3. Update `docs/evidence_ledger.md` and `CHANGELOG.md` after every new result.
4. Expand the manuscript only after the claim boundary is frozen.
