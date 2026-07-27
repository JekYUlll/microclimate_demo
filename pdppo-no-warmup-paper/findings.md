# Findings: No-Warmup PD-PPO Paper Track

## Initial Evidence Imported From Prior Track

- No-warmup base grid partially completed:
  - B=1.65: 10 completed seeds; PD-PPO beat selected/static in 9/10 but beat
    the best fair non-PPO baseline in 0/10.
  - B=1.70: 5 completed seeds; PD-PPO beat selected/static in 5/5 but beat the
    best fair non-PPO baseline in 1/5.
  - Duty often failed the clarified behavior target, typically with multiple
    always-on/off sensors.
- No-warmup hard-duty reduced probe:
  - seed41: dynamic duty valid; PD-PPO beat static but lost to round-robin and
    the best constrained baseline.
  - seed42: dynamic duty valid; PD-PPO lost to selected/static and round-robin,
    but beat the best duty-constrained baseline.
  - seed43: dynamic duty valid; PD-PPO beat selected/static but lost to
    round-robin, AoI, and the best duty-constrained baseline.

## Current Interpretation

No-warmup is promising only for a narrower claim: it can weaken selected/static
allocations. It is not yet evidence for broad learned-scheduler superiority.
The next experiments must either find a hard-duty setting that preserves the
static advantage or freeze the paper around a regime-map claim instead of a
dominance claim.

## Aggregated Claim Gates

- Base no-warmup:
  - runs: 15;
  - selected/static wins: 14/15;
  - best original fair baseline wins: 1/15;
  - mean duty remains invalid: mid-duty sensors about 3.85, always-on about
    1.00, always-off about 3.00.
- Hard-duty reduced:
  - runs: 3;
  - selected/static wins: 2/3;
  - best original fair baseline wins: 0/3;
  - best constrained baseline wins: 1/3;
  - dynamic duty valid in all 3 runs: mid=8, always_on=0, always_off=0.

## Paper-Framing Consequence

The no-warmup paper cannot honestly be framed as "PD-PPO dominates dynamic
heuristics" with current evidence. The defensible paper direction is:

1. no-warmup removes one structural reason static allocations dominate;
2. PD-PPO frequently beats selected/static allocations in that regime;
3. enforcing operationally valid dynamic duty is possible;
4. performance against dynamic heuristics remains the limiting condition and
   must be reported as a negative or mixed result unless further補跑 changes it.

## Rejected Shortcuts

- Do not relabel duty-constrained heuristic wins as general heuristic wins.
- Do not hide unconstrained round-robin/AoI if they beat PD-PPO.
- Do not promote a no-warmup run with multiple always-on/off sensors as the
  final behavioral result.

## DQN Diagnostic Design

The DQN diagnostic should not use the older all-sequence `29_v2_train_dqn.py`
protocol as final evidence. The implemented diagnostic instead reuses the
existing no-warmup split artifacts:

- same truth CSV;
- same frozen TCN oracle checkpoint;
- same chronological split manifest;
- same final-test start indices;
- same validation-selected static candidate.

This makes the DQN comparison a credible answer to whether the policy optimizer
is the bottleneck. If DQN still loses round-robin/AoI, the remaining problem is
mostly scene/objective structure rather than PPO alone.

## DQN Diagnostic Interim Result

After the first two strict split runs:

- DQN does not beat source PD-PPO: `0/2`.
- DQN does not beat AoI or round-robin: `0/2`.
- DQN beats validation/static in only `1/2`.
- DQN's duty behavior is better than base no-warmup PD-PPO:
  - mean mid-duty sensors `7`;
  - always-on sensors `0`;
  - always-off sensors `0`.

Interpretation: plain DQN helps schedule diversity but does not solve the loss
advantage problem. If the remaining four runs follow this pattern, the no-warmup
failure against dynamic heuristics is not primarily a PPO-specific failure.

After four strict split DQN runs, the pattern strengthened:

- wins source PD-PPO: `1/4`;
- wins validation/static: `2/4`;
- wins round-robin: `0/4`;
- wins AoI: `0/4`;
- wins best non-DQN policy: `0/4`;
- duty remains valid with no always-on/off collapse.

This is strong interim evidence that replacing PD-PPO with plain DQN does not
solve the no-warmup dynamic-heuristic gap.

## DQN Diagnostic Final Six-Run Result

After the strict split DQN pilot completed all six planned runs:

- wins source PD-PPO: `1/6`;
- wins validation/static: `3/6`;
- wins round-robin: `0/6`;
- wins AoI: `1/6`;
- wins best non-DQN policy: `0/6`;
- duty remains valid in all six runs:
  `mean mid=7.5`, `always_on=0`, `always_off=0`.

Interpretation: ordinary DQN is not a fast replacement for PD-PPO. It fixes
the always-on/off collapse better than base PD-PPO, but it does not beat the
dynamic heuristic baselines. This shifts the bottleneck back to scenario and
objective design, not merely optimizer choice.

## Env-Dwell12 Reduced Result

The first split-protocol hard-duty plus environment-dwell12 trained run is
positive:

- run:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_reduced/raw/budget1p70_seed41`;
- custom PPO loss `0.132886`;
- validation-selected static `0.137648`;
- feasible static projected `0.146423`;
- round-robin `0.158871`;
- AoI `0.141682`;
- best duty-constrained baseline
  `duty_constrained_round_robin=0.134288`;
- duty valid: `mid=8`, `always_on=0`, `always_off=0`;
- switch rate `0.024377`.

This was the first no-warmup result that satisfies all promotion gates in a
single split-protocol run.

Seed42 completed later and is mixed:

- custom PPO loss `0.149620`;
- validation/static `0.138138`, so static still wins this seed;
- round-robin `0.154632`, AoI `0.166598`, random `0.161948`;
- best duty-constrained baseline
  `duty_constrained_feasible_static_projected=0.160709`;
- duty valid: `mid=8`, `always_on=0`, `always_off=0`;
- switch rate `0.026860`.

Two-seed interpretation: env-dwell12 reduced reliably removes the dynamic
heuristic advantage so far (`2/2` vs round-robin and AoI), while static
dominance remains unresolved (`1/2` vs validation/static). This is useful for
an operational dynamic-baseline claim, not yet for a full first-paper main
claim.

Seed43 was later available in the local synced artifacts:

- custom PPO loss `0.140702`;
- validation/static `0.144098`;
- round-robin `0.168811`, AoI `0.150172`, random `0.150048`;
- best duty-constrained baseline `duty_constrained_aoi=0.151801`;
- duty valid: `mid=8`, `always_on=0`, `always_off=0`;
- switch rate `0.029383`.

Three-seed interpretation: env-dwell12 reduced is the first setting with a
coherent positive story:

- `3/3` against round-robin and AoI;
- `3/3` against duty-constrained baselines;
- `2/3` against validation/static;
- `3/3` operationally valid duty.

It supports a constrained-deployment dynamic-baseline claim. It still should
not be overstated as unconditional dominance over static.

## Base No-Warmup B=1.75 Monitor

B=1.75 seed41 completed on the remote server and was synced locally:

- run:
  `reports/v31_split_protocol_no_warmup/raw/budget1p75_seed41`;
- custom PPO loss `0.126576`;
- validation/static `0.144749`;
- round-robin `0.115072`;
- AoI `0.118588`;
- random `0.121769`;
- duty invalid: `always_on=1`, `always_off=3`, `mid=2`;
- switch rate `0.074516`.

Interpretation: B=1.75 weakens static but does not solve the base no-warmup
failure mode. The learned policy still loses to dynamic heuristics and remains
quasi-static. This should remain diagnostic/auxiliary, not a first-paper main
result.

B=1.75 seed42 later completed:

- custom PPO loss `0.147619`;
- validation/static `0.146270`;
- round-robin `0.147178`;
- AoI `0.151523`;
- random `0.154893`;
- best non-learned comparator `feasible_static_projected=0.146270`;
- duty invalid: `always_on=1`, `always_off=3`, `mid=4`;
- switch rate `0.160549`.

Two-seed interpretation: B=1.75 does not improve the base no-warmup claim. It
is `1/2` against validation/static, `0/2` against best non-learned, and `0/2`
on deployment-valid duty. Keep it diagnostic only.
