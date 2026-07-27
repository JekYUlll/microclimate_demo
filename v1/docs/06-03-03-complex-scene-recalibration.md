# Complex Scene Recalibration Report

Date: 2026-06-03

Scope: v1 forecast-aware constrained scheduling only.

## Current Blockage

The v5/e70 scene is constraint-active but still not complex enough. It prevents
continuous `core+laser+fc4`, yet it still allows `core+laser` to become a very
strong static anchor in some seeds. In those regimes, deployable dynamic
policies can reduce power and sometimes improve oracle/broad metrics, but they
lose the configured snow-task objective because static laser directly observes
the particle variables.

Latest evidence:

- `learned_hybrid_contextual_duty_riskcenter_safe`, v5/e70, `w=0.30`:
  deployable `1/3`, teacher `3/3`.
- `learned_hybrid_teacher_mix_guarded_safe`, v5/e70, `w=0.30`:
  deployable `1/3`, teacher `3/3`.
- Teacher remains positive, so dynamic value exists.
- Existing deployable compression/selection is not enough under laser-anchor
  static regimes.

## Why Static Still Wins

1. `core+laser` remains a near-complete direct snow-task stack.
   The task targets include particle diameter and velocity, and laser observes
   both with low noise. Even without fc4, this is enough to keep task error low.

2. The calibrated constraint mainly removes `laser+fc4`, not `laser`.
   This helps seed41, where static becomes a proxy/fc4 stack and dynamic
   selective laser helps. It does not help seeds42/44, where static still
   executes mostly `core+laser`.

3. The current task objective rewards direct target observations too strongly.
   A dynamic policy that saves energy and improves oracle/broad metrics can
   still lose if it gives up continuous laser observations of the task columns.

4. Proxy complementarity is too weak.
   `snow_particle_counter` is cheaper but noisy/saturated and not central in
   teacher rollouts. The dynamic value is mostly laser/fc4/context mixing, not
   a clean proxy-vs-direct scheduling problem.

## Required v6 Properties

A useful complex scene should make every fixed static mask incomplete.

Required gates:

- `core+laser+fc4` infeasible.
- Continuous `core+laser` should either be infeasible over the eval horizon or
  insufficient for the task objective.
- `core+fc4+context` should be useful but incomplete because it lacks particle
  detail.
- `core+SPC+fc4` should be feasible and useful in some regimes, but not a
  complete replacement for laser.
- Teacher must beat validation-selected static by a non-marginal margin on
  seed41/42/44 before any deployable training resumes.

## Candidate v6 Changes

1. Make laser episodic, not static-dominant.
   Increase laser running/startup cost or reduce energy capacity/reserve so
   continuous `core+laser` cannot last a 256-step evaluation window.

2. Make mass-flux and particle variables jointly necessary.
   Keep `fc4_flux` as the direct mass-flux sensor, and ensure particle
   diameter/velocity alone do not solve the task-composite objective.

3. Strengthen proxy usefulness without making proxy static dominant.
   Lower SPC power moderately or reduce non-event noise, but keep event
   saturation so SPC is a regime-dependent proxy.

4. Add regime-specific sensor relevance.
   During high-transport regimes, laser should be valuable for short bursts;
   during lower-risk regimes, fc4/context/proxy should be preferable. Static
   should not cover both regimes with one mask.

5. Revisit task weighting only after scene calibration.
   Do not use a global weight to force wins. First ensure the physical scene
   has no single static direct stack that solves the task.

## Next Work

1. Add v6 candidate sensor/truth calibration config.
2. Run structural audit only.
3. Run static/teacher-only seed41/42/44 calibration.
4. Accept v6 only if static no longer dominates through `core+laser` and
   teacher wins with clear margin.
5. Resume deployable-policy experiments only after those gates pass.

