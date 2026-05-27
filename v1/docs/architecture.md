# Forecast-Aware Constrained Scheduling Architecture

## Purpose

This v1 implementation is a new research branch, not an incremental patch to
`rl_sensor_scheduling_framework`. The archived framework remains the source of
truth for previous experiments and reusable simulation components.

The new method targets the failure mode observed in corrected PD-PPO:
forecast-loss reward was dynamic, but the policy lacked explicit future event
context and learned high-latency sensor timing through weak, delayed RL credit.

## Reused Components

From `rl_sensor_scheduling_framework/src/v2`:

- `WarmupSchedulingEnv`: deterministic warm-up/SOC replay environment.
- `SensorRuntime`: warm-up and observation readiness state.
- `PowerProjector`: hard max-active, steady-power and startup-peak feasibility.
- `SensorSpecV2`: sensor metadata including warm-up and power costs.
- `TCNFrozenForecastOracle`: frozen forecast-loss scorer.
- `rollout.py` and `evaluation.py`: rollout storage and metrics.

The v1 package imports these modules through a small path shim. It does not edit
their source.

## New Components

### Forecast Context

`forecast_cmdp.features` builds explicit event forecast features:

- future event probabilities over horizon `H`;
- normalized time-to-event;
- confidence / sharpness;
- sensor timing features such as warm-up-to-horizon and power.

Teacher code may use truth-future event labels. Deployed policies should use
causal or forecasted probabilities.

### MPC Teacher

`forecast_cmdp.mpc_teacher` performs short-horizon beam search over feasible
sensor masks. It evaluates candidate schedules in the archived warm-up
environment using frozen oracle loss plus constraint costs.

The teacher is training-only. It can use truth event labels and/or a fixed frozen
oracle to generate labels for behavior cloning.

### Policy Path

The intended training path is:

1. collect MPC teacher labels on the RL-train partition;
2. train a forecast-aware behavior cloning policy;
3. optionally apply advantage-weighted imitation;
4. only then add constrained RL fine-tuning with reward/cost critics and dual
   variables.

This ordering keeps credit assignment controlled before adding full CMDP
complexity.

## Evaluation Contract

Every protocol runner must preserve:

- oracle pretrain split;
- teacher/RL train split;
- validation split for static comparator selection and hyperparameter gating;
- final-test split used only once for reporting.

The strict comparator remains `validation_selected_static`. A new controller is
not considered successful unless it beats that comparator in same-run final-test
evaluation and then scales across seeds.
