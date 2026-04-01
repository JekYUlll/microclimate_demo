````markdown
# RL Sensor Scheduling Framework — Implementation Spec for codeX

## 0. Document purpose

This document specifies a **complete re-architecture** of the current experiment suite into a **power-constrained Markov decision process / deep reinforcement learning framework** for multi-sensor scheduling.

The goal is **not** to continue the previous workflow of:
- generating synthetic data,
- applying several missingness patterns,
- comparing many interpolation methods,
- then training forecasting models.

Instead, the goal is to build a **closed-loop framework** in which:

1. an environment evolves over time,
2. a scheduler decides which sensors to activate under a total power constraint,
3. an estimator updates the latent state using the received observations,
4. a learned policy optimizes long-term sensing decisions,
5. downstream forecasting models are evaluated using the data produced by the learned scheduling policy.

The final output should be a **reusable framework**, not a one-off case implementation.

---

# 1. Core research problem

We model the problem as:

> Under a total power / activation budget, decide online which sensors to sample so that the long-term estimation and forecasting performance is optimized.

The new system must support:
- **MDP formulation**
- **Deep RL policy training**
- **state estimation**
- **downstream forecasting evaluation**
- **business-case-specific adapters** for data generation / real sensors / sensor power / observation shapes

The framework must support:
- synthetic environments now
- real sensor integration later
- multiple estimators
- multiple RL algorithms
- multiple forecasting backbones

---

# 2. Scope boundaries

## 2.1 What this framework must do
- Build a **generic RL-based sensor scheduling framework**
- Use **Kalman filtering as the default estimator**
- Support **discrete sensor subset scheduling under total power constraints**
- Support **business-level environment adapters**
- Support **evaluation using multiple forecasting models**
- Support **rule-based scheduling baselines**
- Preserve cost/coverage/event-based analysis

## 2.2 What this framework should NOT prioritize
- Do **not** continue the previous large-scale interpolation benchmark as a main research axis
- Do **not** make "which interpolation method is best" the central question
- Do **not** hard-code the current synthetic windblown dataset logic into the core framework
- Do **not** make the core framework depend on one specific dataset schema

## 2.3 What should remain as legacy / benchmark
The existing `experiments_scheduling_suite/` should be retained as **Benchmark V1 / legacy pipeline**, but the new framework should be implemented separately and cleanly.

---

# 3. High-level architecture

The new framework must be separated into five layers:

1. **Environment Layer**
   - defines latent system evolution
   - provides true state and observations
   - enforces sensor refresh / availability / noise rules

2. **Sensor Layer**
   - defines sensor specifications
   - power cost
   - measurement shape
   - refresh interval
   - observation model

3. **Estimator Layer**
   - default: Kalman Filter
   - optionally EKF / UKF later
   - consumes observations selected by scheduler
   - produces state summary for RL

4. **Scheduler Layer**
   - rule-based baselines
   - DQN-based scheduler first
   - optional later: Double DQN / PPO / constrained RL
   - outputs sensor subset action

5. **Forecasting Evaluation Layer**
   - takes data produced under a scheduling policy
   - evaluates forecasting models
   - compares multi-step prediction quality

---

# 4. Formal problem definition to implement

## 4.1 Latent environment dynamics
At time step `t`, latent state evolves as:

x_{t+1} = F(x_t, u_t) + w_t

Where:
- `x_t`: latent state
- `u_t`: optional exogenous input / regime variable
- `w_t`: process noise

Implementation note:
- for the first minimal version, implement a **linear Gaussian environment**
- define:
  - `x_{t+1} = A x_t + w_t`
  - `w_t ~ N(0, Q)`

Later, allow a business adapter to implement nonlinear dynamics.

## 4.2 Sensor observation model
Each sensor `j` produces an observation:

y_{j,t} = H_j(x_t) + v_{j,t}

Where:
- `H_j`: sensor-specific observation function
- `v_{j,t}`: measurement noise

Implementation note:
- V1 must support **linear observation**
  - `y_{j,t} = C_j x_t + v_{j,t}`
- Later, allow nonlinear observation functions via adapter methods

## 4.3 Action
At each time step, the scheduler selects a subset of sensors:

a_t ∈ A

Subject to:
- total active sensors <= K
- total power <= B_t or average power budget
- optional minimum coverage ratio constraints
- optional minimum on-time constraints

Implementation note:
- V1 should use a **finite discrete action space**
- action = one valid sensor subset from an enumerated feasible set
- this is required for straightforward DQN

## 4.4 State for RL
The RL state should not be the raw latent truth; it should be the **estimator-side available information summary**.

Recommended state vector components:
1. estimator posterior mean `x_hat_t`
2. estimator uncertainty summary
   - `trace(P_t)`
   - diagonal of `P_t`
   - optionally flattened upper triangle if dimension is small
3. freshness of each sensor
   - `time_since_last_seen_j`
4. coverage ratio of each sensor over a recent window
5. remaining power / normalized budget statistics
6. previous action summary
7. optional event indicators from environment
   - regime id
   - threshold exceeded
   - event flag

Implementation note:
- V1 should use a compact state vector:
  - `diag(P_t)` or `trace(P_t)`
  - `time_since_last_seen`
  - `coverage_ratio`
  - `current_budget_ratio`
  - `previous_action_onehot`

## 4.5 Reward / cost
Use a one-step cost:

C_t = alpha * EstimationLoss_t
    + beta * PredictionLoss_t
    + lambda * PowerCost_t
    + eta * SwitchCost_t
    + rho * CoveragePenalty_t

Reward:
R_t = -C_t

### Recommended training stages
#### Stage A (mandatory first)
Train RL using:
- `EstimationLoss_t = trace(P_t)`
- plus `PowerCost_t`
- optional `SwitchCost_t`

This is the most stable first implementation.

#### Stage B (optional after Stage A works)
Add downstream forecast-oriented loss:
- short horizon forecast error
- event-weighted forecast loss

Implementation note:
- Do NOT start from prediction loss only
- First stabilize the MDP using estimator-side loss

---

# 5. Development roadmap (must follow in order)

## Phase 1 — Core minimal RL framework
Implement the simplest valid closed-loop version:

- linear Gaussian latent environment
- multiple sensors with different `C_j`, `R_j`, power cost
- Kalman filter estimator
- finite discrete action space
- DQN scheduler
- baseline schedulers:
  - random
  - periodic
  - round robin
  - max-uncertainty-first
  - info-priority
- training reward based on `trace(P)` + power cost

Deliverable:
- a complete RL training loop that converges
- plots comparing baselines vs DQN on estimation cost under budget

## Phase 2 — Business adapter interface
Refactor the framework so the environment/sensor logic is not hard-coded.

Deliverable:
- `BaseEnvironment`
- `BaseSensor`
- `BaseEstimator`
- `BaseScheduler`
- `BasePredictor`
- one synthetic linear case
- one windblown business case adapter stub

## Phase 3 — Windblown / microclimate business case integration
Integrate the existing synthetic windblown logic as a **business case**, not as framework core logic.

Deliverable:
- windblown environment adapter
- sensor specs for business case
- power configuration file
- observation mapping file
- event definitions

## Phase 4 — Downstream forecasting evaluation
Use the trajectories produced by each scheduler to create forecasting datasets and evaluate:
- LSTM
- Transformer
- Informer
- TCN
- MLP
- XGBoost
- Naive

Deliverable:
- model comparison under learned scheduling policy
- event vs non-event forecasting comparison
- power-performance tradeoff figures

## Phase 5 — Extended analysis
Deliver:
- Pareto front
- rank stability
- significance tests
- event-based analysis
- scheduler ablations

---

# 6. Directory structure to create

Create a new independent project:

```text
rl_sensor_scheduling_framework/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── env/
│   │   ├── linear_gaussian_case.yaml
│   │   └── windblown_case.yaml
│   ├── sensors/
│   │   ├── linear_gaussian_sensors.yaml
│   │   └── windblown_sensors.yaml
│   ├── estimator/
│   │   ├── kalman.yaml
│   │   ├── ekf.yaml
│   │   └── ukf.yaml
│   ├── scheduler/
│   │   ├── random.yaml
│   │   ├── periodic.yaml
│   │   ├── round_robin.yaml
│   │   ├── info_priority.yaml
│   │   └── dqn.yaml
│   ├── predictor/
│   │   ├── naive.yaml
│   │   ├── mlp.yaml
│   │   ├── lstm.yaml
│   │   ├── transformer.yaml
│   │   ├── informer.yaml
│   │   └── tcn.yaml
│   └── sweeps/
│       ├── phase1_debug.yaml
│       ├── phase1_full.yaml
│       ├── phase3_windblown.yaml
│       └── phase4_forecasting_eval.yaml
├── data/
│   ├── raw/
│   ├── generated/
│   ├── interim/
│   └── processed/
├── src/
│   ├── core/
│   │   ├── types.py
│   │   ├── registry.py
│   │   ├── interfaces.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── seed.py
│   ├── envs/
│   │   ├── base_environment.py
│   │   ├── linear_gaussian_env.py
│   │   └── windblown_env.py
│   ├── sensors/
│   │   ├── base_sensor.py
│   │   ├── linear_sensor.py
│   │   ├── windblown_sensor.py
│   │   ├── power_models.py
│   │   └── observation_models.py
│   ├── estimators/
│   │   ├── base_estimator.py
│   │   ├── kalman_filter.py
│   │   ├── ekf.py
│   │   ├── ukf.py
│   │   └── state_summary.py
│   ├── scheduling/
│   │   ├── base_scheduler.py
│   │   ├── action_space.py
│   │   ├── constraints.py
│   │   ├── baselines/
│   │   │   ├── random_scheduler.py
│   │   │   ├── periodic_scheduler.py
│   │   │   ├── round_robin_scheduler.py
│   │   │   ├── max_uncertainty_scheduler.py
│   │   │   └── info_priority_scheduler.py
│   │   └── rl/
│   │       ├── replay_buffer.py
│   │       ├── q_network.py
│   │       ├── dqn_agent.py
│   │       ├── epsilon_scheduler.py
│   │       └── train_dqn.py
│   ├── forecasting/
│   │   ├── dataset_builder.py
│   │   ├── base_predictor.py
│   │   ├── baselines.py
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── informer.py
│   │   ├── tcn.py
│   │   └── factory.py
│   ├── evaluation/
│   │   ├── estimation_metrics.py
│   │   ├── forecast_metrics.py
│   │   ├── cost_metrics.py
│   │   ├── event_metrics.py
│   │   ├── significance.py
│   │   └── rank_analysis.py
│   ├── visualization/
│   │   ├── training_curves.py
│   │   ├── policy_plots.py
│   │   ├── estimation_plots.py
│   │   ├── forecast_plots.py
│   │   ├── pareto_plots.py
│   │   └── posthoc_plots.py
│   ├── business_cases/
│   │   └── windblown_case/
│   │       ├── adapter.py
│   │       ├── sensor_specs.py
│   │       ├── generator_adapter.py
│   │       ├── event_definitions.py
│   │       └── predictor_targets.py
│   └── pipelines/
│       ├── run_phase1_debug.py
│       ├── run_phase1_training.py
│       ├── run_phase3_windblown.py
│       ├── run_phase4_forecast_eval.py
│       └── run_full_pipeline.py
├── scripts/
│   ├── 00_generate_business_data.py
│   ├── 01_train_rl_scheduler.py
│   ├── 02_evaluate_scheduler.py
│   ├── 03_build_forecast_dataset.py
│   ├── 04_train_predictors.py
│   ├── 05_evaluate_forecasts.py
│   ├── 06_posthoc_analysis.py
│   └── 07_export_report.py
├── reports/
│   ├── runs/
│   └── aggregate/
└── tests/
    ├── test_env.py
    ├── test_sensor.py
    ├── test_kalman.py
    ├── test_action_space.py
    ├── test_scheduler_baselines.py
    ├── test_dqn_agent.py
    └── test_forecast_dataset_builder.py
````

---

# 7. Interface requirements (strict)

## 7.1 BaseEnvironment

File: `src/envs/base_environment.py`

Must define:

```python
class BaseEnvironment(ABC):
    def reset(self) -> dict:
        """Reset environment and return initial latent state / info."""

    def step(self, action: Any) -> dict:
        """
        Advance environment by one time step under the chosen action.
        Return:
            {
                "latent_state": ...,
                "available_observations": ...,
                "event_flags": ...,
                "done": bool,
                "info": dict
            }
        """

    def get_ground_truth(self) -> dict:
        """Return current latent truth for analysis / supervised evaluation."""

    def get_time_index(self) -> int:
        """Return current discrete time step."""
```

## 7.2 BaseSensor

File: `src/sensors/base_sensor.py`

```python
class BaseSensor(ABC):
    @property
    def sensor_id(self) -> str: ...

    def power_cost(self) -> float:
        """Return current activation cost."""

    def can_sample(self, t: int) -> bool:
        """Whether the sensor is available at time t (refresh interval / cooldown)."""

    def observe(self, latent_state: Any) -> dict:
        """Return sensor observation."""
```

## 7.3 BaseEstimator

File: `src/estimators/base_estimator.py`

```python
class BaseEstimator(ABC):
    def reset(self) -> None: ...
    def predict(self) -> None: ...
    def update(self, observations: list[dict]) -> None: ...
    def get_state_estimate(self) -> Any: ...
    def get_uncertainty_summary(self) -> dict: ...
    def get_rl_state_features(self) -> dict: ...
```

## 7.4 BaseScheduler

File: `src/scheduling/base_scheduler.py`

```python
class BaseScheduler(ABC):
    def reset(self) -> None: ...
    def act(self, state: dict) -> Any: ...
    def observe_transition(self, transition: dict) -> None:
        """Optional hook for RL training / logging."""
```

## 7.5 BasePredictor

File: `src/forecasting/base_predictor.py`

```python
class BasePredictor(ABC):
    def fit(self, train_data, val_data=None) -> None: ...
    def predict(self, test_data) -> Any: ...
    def evaluate(self, test_data) -> dict: ...
```

---

# 8. Action space design (must be concrete)

## 8.1 Phase 1 action space

Use a discrete finite action space:

* all valid sensor subsets satisfying:

  * total sensors <= K
  * total power cost <= B_step

Implementation:

* `src/scheduling/action_space.py`
* precompute all feasible subsets
* map each subset to an integer action id

Example:

* if 5 sensors, max active 2:

  * actions include all 1-sensor and 2-sensor subsets
* if power costs differ:

  * prune subsets whose cost exceeds step budget

This is mandatory for first DQN implementation.

## 8.2 Phase 2 optional extension

If action space becomes too large:

* implement parameterized subset selection
* or sequential action selection
* or actor-critic with masked action head

Do NOT implement this before Phase 1 works.

---

# 9. RL algorithm requirements

## 9.1 Mandatory first algorithm

Implement **DQN** first.

Required components:

* Q-network
* target network
* replay buffer
* epsilon-greedy exploration
* periodic target update
* gradient clipping
* deterministic seeding

State input:

* flattened numeric state vector

Action output:

* Q-values over feasible subset actions

## 9.2 Required baseline schedulers

Implement:

1. `random_scheduler.py`
2. `periodic_scheduler.py`
3. `round_robin_scheduler.py`
4. `max_uncertainty_scheduler.py`
5. `info_priority_scheduler.py`

### max_uncertainty_scheduler

Use estimator uncertainty summary to prioritize sensing the most uncertain dimensions / sensors.
(Implementation method: map sensor to the subset of state variables it observes; use corresponding posterior variance as score.)

### info_priority_scheduler

Use a weighted heuristic:

* uncertainty gain proxy
* freshness
* event weight
* coverage deficit
* switch penalty

(Implementation method: fixed weighted sum with config-defined coefficients.)

---

# 10. Estimator implementation requirements

## 10.1 Kalman filter is the default

Implement standard linear KF in:

* `src/estimators/kalman_filter.py`

Must support:

* arbitrary `A`, `Q`
* sensor-specific `C_j`, `R_j`
* subset observation update
* missing observation handling

## 10.2 State summary for RL

Implement:

* `trace(P)`
* `diag(P)`
* optionally `logdet(P)` if stable
* per-sensor freshness
* per-sensor recent coverage ratio
* budget ratio
* previous action encoding

This summary goes to RL, not the full latent truth.

---

# 11. Business case integration requirements

## 11.1 Windblown case should be an adapter, not framework core

Create:

* `src/business_cases/windblown_case/adapter.py`
* `src/business_cases/windblown_case/sensor_specs.py`
* `src/business_cases/windblown_case/generator_adapter.py`
* `src/business_cases/windblown_case/event_definitions.py`
* `src/business_cases/windblown_case/predictor_targets.py`

### windblown_case responsibilities

* wrap existing synthetic generator logic
* define latent state fields
* define sensor observation mapping
* define sensor power costs
* define event logic
* define downstream prediction targets

### Important constraint

Do NOT directly copy the old experiment pipeline logic into framework core.

---

# 12. Sensor power / data shape definitions

These belong to business case code, not framework core.

For each sensor, define:

* sensor id
* observation dimension
* observation variables
* refresh interval
* power cost
* optional startup/warmup delay
* measurement noise
* availability rule

Implementation location:

* `src/business_cases/windblown_case/sensor_specs.py`

Use a structured dataclass, for example:

```python
@dataclass
class SensorSpec:
    sensor_id: str
    obs_dim: int
    variables: list[str]
    refresh_interval: int
    power_cost: float
    startup_delay: int = 0
    noise_std: float | list[float] = 0.0
```

---

# 13. Forecasting evaluation design

## 13.1 Why forecasting is still needed

The framework should not stop at estimator-side metrics. The final scientific value is:

* does RL scheduling improve downstream forecasting under cost constraints?

## 13.2 Dataset construction under a learned scheduler

Implement:

* rollout an environment under a fixed trained scheduler
* collect resulting observation histories
* build forecasting datasets from the scheduler-generated data stream

This means forecasting datasets are no longer based on manually imposed missingness masks.
They are generated by actual policy decisions.

## 13.3 Mandatory forecasting models

Implement / reuse minimal versions of:

* Naive
* MLP
* LSTM
* Transformer
* Informer
* TCN
* XGBoost (optional if already easy to reuse)

## 13.4 Mandatory metrics

* RMSE
* MAE
* MAPE
* event vs non-event metrics
* cost-performance tradeoff

---

# 14. Evaluation protocol

## 14.1 Phase 1 (estimation-focused)

Compare schedulers on:

* average `trace(P)`
* average power cost
* average reward
* coverage ratio
* switching frequency

## 14.2 Phase 2 (forecast-focused)

For each scheduler:

* build forecasting dataset
* train forecasting models
* compare:

  * H=1,2,3
  * later longer horizons (10, 30, 60 steps if needed)
* event vs non-event
* performance-cost Pareto

## 14.3 Statistical analysis

Keep and port the existing aggregate analysis mindset:

* rank correlation across schedulers
* paired significance tests
* sensitivity to budget
* sensitivity to action space size
* sensitivity to event frequency

---

# 15. Scripts to implement

## 15.1 `01_train_rl_scheduler.py`

Train DQN or baseline scheduler in a specified environment config.

Required CLI arguments:

* `--env_cfg`
* `--sensor_cfg`
* `--estimator_cfg`
* `--scheduler_cfg`
* `--run_id`

Outputs:

* trained scheduler weights
* training logs
* reward curves
* estimation metrics
* config snapshot

## 15.2 `02_evaluate_scheduler.py`

Evaluate a trained scheduler without forecasting.

Outputs:

* estimation metrics tables
* power/coverage tables
* scheduler action statistics
* plots

## 15.3 `03_build_forecast_dataset.py`

Run environment under a trained scheduler and export dataset for forecasting.

Outputs:

* train/val/test forecasting arrays
* metadata
* scheduler rollout info

## 15.4 `04_train_predictors.py`

Train forecasting models on scheduler-generated data.

Outputs:

* trained predictors
* preds
* metrics

## 15.5 `05_evaluate_forecasts.py`

Aggregate forecasting metrics.

Outputs:

* per-model metrics
* event/non-event metrics
* summary plots

## 15.6 `06_posthoc_analysis.py`

Aggregate across multiple scheduler runs / budgets / environments.

Outputs:

* rank correlation
* Pareto front
* boxplots
* sensitivity curves
* heatmaps

---

# 16. Logging / artifact requirements

Every run must save:

* `config_used.yaml`
* `metrics_estimation.csv`
* `metrics_forecast.csv` (if applicable)
* `scheduler_stats.csv`
* `coverage_stats.csv`
* `power_stats.csv`
* training curves
* model checkpoints

Directory:

* `reports/runs/<run_id>/...`

Aggregate outputs:

* `reports/aggregate/...`

---

# 17. Mandatory plots

## Scheduler / estimation plots

* reward vs episode
* average `trace(P)` vs episode
* average power cost vs episode
* action frequency histogram
* coverage ratio per sensor
* Pareto front: estimation quality vs power

## Forecasting plots

* per-model `pred_h123.png`
* scheduler comparison summary plot
* event vs non-event performance
* rank correlation heatmap
* cost-performance summary

---

# 18. Tests to implement

Minimum tests:

1. KF predict/update correctness for a toy linear system
2. action space feasibility under power constraints
3. baseline scheduler outputs valid actions
4. DQN agent can overfit a tiny toy environment
5. dataset builder creates aligned forecast targets
6. coverage / power stats are computed correctly

All tests should run with:

* deterministic seed
* small toy configs

---

# 19. Explicit implementation constraints

1. The new framework must be **independent** of the old missingness/interpolation suite.
2. The old suite can be reused only as a conceptual reference, not as a dependency.
3. Business logic must be implemented through adapters and specs.
4. The first complete milestone must be a **linear Gaussian + KF + DQN** end-to-end working pipeline.
5. The first RL reward must be **estimation-oriented**, not prediction-oriented.
6. Forecasting evaluation comes after the RL scheduling loop is stable.

---

# 20. Deliverables and acceptance criteria

## Deliverable A — Phase 1 complete

A fully running minimal RL scheduling system with:

* linear Gaussian environment
* multiple sensors
* power-constrained discrete subset actions
* KF estimator
* random / periodic / round-robin / info-priority / DQN comparison

Acceptance:

* DQN training converges stably
* scheduler outputs valid actions
* DQN beats random baseline under same budget
* all plots and logs are saved

## Deliverable B — Phase 3 complete

A business-case integration of windblown synthetic environment.

Acceptance:

* framework can swap from linear toy case to windblown adapter without modifying core code
* sensor specs and power costs are defined through adapter
* scheduler rollouts can generate forecasting datasets

## Deliverable C — Phase 4 complete

Forecasting evaluation under learned scheduling policies.

Acceptance:

* all required forecasting models train and evaluate
* multi-step metrics are produced
* event-based metrics are produced
* scheduler-vs-forecast tradeoff can be plotted

---

# 21. Recommended implementation order

1. Define interfaces and core types
2. Implement linear Gaussian environment
3. Implement sensor specs and action space
4. Implement Kalman filter
5. Implement baseline schedulers
6. Implement DQN
7. Implement training / evaluation loop for Phase 1
8. Implement business case adapter interface
9. Integrate windblown case
10. Build forecasting dataset exporter
11. Train and evaluate forecasting models
12. Implement aggregate posthoc analysis

Do NOT change this order unless blocked by a hard engineering dependency.

---

# 22. Final note for codeX

This is a **framework project**, not a one-off experiment.

Design choices must prioritize:

* interface clarity
* modularity
* clean separation between framework core and business case
* reproducibility
* easy extension to real sensors later

Where a design choice is ambiguous, prefer:

* simpler first implementation,
* explicit interfaces,
* finite discrete action spaces,
* deterministic reproducible behavior,
* metrics that match the optimization objective.

End of spec.

```
```
