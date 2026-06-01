from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.features import ForecastContextConfig, build_event_forecast, sensor_timing_features
from forecast_cmdp.event_forecaster import (
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    select_event_forecast_columns,
    train_event_forecaster,
)
from forecast_cmdp.cost_policy import (
    ActionCostTrainingConfig,
    ForecastAwareAdvantageResidualPolicy,
    ForecastAwareCostPolicy,
    ForecastAwareEnsembleValuePolicy,
    collect_anchor_advantage_dataset,
    collect_action_cost_dataset,
    train_anchor_advantage_model,
    train_action_cost_ensemble,
    train_action_cost_model,
)
from forecast_cmdp.dataset import collect_dagger_dataset, collect_teacher_dataset, concat_teacher_datasets
from forecast_cmdp.mpc_teacher import (
    MpcTeacherConfig,
    beam_search_teacher_action,
    enumerate_action_masks,
    snapshot_env,
)
from forecast_cmdp.policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    ForecastAwareCyclePolicy,
    ForecastAwareEventSupportCyclePolicy,
    ForecastAwareEventThresholdPolicy,
    ForecastAwareKNNPolicy,
    ForecastAwareMaskBCPolicy,
    ForecastAwareTeacherRatePolicy,
    load_bc_policy_checkpoint,
    save_bc_policy_checkpoint,
    train_bc_classifier,
    train_mask_bc,
)
from forecast_cmdp.protocol import task_focus_metrics
from forecast_cmdp.reuse import ensure_archive_src
from forecast_cmdp.selection import choose_deployable_validation_row

ensure_archive_src()

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import RolloutResult  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)


class DummyOracle:
    is_fitted = True

    class Cfg:
        horizon = 3

    cfg = Cfg()

    def loss(self, feature, future):
        del feature
        return float(np.mean(np.abs(future)))


class SaturatingOracle(DummyOracle):
    def loss(self, feature, future):
        del feature, future
        return 10.0


def make_truth(n: int = 64) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    event = (t >= 20) & (t < 32)
    return pd.DataFrame(
        {
            "wind_speed_ms": 6.0 + 5.0 * event.astype(float),
            "wind_direction_deg": np.full(n, 180.0),
            "air_temperature_c": -20.0 + 0.01 * t,
            "relative_humidity": np.full(n, 70.0),
            "air_pressure_pa": np.full(n, 70000.0),
            "solar_radiation_wm2": np.zeros(n),
            "snow_surface_temperature_c": -22.0 + 0.01 * t,
            "snow_particle_mean_diameter_mm": 0.2 * event.astype(float),
            "snow_particle_mean_velocity_ms": 5.0 * event.astype(float),
            "snow_mass_flux_kg_m2_s": 1.0e-5 * event.astype(float),
            "event_flag": event,
        }
    )


def make_sensors() -> list[SensorSpecV2]:
    return [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.2, 0.2, warmup_steps=0),
        SensorSpecV2("snow", ("snow_particle_mean_velocity_ms",), 0.4, 0.5, warmup_steps=2),
        SensorSpecV2("flux", ("snow_mass_flux_kg_m2_s",), 0.1, 0.1, warmup_steps=0),
    ]


def make_env() -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        make_truth(),
        make_sensors(),
        PowerConstraintsV2(max_active=2, per_step_budget=0.6, startup_peak_budget=0.7, required_sensor_ids=("met",)),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=16,
            seed=7,
            energy_account_enabled=True,
            energy_capacity=10.0,
            initial_energy=10.0,
            harvest_per_step=0.3,
            reserve_energy=1.0,
        ),
        oracle=DummyOracle(),
    )


def make_saturating_env() -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        make_truth(),
        make_sensors(),
        PowerConstraintsV2(max_active=2, per_step_budget=0.6, startup_peak_budget=0.7),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=16,
            seed=7,
        ),
        oracle=SaturatingOracle(),
    )


def test_event_forecast_truth_future_vector_shape():
    truth = make_truth()
    forecast = build_event_forecast(truth, 18, ForecastContextConfig(horizon=5, truth_future=True))
    assert forecast.probabilities.shape == (5,)
    assert forecast.confidence.shape == (5,)
    assert forecast.as_vector().shape == (11,)
    assert forecast.probabilities[1] == 1.0
    assert 0.0 <= forecast.time_to_event <= 1.0


def test_event_forecast_can_read_learned_probability_columns():
    truth = make_truth()
    truth["learned_event_p_h1"] = np.linspace(0.0, 1.0, len(truth))
    truth["learned_event_p_h2"] = 0.25
    forecast = build_event_forecast(
        truth,
        10,
        ForecastContextConfig(
            horizon=3,
            learned_event_probability_columns=("learned_event_p_h1", "learned_event_p_h2"),
        ),
    )
    assert forecast.probabilities.shape == (3,)
    assert np.isclose(forecast.probabilities[0], truth.loc[10, "learned_event_p_h1"])
    assert np.isclose(forecast.probabilities[1], 0.25)
    assert forecast.probabilities[2] == 0.0


def test_learned_event_forecaster_augments_truth_without_future_columns():
    truth = make_truth(72)
    cfg = EventForecasterTrainingConfig(
        horizon=3,
        lookback=4,
        hidden_dim=16,
        epochs=2,
        batch_size=16,
        seed=11,
        device="cpu",
        period_steps=8,
    )
    columns = select_event_forecast_columns(
        truth,
        preferred_columns=("wind_speed_ms", "snow_particle_mean_velocity_ms"),
        event_column="event_flag",
    )
    dataset = build_event_forecast_dataset(
        truth,
        bounds=(0, 48),
        feature_columns=columns,
        event_column="event_flag",
        cfg=cfg,
    )
    assert dataset.features.shape[0] == dataset.targets.shape[0]
    assert dataset.targets.shape[1] == 3
    bundle = train_event_forecaster(dataset, cfg)
    augmented, probability_columns = augment_truth_with_event_forecasts(truth, bundle)
    assert len(probability_columns) == 3
    for column in probability_columns:
        assert column in augmented.columns
        assert np.all((augmented[column].to_numpy() >= 0.0) & (augmented[column].to_numpy() <= 1.0))
    forecast = build_event_forecast(
        augmented,
        20,
        ForecastContextConfig(horizon=3, learned_event_probability_columns=probability_columns),
    )
    assert forecast.probabilities.shape == (3,)


def test_sensor_timing_features_shape():
    features = sensor_timing_features(
        warmup_steps=[0, 2, 3],
        power_costs=[0.2, 0.4, 0.9],
        startup_peaks=[0.2, 0.5, 1.2],
        horizon=8,
    )
    assert features.shape == (3, 3)
    assert np.all(features >= 0.0)


def test_mpc_teacher_returns_feasible_action_and_restores_env():
    env = make_env()
    env.reset(start_idx=18)
    before = snapshot_env(env)
    masks = enumerate_action_masks(3, max_active=2)
    action = beam_search_teacher_action(env, masks, MpcTeacherConfig(planning_horizon=3, beam_width=3, max_branch=4))
    assert action.shape == (3,)
    projection = env.projector.project_mask(action, env.runtimes)
    assert projection.feasible
    after = snapshot_env(env)
    assert after["current_idx"] == before["current_idx"]
    assert np.allclose(after["history"], before["history"])
    assert np.allclose(after["previous_action_mask"], before["previous_action_mask"])


def test_mpc_teacher_bootstraps_coverage_when_oracle_loss_saturates():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    action = beam_search_teacher_action(env, masks, MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4))
    assert np.any(action)
    assert action[0]


def test_mpc_teacher_candidate_prior_breaks_saturated_ties():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    desired_idx = int(np.flatnonzero(np.all(masks == np.asarray([[0, 0, 1]], dtype=bool), axis=1))[0])
    costs = np.ones(masks.shape[0], dtype=float)
    costs[desired_idx] = 0.0
    action = beam_search_teacher_action(
        env,
        masks,
        MpcTeacherConfig(
            planning_horizon=1,
            beam_width=2,
            max_branch=masks.shape[0],
            saturated_coverage_bonus=0.0,
            candidate_prior_weight=5.0,
            candidate_prior_costs=tuple(float(x) for x in costs),
        ),
    )
    assert action.tolist() == [False, False, True]


def test_mpc_teacher_anchor_guard_defaults_to_anchor_without_regret_gain():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (False, False, True)
    action = beam_search_teacher_action(
        env,
        masks,
        MpcTeacherConfig(
            planning_horizon=1,
            beam_width=2,
            max_branch=masks.shape[0],
            saturated_coverage_bonus=0.0,
            anchor_mask=anchor,
            anchor_regret_guard=True,
            anchor_improvement_margin=0.1,
        ),
    )
    assert action.tolist() == [False, False, True]


def test_task_focus_metrics_uses_event_filtered_normalized_error():
    result = RolloutResult(
        policy_name="p",
        observations=np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=float),
        masks=np.zeros((2, 2), dtype=float),
        truth=np.asarray([[1.0, 1.0], [4.0, 10.0]], dtype=float),
        rewards=np.zeros(2, dtype=float),
        scores=np.zeros((2, 1), dtype=float),
        powers=np.zeros(2, dtype=float),
        peaks=np.zeros(2, dtype=float),
        selected_masks=np.zeros((2, 1), dtype=bool),
        mode_ids=np.zeros((2, 1), dtype=int),
        event_flags=np.asarray([0.0, 1.0], dtype=float),
        oracle_losses=np.zeros(2, dtype=float),
        step_indices=np.asarray([0, 1], dtype=int),
        warmup_abort_count=0,
        warmup_abort_deltas=np.zeros(2, dtype=int),
        energy_guard_dropped=np.zeros(2, dtype=int),
        soc=np.zeros(2, dtype=float),
    )
    metrics = task_focus_metrics(
        result,
        state_columns=("a", "b"),
        task_error_columns=("a", "b"),
        task_error_scales=(2.0, 3.0),
        event_only=True,
    )
    assert np.isclose(metrics["task_error_event_mean"], 1.5)
    assert np.isclose(metrics["task_error_mean"], 1.5)


def test_static_margin_guard_prefers_robust_validation_policy():
    rows = [
        {
            "policy": "forecast_aware_advantage_residual",
            "objective": 0.91,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.03,
            "objective_margin_min": -0.05,
            "negative_start_count": 2,
        },
        {
            "policy": "forecast_aware_value_residual",
            "objective": 0.93,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.01,
            "objective_margin_min": -0.005,
            "negative_start_count": 1,
        },
    ]
    selected_mean = choose_deployable_validation_row(rows.copy(), criterion="mean_objective")
    assert selected_mean["policy"] == "forecast_aware_advantage_residual"
    selected_guard = choose_deployable_validation_row(
        rows.copy(),
        criterion="static_margin_guard",
        min_mean_margin=0.0,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_guard["policy"] == "forecast_aware_value_residual"
    assert selected_guard["static_margin_guard_pass"]


def test_teacher_dataset_and_bc_policy_smoke(tmp_path):
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=2, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=True)
    dataset = collect_teacher_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=6,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.features.shape[0] == 6
    assert dataset.action_masks.shape == (6, masks.shape[0])
    model, history = train_bc_classifier(
        dataset.features,
        dataset.labels,
        dataset.action_masks,
        cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareBCPolicy(model=model, candidate_masks=masks, forecast_cfg=forecast_cfg, device="cpu")
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible
    fallback = tuple(bool(x) for x in masks[1])
    guarded_policy = ForecastAwareBCPolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        fallback_mask=fallback,
        min_logit_margin=1.0e9,
    )
    env.reset(start_idx=18)
    guarded_action = guarded_policy.act_mask(env)
    assert guarded_action.tolist() == list(fallback)
    env = make_env()
    dagger = collect_dagger_dataset(
        env,
        masks,
        policy=policy,
        start_indices=(18,),
        steps_per_start=2,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    combined = concat_teacher_datasets([dataset, dagger])
    assert combined.features.shape[0] == dataset.features.shape[0] + dagger.features.shape[0]
    checkpoint = tmp_path / "bc_policy.pt"
    save_bc_policy_checkpoint(
        checkpoint,
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        train_cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
        history=history,
    )
    loaded = load_bc_policy_checkpoint(checkpoint, device="cpu")
    env.reset(start_idx=18)
    loaded_action = loaded.act_mask(env)
    assert loaded_action.shape == (3,)
    assert env.projector.project_mask(loaded_action, env.runtimes).feasible
    knn_policy = ForecastAwareKNNPolicy(
        features=combined.features,
        labels=combined.labels,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        k=3,
    )
    env.reset(start_idx=18)
    knn_action = knn_policy.act_mask(env)
    assert knn_action.shape == (3,)
    assert env.projector.project_mask(knn_action, env.runtimes).feasible
    cycle_policy = ForecastAwareCyclePolicy(labels=combined.labels, candidate_masks=masks)
    env.reset(start_idx=18)
    cycle_action = cycle_policy.act_mask(env)
    assert cycle_action.shape == (3,)
    assert env.projector.project_mask(cycle_action, env.runtimes).feasible
    mask_model, mask_history = train_mask_bc(
        dataset.features,
        dataset.labels,
        dataset.candidate_masks,
        cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
    )
    assert len(mask_history["loss"]) == 2
    mask_policy = ForecastAwareMaskBCPolicy(
        model=mask_model,
        forecast_cfg=forecast_cfg,
        device="cpu",
        required_sensor_indices=(0,),
    )
    env.reset(start_idx=18)
    mask_action = mask_policy.act_mask(env)
    assert mask_action.shape == (3,)
    assert mask_action[0]
    assert env.projector.project_mask(mask_action, env.runtimes).feasible


def test_bc_policy_can_preserve_warming_sensor():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    met_only_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    met_snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])

    class FixedLogitModel:
        input_dim = env._state().shape[0] + forecast_cfg.horizon * 2 + 1
        hidden_dim = 1
        n_actions = masks.shape[0]

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            logits = torch.zeros((x.shape[0], masks.shape[0]), dtype=torch.float32, device=x.device)
            logits[:, met_only_idx] = 10.0
            logits[:, met_snow_idx] = 1.0
            return logits

    env.reset(start_idx=18)
    env.step_mask(np.asarray([True, True, False], dtype=bool))
    assert str(env.runtimes["snow"].mode.name).lower() == "warming"
    unguarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=False,
    )
    guarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=True,
    )
    assert unguarded.act_mask(env).tolist() == [True, False, False]
    assert guarded.act_mask(env).tolist() == [True, True, False]
    support_guarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=False,
        allowed_action_indices=(met_snow_idx,),
    )
    assert support_guarded.act_mask(env).tolist() == [True, True, False]


def test_event_threshold_policy_switches_on_forecast_probability():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.0
    truth.loc[20:, "learned_event_p_h1"] = 0.8
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    event_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventThresholdPolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_idx=event_idx,
        threshold=0.5,
        aggregation="max",
    )
    env.reset(start_idx=18)
    assert policy.act_mask(env).tolist() == [True, False, False]
    env.reset(start_idx=22)
    assert policy.act_mask(env).tolist() == [True, False, True]


def test_event_support_cycle_policy_rotates_teacher_supported_actions():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventSupportCyclePolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_indices=(snow_idx, flux_idx),
        threshold=0.5,
        aggregation="max",
        cycle_period=1,
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env).tolist()
    env.reset(start_idx=19)
    second = policy.act_mask(env).tolist()
    assert first != second
    assert first in ([True, True, False], [True, False, True])
    assert second in ([True, True, False], [True, False, True])


def test_event_support_cycle_policy_can_select_by_freshness():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventSupportCyclePolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_indices=(flux_idx, snow_idx),
        threshold=0.5,
        aggregation="max",
        selection_mode="freshness",
    )
    env.reset(start_idx=50)
    env.runtimes["met"].last_observed_step = 50
    env.runtimes["snow"].last_observed_step = None
    env.runtimes["flux"].last_observed_step = 50
    assert policy.act_mask(env).tolist() == [True, True, False]


def test_teacher_rate_policy_tracks_target_duty_cycle():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareTeacherRatePolicy(
        candidate_masks=masks,
        target_rates=(1.0, 0.5, 0.5),
        allowed_action_indices=(snow_idx, flux_idx),
        freshness_weight=0.0,
        power_weight=0.0,
        preserve_warming=False,
        anchor_mask=(True, False, False),
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env)
    env.step_mask(first)
    second = policy.act_mask(env)
    assert first.tolist() in ([True, True, False], [True, False, True])
    assert second.tolist() in ([True, True, False], [True, False, True])
    assert first.tolist() != second.tolist()


def test_action_cost_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    dataset = collect_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    model, history = train_action_cost_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareCostPolicy(model=model, candidate_masks=masks, forecast_cfg=forecast_cfg, device="cpu")
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible

    import torch

    met_snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])

    class FixedCostModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            # Prefer actions without the snow sensor unless the support guard
            # removes them from the runtime candidate set.
            return x[:, -2:-1].clone()

    guarded_policy = ForecastAwareCostPolicy(
        model=FixedCostModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        allowed_action_indices=(met_snow_idx,),
    )
    env.reset(start_idx=18)
    assert guarded_policy.act_mask(env).tolist() == [True, True, False]


def test_ensemble_value_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    dataset = collect_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    models, histories = train_action_cost_ensemble(
        dataset,
        ActionCostTrainingConfig(
            epochs=2,
            batch_size=8,
            hidden_dim=16,
            device="cpu",
            ensemble_size=2,
            bootstrap_fraction=0.75,
        ),
    )
    assert len(models) == 2
    assert len(histories) == 2
    anchor = tuple(bool(x) for x in masks[0])
    policy = ForecastAwareEnsembleValuePolicy(
        models=models,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        uncertainty_beta=0.5,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_anchor_advantage_residual_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    dataset = collect_anchor_advantage_dataset(
        env,
        masks,
        anchor_mask=anchor,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    assert dataset.anchor_idx == anchor_idx
    model, history = train_anchor_advantage_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareAdvantageResidualPolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_anchor_advantage_dataset_handles_projected_anchor():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor = (True, True, True)
    env.reset(start_idx=18)
    assert not np.array_equal(env.projector.project_mask(np.asarray(anchor), env.runtimes).selected_mask, np.asarray(anchor))
    dataset = collect_anchor_advantage_dataset(
        env,
        masks,
        anchor_mask=anchor,
        start_indices=(18,),
        steps_per_start=2,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    assert dataset.anchor_idx is not None
    assert np.any(np.isclose(dataset.advantages, 0.0))


def test_advantage_residual_policy_falls_back_to_projected_anchor():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    anchor = (True, True, True)
    met_only_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)

    class NegativeAdvantageModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            out = torch.full((x.shape[0], 1), -1.0, dtype=torch.float32, device=x.device)
            # The deployable candidate set still contains a valid action, but a
            # non-positive predicted advantage should choose the static anchor
            # mask and let the environment projector execute it.
            out[:, 0] = -1.0
            return out

    policy = ForecastAwareAdvantageResidualPolicy(
        model=NegativeAdvantageModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(met_only_idx,),
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.tolist() == [True, True, True]
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_advantage_residual_policy_does_not_open_full_space_when_support_invalid():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    anchor = (True, True, True)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([anchor], dtype=bool), axis=1))[0])
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)

    class PositiveAdvantageModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            return torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)

    policy = ForecastAwareAdvantageResidualPolicy(
        model=PositiveAdvantageModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(anchor_idx,),
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    projection = env.projector.project_mask(np.asarray(anchor), env.runtimes)
    assert not np.array_equal(projection.selected_mask, np.asarray(anchor))
    action = policy.act_mask(env)
    assert action.tolist() == [True, True, True]


def test_budget_matrix_parse_budget_ignores_matrix_root_name():
    module_path = ROOT / "v1" / "scripts" / "aggregate_budget_matrix.py"
    spec = importlib.util.spec_from_file_location("aggregate_budget_matrix_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path(
        "v1/artifacts/budget_matrix_learned_hybrid_event_guarded_paired/"
        "budget1p20/learned_hybrid_event_guarded_safe_seed41/gate_summary.json"
    )
    assert module.parse_budget(path) == 1.20


def test_claim_assessment_enforces_win_rate_against_actual_n():
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = pd.DataFrame(
        {
            "preset": ["main"] * 10,
            "deployable_margin": [1.0] * 7 + [-1.0] * 3,
            "teacher_margin": [1.0] * 10,
        }
    )
    assessment = module.assess_claim(runs, main_preset="main", min_seeds=5, min_win_rate=0.8)
    assert assessment["required_wins"] == 8
    assert assessment["deployable_wins"] == 7
    assert not assessment["claim_pass"]


def test_claim_aggregate_collects_multiple_roots(tmp_path):
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for idx, root in enumerate((tmp_path / "suite_a", tmp_path / "suite_b"), start=1):
        run_dir = root / f"main_seed{idx}"
        run_dir.mkdir(parents=True)
        (run_dir / "gate_summary.json").write_text(
            """
{
  "validation_selected_static_objective": 1.0,
  "teacher_reference_objective": 0.8,
  "best_deployable_objective": 0.9,
  "teacher_beats_static": true,
  "gate_pass": true,
  "objective_metric": "task_composite",
  "best_deployable_policy": "forecast_aware_event_threshold"
}
""".strip(),
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text('{"seed": %d}' % idx, encoding="utf-8")
    run_rows, policy_rows = module.collect_suite_roots([tmp_path / "suite_a", tmp_path / "suite_b"])
    assert len(run_rows) == 2
    assert not policy_rows


def test_budget_matrix_assessment_enforces_win_rate_against_actual_n():
    module_path = ROOT / "v1" / "scripts" / "aggregate_budget_matrix.py"
    spec = importlib.util.spec_from_file_location("aggregate_budget_matrix_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = pd.DataFrame(
        {
            "budget": [1.2],
            "preset": ["main"],
            "n": [10],
            "deployable_wins": [7],
            "deployable_margin_mean": [0.1],
            "teacher_wins": [10],
        }
    )
    assessment = module.assess(summary, min_seeds=5, min_win_rate=0.8)
    assert assessment["required_wins_by_group"]["budget=1.20 preset=main"] == 8
    assert not assessment["matrix_pass"]
