from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .features import (
    ForecastContextConfig,
    build_event_forecast,
    event_forecast_feature_names,
    learned_continuous_column_name,
)
from .policy import ForecastAwareProxyMPCPolicy
from .reuse import ensure_archive_src
from .window_risk import ControllerSpec
from .window_risk_model import WindowRiskModelBundle

ensure_archive_src()

from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


CONTROLLER_PARAMETER_NAMES = (
    "event_weight",
    "magnitude_weight",
    "variability_weight",
    "freshness_weight",
    "target_rate_weight",
    "anchor_bias",
    "power_weight",
    "switch_weight",
    "min_soc",
    "min_dwell",
    "planning_depth",
    "beam_width",
    "max_branch",
    "age_weight",
    "anchor_improvement_threshold",
    "max_anchor_hamming",
)

CAUSAL_HISTORY_CORE_COLUMNS = (
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
)
CAUSAL_HISTORY_WINDOWS = (64, 256, 1024)
CAUSAL_HISTORY_STATS = ("mean", "std", "max", "last", "delta")
RESIDUAL_HISTORY_WINDOWS = (64, 256)
RESIDUAL_HISTORY_STATS = ("mean", "std", "last", "delta")


def causal_window_agent_state(env: WarmupSchedulingEnv) -> np.ndarray:
    """Remove the simulator-only current event label from the policy state."""

    state = np.asarray(env._state(), dtype=float).reshape(-1)
    event_offset = 2 if bool(env._energy_enabled()) else 1
    event_idx = int(state.size - event_offset)
    if event_idx < 0:
        raise ValueError("Agent state is too short to remove the current event label")
    return np.delete(state, event_idx)


def compact_window_state(env: WarmupSchedulingEnv) -> tuple[np.ndarray, tuple[str, ...]]:
    seconds = float(int(env.current_idx) * int(env.cfg.base_freq_s))
    theta = 2.0 * np.pi * ((seconds % 86400.0) / 86400.0)
    return (
        np.asarray([np.sin(theta), np.cos(theta), float(env._soc_ratio())], dtype=np.float32),
        ("window_phase_sin", "window_phase_cos", "window_initial_soc_ratio"),
    )


def residual_boundary_state(
    env: WarmupSchedulingEnv,
) -> tuple[np.ndarray, tuple[str, ...]]:
    normalized_observation = (
        np.asarray(env.last_observation, dtype=float)
        - np.asarray(env.state_mean, dtype=float)
    ) / np.asarray(env.state_std, dtype=float)
    observed_coverage = np.mean(
        np.asarray(env.mask_history, dtype=float),
        axis=0,
    )
    max_warmup = max(
        1,
        max(
            (int(sensor.warmup_steps) for sensor in env.sensor_specs),
            default=1,
        ),
    )
    modes = np.asarray(
        [float(env.runtimes[sid].mode) / 2.0 for sid in env.sensor_ids],
        dtype=float,
    )
    warm_remaining = np.asarray(
        [
            float(env.runtimes[sid].warm_remaining) / float(max_warmup)
            for sid in env.sensor_ids
        ],
        dtype=float,
    )
    freshness = np.asarray(
        [
            np.log1p(float(env.runtimes[sid].freshness(env.current_idx)))
            for sid in env.sensor_ids
        ],
        dtype=float,
    )
    previous_mask = np.asarray(env.previous_action_mask, dtype=float)
    feature = np.concatenate(
        [
            normalized_observation,
            observed_coverage,
            modes,
            warm_remaining,
            freshness,
            previous_mask,
        ]
    )
    names = (
        tuple(f"residual_boundary_observation_{name}" for name in env.state_columns)
        + tuple(f"residual_boundary_coverage_{name}" for name in env.state_columns)
        + tuple(f"residual_boundary_mode_{sid}" for sid in env.sensor_ids)
        + tuple(f"residual_boundary_warm_remaining_{sid}" for sid in env.sensor_ids)
        + tuple(f"residual_boundary_log_freshness_{sid}" for sid in env.sensor_ids)
        + tuple(f"residual_boundary_previous_mask_{sid}" for sid in env.sensor_ids)
    )
    return feature.astype(np.float32), names


def causal_history_summary(
    env: WarmupSchedulingEnv,
    forecast_cfg: ForecastContextConfig,
    *,
    windows: Sequence[int] = CAUSAL_HISTORY_WINDOWS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    horizon = max(1, int(forecast_cfg.horizon))
    horizon_indices = tuple(sorted({1, (horizon + 1) // 2, horizon}))
    columns = list(CAUSAL_HISTORY_CORE_COLUMNS)
    event_columns = tuple(str(x) for x in forecast_cfg.learned_event_probability_columns)
    columns.extend(
        event_columns[idx - 1]
        for idx in horizon_indices
        if idx - 1 < len(event_columns)
    )
    for target in tuple(str(x) for x in forecast_cfg.continuous_columns):
        columns.extend(
            learned_continuous_column_name(
                str(forecast_cfg.learned_continuous_prefix),
                target,
                horizon_idx,
            )
            for horizon_idx in horizon_indices
        )
    values: list[float] = []
    names: list[str] = []
    current_idx = int(env.current_idx)
    for window in tuple(int(x) for x in windows):
        if window <= 0:
            raise ValueError("Causal history windows must be positive")
        begin = max(0, current_idx - window)
        for column in columns:
            if column in env.truth_df.columns and current_idx > begin:
                series = env.truth_df.iloc[begin:current_idx][column].to_numpy(dtype=float)
                series = series[np.isfinite(series)]
            else:
                series = np.zeros(0, dtype=float)
            if series.size:
                stats = {
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series)),
                    "max": float(np.max(series)),
                    "last": float(series[-1]),
                    "delta": float(series[-1] - series[0]),
                }
            else:
                stats = {name: 0.0 for name in CAUSAL_HISTORY_STATS}
            for stat_name in CAUSAL_HISTORY_STATS:
                values.append(float(stats[stat_name]))
                names.append(f"causal_history_w{window}_{column}_{stat_name}")
    return np.asarray(values, dtype=np.float32), tuple(names)


def anchor_neighborhood_support(
    candidate_masks: np.ndarray,
    *,
    anchor_mask: Sequence[bool],
    support: Sequence[int],
    max_hamming_distance: int,
) -> tuple[int, ...]:
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    if masks.ndim != 2 or masks.shape[1] != anchor.size:
        raise ValueError("Candidate masks and anchor mask must have matching widths")
    valid_support = tuple(
        sorted(
            {
                int(idx)
                for idx in support
                if 0 <= int(idx) < int(masks.shape[0])
            }
        )
    )
    if int(max_hamming_distance) < 0:
        return valid_support
    distances = np.count_nonzero(masks != anchor[None, :], axis=1)
    selected = tuple(
        idx
        for idx in valid_support
        if int(distances[int(idx)]) <= int(max_hamming_distance)
    )
    if not selected:
        raise ValueError("Anchor neighborhood removed every supported action")
    if not any(bool(np.array_equal(masks[idx], anchor)) for idx in selected):
        raise ValueError("Anchor neighborhood support must retain the static anchor")
    return selected


def residual_action_controller_specs(
    action_indices: Sequence[int],
) -> tuple[ControllerSpec, ...]:
    return tuple(
        ControllerSpec(
            controller_id=f"residual_action_{int(action_idx):03d}",
            parameters={"target_action_idx": int(action_idx)},
        )
        for action_idx in sorted({int(x) for x in action_indices})
    )


def causal_residual_history_summary(
    env: WarmupSchedulingEnv,
    *,
    windows: Sequence[int] = RESIDUAL_HISTORY_WINDOWS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    names: list[str] = []
    current_idx = int(env.current_idx)
    for window in tuple(int(x) for x in windows):
        if window <= 0:
            raise ValueError("Residual history windows must be positive")
        begin = max(0, current_idx - window)
        for column in CAUSAL_HISTORY_CORE_COLUMNS:
            if column in env.truth_df.columns and current_idx > begin:
                series = env.truth_df.iloc[begin:current_idx][column].to_numpy(
                    dtype=float
                )
                series = series[np.isfinite(series)]
            else:
                series = np.zeros(0, dtype=float)
            if series.size:
                stats = {
                    "mean": float(np.mean(series)),
                    "std": float(np.std(series)),
                    "last": float(series[-1]),
                    "delta": float(series[-1] - series[0]),
                }
            else:
                stats = {name: 0.0 for name in RESIDUAL_HISTORY_STATS}
            for stat_name in RESIDUAL_HISTORY_STATS:
                values.append(float(stats[stat_name]))
                names.append(
                    f"residual_history_w{window}_{column}_{stat_name}"
                )
    return np.asarray(values, dtype=np.float32), tuple(names)


def valid_anchor_residual_action(
    candidate_masks: np.ndarray,
    *,
    anchor_idx: int,
    controller: ControllerSpec,
    allowed_action_indices: Sequence[int],
    required_sensor_indices: Sequence[int] = (),
) -> bool:
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor_idx = int(anchor_idx)
    target_idx = int(controller.parameters["target_action_idx"])
    if not (
        0 <= anchor_idx < int(masks.shape[0])
        and 0 <= target_idx < int(masks.shape[0])
    ):
        return False
    if target_idx not in {int(x) for x in allowed_action_indices}:
        return False
    target = masks[target_idx]
    required = tuple(int(x) for x in required_sensor_indices)
    if required and not bool(np.all(target[list(required)])):
        return False
    return int(np.count_nonzero(target != masks[anchor_idx])) == 1


def build_residual_risk_feature(
    *,
    env: WarmupSchedulingEnv,
    forecast_cfg: ForecastContextConfig,
    candidate_masks: np.ndarray,
    anchor_idx: int,
    controller: ControllerSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    compact_state, compact_state_names = compact_window_state(env)
    boundary_state, boundary_state_names = residual_boundary_state(env)
    history, history_names = causal_residual_history_summary(env)
    forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor = masks[int(anchor_idx)].astype(bool)
    target_idx = int(controller.parameters["target_action_idx"])
    target = masks[target_idx].astype(bool)
    delta = target.astype(float) - anchor.astype(float)
    hamming = int(np.count_nonzero(delta))
    if hamming > 1:
        raise ValueError("Residual action must change at most one sensor")
    power = np.asarray(
        [float(sensor.power_cost) for sensor in env.sensor_specs],
        dtype=float,
    )
    peaks = np.asarray(
        [float(sensor.startup_peak_power) for sensor in env.sensor_specs],
        dtype=float,
    )
    warmup = np.asarray(
        [float(sensor.warmup_steps) for sensor in env.sensor_specs],
        dtype=float,
    )
    operation = np.asarray(
        [
            float(hamming == 0),
            float(np.any(delta > 0.0)),
            float(np.any(delta < 0.0)),
        ],
        dtype=float,
    )
    feature = np.concatenate(
        [
            compact_state.astype(float),
            boundary_state.astype(float),
            history.astype(float),
            forecast.as_vector().astype(float),
            anchor.astype(float),
            delta.astype(float),
            operation,
            np.asarray(
                [
                    float(np.sum(power[anchor])),
                    float(np.sum(power[target])),
                    float(np.sum(power * delta)),
                    float(np.sum(peaks[anchor])),
                    float(np.sum(peaks[target])),
                    float(np.sum(peaks * np.maximum(delta, 0.0))),
                    float(np.sum(warmup * np.maximum(delta, 0.0))),
                    float(np.mean(anchor)),
                    float(np.mean(target)),
                    float(hamming),
                ],
                dtype=float,
            ),
        ]
    )
    names = (
        compact_state_names
        + boundary_state_names
        + history_names
        + event_forecast_feature_names(
            horizon=int(forecast_cfg.horizon),
            continuous_columns=tuple(
                str(x) for x in forecast_cfg.continuous_columns
            ),
        )
        + tuple(f"residual_anchor_mask_{sensor.sensor_id}" for sensor in env.sensor_specs)
        + tuple(f"residual_delta_{sensor.sensor_id}" for sensor in env.sensor_specs)
        + ("residual_noop", "residual_add", "residual_drop")
        + (
            "residual_anchor_power",
            "residual_target_power",
            "residual_delta_power",
            "residual_anchor_startup_peak",
            "residual_target_startup_peak",
            "residual_added_startup_peak",
            "residual_added_warmup_steps",
            "residual_anchor_duty",
            "residual_target_duty",
            "residual_hamming",
        )
    )
    return feature.astype(np.float32), names


def make_proxy_mpc_controller(
    *,
    controller: ControllerSpec,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: Sequence[bool],
    support: Sequence[int],
    target_rates: np.ndarray,
    preserve_warming: bool,
) -> ForecastAwareProxyMPCPolicy:
    parameters = controller.parameters
    neighborhood_support = anchor_neighborhood_support(
        np.asarray(candidate_masks, dtype=bool),
        anchor_mask=anchor_mask,
        support=support,
        max_hamming_distance=int(parameters.get("max_anchor_hamming", -1)),
    )
    return ForecastAwareProxyMPCPolicy(
        candidate_masks=np.asarray(candidate_masks, dtype=bool),
        forecast_cfg=forecast_cfg,
        anchor_mask=tuple(bool(x) for x in anchor_mask),
        allowed_action_indices=neighborhood_support,
        target_rates=np.asarray(target_rates, dtype=float),
        event_weight=float(parameters["event_weight"]),
        magnitude_weight=float(parameters["magnitude_weight"]),
        variability_weight=float(parameters["variability_weight"]),
        freshness_weight=float(parameters["freshness_weight"]),
        target_rate_weight=float(parameters["target_rate_weight"]),
        anchor_bias=float(parameters["anchor_bias"]),
        power_weight=float(parameters["power_weight"]),
        switch_weight=float(parameters["switch_weight"]),
        min_soc=float(parameters["min_soc"]),
        min_dwell=int(parameters["min_dwell"]),
        aggregation=str(parameters["aggregation"]),
        planning_depth=int(parameters["planning_depth"]),
        beam_width=int(parameters["beam_width"]),
        max_branch=int(parameters["max_branch"]),
        age_weight=float(parameters["age_weight"]),
        anchor_improvement_threshold=float(parameters["anchor_improvement_threshold"]),
        preserve_warming=bool(preserve_warming),
        name=str(controller.controller_id),
    )


def build_window_risk_feature(
    *,
    env: WarmupSchedulingEnv,
    forecast_cfg: ForecastContextConfig,
    candidate_masks: np.ndarray,
    anchor_idx: int,
    controller: ControllerSpec,
    support: Sequence[int],
    target_rates: np.ndarray,
    preserve_warming: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    compact_state, compact_state_names = compact_window_state(env)
    history, history_names = causal_history_summary(env, forecast_cfg)
    forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor_mask = np.asarray(masks[int(anchor_idx)], dtype=bool)
    policy = make_proxy_mpc_controller(
        controller=controller,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor_mask,
        support=support,
        target_rates=target_rates,
        preserve_warming=preserve_warming,
    )
    sensor_coverage = policy._sensor_column_coverage(env)
    anchor_coverage = (
        np.clip(anchor_mask.astype(float) @ sensor_coverage, 0.0, 1.0)
        if sensor_coverage.size
        else np.zeros(0, dtype=float)
    )
    planned_first_mask = np.asarray(policy.act_mask(env), dtype=float)
    power = np.asarray([float(sensor.power_cost) for sensor in env.sensor_specs], dtype=float)
    peaks = np.asarray([float(sensor.startup_peak_power) for sensor in env.sensor_specs], dtype=float)
    parameter_values = np.asarray(
        [
            float(
                controller.parameters.get(
                    name,
                    -1.0 if name == "max_anchor_hamming" else 0.0,
                )
            )
            for name in CONTROLLER_PARAMETER_NAMES
        ],
        dtype=float,
    )
    feature = np.concatenate(
        [
            compact_state.astype(float),
            history.astype(float),
            forecast.as_vector().astype(float),
            anchor_mask.astype(float),
            np.asarray(
                [
                    float(np.sum(power[anchor_mask])),
                    float(np.sum(peaks[anchor_mask])),
                    float(np.mean(anchor_mask)),
                ],
                dtype=float,
            ),
            anchor_coverage.astype(float),
            np.asarray(target_rates, dtype=float),
            parameter_values,
            planned_first_mask,
            np.asarray(
                [
                    float(np.sum(power[planned_first_mask.astype(bool)])),
                    float(np.mean(np.abs(planned_first_mask - anchor_mask.astype(float)))),
                ],
                dtype=float,
            ),
        ]
    )
    names = (
        compact_state_names
        + history_names
        + event_forecast_feature_names(
            horizon=int(forecast_cfg.horizon),
            continuous_columns=tuple(str(x) for x in forecast_cfg.continuous_columns),
        )
        + tuple(f"anchor_mask_{sensor.sensor_id}" for sensor in env.sensor_specs)
        + ("anchor_power", "anchor_startup_peak", "anchor_duty")
        + tuple(f"anchor_coverage_{column}" for column in forecast_cfg.continuous_columns)
        + tuple(f"teacher_target_rate_{sensor.sensor_id}" for sensor in env.sensor_specs)
        + tuple(f"controller_{name}" for name in CONTROLLER_PARAMETER_NAMES)
        + tuple(f"planned_first_mask_{sensor.sensor_id}" for sensor in env.sensor_specs)
        + ("planned_first_power", "planned_first_switch_from_anchor")
    )
    return feature.astype(np.float32), names


@dataclass
class ForecastAwareMeanRiskControllerPolicy(V2Policy):
    model_bundle: WindowRiskModelBundle
    controllers: tuple[ControllerSpec, ...]
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_action_idx: int
    support: tuple[int, ...]
    target_rates: np.ndarray
    min_risk_lower_bound: float = 0.0
    max_negative_probability: float = 0.25
    min_mean_margin: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_mean_risk_controller"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.target_rates = np.asarray(self.target_rates, dtype=float).reshape(-1)
        self.anchor_action_idx = int(self.anchor_action_idx)
        if not 0 <= self.anchor_action_idx < int(self.candidate_masks.shape[0]):
            raise ValueError("anchor_action_idx is outside candidate_masks")
        if self.target_rates.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("target_rates must match candidate mask width")
        if not self.controllers:
            raise ValueError("Mean-risk controller requires at least one controller")
        self.anchor_mask = self.candidate_masks[self.anchor_action_idx].astype(bool).copy()
        self.reset()

    def reset(self) -> None:
        self.selected_policy: ForecastAwareProxyMPCPolicy | None = None
        self.selected_controller_id: str | None = None
        self.static_fallback = False
        self.selection_rows: list[dict[str, object]] = []

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self.selected_policy is None and not self.static_fallback:
            self._select_controller(env)
        if self.selected_policy is None:
            return self.anchor_mask.copy()
        return np.asarray(self.selected_policy.act_mask(env), dtype=bool)

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_controller(self, env: WarmupSchedulingEnv) -> None:
        features = []
        policies = []
        for controller in self.controllers:
            feature, names = build_window_risk_feature(
                env=env,
                forecast_cfg=self.forecast_cfg,
                candidate_masks=self.candidate_masks,
                anchor_idx=self.anchor_action_idx,
                controller=controller,
                support=self.support,
                target_rates=self.target_rates,
                preserve_warming=self.preserve_warming,
            )
            if names != self.model_bundle.feature_names:
                raise ValueError("Runtime mean-risk feature schema differs from training")
            features.append(feature)
            policies.append(
                make_proxy_mpc_controller(
                    controller=controller,
                    candidate_masks=self.candidate_masks,
                    forecast_cfg=self.forecast_cfg,
                    anchor_mask=self.anchor_mask,
                    support=self.support,
                    target_rates=self.target_rates,
                    preserve_warming=self.preserve_warming,
                )
            )
        predictions = self.model_bundle.predict(np.asarray(features, dtype=np.float32))
        negative_enabled = self.model_bundle.negative_model is not None
        safe_indices = []
        for idx, controller in enumerate(self.controllers):
            mean_margin = float(predictions["mean_margin"][idx])
            q25_margin = float(predictions["q25_margin"][idx])
            lower = float(predictions["risk_lower_bound"][idx])
            negative_probability = float(predictions["negative_probability"][idx])
            safe = (
                mean_margin >= float(self.min_mean_margin)
                and lower >= float(self.min_risk_lower_bound)
                and (
                    not negative_enabled
                    or negative_probability <= float(self.max_negative_probability)
                )
            )
            self.selection_rows.append(
                {
                    "controller_id": str(controller.controller_id),
                    "mean_margin": mean_margin,
                    "q25_margin": q25_margin,
                    "risk_lower_bound": lower,
                    "negative_probability": negative_probability,
                    "safe": bool(safe),
                }
            )
            if safe:
                safe_indices.append(int(idx))
        if not safe_indices:
            self.static_fallback = True
            return
        selected_idx = max(
            safe_indices,
            key=lambda idx: (
                float(predictions["mean_margin"][idx]),
                float(predictions["risk_lower_bound"][idx]),
                -int(idx),
            ),
        )
        self.selected_policy = policies[int(selected_idx)]
        self.selected_policy.reset()
        self.selected_controller_id = str(self.controllers[int(selected_idx)].controller_id)


@dataclass
class RecedingForecastAwareMeanRiskControllerPolicy(ForecastAwareMeanRiskControllerPolicy):
    decision_interval: int = 64
    name: str = "receding_forecast_aware_mean_risk_controller"

    def __post_init__(self) -> None:
        self.decision_interval = max(1, int(self.decision_interval))
        super().__post_init__()

    def reset(self) -> None:
        super().reset()
        self.macro_step = 0
        self.block_history: list[dict[str, object]] = []
        self.all_selection_rows: list[dict[str, object]] = []

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if int(self.macro_step) % int(self.decision_interval) == 0:
            self.selected_policy = None
            self.selected_controller_id = None
            self.static_fallback = False
            self.selection_rows = []
            self._select_controller(env)
            if self.selected_policy is not None:
                self.selected_policy.prev_mask = np.asarray(
                    env.previous_action_mask,
                    dtype=bool,
                ).copy()
            block_rows = [
                {
                    **row,
                    "block_start_idx": int(env.current_idx),
                    "block_index": int(self.macro_step // self.decision_interval),
                }
                for row in self.selection_rows
            ]
            self.all_selection_rows.extend(block_rows)
            self.block_history.append(
                {
                    "block_start_idx": int(env.current_idx),
                    "block_index": int(self.macro_step // self.decision_interval),
                    "selected_controller_id": (
                        str(self.selected_controller_id)
                        if self.selected_controller_id is not None
                        else "static_fallback"
                    ),
                    "static_fallback": bool(self.static_fallback),
                }
            )
        if self.selected_policy is None:
            mask = self.anchor_mask.copy()
        else:
            mask = np.asarray(self.selected_policy.act_mask(env), dtype=bool)
        self.macro_step += 1
        return mask


@dataclass
class ForecastAwareResidualRiskControllerPolicy(V2Policy):
    model_bundle: WindowRiskModelBundle
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_action_idx: int
    support: tuple[int, ...]
    required_sensor_indices: tuple[int, ...] = ()
    decision_interval: int = 64
    conditioning_blocks: int = 1
    min_risk_lower_bound: float = 0.0
    max_negative_probability: float = 0.25
    min_mean_margin: float = 0.0
    name: str = "forecast_aware_residual_risk_controller"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_action_idx = int(self.anchor_action_idx)
        self.support = tuple(sorted({int(x) for x in self.support}))
        self.required_sensor_indices = tuple(
            int(x) for x in self.required_sensor_indices
        )
        self.decision_interval = max(1, int(self.decision_interval))
        self.conditioning_blocks = max(0, int(self.conditioning_blocks))
        if not 0 <= self.anchor_action_idx < int(self.candidate_masks.shape[0]):
            raise ValueError("anchor_action_idx is outside candidate_masks")
        self.anchor_mask = self.candidate_masks[
            self.anchor_action_idx
        ].astype(bool).copy()
        self.controllers = residual_action_controller_specs(self.support)
        self.reset()

    def reset(self) -> None:
        self.macro_step = 0
        self.selected_action_idx = int(self.anchor_action_idx)
        self.block_history: list[dict[str, object]] = []
        self.selection_rows: list[dict[str, object]] = []

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if int(self.macro_step) % int(self.decision_interval) == 0:
            block_index = int(self.macro_step // self.decision_interval)
            cycle_length = int(self.conditioning_blocks) + 1
            conditioning = (
                int(self.conditioning_blocks) > 0
                and block_index % cycle_length < int(self.conditioning_blocks)
            )
            if conditioning:
                self.selected_action_idx = int(self.anchor_action_idx)
                self.block_history.append(
                    {
                        "block_index": block_index,
                        "block_start_idx": int(env.current_idx),
                        "selected_action_idx": int(self.anchor_action_idx),
                        "static_fallback": True,
                        "conditioning_block": True,
                    }
                )
            else:
                self._select_residual(env)
        mask = self.candidate_masks[int(self.selected_action_idx)].copy()
        self.macro_step += 1
        return mask

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_residual(self, env: WarmupSchedulingEnv) -> None:
        block_index = int(self.macro_step // self.decision_interval)
        block_start_idx = int(env.current_idx)
        boundary_exact = bool(
            np.array_equal(
                np.asarray(env.previous_action_mask, dtype=bool),
                self.anchor_mask,
            )
        )
        self.selected_action_idx = int(self.anchor_action_idx)
        if not boundary_exact:
            self.block_history.append(
                {
                    "block_index": block_index,
                    "block_start_idx": block_start_idx,
                    "selected_action_idx": int(self.anchor_action_idx),
                    "static_fallback": True,
                    "conditioning_block": False,
                    "anchor_boundary_exact": False,
                }
            )
            return
        valid_controllers = [
            controller
            for controller in self.controllers
            if valid_anchor_residual_action(
                self.candidate_masks,
                anchor_idx=self.anchor_action_idx,
                controller=controller,
                allowed_action_indices=self.support,
                required_sensor_indices=self.required_sensor_indices,
            )
        ]
        if not valid_controllers:
            self.block_history.append(
                {
                    "block_index": block_index,
                    "block_start_idx": block_start_idx,
                    "selected_action_idx": int(self.anchor_action_idx),
                    "static_fallback": True,
                    "conditioning_block": False,
                    "anchor_boundary_exact": True,
                }
            )
            return
        features = []
        for controller in valid_controllers:
            feature, names = build_residual_risk_feature(
                env=env,
                forecast_cfg=self.forecast_cfg,
                candidate_masks=self.candidate_masks,
                anchor_idx=self.anchor_action_idx,
                controller=controller,
            )
            if names != self.model_bundle.feature_names:
                raise ValueError(
                    "Runtime residual-risk feature schema differs from training"
                )
            features.append(feature)
        predictions = self.model_bundle.predict(
            np.asarray(features, dtype=np.float32)
        )
        negative_enabled = self.model_bundle.negative_model is not None
        safe_indices: list[int] = []
        for idx, controller in enumerate(valid_controllers):
            mean_margin = float(predictions["mean_margin"][idx])
            q25_margin = float(predictions["q25_margin"][idx])
            lower = float(predictions["risk_lower_bound"][idx])
            negative_probability = float(
                predictions["negative_probability"][idx]
            )
            safe = bool(
                mean_margin >= float(self.min_mean_margin)
                and lower >= float(self.min_risk_lower_bound)
                and (
                    not negative_enabled
                    or negative_probability
                    <= float(self.max_negative_probability)
                )
            )
            target_idx = int(controller.parameters["target_action_idx"])
            self.selection_rows.append(
                {
                    "block_index": block_index,
                    "block_start_idx": block_start_idx,
                    "target_action_idx": target_idx,
                    "mean_margin": mean_margin,
                    "q25_margin": q25_margin,
                    "risk_lower_bound": lower,
                    "negative_probability": negative_probability,
                    "safe": safe,
                }
            )
            if safe:
                safe_indices.append(idx)
        if safe_indices:
            selected_idx = max(
                safe_indices,
                key=lambda idx: (
                    float(predictions["mean_margin"][idx]),
                    float(predictions["risk_lower_bound"][idx]),
                    -int(valid_controllers[idx].parameters["target_action_idx"]),
                ),
            )
            self.selected_action_idx = int(
                valid_controllers[selected_idx].parameters["target_action_idx"]
            )
        self.block_history.append(
            {
                "block_index": block_index,
                "block_start_idx": block_start_idx,
                "selected_action_idx": int(self.selected_action_idx),
                "static_fallback": bool(
                    self.selected_action_idx == self.anchor_action_idx
                ),
                "conditioning_block": False,
                "anchor_boundary_exact": True,
            }
        )
