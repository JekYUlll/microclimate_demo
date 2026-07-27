from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import ForecastContextConfig, append_event_forecast, build_event_forecast
from .mpc_teacher import (
    MpcTeacherConfig,
    _rollout_repeated_mask_cost,
    _step_cost_from_info,
    beam_search_first_action_costs,
    beam_search_teacher_action,
    restore_env,
    snapshot_env,
)
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.custom_ppo import feasible_candidate_mask  # noqa: E402
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class ActionCostTrainingConfig:
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    device: str = "auto"
    ensemble_size: int = 1
    bootstrap_fraction: float = 0.85
    rank_weight: float = 0.0


@dataclass
class ActionCostDataset:
    inputs: np.ndarray
    costs: np.ndarray
    feature_dim: int
    n_sensors: int


@dataclass
class RecurrentActionCostDataset:
    features: np.ndarray
    costs: np.ndarray
    action_masks: np.ndarray
    labels: np.ndarray
    candidate_masks: np.ndarray
    step_indices: np.ndarray
    feature_dim: int
    n_sensors: int


@dataclass
class RecurrentAnchorAdvantageDataset:
    features: np.ndarray
    advantages: np.ndarray
    action_masks: np.ndarray
    labels: np.ndarray
    candidate_masks: np.ndarray
    step_indices: np.ndarray
    feature_dim: int
    n_sensors: int
    anchor_idx: int


@dataclass
class AnchorAdvantageDataset:
    inputs: np.ndarray
    advantages: np.ndarray
    feature_dim: int
    n_sensors: int
    anchor_idx: int


@dataclass
class FeatureTransitionDataset:
    inputs: np.ndarray
    deltas: np.ndarray
    feature_dim: int
    n_sensors: int


@dataclass
class SequenceValueDataset:
    inputs: np.ndarray
    advantages: np.ndarray
    sequence_bank: np.ndarray
    feature_dim: int
    n_sensors: int
    sequence_len: int


def concat_action_cost_datasets(
    datasets: list[ActionCostDataset] | tuple[ActionCostDataset, ...],
) -> ActionCostDataset:
    if not datasets:
        raise ValueError("No action-cost datasets to concatenate")
    feature_dim = int(datasets[0].feature_dim)
    n_sensors = int(datasets[0].n_sensors)
    for dataset in datasets:
        if int(dataset.feature_dim) != feature_dim or int(dataset.n_sensors) != n_sensors:
            raise ValueError("Action-cost datasets have incompatible dimensions")
    return ActionCostDataset(
        inputs=np.vstack([np.asarray(dataset.inputs, dtype=np.float32) for dataset in datasets]).astype(np.float32),
        costs=np.concatenate([np.asarray(dataset.costs, dtype=np.float32).reshape(-1) for dataset in datasets]).astype(
            np.float32
        ),
        feature_dim=feature_dim,
        n_sensors=n_sensors,
    )


def concat_feature_transition_datasets(
    datasets: list[FeatureTransitionDataset] | tuple[FeatureTransitionDataset, ...],
) -> FeatureTransitionDataset:
    if not datasets:
        raise ValueError("No feature-transition datasets to concatenate")
    feature_dim = int(datasets[0].feature_dim)
    n_sensors = int(datasets[0].n_sensors)
    for dataset in datasets:
        if int(dataset.feature_dim) != feature_dim or int(dataset.n_sensors) != n_sensors:
            raise ValueError("Feature-transition datasets have incompatible dimensions")
    return FeatureTransitionDataset(
        inputs=np.vstack([np.asarray(dataset.inputs, dtype=np.float32) for dataset in datasets]).astype(np.float32),
        deltas=np.vstack([np.asarray(dataset.deltas, dtype=np.float32) for dataset in datasets]).astype(np.float32),
        feature_dim=feature_dim,
        n_sensors=n_sensors,
    )


def collect_action_cost_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    normalize_costs: bool = True,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None,
    rollout_policy: V2Policy | None = None,
) -> ActionCostDataset:
    masks = np.asarray(candidate_masks, dtype=bool)
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    if anchor_mask is not None:
        anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
        if anchor_idx is not None:
            allowed[int(anchor_idx)] = True
    rows: list[np.ndarray] = []
    costs_out: list[float] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        if rollout_policy is not None:
            rollout_policy.reset()
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(env._state().astype(np.float32), forecast)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            finite = np.flatnonzero(np.isfinite(costs) & allowed)
            if finite.size:
                finite_costs = costs[finite].astype(float)
                center = float(np.min(finite_costs))
                spread = float(np.std(finite_costs))
                scale = spread if spread > 1.0e-6 else 1.0
                for action_idx in finite:
                    action_features = masks[int(action_idx)].astype(np.float32)
                    rows.append(np.concatenate([feature, action_features], axis=0).astype(np.float32))
                    if bool(normalize_costs):
                        costs_out.append(float((costs[int(action_idx)] - center) / scale))
                    else:
                        costs_out.append(float(costs[int(action_idx)]))
            if rollout_policy is None:
                action = beam_search_teacher_action(env, masks, teacher_cfg)
            else:
                action = rollout_policy.act_mask(env)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No action-cost rows were collected")
    return ActionCostDataset(
        inputs=np.vstack(rows).astype(np.float32),
        costs=np.asarray(costs_out, dtype=np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
    )


def collect_recurrent_action_cost_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    normalize_costs: bool = True,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None,
    rollout_policy: V2Policy | None = None,
) -> RecurrentActionCostDataset:
    """Collect ordered per-state candidate costs for a recurrent value student."""

    masks = np.asarray(candidate_masks, dtype=bool)
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    if anchor_mask is not None:
        anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
        if anchor_idx is not None:
            allowed[int(anchor_idx)] = True
    feature_rows: list[np.ndarray] = []
    cost_rows: list[np.ndarray] = []
    action_mask_rows: list[np.ndarray] = []
    labels: list[int] = []
    step_rows: list[int] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        if rollout_policy is not None:
            rollout_policy.reset()
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            finite = np.isfinite(costs) & allowed
            if np.any(finite):
                values = np.zeros(int(masks.shape[0]), dtype=np.float32)
                finite_costs = costs[finite].astype(float)
                if bool(normalize_costs):
                    center = float(np.min(finite_costs))
                    spread = float(np.std(finite_costs))
                    scale = spread if spread > 1.0e-6 else 1.0
                    values[finite] = ((costs[finite].astype(float) - center) / scale).astype(np.float32)
                else:
                    values[finite] = costs[finite].astype(np.float32)
                feature_rows.append(np.asarray(feature, dtype=np.float32))
                cost_rows.append(values)
                action_mask_rows.append(np.asarray(finite, dtype=bool))
                labels.append(int(np.flatnonzero(finite)[int(np.argmin(costs[finite]))]))
                step_rows.append(int(env.current_idx))
            if rollout_policy is None:
                action = beam_search_teacher_action(env, masks, teacher_cfg)
            else:
                action = rollout_policy.act_mask(env)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not feature_rows or feature_dim is None:
        raise ValueError("No recurrent action-cost rows were collected")
    return RecurrentActionCostDataset(
        features=np.vstack(feature_rows).astype(np.float32),
        costs=np.vstack(cost_rows).astype(np.float32),
        action_masks=np.vstack(action_mask_rows).astype(bool),
        labels=np.asarray(labels, dtype=np.int64),
        candidate_masks=masks.astype(bool),
        step_indices=np.asarray(step_rows, dtype=np.int64),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
    )


def concat_recurrent_action_cost_datasets(
    datasets: list[RecurrentActionCostDataset] | tuple[RecurrentActionCostDataset, ...],
) -> RecurrentActionCostDataset:
    """Concatenate recurrent cost datasets while preserving sequence breaks."""

    if not datasets:
        raise ValueError("No recurrent action-cost datasets to concatenate")
    base = datasets[0]
    base_masks = np.asarray(base.candidate_masks, dtype=bool)
    feature_dim = int(base.feature_dim)
    n_sensors = int(base.n_sensors)
    features: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    offset = 0
    for dataset in datasets:
        if int(dataset.feature_dim) != feature_dim or int(dataset.n_sensors) != n_sensors:
            raise ValueError("Recurrent action-cost datasets have incompatible feature/sensor dimensions")
        masks = np.asarray(dataset.candidate_masks, dtype=bool)
        if masks.shape != base_masks.shape or not np.array_equal(masks, base_masks):
            raise ValueError("Recurrent action-cost datasets must share candidate_masks")
        steps = np.asarray(dataset.step_indices, dtype=np.int64).reshape(-1)
        if steps.size:
            adjusted_steps = steps - int(np.min(steps)) + int(offset)
            offset = int(np.max(adjusted_steps)) + 2
        else:
            adjusted_steps = steps
        features.append(np.asarray(dataset.features, dtype=np.float32))
        costs.append(np.asarray(dataset.costs, dtype=np.float32))
        action_masks.append(np.asarray(dataset.action_masks, dtype=bool))
        labels.append(np.asarray(dataset.labels, dtype=np.int64))
        step_indices.append(adjusted_steps.astype(np.int64))
    return RecurrentActionCostDataset(
        features=np.vstack(features).astype(np.float32),
        costs=np.vstack(costs).astype(np.float32),
        action_masks=np.vstack(action_masks).astype(bool),
        labels=np.concatenate(labels).astype(np.int64),
        candidate_masks=base_masks.astype(bool),
        step_indices=np.concatenate(step_indices).astype(np.int64),
        feature_dim=feature_dim,
        n_sensors=n_sensors,
    )


def collect_recurrent_anchor_advantage_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    anchor_mask: tuple[bool, ...] | np.ndarray,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
) -> RecurrentAnchorAdvantageDataset:
    """Collect ordered candidate advantages relative to the static anchor."""

    masks = np.asarray(candidate_masks, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    anchor_idx = _candidate_index(masks, anchor)
    if anchor_idx is None:
        raise ValueError("Anchor mask is not present in candidate_masks")
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    allowed[int(anchor_idx)] = True
    feature_rows: list[np.ndarray] = []
    advantage_rows: list[np.ndarray] = []
    action_mask_rows: list[np.ndarray] = []
    labels: list[int] = []
    step_rows: list[int] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            anchor_cost = _anchor_rollout_cost(env, anchor, masks, teacher_cfg)
            finite = np.isfinite(costs) & allowed
            if np.isfinite(anchor_cost):
                finite[int(anchor_idx)] = True
            if np.any(finite) and np.isfinite(anchor_cost):
                values = np.zeros(int(masks.shape[0]), dtype=np.float32)
                finite_ids = np.flatnonzero(finite)
                scale_values = costs[np.isfinite(costs) & finite].astype(float)
                scale_values = np.concatenate(
                    [scale_values, np.asarray([float(anchor_cost)], dtype=float)],
                    axis=0,
                )
                spread = float(np.std(scale_values))
                scale = spread if spread > 1.0e-6 else 1.0
                for action_idx in finite_ids:
                    if int(action_idx) == int(anchor_idx):
                        values[int(action_idx)] = 0.0
                    elif np.isfinite(costs[int(action_idx)]):
                        values[int(action_idx)] = float((anchor_cost - float(costs[int(action_idx)])) / scale)
                    else:
                        finite[int(action_idx)] = False
                finite_ids = np.flatnonzero(finite)
                if finite_ids.size:
                    feature_rows.append(np.asarray(feature, dtype=np.float32))
                    advantage_rows.append(values)
                    action_mask_rows.append(np.asarray(finite, dtype=bool))
                    labels.append(int(finite_ids[int(np.argmax(values[finite_ids]))]))
                    step_rows.append(int(env.current_idx))
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not feature_rows or feature_dim is None:
        raise ValueError("No recurrent anchor-advantage rows were collected")
    return RecurrentAnchorAdvantageDataset(
        features=np.vstack(feature_rows).astype(np.float32),
        advantages=np.vstack(advantage_rows).astype(np.float32),
        action_masks=np.vstack(action_mask_rows).astype(bool),
        labels=np.asarray(labels, dtype=np.int64),
        candidate_masks=masks.astype(bool),
        step_indices=np.asarray(step_rows, dtype=np.int64),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
        anchor_idx=int(anchor_idx),
    )


def collect_anchor_advantage_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    anchor_mask: tuple[bool, ...] | np.ndarray,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
) -> AnchorAdvantageDataset:
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    anchor_idx = _candidate_index(masks, anchor)
    if anchor_idx is None:
        raise ValueError("Anchor mask is not present in candidate_masks")
    anchor_features = anchor.astype(np.float32)
    rows: list[np.ndarray] = []
    advantages_out: list[float] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(env._state().astype(np.float32), forecast)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            finite = np.flatnonzero(np.isfinite(costs))
            anchor_cost = _anchor_rollout_cost(env, anchor, masks, teacher_cfg)
            if np.isfinite(anchor_cost):
                scale_values = costs[finite].astype(float) if finite.size else np.asarray([], dtype=float)
                scale_values = np.concatenate([scale_values, np.asarray([float(anchor_cost)], dtype=float)])
                spread = float(np.std(scale_values))
                scale = spread if spread > 1.0e-6 else 1.0
                for action_idx in finite:
                    action_features = masks[int(action_idx)].astype(np.float32)
                    advantage = 0.0 if int(action_idx) == int(anchor_idx) else (anchor_cost - float(costs[int(action_idx)])) / scale
                    rows.append(
                        np.concatenate(
                            [
                                feature,
                                action_features,
                                anchor_features,
                                action_features - anchor_features,
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    advantages_out.append(float(advantage))
                if not np.any(finite == int(anchor_idx)):
                    rows.append(
                        np.concatenate(
                            [
                                feature,
                                anchor_features,
                                anchor_features,
                                np.zeros_like(anchor_features),
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    advantages_out.append(0.0)
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No anchor-advantage rows were collected")
    return AnchorAdvantageDataset(
        inputs=np.vstack(rows).astype(np.float32),
        advantages=np.asarray(advantages_out, dtype=np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
        anchor_idx=int(anchor_idx),
    )


def collect_feature_transition_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None,
    rollout_policy: V2Policy | None = None,
) -> FeatureTransitionDataset:
    """Collect causal one-step feature transitions for a learned planner.

    Training may use the simulator/truth split to observe next features under
    candidate actions. Deployed policies only use the fitted transition model.
    """

    masks = np.asarray(candidate_masks, dtype=bool)
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    if anchor_mask is not None:
        anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
        if anchor_idx is not None:
            allowed[int(anchor_idx)] = True
    rows: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        if rollout_policy is not None:
            rollout_policy.reset()
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            valid = feasible_candidate_mask(env, masks) & allowed
            valid_ids = np.flatnonzero(valid)
            if valid_ids.size:
                state_snapshot = snapshot_env(env)
                for action_idx in valid_ids:
                    restore_env(env, state_snapshot)
                    _, _, _, _ = env.step_mask(masks[int(action_idx)])
                    next_feature = _current_policy_feature(env, forecast_cfg)
                    action_features = masks[int(action_idx)].astype(np.float32)
                    rows.append(np.concatenate([feature, action_features], axis=0).astype(np.float32))
                    deltas.append((next_feature - feature).astype(np.float32))
                restore_env(env, state_snapshot)
            if rollout_policy is None:
                action = beam_search_teacher_action(env, masks, teacher_cfg)
            else:
                action = rollout_policy.act_mask(env)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No feature-transition rows were collected")
    return FeatureTransitionDataset(
        inputs=np.vstack(rows).astype(np.float32),
        deltas=np.vstack(deltas).astype(np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
    )


def collect_executed_outcome_datasets(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    rollout_policy: V2Policy | None = None,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None,
    seed: int = 42,
) -> tuple[ActionCostDataset, FeatureTransitionDataset]:
    """Collect actual executed step costs and feature transitions.

    Unlike ``collect_action_cost_dataset()``, targets here are not teacher beam
    scores. The collector executes a policy or random feasible actions in the
    replay environment, records the realized selected action, computes the same
    per-step objective used by the teacher, and learns the next-feature delta.
    This is the lightweight learned-digital-twin data path.
    """

    masks = np.asarray(candidate_masks, dtype=bool)
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    if anchor_mask is not None:
        anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
        if anchor_idx is not None:
            allowed[int(anchor_idx)] = True
    rng = np.random.default_rng(int(seed))
    rows: list[np.ndarray] = []
    costs: list[float] = []
    deltas: list[np.ndarray] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        if rollout_policy is not None:
            rollout_policy.reset()
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            if rollout_policy is None:
                valid = feasible_candidate_mask(env, masks)
                supported = np.asarray(valid, dtype=bool) & allowed
                if np.any(supported):
                    valid = supported
                valid_ids = np.flatnonzero(valid)
                if valid_ids.size == 0:
                    break
                action_idx = int(rng.choice(valid_ids))
                desired = masks[action_idx]
            else:
                desired = rollout_policy.act_mask(env)
            step_idx = int(env.current_idx)
            _, reward, done, info = env.step_mask(desired)
            selected = np.asarray(info.get("selected_mask", desired), dtype=bool).reshape(-1)
            selected_idx = _candidate_index(masks, selected)
            if selected_idx is None:
                if done:
                    break
                continue
            loss = float(info.get("oracle_loss", np.nan))
            if not np.isfinite(loss):
                loss = -float(reward)
            cost = _step_cost_from_info(
                env,
                selected,
                info,
                float(loss),
                teacher_cfg,
                masks,
                step_idx=step_idx,
            )
            next_feature = feature.copy() if bool(done) else _current_policy_feature(env, forecast_cfg)
            action_features = masks[int(selected_idx)].astype(np.float32)
            rows.append(np.concatenate([feature, action_features], axis=0).astype(np.float32))
            costs.append(float(cost))
            deltas.append((next_feature - feature).astype(np.float32))
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No executed-outcome rows were collected")
    inputs = np.vstack(rows).astype(np.float32)
    return (
        ActionCostDataset(
            inputs=inputs,
            costs=np.asarray(costs, dtype=np.float32),
            feature_dim=int(feature_dim),
            n_sensors=int(masks.shape[1]),
        ),
        FeatureTransitionDataset(
            inputs=inputs,
            deltas=np.vstack(deltas).astype(np.float32),
            feature_dim=int(feature_dim),
            n_sensors=int(masks.shape[1]),
        ),
    )


def collect_sequence_value_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...] | np.ndarray,
    sequence_len: int = 8,
    snippet_stride: int = 4,
    negatives_per_state: int = 3,
    max_rows: int = 4096,
    extra_sequence_bank: np.ndarray | None = None,
    seed: int = 42,
) -> SequenceValueDataset:
    """Collect sequence-level static-anchor margins for deployable planning.

    Each row is ``current causal feature + flattened candidate mask sequence``.
    The target is ``anchor_cost - sequence_cost`` under the teacher objective,
    so positive values mean the sequence should beat the static anchor from the
    same causal state. This is intentionally different from one-step teacher
    labels and recurrent action-cost targets.
    """

    del features
    masks = np.asarray(candidate_masks, dtype=bool)
    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    if label_arr.shape[0] != steps.shape[0]:
        raise ValueError("labels and step_indices must have matching rows")
    seq_len = max(1, int(sequence_len))
    stride = max(1, int(snippet_stride))
    negatives = max(0, int(negatives_per_state))
    max_count = max(1, int(max_rows))
    anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
    if anchor_idx is None:
        raise ValueError("anchor_mask must be present in candidate_masks for sequence-value collection")
    anchor_sequence = np.full(seq_len, int(anchor_idx), dtype=np.int64)
    sequence_bank = _build_label_sequence_bank(
        label_arr,
        steps,
        sequence_len=seq_len,
        anchor_idx=int(anchor_idx),
        stride=stride,
    )
    sequence_bank = _merge_sequence_banks(
        sequence_bank,
        extra_sequence_bank,
        sequence_len=seq_len,
        anchor_idx=int(anchor_idx),
        n_actions=int(masks.shape[0]),
    )
    if sequence_bank.size == 0:
        raise ValueError("No sequence-value snippets available")
    rng = np.random.default_rng(int(seed))
    rows: list[np.ndarray] = []
    advantages: list[float] = []
    feature_dim: int | None = None
    slices = _contiguous_sequence_slices(steps)
    for begin, end in slices:
        if len(rows) >= max_count:
            break
        start_idx = int(steps[int(begin)])
        env.reset(start_idx=start_idx)
        for row in range(int(begin), int(end)):
            offset = int(row - begin)
            if offset > 0:
                prev_label = int(label_arr[int(row) - 1])
                if 0 <= prev_label < int(masks.shape[0]):
                    _, _, done, _ = env.step_mask(masks[prev_label])
                    if done:
                        break
            if offset % stride != 0:
                continue
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            state_snapshot = snapshot_env(env)
            anchor_cost = _rollout_label_sequence_cost(
                env,
                anchor_sequence,
                masks,
                teacher_cfg,
            )
            restore_env(env, state_snapshot)
            candidates: list[np.ndarray] = [
                _pad_label_sequence(label_arr[int(row) : min(int(row) + seq_len, int(end))], seq_len, int(anchor_idx)),
                anchor_sequence.copy(),
            ]
            if sequence_bank.shape[0] > 0 and negatives > 0:
                sampled = rng.choice(
                    int(sequence_bank.shape[0]),
                    size=min(negatives, int(sequence_bank.shape[0])),
                    replace=False,
                )
                candidates.extend(sequence_bank[np.asarray(sampled, dtype=int)].astype(np.int64))
            seen: set[tuple[int, ...]] = set()
            for sequence in candidates:
                if len(rows) >= max_count:
                    break
                seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
                key = tuple(int(x) for x in seq)
                if key in seen:
                    continue
                seen.add(key)
                restore_env(env, state_snapshot)
                sequence_cost = _rollout_label_sequence_cost(env, seq, masks, teacher_cfg)
                restore_env(env, state_snapshot)
                if not (np.isfinite(anchor_cost) and np.isfinite(sequence_cost)):
                    continue
                rows.append(_sequence_value_input(feature, masks, seq).astype(np.float32))
                advantages.append(float(anchor_cost - sequence_cost) / float(max(1, seq_len)))
            restore_env(env, state_snapshot)
    if not rows or feature_dim is None:
        raise ValueError("No sequence-value rows were collected")
    return SequenceValueDataset(
        inputs=np.vstack(rows).astype(np.float32),
        advantages=np.asarray(advantages, dtype=np.float32),
        sequence_bank=sequence_bank.astype(np.int64),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
        sequence_len=int(seq_len),
    )


def train_recurrent_action_cost_model(
    dataset: RecurrentActionCostDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, _, _ = _torch_modules()
    features = np.asarray(dataset.features, dtype=np.float32)
    targets = np.asarray(dataset.costs, dtype=np.float32)
    valid = np.asarray(dataset.action_masks, dtype=bool)
    labels = np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
    masks = np.asarray(dataset.candidate_masks, dtype=np.float32)
    steps = np.asarray(dataset.step_indices, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or targets.ndim != 2 or valid.ndim != 2:
        raise ValueError("features, costs and action_masks must be 2D")
    if features.shape[0] != targets.shape[0] or valid.shape != targets.shape or labels.shape[0] != features.shape[0]:
        raise ValueError("recurrent action-cost dataset arrays have incompatible shapes")
    sequence_slices = _contiguous_sequence_slices(steps)
    if not sequence_slices:
        raise ValueError("No recurrent action-cost sequences available")

    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = RecurrentActionCostNet(
        feature_dim=int(dataset.feature_dim),
        hidden_dim=int(cfg.hidden_dim),
        n_sensors=int(dataset.n_sensors),
    ).to(device)
    masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    rng = np.random.default_rng(int(cfg.seed))
    history = {"loss": [], "best_action_accuracy": [], "sequence_count": [float(len(sequence_slices))]}
    for _ in range(int(cfg.epochs)):
        order = np.arange(len(sequence_slices))
        rng.shuffle(order)
        losses: list[float] = []
        hits = 0
        rows = 0
        model.train()
        for seq_idx in order:
            start, end = sequence_slices[int(seq_idx)]
            hidden = model.initial_hidden(batch_size=1, device=device)
            prev_mask = torch.zeros((1, int(dataset.n_sensors)), dtype=torch.float32, device=device)
            seq_losses: list[Any] = []
            for row_idx in range(int(start), int(end)):
                feature_t = torch.as_tensor(
                    features[row_idx : row_idx + 1],
                    dtype=torch.float32,
                    device=device,
                )
                hidden = model.forward_state(feature_t, prev_mask, hidden)
                pred = model.score_candidates(hidden, masks_t).reshape(-1)
                valid_t = torch.as_tensor(valid[row_idx], dtype=torch.bool, device=device)
                if bool(torch.any(valid_t).detach().cpu().item()):
                    target_t = torch.as_tensor(targets[row_idx], dtype=torch.float32, device=device)
                    row_loss = nn.functional.smooth_l1_loss(pred[valid_t], target_t[valid_t])
                    if float(cfg.rank_weight) > 0.0:
                        valid_ids_np = np.flatnonzero(valid[row_idx])
                        target_matches = np.flatnonzero(valid_ids_np == int(labels[row_idx]))
                        if target_matches.size:
                            target_pos = int(target_matches[0])
                        else:
                            target_pos = int(np.argmin(targets[row_idx, valid[row_idx]]))
                        logits = -pred[valid_t].reshape(1, -1)
                        target_idx = torch.as_tensor([target_pos], dtype=torch.long, device=device)
                        row_loss = row_loss + float(cfg.rank_weight) * nn.functional.cross_entropy(logits, target_idx)
                    seq_losses.append(row_loss)
                    pred_masked = pred.detach().masked_fill(~valid_t, 1.0e9)
                    best_pred = int(torch.argmin(pred_masked).detach().cpu().item())
                    hits += int(best_pred == int(labels[row_idx]))
                    rows += 1
                prev_mask = masks_t[int(labels[row_idx]) : int(labels[row_idx]) + 1].detach()
            if not seq_losses:
                continue
            loss = torch.stack(seq_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["best_action_accuracy"].append(float(hits) / max(rows, 1))
    return model.eval(), history


def train_recurrent_anchor_advantage_model(
    dataset: RecurrentAnchorAdvantageDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, _, _ = _torch_modules()
    features = np.asarray(dataset.features, dtype=np.float32)
    targets = np.asarray(dataset.advantages, dtype=np.float32)
    valid = np.asarray(dataset.action_masks, dtype=bool)
    labels = np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
    masks = np.asarray(dataset.candidate_masks, dtype=np.float32)
    steps = np.asarray(dataset.step_indices, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or targets.ndim != 2 or valid.ndim != 2:
        raise ValueError("features, advantages and action_masks must be 2D")
    if features.shape[0] != targets.shape[0] or valid.shape != targets.shape or labels.shape[0] != features.shape[0]:
        raise ValueError("recurrent anchor-advantage dataset arrays have incompatible shapes")
    sequence_slices = _contiguous_sequence_slices(steps)
    if not sequence_slices:
        raise ValueError("No recurrent anchor-advantage sequences available")

    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = RecurrentActionCostNet(
        feature_dim=int(dataset.feature_dim),
        hidden_dim=int(cfg.hidden_dim),
        n_sensors=int(dataset.n_sensors),
    ).to(device)
    masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    rng = np.random.default_rng(int(cfg.seed))
    history = {"loss": [], "best_action_accuracy": [], "sequence_count": [float(len(sequence_slices))]}
    for _ in range(int(cfg.epochs)):
        order = np.arange(len(sequence_slices))
        rng.shuffle(order)
        losses: list[float] = []
        hits = 0
        rows = 0
        model.train()
        for seq_idx in order:
            start, end = sequence_slices[int(seq_idx)]
            hidden = model.initial_hidden(batch_size=1, device=device)
            prev_mask = torch.zeros((1, int(dataset.n_sensors)), dtype=torch.float32, device=device)
            seq_losses: list[Any] = []
            for row_idx in range(int(start), int(end)):
                feature_t = torch.as_tensor(
                    features[row_idx : row_idx + 1],
                    dtype=torch.float32,
                    device=device,
                )
                hidden = model.forward_state(feature_t, prev_mask, hidden)
                pred = model.score_candidates(hidden, masks_t).reshape(-1)
                valid_t = torch.as_tensor(valid[row_idx], dtype=torch.bool, device=device)
                if bool(torch.any(valid_t).detach().cpu().item()):
                    target_t = torch.as_tensor(targets[row_idx], dtype=torch.float32, device=device)
                    row_loss = nn.functional.smooth_l1_loss(pred[valid_t], target_t[valid_t])
                    if float(cfg.rank_weight) > 0.0:
                        valid_ids_np = np.flatnonzero(valid[row_idx])
                        target_matches = np.flatnonzero(valid_ids_np == int(labels[row_idx]))
                        if target_matches.size:
                            target_pos = int(target_matches[0])
                        else:
                            target_pos = int(np.argmax(targets[row_idx, valid[row_idx]]))
                        logits = pred[valid_t].reshape(1, -1)
                        target_idx = torch.as_tensor([target_pos], dtype=torch.long, device=device)
                        row_loss = row_loss + float(cfg.rank_weight) * nn.functional.cross_entropy(logits, target_idx)
                    seq_losses.append(row_loss)
                    pred_masked = pred.detach().masked_fill(~valid_t, -1.0e9)
                    best_pred = int(torch.argmax(pred_masked).detach().cpu().item())
                    hits += int(best_pred == int(labels[row_idx]))
                    rows += 1
                prev_mask = masks_t[int(labels[row_idx]) : int(labels[row_idx]) + 1].detach()
            if not seq_losses:
                continue
            loss = torch.stack(seq_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["best_action_accuracy"].append(float(hits) / max(rows, 1))
    return model.eval(), history


def train_action_cost_model(dataset: ActionCostDataset, cfg: ActionCostTrainingConfig) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.costs, dtype=np.float32).reshape(-1, 1)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ActionCostNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_feature_transition_model(
    dataset: FeatureTransitionDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.deltas, dtype=np.float32)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = FeatureTransitionNet(
        input_dim=x.shape[1],
        output_dim=int(dataset.feature_dim),
        hidden_dim=int(cfg.hidden_dim),
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_anchor_advantage_model(
    dataset: AnchorAdvantageDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.advantages, dtype=np.float32).reshape(-1, 1)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ActionCostNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_sequence_value_model(
    dataset: SequenceValueDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.advantages, dtype=np.float32).reshape(-1, 1)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("sequence-value inputs and advantages have incompatible shapes")
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ActionCostNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "target_mean": [float(np.mean(y))], "positive_rate": [float(np.mean(y > 0.0))]}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_action_cost_ensemble(
    dataset: ActionCostDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[list[Any], list[dict[str, list[float]]]]:
    size = max(1, int(cfg.ensemble_size))
    if size == 1:
        model, history = train_action_cost_model(dataset, cfg)
        return [model], [history]
    rng = np.random.default_rng(int(cfg.seed))
    n = int(dataset.inputs.shape[0])
    sample_size = max(1, int(round(float(cfg.bootstrap_fraction) * float(n))))
    models: list[Any] = []
    histories: list[dict[str, list[float]]] = []
    for member in range(size):
        ids = rng.integers(0, n, size=sample_size, endpoint=False)
        member_dataset = ActionCostDataset(
            inputs=np.asarray(dataset.inputs[ids], dtype=np.float32),
            costs=np.asarray(dataset.costs[ids], dtype=np.float32),
            feature_dim=int(dataset.feature_dim),
            n_sensors=int(dataset.n_sensors),
        )
        member_cfg = ActionCostTrainingConfig(
            hidden_dim=int(cfg.hidden_dim),
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            seed=int(cfg.seed) + 997 * int(member + 1),
            device=str(cfg.device),
            ensemble_size=1,
            bootstrap_fraction=float(cfg.bootstrap_fraction),
        )
        model, history = train_action_cost_model(member_dataset, member_cfg)
        models.append(model)
        histories.append(history)
    return models, histories


class ActionCostNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _ActionCostNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), 1),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _ActionCostNet()


class FeatureTransitionNet:
    def __new__(cls, *, input_dim: int, output_dim: int, hidden_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _FeatureTransitionNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.output_dim = int(output_dim)
                self.hidden_dim = int(hidden_dim)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(output_dim)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _FeatureTransitionNet()


class RecurrentActionCostNet:
    def __new__(cls, *, feature_dim: int, hidden_dim: int, n_sensors: int) -> Any:
        torch, nn, _, _ = _torch_modules()

        class _RecurrentActionCostNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_dim = int(feature_dim)
                self.hidden_dim = int(hidden_dim)
                self.n_sensors = int(n_sensors)
                self.gru = nn.GRUCell(int(feature_dim) + int(n_sensors), int(hidden_dim))
                self.scorer = nn.Sequential(
                    nn.Linear(int(hidden_dim) + int(n_sensors), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), 1),
                )

            def initial_hidden(self, *, batch_size: int, device: Any) -> Any:
                return torch.zeros((int(batch_size), int(hidden_dim)), dtype=torch.float32, device=device)

            def forward_state(self, feature: Any, prev_mask: Any, hidden: Any) -> Any:
                x = torch.cat([feature, prev_mask], dim=1)
                return self.gru(x, hidden)

            def score_candidates(self, hidden: Any, candidate_masks: Any) -> Any:
                if int(hidden.shape[0]) != 1:
                    raise ValueError("score_candidates currently expects batch_size=1")
                h = hidden.expand(int(candidate_masks.shape[0]), -1)
                return self.scorer(torch.cat([h, candidate_masks], dim=1)).reshape(-1)

        return _RecurrentActionCostNet()


@dataclass
class ForecastAwareCostPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    preserve_warming: bool = True
    name: str = "forecast_aware_cost"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return np.zeros(self.candidate_masks.shape[1], dtype=bool)
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            costs = self.model(x).reshape(-1).detach().cpu().numpy()
        best = int(valid_ids[int(np.argmin(costs))])
        return self.candidate_masks[best].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareCostKNNPolicy(V2Policy):
    features: np.ndarray
    costs: np.ndarray
    action_masks: np.ndarray
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    k: int = 16
    advantage_threshold: float = 0.0
    distance_weighting: str = "inverse"
    preserve_warming: bool = True
    name: str = "forecast_aware_cost_knn"

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.costs = np.asarray(self.costs, dtype=np.float32)
        self.action_masks = np.asarray(self.action_masks, dtype=bool)
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        if self.features.ndim != 2:
            raise ValueError("features must be 2D")
        if self.costs.ndim != 2 or self.action_masks.shape != self.costs.shape:
            raise ValueError("costs and action_masks must be matching 2D arrays")
        if self.candidate_masks.ndim != 2 or self.candidate_masks.shape[0] != self.costs.shape[1]:
            raise ValueError("candidate_masks must match cost action dimension")
        if self.anchor_mask_arr.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("anchor_mask must match candidate sensor width")
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True
        if str(self.distance_weighting) not in {"uniform", "inverse"}:
            raise ValueError(f"Unsupported distance_weighting: {self.distance_weighting}")
        self.k = max(1, int(self.k))
        self.feature_mean = np.mean(self.features, axis=0).astype(np.float32)
        self.feature_std = np.std(self.features, axis=0).astype(np.float32)
        self.feature_std = np.where(self.feature_std > 1.0e-6, self.feature_std, 1.0).astype(np.float32)
        self.normalized_features = ((self.features - self.feature_mean) / self.feature_std).astype(np.float32)

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        feature = _current_policy_feature(env, self.forecast_cfg)
        if int(feature.shape[0]) != int(self.features.shape[1]):
            raise ValueError(
                f"feature dimension mismatch: policy={self.features.shape[1]} env={feature.shape[0]}"
            )
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self._anchor_action(env)
        scores = self._predict_costs(feature, valid)
        finite_valid = np.asarray(valid, dtype=bool) & np.isfinite(scores)
        if not np.any(finite_valid):
            return self._anchor_action(env)
        best_idx = int(np.flatnonzero(finite_valid)[int(np.argmin(scores[finite_valid]))])
        anchor_score = self._anchor_score(scores)
        best_score = float(scores[int(best_idx)])
        if not np.isfinite(anchor_score) or best_idx == self.anchor_idx:
            return self._anchor_action(env)
        advantage = float(anchor_score - best_score)
        if advantage <= float(self.advantage_threshold):
            return self._anchor_action(env)
        return self.candidate_masks[int(best_idx)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _predict_costs(self, feature: np.ndarray, valid: np.ndarray) -> np.ndarray:
        query = ((np.asarray(feature, dtype=np.float32) - self.feature_mean) / self.feature_std).astype(np.float32)
        distances = np.sqrt(np.sum((self.normalized_features - query.reshape(1, -1)) ** 2, axis=1))
        k = min(int(self.k), int(distances.shape[0]))
        if k <= 0:
            return np.full(self.candidate_masks.shape[0], np.inf, dtype=float)
        nearest = np.argpartition(distances, kth=k - 1)[:k]
        if str(self.distance_weighting) == "inverse":
            weights = 1.0 / (distances[nearest].astype(float) + 1.0e-6)
        else:
            weights = np.ones(k, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        scores = np.full(self.candidate_masks.shape[0], np.inf, dtype=float)
        neighbor_costs = self.costs[nearest].astype(float)
        neighbor_masks = self.action_masks[nearest].astype(bool)
        for action_idx in np.flatnonzero(np.asarray(valid, dtype=bool)):
            support = neighbor_masks[:, int(action_idx)]
            if not np.any(support):
                continue
            local_weights = weights[support]
            local_weights = local_weights / max(float(np.sum(local_weights)), 1.0e-12)
            scores[int(action_idx)] = float(np.sum(local_weights * neighbor_costs[support, int(action_idx)]))
        return scores

    def _anchor_score(self, scores: np.ndarray) -> float:
        if self.anchor_idx is None:
            return float("inf")
        value = float(scores[int(self.anchor_idx)])
        return value if np.isfinite(value) else float("inf")

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = self._preserve_warming_mask(env, self.anchor_mask_arr)
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareValueResidualPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_value_residual"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            costs = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmin(costs))
        best_idx = int(valid_ids[best_local])
        anchor_cost = _predict_single_action_cost(
            self.model,
            feature=feature,
            action_features=self.anchor_mask_arr.astype(np.float32),
            device_obj=self.device_obj,
            torch=torch,
        )
        best_cost = float(costs[best_local])
        advantage = anchor_cost - best_cost
        if best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareEnsembleValuePolicy(V2Policy):
    models: list[Any]
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    uncertainty_beta: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_ensemble_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.models = list(self.models)
        if not self.models:
            raise ValueError("ForecastAwareEnsembleValuePolicy requires at least one model")
        for model in self.models:
            model.to(self.device_obj)
            model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            preds = [
                model(x).reshape(-1).detach().cpu().numpy().astype(float)
                for model in self.models
            ]
        pred = np.vstack(preds).astype(float)
        mean = np.mean(pred, axis=0)
        std = np.std(pred, axis=0)
        score = mean + float(self.uncertainty_beta) * std
        best_local = int(np.argmin(score))
        best_idx = int(valid_ids[best_local])
        anchor_preds = _predict_single_action_ensemble_costs(
            self.models,
            feature=feature,
            action_features=self.anchor_mask_arr.astype(np.float32),
            device_obj=self.device_obj,
            torch=torch,
        )
        anchor_score = float(np.mean(anchor_preds) + float(self.uncertainty_beta) * np.std(anchor_preds))
        best_score = float(score[best_local])
        advantage = anchor_score - best_score
        if best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareAdvantageResidualPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_advantage_residual"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        anchor_features = self.anchor_mask_arr.astype(np.float32)
        rows = []
        for action_idx in valid_ids:
            action_features = self.candidate_masks[int(action_idx)].astype(np.float32)
            rows.append(
                np.concatenate(
                    [
                        feature,
                        action_features,
                        anchor_features,
                        action_features - anchor_features,
                    ],
                    axis=0,
                )
            )
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            advantages = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmax(advantages))
        best_idx = int(valid_ids[best_local])
        if best_idx == self.anchor_idx:
            return self.anchor_mask_arr.astype(bool).copy()
        if float(advantages[best_local]) <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareRolloutValuePolicy(V2Policy):
    cost_model: Any
    transition_model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    planning_depth: int = 2
    beam_width: int = 4
    max_branch: int = 6
    discount: float = 0.95
    preserve_warming: bool = True
    name: str = "forecast_aware_rollout_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.cost_model.to(self.device_obj)
        self.transition_model.to(self.device_obj)
        self.cost_model.eval()
        self.transition_model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True
        self.future_action_ids = np.flatnonzero(self.allowed_action_mask)
        if self.future_action_ids.size == 0:
            self.future_action_ids = np.arange(self.candidate_masks.shape[0], dtype=int)

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        feature = _current_policy_feature(env, self.forecast_cfg)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        best_idx, best_score = self._plan_from_feature(feature, valid_ids)
        anchor_score = self._score_repeated_anchor(feature)
        advantage = float(anchor_score - best_score)
        if best_idx is None or best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[int(best_idx)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _plan_from_feature(self, feature: np.ndarray, valid_ids: np.ndarray) -> tuple[int | None, float]:
        first_ids = np.asarray(valid_ids, dtype=int).reshape(-1)
        beams: list[tuple[float, int | None, np.ndarray]] = [(0.0, None, np.asarray(feature, dtype=np.float32))]
        depth = max(1, int(self.planning_depth))
        for step in range(depth):
            expanded: list[tuple[float, int | None, np.ndarray]] = []
            for score_so_far, first_idx, beam_feature in beams:
                action_ids = first_ids if step == 0 else self.future_action_ids
                costs = self._predict_costs(beam_feature, action_ids)
                if costs.size == 0:
                    continue
                order = np.argsort(costs, kind="stable")[: max(1, int(self.max_branch))]
                for local_idx in order:
                    action_idx = int(action_ids[int(local_idx)])
                    action_features = self.candidate_masks[action_idx].astype(np.float32)
                    step_cost = float(costs[int(local_idx)])
                    next_feature = self._predict_next_feature(beam_feature, action_features)
                    expanded.append(
                        (
                            float(score_so_far + (float(self.discount) ** step) * step_cost),
                            int(action_idx) if first_idx is None else int(first_idx),
                            next_feature,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            beams = expanded[: max(1, int(self.beam_width))]
        completed = [beam for beam in beams if beam[1] is not None]
        if not completed:
            return None, float("inf")
        best = min(completed, key=lambda item: item[0])
        return int(best[1]), float(best[0])

    def _score_repeated_anchor(self, feature: np.ndarray) -> float:
        current = np.asarray(feature, dtype=np.float32)
        action_features = self.anchor_mask_arr.astype(np.float32)
        total = 0.0
        for step in range(max(1, int(self.planning_depth))):
            total += (float(self.discount) ** step) * self._predict_one_cost(current, action_features)
            current = self._predict_next_feature(current, action_features)
        return float(total)

    def _predict_costs(self, feature: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        ids = np.asarray(action_ids, dtype=int).reshape(-1)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            return self.cost_model(x).reshape(-1).detach().cpu().numpy().astype(float)

    def _predict_one_cost(self, feature: np.ndarray, action_features: np.ndarray) -> float:
        torch, _, _, _ = _torch_modules()
        row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            return float(self.cost_model(x).reshape(-1).detach().cpu().numpy()[0])

    def _predict_next_feature(self, feature: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            delta = self.transition_model(x).reshape(-1).detach().cpu().numpy().astype(np.float32)
        return (np.asarray(feature, dtype=np.float32) + delta).astype(np.float32)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareSequenceValuePolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    sequence_bank: np.ndarray
    device: str = "auto"
    advantage_threshold: float = 0.0
    top_k_sequences: int = 128
    preserve_warming: bool = True
    name: str = "forecast_aware_sequence_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        bank = np.asarray(self.sequence_bank, dtype=np.int64)
        if bank.ndim != 2 or bank.shape[0] == 0:
            raise ValueError("sequence_bank must be a nonempty 2D integer array")
        bank = np.clip(bank, 0, int(self.candidate_masks.shape[0]) - 1)
        self.sequence_bank = bank.astype(np.int64)
        self.sequence_len = int(bank.shape[1])
        self.top_k_sequences = max(1, int(self.top_k_sequences))
        self.reset()

    def reset(self) -> None:
        self.active_sequence: np.ndarray | None = None
        self.cursor = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self.active_sequence is not None and self.cursor < int(self.active_sequence.shape[0]):
            label = self._next_valid_label(env)
            if label is not None:
                return self.candidate_masks[int(label)].astype(bool).copy()
        feature = _current_policy_feature(env, self.forecast_cfg)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, valid)
        sequence_ids = self._valid_sequence_ids(valid)
        if sequence_ids.size == 0:
            self.active_sequence = None
            self.cursor = 0
            return self._anchor_action(env)
        best_sequence, best_advantage = self._score_sequences(feature, sequence_ids)
        if best_sequence is None or float(best_advantage) <= float(self.advantage_threshold):
            self.active_sequence = None
            self.cursor = 0
            return self._anchor_action(env)
        self.active_sequence = np.asarray(best_sequence, dtype=np.int64).copy()
        self.cursor = 0
        label = self._next_valid_label(env)
        if label is None:
            return self._anchor_action(env)
        return self.candidate_masks[int(label)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _valid_sequence_ids(self, valid: np.ndarray) -> np.ndarray:
        first = self.sequence_bank[:, 0].astype(int)
        keep = np.asarray(valid, dtype=bool)[first]
        ids = np.flatnonzero(keep)
        if ids.size <= int(self.top_k_sequences):
            return ids.astype(int)
        return ids[: int(self.top_k_sequences)].astype(int)

    def _score_sequences(self, feature: np.ndarray, sequence_ids: np.ndarray) -> tuple[np.ndarray | None, float]:
        torch, _, _, _ = _torch_modules()
        ids = np.asarray(sequence_ids, dtype=int).reshape(-1)
        if ids.size == 0:
            return None, float("-inf")
        rows = [_sequence_value_input(feature, self.candidate_masks, self.sequence_bank[int(idx)]) for idx in ids]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            values = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmax(values))
        best_id = int(ids[best_local])
        return self.sequence_bank[best_id].astype(np.int64).copy(), float(values[best_local])

    def _next_valid_label(self, env: WarmupSchedulingEnv) -> int | None:
        if self.active_sequence is None:
            return None
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, valid)
        while self.cursor < int(self.active_sequence.shape[0]):
            label = int(self.active_sequence[int(self.cursor)])
            self.cursor += 1
            if 0 <= label < int(self.candidate_masks.shape[0]) and bool(valid[int(label)]):
                return int(label)
        return None

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        projected = env.projector.project_mask(self.anchor_mask_arr, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareRecurrentValuePolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_recurrent_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True
        self.reset()

    def reset(self) -> None:
        self.hidden = None
        self.prev_mask = np.zeros(int(self.candidate_masks.shape[1]), dtype=np.float32)

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        feature = _current_policy_feature(env, self.forecast_cfg)
        with torch.no_grad():
            feature_t = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            prev_t = torch.as_tensor(self.prev_mask.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            if self.hidden is None:
                self.hidden = self.model.initial_hidden(batch_size=1, device=self.device_obj)
            next_hidden = self.model.forward_state(feature_t, prev_t, self.hidden)
            masks_t = torch.as_tensor(
                self.candidate_masks.astype(np.float32),
                dtype=torch.float32,
                device=self.device_obj,
            )
            costs = self.model.score_candidates(next_hidden, masks_t).detach().cpu().numpy().astype(float)
            self.hidden = next_hidden.detach()
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            mask = self.anchor_mask_arr.astype(bool).copy()
        else:
            best_idx = int(valid_ids[int(np.argmin(costs[valid_ids]))])
            anchor_cost = float(costs[int(self.anchor_idx)]) if self.anchor_idx is not None else float("inf")
            best_cost = float(costs[int(best_idx)])
            advantage = anchor_cost - best_cost
            if best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
                mask = self.anchor_mask_arr.astype(bool).copy()
            else:
                mask = self.candidate_masks[best_idx].astype(bool).copy()
        projected = env.projector.project_mask(mask, env.runtimes)
        executed = np.asarray(projected.selected_mask, dtype=bool).copy()
        self.prev_mask = executed.astype(np.float32)
        return executed

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareRecurrentAdvantagePolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_recurrent_advantage"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True
        self.reset()

    def reset(self) -> None:
        self.hidden = None
        self.prev_mask = np.zeros(int(self.candidate_masks.shape[1]), dtype=np.float32)

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        feature = _current_policy_feature(env, self.forecast_cfg)
        with torch.no_grad():
            feature_t = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            prev_t = torch.as_tensor(self.prev_mask.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            if self.hidden is None:
                self.hidden = self.model.initial_hidden(batch_size=1, device=self.device_obj)
            next_hidden = self.model.forward_state(feature_t, prev_t, self.hidden)
            masks_t = torch.as_tensor(
                self.candidate_masks.astype(np.float32),
                dtype=torch.float32,
                device=self.device_obj,
            )
            advantages = self.model.score_candidates(next_hidden, masks_t).detach().cpu().numpy().astype(float)
            self.hidden = next_hidden.detach()
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            mask = self.anchor_mask_arr.astype(bool).copy()
        else:
            best_idx = int(valid_ids[int(np.argmax(advantages[valid_ids]))])
            best_advantage = float(advantages[int(best_idx)])
            if best_idx == self.anchor_idx or best_advantage <= float(self.advantage_threshold):
                mask = self.anchor_mask_arr.astype(bool).copy()
            else:
                mask = self.candidate_masks[best_idx].astype(bool).copy()
        projected = env.projector.project_mask(mask, env.runtimes)
        executed = np.asarray(projected.selected_mask, dtype=bool).copy()
        self.prev_mask = executed.astype(np.float32)
        return executed

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _anchor_rollout_cost(
    env: WarmupSchedulingEnv,
    anchor: np.ndarray,
    candidate_masks: np.ndarray,
    teacher_cfg: MpcTeacherConfig,
) -> float:
    snapshot = snapshot_env(env)
    try:
        return float(
            _rollout_repeated_mask_cost(
                env,
                np.asarray(anchor, dtype=bool).reshape(-1),
                max(1, int(teacher_cfg.planning_horizon)),
                np.asarray(candidate_masks, dtype=bool),
                teacher_cfg,
            )
        )
    finally:
        restore_env(env, snapshot)


def _rollout_label_sequence_cost(
    env: WarmupSchedulingEnv,
    sequence: np.ndarray,
    candidate_masks: np.ndarray,
    teacher_cfg: MpcTeacherConfig,
) -> float:
    masks = np.asarray(candidate_masks, dtype=bool)
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    total = 0.0
    for label in seq:
        idx = int(label)
        if idx < 0 or idx >= int(masks.shape[0]):
            continue
        step_idx = int(env.current_idx)
        _, reward, done, info = env.step_mask(masks[idx])
        loss = float(info.get("oracle_loss", np.nan))
        if not np.isfinite(loss):
            loss = -float(reward)
        selected = np.asarray(info.get("selected_mask", masks[idx]), dtype=bool).reshape(-1)
        total += _step_cost_from_info(
            env,
            selected,
            info,
            float(loss),
            teacher_cfg,
            masks,
            step_idx=step_idx,
        )
        if bool(done):
            break
    return float(total)


def _pad_label_sequence(labels: np.ndarray, sequence_len: int, anchor_idx: int) -> np.ndarray:
    seq_len = max(1, int(sequence_len))
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = values[values >= 0]
    if values.size >= seq_len:
        return values[:seq_len].astype(np.int64)
    pad_value = int(values[-1]) if values.size else int(anchor_idx)
    padded = np.full(seq_len, int(pad_value), dtype=np.int64)
    if values.size:
        padded[: int(values.size)] = values.astype(np.int64)
    return padded


def _sequence_value_input(feature: np.ndarray, candidate_masks: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool)
    seq = np.asarray(sequence, dtype=np.int64).reshape(-1)
    seq = np.clip(seq, 0, int(masks.shape[0]) - 1)
    return np.concatenate(
        [
            np.asarray(feature, dtype=np.float32).reshape(-1),
            masks[seq].astype(np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def _build_label_sequence_bank(
    labels: np.ndarray,
    step_indices: np.ndarray,
    *,
    sequence_len: int,
    anchor_idx: int,
    stride: int,
) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    seq_len = max(1, int(sequence_len))
    stride = max(1, int(stride))
    rows: list[np.ndarray] = []
    for begin, end in _contiguous_sequence_slices(steps):
        for row in range(int(begin), int(end), stride):
            seq = _pad_label_sequence(label_arr[int(row) : min(int(row) + seq_len, int(end))], seq_len, int(anchor_idx))
            rows.append(seq.astype(np.int64))
    if not rows:
        return np.zeros((0, seq_len), dtype=np.int64)
    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for row in rows:
        key = tuple(int(x) for x in row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row.astype(np.int64))
    return np.vstack(unique).astype(np.int64)


def _merge_sequence_banks(
    base: np.ndarray,
    extra: np.ndarray | None,
    *,
    sequence_len: int,
    anchor_idx: int,
    n_actions: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for bank in (base, extra):
        if bank is None:
            continue
        arr = np.asarray(bank, dtype=np.int64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row in arr:
            seq = _pad_label_sequence(row.reshape(-1), int(sequence_len), int(anchor_idx))
            seq = np.clip(seq, 0, int(n_actions) - 1).astype(np.int64)
            rows.append(seq)
    if not rows:
        return np.zeros((0, max(1, int(sequence_len))), dtype=np.int64)
    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for row in rows:
        key = tuple(int(x) for x in row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row.astype(np.int64))
    return np.vstack(unique).astype(np.int64)


def _predict_single_action_cost(
    model: Any,
    *,
    feature: np.ndarray,
    action_features: np.ndarray,
    device_obj: Any,
    torch: Any,
) -> float:
    row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=device_obj)
        return float(model(x).reshape(-1).detach().cpu().numpy()[0])


def _predict_single_action_ensemble_costs(
    models: list[Any],
    *,
    feature: np.ndarray,
    action_features: np.ndarray,
    device_obj: Any,
    torch: Any,
) -> np.ndarray:
    row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=device_obj)
        preds = [float(model(x).reshape(-1).detach().cpu().numpy()[0]) for model in models]
    return np.asarray(preds, dtype=float)


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _allowed_action_mask(indices: tuple[int, ...] | np.ndarray | None, n_actions: int) -> np.ndarray:
    mask = np.ones(int(n_actions), dtype=bool)
    if indices is None:
        return mask
    mask[:] = False
    values = np.asarray(indices, dtype=int).reshape(-1)
    values = values[(values >= 0) & (values < int(n_actions))]
    if values.size == 0:
        mask[:] = True
    else:
        mask[values] = True
    return mask


def _candidate_index(candidates: np.ndarray, mask: np.ndarray) -> int | None:
    matches = np.all(np.asarray(candidates, dtype=bool) == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        return None
    return int(ids[0])


def _current_policy_feature(env: WarmupSchedulingEnv, forecast_cfg: ForecastContextConfig) -> np.ndarray:
    state = env._state().astype(np.float32)
    forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
    return append_event_forecast(state, forecast)


def _contiguous_sequence_slices(step_indices: np.ndarray) -> list[tuple[int, int]]:
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    if steps.size == 0:
        return []
    slices: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, int(steps.size)):
        if int(steps[idx]) != int(steps[idx - 1]) + 1:
            if idx > start:
                slices.append((int(start), int(idx)))
            start = idx
    if int(steps.size) > start:
        slices.append((int(start), int(steps.size)))
    return slices
