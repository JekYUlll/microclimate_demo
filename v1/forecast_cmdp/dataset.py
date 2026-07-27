from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import (
    ForecastContextConfig,
    append_event_forecast,
    build_event_forecast,
    event_forecast_feature_names,
)
from .mpc_teacher import MpcTeacherConfig, beam_search_teacher_action, feasible_masks
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class TeacherDataset:
    features: np.ndarray
    labels: np.ndarray
    action_masks: np.ndarray
    candidate_masks: np.ndarray
    step_indices: np.ndarray
    event_flags: np.ndarray
    feature_names: tuple[str, ...] = ()

    def save_npz(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            features=self.features,
            labels=self.labels,
            action_masks=self.action_masks,
            candidate_masks=self.candidate_masks,
            step_indices=self.step_indices,
            event_flags=self.event_flags,
            feature_names=np.asarray(self.feature_names, dtype=str),
        )

    @classmethod
    def load_npz(cls, path: str) -> "TeacherDataset":
        data = np.load(path, allow_pickle=False)
        return cls(
            features=np.asarray(data["features"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int64),
            action_masks=np.asarray(data["action_masks"], dtype=bool),
            candidate_masks=np.asarray(data["candidate_masks"], dtype=bool),
            step_indices=np.asarray(data["step_indices"], dtype=np.int64),
            event_flags=np.asarray(data["event_flags"], dtype=np.float32),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist())
            if "feature_names" in data.files
            else (),
        )


def collect_teacher_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: list[int] | tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
) -> TeacherDataset:
    """Collect supervised labels from the MPC teacher.

    The collector advances ``env`` only along the teacher trajectory for each
    start index. The teacher itself snapshots/restores internally while searching.
    """

    candidates = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    action_mask_rows: list[np.ndarray] = []
    step_rows: list[int] = []
    event_rows: list[float] = []
    feature_names: tuple[str, ...] = ()

    for start in start_indices:
        env.reset(start_idx=int(start))
        for _ in range(int(steps_per_start)):
            state = env._state().astype(np.float32)
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(state, forecast)
            if not feature_names:
                feature_names = build_policy_feature_names(env, forecast_cfg, feature_dim=int(feature.shape[0]))
            feature_rows.append(feature)
            valid = _valid_action_mask(env, candidates)
            action = beam_search_teacher_action(env, candidates, teacher_cfg)
            label_rows.append(_candidate_index(candidates, action))
            action_mask_rows.append(valid)
            step_rows.append(int(env.current_idx))
            event_rows.append(float(env.event_flags[env.current_idx]))
            _, _, done, _ = env.step_mask(action)
            if done:
                break

    if not feature_rows:
        raise ValueError("No teacher samples collected")
    return TeacherDataset(
        features=np.vstack(feature_rows).astype(np.float32),
        labels=np.asarray(label_rows, dtype=np.int64),
        action_masks=np.vstack(action_mask_rows).astype(bool),
        candidate_masks=candidates.astype(bool),
        step_indices=np.asarray(step_rows, dtype=np.int64),
        event_flags=np.asarray(event_rows, dtype=np.float32),
        feature_names=feature_names,
    )


def collect_dagger_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    policy: V2Policy,
    start_indices: list[int] | tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
) -> TeacherDataset:
    """Label states visited by a deployable policy with the MPC teacher."""

    candidates = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    action_mask_rows: list[np.ndarray] = []
    step_rows: list[int] = []
    event_rows: list[float] = []
    feature_names: tuple[str, ...] = ()

    for start in start_indices:
        policy.reset()
        env.reset(start_idx=int(start))
        for _ in range(int(steps_per_start)):
            state = env._state().astype(np.float32)
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(state, forecast)
            if not feature_names:
                feature_names = build_policy_feature_names(env, forecast_cfg, feature_dim=int(feature.shape[0]))
            feature_rows.append(feature)
            valid = _valid_action_mask(env, candidates)
            teacher_action = beam_search_teacher_action(env, candidates, teacher_cfg)
            label_rows.append(_candidate_index(candidates, teacher_action))
            action_mask_rows.append(valid)
            step_rows.append(int(env.current_idx))
            event_rows.append(float(env.event_flags[env.current_idx]))
            action = policy.act_mask(env)
            _, _, done, _ = env.step_mask(action)
            if done:
                break

    if not feature_rows:
        raise ValueError("No DAgger samples collected")
    return TeacherDataset(
        features=np.vstack(feature_rows).astype(np.float32),
        labels=np.asarray(label_rows, dtype=np.int64),
        action_masks=np.vstack(action_mask_rows).astype(bool),
        candidate_masks=candidates.astype(bool),
        step_indices=np.asarray(step_rows, dtype=np.int64),
        event_flags=np.asarray(event_rows, dtype=np.float32),
        feature_names=feature_names,
    )


def concat_teacher_datasets(datasets: list[TeacherDataset] | tuple[TeacherDataset, ...]) -> TeacherDataset:
    if not datasets:
        raise ValueError("No datasets to concatenate")
    base_masks = datasets[0].candidate_masks
    for dataset in datasets[1:]:
        if dataset.candidate_masks.shape != base_masks.shape or not np.array_equal(dataset.candidate_masks, base_masks):
            raise ValueError("All teacher datasets must share candidate_masks")
    feature_names = datasets[0].feature_names
    if any(dataset.feature_names != feature_names for dataset in datasets[1:]):
        feature_names = ()
    return TeacherDataset(
        features=np.vstack([dataset.features for dataset in datasets]).astype(np.float32),
        labels=np.concatenate([dataset.labels for dataset in datasets]).astype(np.int64),
        action_masks=np.vstack([dataset.action_masks for dataset in datasets]).astype(bool),
        candidate_masks=base_masks.astype(bool),
        step_indices=np.concatenate([dataset.step_indices for dataset in datasets]).astype(np.int64),
        event_flags=np.concatenate([dataset.event_flags for dataset in datasets]).astype(np.float32),
        feature_names=feature_names,
    )


def build_policy_feature_names(
    env: WarmupSchedulingEnv,
    forecast_cfg: ForecastContextConfig,
    *,
    feature_dim: int | None = None,
) -> tuple[str, ...]:
    state_columns = tuple(str(name) for name in env.state_columns)
    sensor_ids = tuple(str(sensor_id) for sensor_id in env.sensor_ids)
    lookback = max(1, int(env.cfg.lookback))
    names: list[str] = []
    for row in range(lookback):
        lag = lookback - row - 1
        names.extend(f"agent_history_lag{lag}_{name}" for name in state_columns)
    for row in range(lookback):
        lag = lookback - row - 1
        names.extend(f"observed_history_lag{lag}_{name}" for name in state_columns)
    names.extend(f"sensor_mode_{sensor_id}" for sensor_id in sensor_ids)
    names.extend(f"warm_remaining_{sensor_id}" for sensor_id in sensor_ids)
    names.extend(f"freshness_{sensor_id}" for sensor_id in sensor_ids)
    names.extend(f"previous_action_{sensor_id}" for sensor_id in sensor_ids)
    tail_names = ["power_ratio", "time_of_day_sin", "time_of_day_cos", "event_flag"]
    if bool(env._energy_enabled()):
        tail_names.append("soc_ratio")
    names.extend(tail_names)
    names.extend(
        event_forecast_feature_names(
            horizon=int(forecast_cfg.horizon),
            continuous_columns=tuple(str(x) for x in forecast_cfg.continuous_columns),
        )
    )
    if feature_dim is not None and len(names) != int(feature_dim):
        return tuple(f"feature_{idx}" for idx in range(int(feature_dim)))
    return tuple(names)


def _valid_action_mask(env: WarmupSchedulingEnv, candidates: np.ndarray) -> np.ndarray:
    valid_masks = feasible_masks(env, candidates)
    valid = np.zeros(candidates.shape[0], dtype=bool)
    for mask in valid_masks:
        valid[_candidate_index(candidates, mask)] = True
    return valid


def _candidate_index(candidates: np.ndarray, mask: np.ndarray) -> int:
    matches = np.all(np.asarray(candidates, dtype=bool) == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        raise ValueError(f"Teacher returned mask outside candidate set: {mask}")
    return int(ids[0])
