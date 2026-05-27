from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .reuse import ensure_archive_src

ensure_archive_src()

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.evaluation import LoadedRollout, overall_metrics  # noqa: E402
from v2.policies import StaticMaskPolicy, V2Policy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import RolloutResult, concat_rollout_results, rollout_metrics, run_policy_rollout  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


SPLIT_NAMES = ("oracle_pretrain", "rl_train", "validation", "final_test")


@dataclass(frozen=True)
class SelectedStarts:
    starts: tuple[int, ...]
    diagnostics: dict[str, object]


def partition_bounds(steps: int, ratios: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
    total = float(sum(ratios))
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to one, got {total}")
    edges = [0]
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += float(ratio)
        edges.append(int(round(int(steps) * cumulative)))
    edges.append(int(steps))
    bounds = {name: (int(edges[idx]), int(edges[idx + 1])) for idx, name in enumerate(SPLIT_NAMES)}
    if any(end <= start for start, end in bounds.values()):
        raise ValueError(f"All partitions must be nonempty, got {bounds}")
    return bounds


def choose_non_overlapping_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    selection: str,
    stride: int,
    event_column: str,
    seed: int,
) -> SelectedStarts:
    if str(selection) == "event_rich":
        starts, diagnostics = event_rich_non_overlapping_starts(
            truth,
            bounds=bounds,
            window_steps=int(window_steps),
            horizon=int(horizon),
            count=int(count),
            stride=int(stride),
            event_column=str(event_column),
        )
        return SelectedStarts(starts=starts, diagnostics=diagnostics)
    starts = random_non_overlapping_starts(
        bounds=bounds,
        window_steps=int(window_steps),
        horizon=int(horizon),
        count=int(count),
        seed=int(seed),
    )
    return SelectedStarts(
        starts=starts,
        diagnostics={"selection": "uniform_non_overlapping", "count": len(starts), "seed": int(seed)},
    )


def random_non_overlapping_starts(
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    start, end = (int(bounds[0]), int(bounds[1]))
    required_span = int(count) * int(window_steps) + int(horizon) + 1
    if end - start < required_span:
        raise ValueError(f"Partition [{start}, {end}) is too short for {count} windows")
    rng = np.random.default_rng(int(seed))
    slack = int(end - start - required_span)
    gaps = rng.multinomial(slack, np.full(int(count) + 1, 1.0 / float(int(count) + 1)))
    starts: list[int] = []
    cursor = start + int(gaps[0])
    for idx in range(int(count)):
        starts.append(int(cursor))
        cursor += int(window_steps) + int(gaps[idx + 1])
    return tuple(starts)


def event_rich_non_overlapping_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    stride: int,
    event_column: str,
) -> tuple[tuple[int, ...], dict[str, object]]:
    start, end = (int(bounds[0]), int(bounds[1]))
    max_start = end - int(window_steps) - int(horizon) - 1
    if max_start < start:
        raise ValueError(f"Partition [{start}, {end}) cannot contain one requested window")
    if event_column not in truth.columns:
        raise ValueError(f"Truth data do not contain event column {event_column!r}")
    span = int(window_steps) + int(horizon) + 1
    candidate_starts = list(range(start, max_start + 1, max(1, int(stride))))
    if not candidate_starts or candidate_starts[-1] != max_start:
        candidate_starts.append(max_start)
    flags = truth[event_column].astype(bool).to_numpy()
    rates = np.asarray(
        [float(np.mean(flags[item : item + int(window_steps)])) for item in candidate_starts],
        dtype=float,
    )

    previous = [bisect.bisect_right(candidate_starts, value - span) - 1 for value in candidate_starts]
    n = len(candidate_starts)
    wanted = int(count)
    scores = np.full((n + 1, wanted + 1), -np.inf, dtype=float)
    selected = np.zeros((n + 1, wanted + 1), dtype=bool)
    scores[:, 0] = 0.0
    for idx in range(1, n + 1):
        for number in range(1, wanted + 1):
            skip = scores[idx - 1, number]
            take = rates[idx - 1] + scores[previous[idx - 1] + 1, number - 1]
            if take > skip:
                scores[idx, number] = take
                selected[idx, number] = True
            else:
                scores[idx, number] = skip
    if not np.isfinite(scores[n, wanted]):
        raise ValueError(f"Partition [{start}, {end}) cannot contain {wanted} non-overlapping windows")
    chosen: list[int] = []
    idx = n
    number = wanted
    while number > 0:
        if selected[idx, number]:
            chosen.append(int(candidate_starts[idx - 1]))
            idx = previous[idx - 1] + 1
            number -= 1
        else:
            idx -= 1
    chosen = sorted(chosen)
    selected_rates = [float(np.mean(flags[value : value + int(window_steps)])) for value in chosen]
    return tuple(chosen), {
        "selection": "maximum_total_event_rate_non_overlapping_within_declared_partition",
        "event_column": str(event_column),
        "stride": int(stride),
        "candidate_count": int(n),
        "selected_event_rates": selected_rates,
        "selected_event_rate_mean": float(np.mean(selected_rates)),
    }


def evaluate_policy_over_starts(
    *,
    truth: pd.DataFrame,
    sensors: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: object | None,
    policy: V2Policy,
    steps: int,
    start_indices: tuple[int, ...],
    seed_offset: int = 0,
) -> tuple[RolloutResult, dict[str, float | str | int]]:
    rollouts: list[RolloutResult] = []
    for offset, start_idx in enumerate(start_indices):
        env_cfg = replace(cfg, episode_len=int(steps), seed=int(cfg.seed) + int(seed_offset) + int(offset))
        env = WarmupSchedulingEnv(truth, sensors, constraints, env_cfg, oracle=oracle)
        rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx)))
    result = concat_rollout_results(rollouts, policy_name=str(policy.name))
    metrics = rollout_metrics(result)
    metrics["objective_loss_mean"] = final_objective(metrics)
    return result, metrics


def evaluate_static_candidates(
    *,
    truth: pd.DataFrame,
    sensors: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: object | None,
    candidate_masks: np.ndarray,
    steps: int,
    start_indices: tuple[int, ...],
    objective_mode: str = "oracle",
    task_error_columns: tuple[str, ...] = (),
    task_error_scales: tuple[float, ...] | None = None,
    task_error_event_only: bool = True,
    task_error_weight: float = 0.0,
) -> pd.DataFrame:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(sensors))
    rows: list[dict[str, object]] = []
    for action_idx, mask in enumerate(masks):
        policy = StaticMaskPolicy(mask=tuple(bool(x) for x in mask), name=f"static_candidate_{int(action_idx)}")
        result, metrics = evaluate_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=policy,
            steps=int(steps),
            start_indices=start_indices,
            seed_offset=30_000 + int(action_idx) * 17,
        )
        metrics.update(
            task_focus_metrics(
                result,
                state_columns=tuple(cfg.state_columns),
                task_error_columns=task_error_columns,
                task_error_scales=task_error_scales,
                event_only=bool(task_error_event_only),
            )
        )
        objective = final_objective(metrics, mode=str(objective_mode), task_error_weight=float(task_error_weight))
        rows.append(
            {
                "action_idx": int(action_idx),
                "objective_loss_mean": float(objective),
                "oracle_loss_mean": float(metrics["oracle_loss_mean"]),
                "instant_mae": float(metrics["instant_mae"]),
                "task_error_mean": float(metrics.get("task_error_mean", np.nan)),
                "task_error_event_mean": float(metrics.get("task_error_event_mean", np.nan)),
                "power_mean": float(metrics["power_mean"]),
                "peak_power_max": float(metrics["peak_power_max"]),
                "warmup_abort_count": int(metrics["warmup_abort_count"]),
                "sensor_ids": "|".join(str(sensors[i].sensor_id) for i in np.flatnonzero(mask)),
                **{f"sensor_{i}": int(v) for i, v in enumerate(mask.astype(int))},
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["objective_loss_mean", "power_mean", "warmup_abort_count", "action_idx"])
        .reset_index(drop=True)
    )


def normalized_candidate_costs(candidate_table: pd.DataFrame, *, n_actions: int) -> tuple[float, ...]:
    by_action = candidate_table.sort_values("action_idx")
    values = np.full(int(n_actions), np.nan, dtype=float)
    for _, row in by_action.iterrows():
        values[int(row["action_idx"])] = float(row["objective_loss_mean"])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return tuple(float(0.0) for _ in range(int(n_actions)))
    fill = float(np.nanmax(finite))
    values = np.where(np.isfinite(values), values, fill)
    lo = float(np.min(values))
    hi = float(np.percentile(values, 90.0))
    denom = max(hi - lo, 1e-6)
    normalized = np.clip((values - lo) / denom, 0.0, 4.0)
    return tuple(float(x) for x in normalized)


def final_objective(metrics: dict[str, object], *, mode: str = "oracle", task_error_weight: float = 0.0) -> float:
    if str(mode) == "task_composite":
        oracle_loss = float(metrics.get("oracle_loss_mean", float("nan")))
        task_error = float(metrics.get("task_error_mean", float("nan")))
        if np.isfinite(oracle_loss) and np.isfinite(task_error):
            return float(oracle_loss + float(task_error_weight) * task_error)
    oracle_loss = float(metrics.get("oracle_loss_mean", float("nan")))
    if np.isfinite(oracle_loss):
        return oracle_loss
    instant_mae = float(metrics.get("instant_mae", float("nan")))
    if np.isfinite(instant_mae):
        return instant_mae
    return float(metrics.get("mae", float("nan")))


def rollout_to_loaded(result: RolloutResult, *, sensor_ids: tuple[str, ...], state_columns: tuple[str, ...]) -> LoadedRollout:
    return LoadedRollout(
        policy=str(result.policy_name),
        observations=result.observations,
        masks=result.masks,
        truth=result.truth,
        rewards=result.rewards,
        scores=result.scores,
        powers=result.powers,
        peaks=result.peaks,
        selected_masks=result.selected_masks,
        mode_ids=result.mode_ids,
        event_flags=result.event_flags,
        oracle_losses=result.oracle_losses,
        step_indices=result.step_indices,
        warmup_abort_count=int(result.warmup_abort_count),
        sensor_ids=tuple(str(x) for x in sensor_ids),
        state_columns=tuple(str(x) for x in state_columns),
    )


def rich_metrics(
    result: RolloutResult,
    *,
    sensor_ids: tuple[str, ...],
    state_columns: tuple[str, ...],
    per_step_budget: float | None,
    startup_peak_budget: float | None,
) -> dict[str, float | int | str]:
    loaded = rollout_to_loaded(result, sensor_ids=sensor_ids, state_columns=state_columns)
    metrics = overall_metrics(
        loaded,
        per_step_budget=per_step_budget,
        startup_peak_budget=startup_peak_budget,
    )
    metrics["objective_loss_mean"] = final_objective(metrics)
    return metrics


def task_focus_metrics(
    result: RolloutResult,
    *,
    state_columns: tuple[str, ...],
    task_error_columns: tuple[str, ...] = (),
    task_error_scales: tuple[float, ...] | None = None,
    event_only: bool = True,
) -> dict[str, float]:
    columns = tuple(str(name) for name in task_error_columns)
    if not columns:
        return {
            "task_error_mean": float("nan"),
            "task_error_event_mean": float("nan"),
            "task_error_all_mean": float("nan"),
        }
    index = {str(name): idx for idx, name in enumerate(state_columns)}
    positions = [index[name] for name in columns if name in index]
    if not positions:
        return {
            "task_error_mean": float("nan"),
            "task_error_event_mean": float("nan"),
            "task_error_all_mean": float("nan"),
        }
    if task_error_scales is None:
        scales = np.ones(len(positions), dtype=float)
    else:
        raw = np.asarray(task_error_scales, dtype=float).reshape(-1)
        if raw.size != len(columns):
            scales = np.ones(len(positions), dtype=float)
        else:
            scales = np.asarray([raw[pos] for pos, name in enumerate(columns) if name in index], dtype=float)
    scales = np.maximum(scales.reshape(1, -1), 1e-12)
    err = np.abs(result.observations[:, positions] - result.truth[:, positions]) / scales
    all_mean = float(np.mean(err)) if err.size else float("nan")
    event_mask = np.asarray(result.event_flags, dtype=bool)
    event_mean = float(np.mean(err[event_mask])) if err.size and np.any(event_mask) else float("nan")
    selected = event_mean if bool(event_only) else all_mean
    return {
        "task_error_mean": float(selected),
        "task_error_event_mean": event_mean,
        "task_error_all_mean": all_mean,
    }


def save_rollout(path: str | Path, result: RolloutResult, *, sensor_ids: tuple[str, ...], state_columns: tuple[str, ...]) -> None:
    from v2.rollout import save_rollout_npz

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_rollout_npz(target, result, sensor_ids=sensor_ids, state_columns=state_columns)
