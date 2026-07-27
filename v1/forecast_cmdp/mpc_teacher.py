from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .reuse import ensure_archive_src

ensure_archive_src()

from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class MpcTeacherConfig:
    planning_horizon: int = 6
    beam_width: int = 8
    max_branch: int = 16
    oracle_loss_weight: float = 1.0
    event_weight_alpha: float = 1.0
    lambda_warmup_abort: float = 0.16
    lambda_switch: float = 0.002
    lambda_energy_deficit: float = 1.0
    prefer_low_power_tiebreak: float = 1.0e-4
    saturated_loss_threshold: float = 9.999
    saturated_coverage_bonus: float = 0.25
    candidate_prior_weight: float = 0.0
    candidate_prior_costs: tuple[float, ...] | None = None
    candidate_prefilter_top_k: int = 0
    anchor_mask: tuple[bool, ...] | None = None
    anchor_regret_guard: bool = False
    anchor_improvement_margin: float = 0.0
    task_error_weight: float = 0.0
    task_error_columns: tuple[str, ...] = ()
    task_error_scales: tuple[float, ...] | None = None
    task_error_event_only: bool = True


def enumerate_action_masks(n_sensors: int, max_active: int | None = None) -> np.ndarray:
    from itertools import combinations

    n = int(n_sensors)
    max_size = n if max_active is None else min(n, int(max_active))
    masks: list[np.ndarray] = []
    for size in range(0, max_size + 1):
        for combo in combinations(range(n), size):
            mask = np.zeros(n, dtype=bool)
            if combo:
                mask[list(combo)] = True
            masks.append(mask)
    return np.vstack(masks).astype(bool)


def feasible_masks(env: WarmupSchedulingEnv, candidate_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    keep = np.zeros(masks.shape[0], dtype=bool)
    for idx, mask in enumerate(masks):
        result = env.projector.project_mask(mask, env.runtimes)
        keep[idx] = bool(result.feasible and np.array_equal(result.selected_mask.astype(bool), mask))
    if not np.any(keep):
        keep[:] = True
    return masks[keep]


def beam_search_teacher_action(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig | None = None,
) -> np.ndarray:
    """Return the first action of a low-cost short-horizon schedule.

    The function snapshots and restores ``env``; callers can use it inside
    rollout code without advancing the real environment.
    """

    config = cfg or MpcTeacherConfig()
    original = snapshot_env(env)
    try:
        start_snapshot = snapshot_env(env)
        beams: list[tuple[float, list[np.ndarray], dict[str, object]]] = [(0.0, [], start_snapshot)]
        depth = max(1, int(config.planning_horizon))
        for _ in range(depth):
            expanded: list[tuple[float, list[np.ndarray], dict[str, object]]] = []
            for cost_so_far, seq, snap in beams:
                restore_env(env, snap)
                branches = _rank_one_step_branches(env, candidate_masks, config)
                for step_cost, mask, next_snap in branches[: max(1, int(config.max_branch))]:
                    expanded.append((float(cost_so_far + step_cost), [*seq, mask.copy()], next_snap))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            beams = expanded[: max(1, int(config.beam_width))]
        if not beams or not beams[0][1]:
            restore_env(env, start_snapshot)
            fm = feasible_masks(env, candidate_masks)
            return fm[0].copy()
        best_cost, best_sequence, _ = beams[0]
        best_action = best_sequence[0].astype(bool).copy()
        if bool(config.anchor_regret_guard) and config.anchor_mask is not None:
            anchor = _valid_anchor_mask(env, config.anchor_mask)
            if anchor is not None:
                restore_env(env, start_snapshot)
                anchor_cost = _rollout_repeated_mask_cost(env, anchor, depth, candidate_masks, config)
                margin = max(0.0, float(config.anchor_improvement_margin))
                if not np.isfinite(anchor_cost) or not (float(best_cost) + margin < float(anchor_cost)):
                    return anchor.astype(bool).copy()
        return best_action
    finally:
        restore_env(env, original)


def beam_search_first_action_costs(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig | None = None,
) -> np.ndarray:
    """Return approximate short-horizon cost for each feasible first action.

    Costs are computed with the same beam-search objective as the teacher, but
    grouped by the first action of the planned sequence. Non-evaluated or
    infeasible actions are returned as ``inf``.
    """

    config = cfg or MpcTeacherConfig()
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    original = snapshot_env(env)
    costs = np.full(masks.shape[0], np.inf, dtype=float)
    try:
        start_snapshot = snapshot_env(env)
        beams: list[tuple[float, int | None, dict[str, object]]] = [(0.0, None, start_snapshot)]
        depth = max(1, int(config.planning_horizon))
        for _ in range(depth):
            expanded: list[tuple[float, int | None, dict[str, object]]] = []
            for cost_so_far, first_idx, snap in beams:
                restore_env(env, snap)
                branches = _rank_one_step_branches_with_indices(env, masks, config)
                for action_idx, step_cost, _, next_snap in branches[: max(1, int(config.max_branch))]:
                    expanded.append(
                        (
                            float(cost_so_far + step_cost),
                            int(action_idx) if first_idx is None else int(first_idx),
                            next_snap,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            for total_cost, first_idx, _ in expanded:
                if first_idx is not None:
                    costs[int(first_idx)] = min(float(costs[int(first_idx)]), float(total_cost))
            beams = expanded[: max(1, int(config.beam_width))]
        if bool(config.anchor_regret_guard) and config.anchor_mask is not None:
            restore_env(env, start_snapshot)
            anchor = _valid_anchor_mask(env, config.anchor_mask)
            if anchor is not None:
                anchor_idx = _candidate_index(masks, anchor)
                if anchor_idx is not None:
                    anchor_cost = _rollout_repeated_mask_cost(env, anchor, depth, masks, config)
                    costs[int(anchor_idx)] = min(float(costs[int(anchor_idx)]), float(anchor_cost))
        return costs
    finally:
        restore_env(env, original)


@dataclass
class MpcTeacherPolicy(V2Policy):
    candidate_masks: np.ndarray
    cfg: MpcTeacherConfig = MpcTeacherConfig()
    name: str = "mpc_teacher"

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        return beam_search_teacher_action(env, self.candidate_masks, self.cfg)

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def _rank_one_step_branches(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig,
) -> list[tuple[float, np.ndarray, dict[str, object]]]:
    start = snapshot_env(env)
    rows: list[tuple[float, np.ndarray, dict[str, object]]] = []
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    branch_masks = _prefilter_candidate_masks(env, masks, cfg)
    for mask in feasible_masks(env, branch_masks):
        restore_env(env, start)
        step_idx = int(env.current_idx)
        _, reward, done, info = env.step_mask(mask)
        del done
        loss = float(info.get("oracle_loss", np.nan))
        if not np.isfinite(loss):
            loss = -float(reward)
        cost = _step_cost_from_info(env, mask, info, loss, cfg, masks, step_idx=step_idx)
        rows.append((float(cost), np.asarray(mask, dtype=bool).copy(), snapshot_env(env)))
    restore_env(env, start)
    rows.sort(key=lambda item: item[0])
    return rows


def _rank_one_step_branches_with_indices(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig,
) -> list[tuple[int, float, np.ndarray, dict[str, object]]]:
    start = snapshot_env(env)
    rows: list[tuple[int, float, np.ndarray, dict[str, object]]] = []
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    branch_masks = _prefilter_candidate_masks(env, masks, cfg)
    for mask in feasible_masks(env, branch_masks):
        action_idx = _candidate_index(masks, mask)
        if action_idx is None:
            continue
        restore_env(env, start)
        step_idx = int(env.current_idx)
        _, reward, done, info = env.step_mask(mask)
        del done
        loss = float(info.get("oracle_loss", np.nan))
        if not np.isfinite(loss):
            loss = -float(reward)
        cost = _step_cost_from_info(env, mask, info, loss, cfg, masks, step_idx=step_idx)
        rows.append((int(action_idx), float(cost), np.asarray(mask, dtype=bool).copy(), snapshot_env(env)))
    restore_env(env, start)
    rows.sort(key=lambda item: item[1])
    return rows


def _step_cost_from_info(
    env: WarmupSchedulingEnv,
    mask: np.ndarray,
    info: dict[str, object],
    loss: float,
    cfg: MpcTeacherConfig,
    candidate_masks: np.ndarray,
    step_idx: int,
) -> float:
    event_weight = 1.0 + float(cfg.event_weight_alpha) * float(bool(info.get("event", False)))
    switch_rate = float(info.get("switch_rate", 0.0))
    aborts = float(info.get("warmup_abort_delta", 0.0))
    deficit = float(info.get("energy_deficit", 0.0))
    power = float(info.get("power", 0.0))
    bootstrap_bonus = (
        float(cfg.saturated_coverage_bonus) * _intended_target_coverage(env, mask)
        if float(loss) >= float(cfg.saturated_loss_threshold)
        else 0.0
    )
    return float(
        float(cfg.oracle_loss_weight) * event_weight * float(loss)
        + float(cfg.lambda_switch) * switch_rate
        + float(cfg.lambda_warmup_abort) * aborts
        + float(cfg.lambda_energy_deficit) * deficit
        + float(cfg.prefer_low_power_tiebreak) * power
        + float(cfg.candidate_prior_weight) * _candidate_prior_cost(cfg, candidate_masks, mask)
        + float(cfg.task_error_weight) * _task_error(env, step_idx=step_idx, cfg=cfg)
        - bootstrap_bonus
    )


def _rollout_repeated_mask_cost(
    env: WarmupSchedulingEnv,
    mask: np.ndarray,
    depth: int,
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig,
) -> float:
    total = 0.0
    for _ in range(max(1, int(depth))):
        step_idx = int(env.current_idx)
        _, reward, done, info = env.step_mask(mask)
        loss = float(info.get("oracle_loss", np.nan))
        if not np.isfinite(loss):
            loss = -float(reward)
        total += _step_cost_from_info(env, mask, info, float(loss), cfg, candidate_masks, step_idx=step_idx)
        if bool(done):
            break
    return float(total)


def _task_error(env: WarmupSchedulingEnv, *, step_idx: int, cfg: MpcTeacherConfig) -> float:
    weight = float(cfg.task_error_weight)
    if weight <= 0.0 or not cfg.task_error_columns:
        return 0.0
    idx = int(step_idx)
    if idx < 0 or idx >= len(env.truth_values):
        return 0.0
    if bool(cfg.task_error_event_only) and not bool(env.event_flags[idx]):
        return 0.0
    columns = tuple(str(name) for name in cfg.task_error_columns)
    indices = [env.state_index[name] for name in columns if name in env.state_index]
    if not indices:
        return 0.0
    obs = np.asarray(env.last_observation, dtype=float)[indices]
    truth = np.asarray(env.truth_values[idx], dtype=float)[indices]
    if cfg.task_error_scales is None:
        scales = np.ones(len(indices), dtype=float)
    else:
        raw_scales = np.asarray(cfg.task_error_scales, dtype=float).reshape(-1)
        if raw_scales.size != len(columns):
            scales = np.ones(len(indices), dtype=float)
        else:
            scales = np.asarray([raw_scales[pos] for pos, name in enumerate(columns) if name in env.state_index], dtype=float)
    scales = np.maximum(scales, 1e-12)
    return float(np.mean(np.abs(obs - truth) / scales))


def _valid_anchor_mask(env: WarmupSchedulingEnv, mask: tuple[bool, ...] | np.ndarray) -> np.ndarray | None:
    anchor = np.asarray(mask, dtype=bool).reshape(-1)
    if anchor.shape[0] != len(env.sensor_ids):
        return None
    projected = env.projector.project_mask(anchor, env.runtimes)
    if not bool(projected.feasible):
        return None
    return np.asarray(projected.selected_mask, dtype=bool).reshape(-1)


def _candidate_prior_cost(cfg: MpcTeacherConfig, candidate_masks: np.ndarray, mask: np.ndarray) -> float:
    if cfg.candidate_prior_costs is None or float(cfg.candidate_prior_weight) <= 0.0:
        return 0.0
    costs = np.asarray(cfg.candidate_prior_costs, dtype=float).reshape(-1)
    if costs.shape[0] != candidate_masks.shape[0]:
        return 0.0
    matches = np.all(candidate_masks == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        return 0.0
    value = float(costs[int(ids[0])])
    return value if np.isfinite(value) else 0.0


def _candidate_index(candidate_masks: np.ndarray, mask: np.ndarray) -> int | None:
    matches = np.all(np.asarray(candidate_masks, dtype=bool) == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        return None
    return int(ids[0])


def _prefilter_candidate_masks(env: WarmupSchedulingEnv, candidate_masks: np.ndarray, cfg: MpcTeacherConfig) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))
    top_k = int(cfg.candidate_prefilter_top_k)
    if top_k <= 0 or cfg.candidate_prior_costs is None:
        return masks
    costs = np.asarray(cfg.candidate_prior_costs, dtype=float).reshape(-1)
    if costs.shape[0] != masks.shape[0]:
        return masks
    finite_order = np.argsort(np.where(np.isfinite(costs), costs, np.inf), kind="stable")
    selected: list[np.ndarray] = [masks[int(idx)].copy() for idx in finite_order[: max(1, min(top_k, masks.shape[0]))]]
    if cfg.anchor_mask is not None:
        anchor = np.asarray(cfg.anchor_mask, dtype=bool).reshape(-1)
        if anchor.shape[0] == masks.shape[1]:
            selected.append(anchor.copy())
    previous = np.asarray(env.previous_action_mask, dtype=bool).reshape(-1)
    if previous.shape[0] == masks.shape[1]:
        selected.append(previous.copy())
    dedup: list[np.ndarray] = []
    seen: set[tuple[bool, ...]] = set()
    for mask in selected:
        key = tuple(bool(x) for x in mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask.astype(bool).copy())
    return np.vstack(dedup).astype(bool) if dedup else masks


def _intended_target_coverage(env: WarmupSchedulingEnv, mask: np.ndarray) -> float:
    covered: set[str] = set()
    for idx in np.flatnonzero(np.asarray(mask, dtype=bool)):
        variables = set(str(name) for name in env.sensor_specs[int(idx)].observed_variables)
        covered.update(variables)
        if "wind_direction_deg" in variables:
            covered.update(("wind_dir_sin", "wind_dir_cos"))
    targets = tuple(str(name) for name in env.reward_target_columns)
    return float(sum(name in covered for name in targets)) / max(len(targets), 1)


def snapshot_env(env: WarmupSchedulingEnv) -> dict[str, object]:
    return {
        "episode_start_idx": int(env.episode_start_idx),
        "episode_end_idx": int(env.episode_end_idx),
        "current_idx": int(env.current_idx),
        "last_observation": np.array(env.last_observation, copy=True),
        "observed_mask": np.array(env.observed_mask, copy=True),
        "history": np.array(env.history, copy=True),
        "mask_history": np.array(env.mask_history, copy=True),
        "previous_action_mask": np.array(env.previous_action_mask, copy=True),
        "current_energy": float(getattr(env, "current_energy", 0.0)),
        "energy_deficit_steps": int(getattr(env, "energy_deficit_steps", 0)),
        "energy_deficit_total": float(getattr(env, "energy_deficit_total", 0.0)),
        "last_info": copy.deepcopy(env.last_info),
        "rng_state": copy.deepcopy(env.rng.bit_generator.state),
        "runtimes": {
            sensor_id: {
                "mode": runtime.mode,
                "warm_remaining": int(runtime.warm_remaining),
                "last_observed_step": runtime.last_observed_step,
                "warmup_abort_count": int(runtime.warmup_abort_count),
            }
            for sensor_id, runtime in env.runtimes.items()
        },
    }


def restore_env(env: WarmupSchedulingEnv, snapshot: dict[str, object]) -> None:
    env.episode_start_idx = int(snapshot["episode_start_idx"])
    env.episode_end_idx = int(snapshot["episode_end_idx"])
    env.current_idx = int(snapshot["current_idx"])
    env.last_observation = np.asarray(snapshot["last_observation"], dtype=float).copy()
    env.observed_mask = np.asarray(snapshot["observed_mask"], dtype=float).copy()
    env.history = np.asarray(snapshot["history"], dtype=float).copy()
    env.mask_history = np.asarray(snapshot["mask_history"], dtype=float).copy()
    env.previous_action_mask = np.asarray(snapshot["previous_action_mask"], dtype=float).copy()
    env.current_energy = float(snapshot.get("current_energy", 0.0))
    env.energy_deficit_steps = int(snapshot.get("energy_deficit_steps", 0))
    env.energy_deficit_total = float(snapshot.get("energy_deficit_total", 0.0))
    env.last_info = copy.deepcopy(snapshot["last_info"])
    env.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
    runtime_state = snapshot["runtimes"]
    assert isinstance(runtime_state, dict)
    for sensor_id, state in runtime_state.items():
        assert isinstance(state, dict)
        runtime = env.runtimes[str(sensor_id)]
        runtime.mode = state["mode"]
        runtime.warm_remaining = int(state["warm_remaining"])
        runtime.last_observed_step = state["last_observed_step"]
        runtime.warmup_abort_count = int(state["warmup_abort_count"])
