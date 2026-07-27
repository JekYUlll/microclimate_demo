from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
import pandas as pd

from .mpc_teacher import (
    MpcTeacherConfig,
    _candidate_prior_cost,
    _intended_target_coverage,
    feasible_masks,
    restore_env,
    snapshot_env,
)
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class CausalWorldModelContext:
    current_idx: int
    state_columns: tuple[str, ...]
    history: np.ndarray
    mask_history: np.ndarray
    last_observation: np.ndarray
    observed_mask: np.ndarray
    event_probabilities: np.ndarray


@dataclass(frozen=True)
class ScenarioBatch:
    """Sampled latent trajectories beginning at the current decision step."""

    values: np.ndarray
    event_flags: np.ndarray
    state_columns: tuple[str, ...]

    def validate(self, *, n_scenarios: int, min_horizon: int) -> None:
        values = np.asarray(self.values, dtype=float)
        events = np.asarray(self.event_flags, dtype=bool)
        if values.ndim != 3:
            raise ValueError(f"scenario values must be 3D, got {values.shape}")
        if values.shape[0] != int(n_scenarios):
            raise ValueError(
                f"scenario count mismatch: {values.shape[0]} != {int(n_scenarios)}"
            )
        if values.shape[1] < int(min_horizon):
            raise ValueError(
                f"scenario horizon {values.shape[1]} is shorter than {int(min_horizon)}"
            )
        if values.shape[2] != len(self.state_columns):
            raise ValueError("scenario state width does not match state_columns")
        if events.shape != values.shape[:2]:
            raise ValueError(
                f"event_flags shape {events.shape} != scenario prefix {values.shape[:2]}"
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("scenario values must be finite")


class CausalScenarioModel(Protocol):
    def sample(
        self,
        context: CausalWorldModelContext,
        *,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ScenarioBatch: ...


@dataclass(frozen=True)
class FixedScenarioModel:
    """Deterministic scenario source used for audits and fixed-forecast replay."""

    future_values: np.ndarray
    future_event_flags: np.ndarray | None = None

    def sample(
        self,
        context: CausalWorldModelContext,
        *,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ScenarioBatch:
        del rng
        future = np.asarray(self.future_values, dtype=float)
        if future.ndim == 2:
            future = np.repeat(future.reshape(1, *future.shape), int(n_scenarios), axis=0)
        if future.ndim != 3:
            raise ValueError(
                "future_values must have shape [horizon, state] "
                "or [scenario, horizon, state]"
            )
        if future.shape[0] not in (1, int(n_scenarios)):
            raise ValueError("fixed scenario count does not match requested n_scenarios")
        if future.shape[0] == 1 and int(n_scenarios) > 1:
            future = np.repeat(future, int(n_scenarios), axis=0)
        if future.shape[1] < max(0, int(horizon) - 1):
            raise ValueError("fixed future trajectory is shorter than requested horizon")
        if future.shape[2] != len(context.state_columns):
            raise ValueError("fixed future state width does not match context")
        current = np.repeat(
            np.asarray(context.last_observation, dtype=float).reshape(1, 1, -1),
            int(n_scenarios),
            axis=0,
        )
        values = np.concatenate([current, future[:, : max(0, int(horizon) - 1)]], axis=1)
        if self.future_event_flags is None:
            events = np.zeros(values.shape[:2], dtype=bool)
            if context.event_probabilities.size:
                probs = _extend_probabilities(context.event_probabilities, int(horizon))
                events[:] = probs.reshape(1, -1) >= 0.5
        else:
            future_events = np.asarray(self.future_event_flags, dtype=bool)
            if future_events.ndim == 1:
                future_events = np.repeat(
                    future_events.reshape(1, -1), int(n_scenarios), axis=0
                )
            if future_events.shape[0] == 1 and int(n_scenarios) > 1:
                future_events = np.repeat(future_events, int(n_scenarios), axis=0)
            if future_events.shape[0] != int(n_scenarios):
                raise ValueError("fixed event scenario count does not match request")
            if future_events.shape[1] < max(0, int(horizon) - 1):
                raise ValueError("fixed future event trajectory is shorter than requested horizon")
            current_event = np.zeros((int(n_scenarios), 1), dtype=bool)
            events = np.concatenate(
                [current_event, future_events[:, : max(0, int(horizon) - 1)]],
                axis=1,
            )
        batch = ScenarioBatch(
            values=values,
            event_flags=events,
            state_columns=context.state_columns,
        )
        batch.validate(n_scenarios=int(n_scenarios), min_horizon=int(horizon))
        return batch


@dataclass(frozen=True)
class RobustPlannerConfig:
    planning_horizon: int = 6
    beam_width: int = 8
    max_branch: int = 16
    n_scenarios: int = 16
    cvar_alpha: float = 0.75
    cvar_weight: float = 0.5
    seed: int = 42
    replan_interval: int = 1
    event_probability_columns: tuple[str, ...] = ()
    step_cost: MpcTeacherConfig = MpcTeacherConfig()
    component_guard_mode: str = "sequence"
    component_guard_hold_steps: int = 0
    component_guard_min_task_margin: float | None = None
    component_guard_min_task_q25: float | None = None
    component_guard_min_total_margin: float | None = None
    component_guard_min_total_q25: float | None = None


@dataclass(frozen=True)
class RobustPlanResult:
    action: np.ndarray
    sequence: tuple[np.ndarray, ...]
    scenario_costs: np.ndarray
    expected_cost: float
    cvar_cost: float
    robust_cost: float
    raw_action: np.ndarray | None = None
    raw_sequence: tuple[np.ndarray, ...] = ()
    raw_expected_cost: float | None = None
    raw_cvar_cost: float | None = None
    raw_robust_cost: float | None = None
    anchor_expected_cost: float | None = None
    anchor_cvar_cost: float | None = None
    anchor_robust_cost: float | None = None
    raw_component_costs: dict[str, np.ndarray] | None = None
    anchor_component_costs: dict[str, np.ndarray] | None = None
    component_guard_stats: dict[str, float] = field(default_factory=dict)
    anchor_guard_applied: bool = False
    component_guard_applied: bool = False


@dataclass
class RobustRecedingHorizonPolicy(V2Policy):
    scenario_model: CausalScenarioModel
    candidate_masks: np.ndarray
    cfg: RobustPlannerConfig = RobustPlannerConfig()
    name: str = "robust_world_model_mpc"
    _cached_action: np.ndarray | None = field(default=None, init=False, repr=False)
    _remaining_steps: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._cached_action = None
        self._remaining_steps = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self._cached_action is not None and self._remaining_steps > 0:
            self._remaining_steps -= 1
            return self._cached_action.copy()
        result = robust_beam_search_plan(
            env,
            self.scenario_model,
            self.candidate_masks,
            self.cfg,
        )
        self._cached_action = result.action.copy()
        self._remaining_steps = max(0, int(self.cfg.replan_interval) - 1)
        return result.action.copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        return np.where(self.act_mask(env), 1.0, -1.0)


def build_causal_world_model_context(
    env: WarmupSchedulingEnv,
    *,
    event_probability_columns: tuple[str, ...] = (),
) -> CausalWorldModelContext:
    probabilities: list[float] = []
    for column in event_probability_columns:
        if column not in env.truth_df.columns:
            raise ValueError(f"causal event probability column not found: {column}")
        value = float(env.truth_df.iloc[int(env.current_idx)][column])
        probabilities.append(float(np.clip(value, 0.0, 1.0)))
    return CausalWorldModelContext(
        current_idx=int(env.current_idx),
        state_columns=tuple(str(x) for x in env.state_columns),
        history=np.asarray(env.history, dtype=float).copy(),
        mask_history=np.asarray(env.mask_history, dtype=float).copy(),
        last_observation=np.asarray(env.last_observation, dtype=float).copy(),
        observed_mask=np.asarray(env.observed_mask, dtype=float).copy(),
        event_probabilities=np.asarray(probabilities, dtype=float),
    )


def robust_beam_search_plan(
    env: WarmupSchedulingEnv,
    scenario_model: CausalScenarioModel,
    candidate_masks: np.ndarray,
    cfg: RobustPlannerConfig | None = None,
) -> RobustPlanResult:
    config = cfg or RobustPlannerConfig()
    depth = max(1, int(config.planning_horizon))
    n_scenarios = max(1, int(config.n_scenarios))
    oracle_horizon = (
        max(1, int(env.oracle.cfg.horizon))
        if env.oracle is not None and bool(env.oracle.is_fitted)
        else 1
    )
    hold_depth = _component_guard_hold_depth(config)
    shadow_depth = max(depth, hold_depth)
    scenario_horizon = shadow_depth + oracle_horizon
    context = build_causal_world_model_context(
        env,
        event_probability_columns=tuple(config.event_probability_columns),
    )
    rng = np.random.default_rng(int(config.seed) + int(env.current_idx))
    batch = scenario_model.sample(
        context,
        horizon=scenario_horizon,
        n_scenarios=n_scenarios,
        rng=rng,
    )
    batch.validate(n_scenarios=n_scenarios, min_horizon=scenario_horizon)
    if tuple(batch.state_columns) != tuple(env.state_columns):
        raise ValueError("scenario model state_columns do not match environment state_columns")

    shadows = [
        build_scenario_environment(
            env,
            batch.values[idx],
            batch.event_flags[idx],
            planning_horizon=shadow_depth,
        )
        for idx in range(n_scenarios)
    ]
    start_snapshots = tuple(snapshot_env(shadow) for shadow in shadows)
    beams: list[
        tuple[float, tuple[np.ndarray, ...], np.ndarray, tuple[dict[str, object], ...]]
    ] = [
        (
            0.0,
            (),
            np.zeros(n_scenarios, dtype=float),
            start_snapshots,
        )
    ]
    masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, len(env.sensor_ids))

    for _ in range(depth):
        expanded: list[
            tuple[float, tuple[np.ndarray, ...], np.ndarray, tuple[dict[str, object], ...]]
        ] = []
        for _, sequence, cumulative_costs, snapshots in beams:
            branch_rows: list[
                tuple[
                    float,
                    np.ndarray,
                    np.ndarray,
                    tuple[dict[str, object], ...],
                ]
            ] = []
            restore_env(shadows[0], snapshots[0])
            available = feasible_masks(shadows[0], masks)
            for mask in available:
                step_costs = np.zeros(n_scenarios, dtype=float)
                next_snapshots: list[dict[str, object]] = []
                for scenario_idx, shadow in enumerate(shadows):
                    restore_env(shadow, snapshots[scenario_idx])
                    step_idx = int(shadow.current_idx)
                    _, reward, _, info = shadow.step_mask(mask)
                    loss = float(info.get("oracle_loss", np.nan))
                    if not np.isfinite(loss):
                        loss = -float(reward)
                    step_costs[scenario_idx] = _scenario_step_cost(
                        shadow,
                        mask,
                        info,
                        loss,
                        config.step_cost,
                        masks,
                        step_idx=step_idx,
                    )
                    next_snapshots.append(snapshot_env(shadow))
                total_costs = cumulative_costs + step_costs
                score, _, _ = robust_cost(
                    total_costs,
                    alpha=float(config.cvar_alpha),
                    cvar_weight=float(config.cvar_weight),
                )
                branch_rows.append(
                    (
                        score,
                        np.asarray(mask, dtype=bool).copy(),
                        total_costs,
                        tuple(next_snapshots),
                    )
                )
            branch_rows.sort(key=lambda row: row[0])
            for score, mask, total_costs, next_snapshots in branch_rows[
                : max(1, int(config.max_branch))
            ]:
                expanded.append(
                    (
                        float(score),
                        (*sequence, mask),
                        total_costs,
                        next_snapshots,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda row: row[0])
        beams = expanded[: max(1, int(config.beam_width))]

    if not beams or not beams[0][1]:
        fallback = feasible_masks(env, masks)[0].copy()
        return RobustPlanResult(
            action=fallback,
            sequence=(fallback.copy(),),
            scenario_costs=np.full(n_scenarios, np.inf, dtype=float),
            expected_cost=float("inf"),
            cvar_cost=float("inf"),
            robust_cost=float("inf"),
        )

    best_score, best_sequence, best_costs, _ = beams[0]
    raw_sequence = tuple(mask.copy() for mask in best_sequence)
    raw_costs = best_costs.copy()
    raw_expected, raw_cvar = _cost_components(
        raw_costs,
        alpha=float(config.cvar_alpha),
    )
    raw_score = float(best_score)
    raw_components = _evaluate_sequence_components(
        shadows,
        start_snapshots,
        raw_sequence,
        candidate_masks=masks,
        cfg=config,
    )
    anchor_expected: float | None = None
    anchor_cvar: float | None = None
    anchor_score: float | None = None
    anchor_components: dict[str, np.ndarray] | None = None
    anchor_guard_applied = False
    step_cfg = config.step_cost
    component_guard_stats: dict[str, float] = {}
    component_guard_applied = False
    anchor_guard_required = (
        bool(step_cfg.anchor_regret_guard)
        or _component_guard_enabled(config)
    )
    if anchor_guard_required and step_cfg.anchor_mask is not None:
        anchor = np.asarray(step_cfg.anchor_mask, dtype=bool).reshape(-1)
        if anchor.shape[0] == len(env.sensor_ids):
            anchor_score, anchor_costs = _evaluate_repeated_mask(
                shadows,
                start_snapshots,
                anchor,
                depth=depth,
                candidate_masks=masks,
                cfg=config,
            )
            anchor_expected, anchor_cvar = _cost_components(
                anchor_costs,
                alpha=float(config.cvar_alpha),
            )
            anchor_components = _evaluate_sequence_components(
                shadows,
                start_snapshots,
                tuple(anchor.copy() for _ in range(depth)),
                candidate_masks=masks,
                cfg=config,
            )
            guard_raw_components = raw_components
            guard_anchor_components = anchor_components
            component_guard_stats = _component_guard_stats(
                guard_raw_components,
                guard_anchor_components,
            )
            raw_first_is_anchor = bool(np.array_equal(raw_sequence[0], anchor))
            if (
                (not raw_first_is_anchor)
                and _component_guard_enabled(config)
                and str(config.component_guard_mode) == "hold"
            ):
                hold_sequence = tuple(
                    raw_sequence[0].copy() for _ in range(max(1, hold_depth))
                )
                anchor_hold_sequence = tuple(
                    anchor.copy() for _ in range(max(1, hold_depth))
                )
                guard_raw_components = _evaluate_sequence_components(
                    shadows,
                    start_snapshots,
                    hold_sequence,
                    candidate_masks=masks,
                    cfg=config,
                )
                guard_anchor_components = _evaluate_sequence_components(
                    shadows,
                    start_snapshots,
                    anchor_hold_sequence,
                    candidate_masks=masks,
                    cfg=config,
                )
                component_guard_stats = _component_guard_stats(
                    guard_raw_components,
                    guard_anchor_components,
                )
            component_guard_failed = (
                (not raw_first_is_anchor)
                and _component_guard_enabled(config)
                and (not _component_guard_passes(component_guard_stats, config))
            )
            margin = max(0.0, float(step_cfg.anchor_improvement_margin))
            regret_guard_failed = (
                bool(step_cfg.anchor_regret_guard)
                and not (float(best_score) + margin < float(anchor_score))
            )
            if regret_guard_failed or component_guard_failed:
                best_score = anchor_score
                best_sequence = tuple(anchor.copy() for _ in range(depth))
                best_costs = anchor_costs
                anchor_guard_applied = bool(regret_guard_failed)
                component_guard_applied = bool(component_guard_failed)
    expected, cvar = _cost_components(best_costs, alpha=float(config.cvar_alpha))
    return RobustPlanResult(
        action=best_sequence[0].copy(),
        sequence=tuple(mask.copy() for mask in best_sequence),
        scenario_costs=best_costs.copy(),
        expected_cost=expected,
        cvar_cost=cvar,
        robust_cost=float(best_score),
        raw_action=raw_sequence[0].copy(),
        raw_sequence=raw_sequence,
        raw_expected_cost=raw_expected,
        raw_cvar_cost=raw_cvar,
        raw_robust_cost=raw_score,
        anchor_expected_cost=anchor_expected,
        anchor_cvar_cost=anchor_cvar,
        anchor_robust_cost=anchor_score,
        raw_component_costs=raw_components,
        anchor_component_costs=anchor_components,
        component_guard_stats=component_guard_stats,
        anchor_guard_applied=anchor_guard_applied,
        component_guard_applied=component_guard_applied,
    )


def build_scenario_environment(
    source: WarmupSchedulingEnv,
    scenario_values: np.ndarray,
    scenario_event_flags: np.ndarray,
    *,
    planning_horizon: int,
) -> WarmupSchedulingEnv:
    values = np.asarray(scenario_values, dtype=float)
    events = np.asarray(scenario_event_flags, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(source.state_columns):
        raise ValueError("scenario_values must have shape [horizon, n_state_columns]")
    if events.shape[0] != values.shape[0]:
        raise ValueError("scenario event length does not match values")
    if np.any(~np.isfinite(values)):
        raise ValueError("scenario values must be finite")

    current_idx = int(source.current_idx)
    total_length = current_idx + values.shape[0]
    prefix = np.repeat(
        np.asarray(source.last_observation, dtype=float).reshape(1, -1),
        current_idx,
        axis=0,
    )
    truth_values = np.vstack([prefix, values])
    truth = pd.DataFrame(truth_values, columns=source.state_columns)
    truth[source.cfg.event_column] = np.concatenate(
        [np.zeros(current_idx, dtype=bool), events]
    )
    cfg: WarmupEnvConfig = replace(
        source.cfg,
        episode_len=max(1, int(planning_horizon)),
        normalization_mean=tuple(float(x) for x in source.state_mean),
        normalization_std=tuple(float(x) for x in source.state_std),
    )
    shadow = WarmupSchedulingEnv(
        truth,
        source.sensor_specs,
        source.projector.constraints,
        cfg,
        oracle=source.oracle,
    )
    state = snapshot_env(source)
    state["episode_end_idx"] = min(
        total_length,
        current_idx + max(1, int(planning_horizon)),
    )
    restore_env(shadow, state)
    return shadow


def robust_cost(
    scenario_costs: np.ndarray,
    *,
    alpha: float,
    cvar_weight: float,
) -> tuple[float, float, float]:
    expected, cvar = _cost_components(scenario_costs, alpha=alpha)
    score = expected + max(0.0, float(cvar_weight)) * cvar
    return float(score), expected, cvar


def _cost_components(
    scenario_costs: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    costs = np.asarray(scenario_costs, dtype=float).reshape(-1)
    if costs.size == 0 or np.any(~np.isfinite(costs)):
        return float("inf"), float("inf")
    expected = float(np.mean(costs))
    clipped_alpha = float(np.clip(alpha, 0.0, 0.999999))
    tail_count = max(1, int(np.ceil((1.0 - clipped_alpha) * costs.size)))
    cvar = float(np.mean(np.sort(costs)[-tail_count:]))
    return expected, cvar


def _component_guard_enabled(cfg: RobustPlannerConfig) -> bool:
    return any(
        value is not None
        for value in (
            cfg.component_guard_min_task_margin,
            cfg.component_guard_min_task_q25,
            cfg.component_guard_min_total_margin,
            cfg.component_guard_min_total_q25,
        )
    )


def _component_guard_hold_depth(cfg: RobustPlannerConfig) -> int:
    if str(cfg.component_guard_mode) != "hold" or not _component_guard_enabled(cfg):
        return 0
    configured = int(cfg.component_guard_hold_steps)
    if configured > 0:
        return configured
    return max(1, int(cfg.replan_interval))


def _component_guard_stats(
    raw_components: dict[str, np.ndarray] | None,
    anchor_components: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    for name in ("task_error", "total", "event_weighted_oracle"):
        raw = (
            np.asarray(raw_components.get(name), dtype=float).reshape(-1)
            if raw_components is not None and name in raw_components
            else np.asarray([], dtype=float)
        )
        anchor = (
            np.asarray(anchor_components.get(name), dtype=float).reshape(-1)
            if anchor_components is not None and name in anchor_components
            else np.asarray([], dtype=float)
        )
        if raw.size == 0 or anchor.size == 0 or raw.shape != anchor.shape:
            stats[f"{name}_margin_mean"] = float("nan")
            stats[f"{name}_margin_q25"] = float("nan")
            continue
        margins = anchor - raw
        stats[f"{name}_margin_mean"] = float(np.mean(margins))
        stats[f"{name}_margin_q25"] = float(np.quantile(margins, 0.25))
    return stats


def _component_guard_passes(
    stats: dict[str, float],
    cfg: RobustPlannerConfig,
) -> bool:
    checks = (
        ("task_error_margin_mean", cfg.component_guard_min_task_margin),
        ("task_error_margin_q25", cfg.component_guard_min_task_q25),
        ("total_margin_mean", cfg.component_guard_min_total_margin),
        ("total_margin_q25", cfg.component_guard_min_total_q25),
    )
    for name, threshold in checks:
        if threshold is None:
            continue
        value = float(stats.get(name, float("nan")))
        if not np.isfinite(value) or value < float(threshold):
            return False
    return True


def _scenario_step_cost(
    env: WarmupSchedulingEnv,
    mask: np.ndarray,
    info: dict[str, object],
    loss: float,
    cfg: MpcTeacherConfig,
    candidate_masks: np.ndarray,
    *,
    step_idx: int,
) -> float:
    return float(
        _scenario_step_cost_components(
            env,
            mask,
            info,
            loss,
            cfg,
            candidate_masks,
            step_idx=step_idx,
        )["total"]
    )


def _scenario_step_cost_components(
    env: WarmupSchedulingEnv,
    mask: np.ndarray,
    info: dict[str, object],
    loss: float,
    cfg: MpcTeacherConfig,
    candidate_masks: np.ndarray,
    *,
    step_idx: int,
) -> dict[str, float]:
    event_weight = 1.0 + float(cfg.event_weight_alpha) * float(
        bool(info.get("event", False))
    )
    bootstrap_bonus = (
        float(cfg.saturated_coverage_bonus) * _intended_target_coverage(env, mask)
        if float(loss) >= float(cfg.saturated_loss_threshold)
        else 0.0
    )
    task_error = _scenario_task_error(env, step_idx=step_idx, cfg=cfg)
    components = {
        "event_weighted_oracle": float(cfg.oracle_loss_weight)
        * event_weight
        * float(loss),
        "switch": float(cfg.lambda_switch) * float(info.get("switch_rate", 0.0)),
        "warmup_abort": float(cfg.lambda_warmup_abort)
        * float(info.get("warmup_abort_delta", 0.0)),
        "energy_deficit": float(cfg.lambda_energy_deficit)
        * float(info.get("energy_deficit", 0.0)),
        "power_tiebreak": float(cfg.prefer_low_power_tiebreak)
        * float(info.get("power", 0.0)),
        "candidate_prior": float(cfg.candidate_prior_weight)
        * _candidate_prior_cost(cfg, candidate_masks, mask),
        "task_error": float(cfg.task_error_weight) * task_error,
        "bootstrap_bonus": -float(bootstrap_bonus),
    }
    components["total"] = float(sum(components.values()))
    return {key: float(value) for key, value in components.items()}


def _evaluate_sequence_components(
    shadows: list[WarmupSchedulingEnv],
    start_snapshots: tuple[dict[str, object], ...],
    sequence: tuple[np.ndarray, ...],
    *,
    candidate_masks: np.ndarray,
    cfg: RobustPlannerConfig,
) -> dict[str, np.ndarray]:
    names = (
        "event_weighted_oracle",
        "switch",
        "warmup_abort",
        "energy_deficit",
        "power_tiebreak",
        "candidate_prior",
        "task_error",
        "bootstrap_bonus",
        "total",
    )
    totals = {name: np.zeros(len(shadows), dtype=float) for name in names}
    if not sequence:
        return totals
    for scenario_idx, shadow in enumerate(shadows):
        restore_env(shadow, start_snapshots[scenario_idx])
        for mask in sequence:
            step_idx = int(shadow.current_idx)
            _, reward, done, info = shadow.step_mask(mask)
            loss = float(info.get("oracle_loss", np.nan))
            if not np.isfinite(loss):
                loss = -float(reward)
            components = _scenario_step_cost_components(
                shadow,
                mask,
                info,
                loss,
                cfg.step_cost,
                candidate_masks,
                step_idx=step_idx,
            )
            for name in names:
                totals[name][scenario_idx] += float(components[name])
            if bool(done):
                break
    return totals


def _evaluate_repeated_mask(
    shadows: list[WarmupSchedulingEnv],
    start_snapshots: tuple[dict[str, object], ...],
    mask: np.ndarray,
    *,
    depth: int,
    candidate_masks: np.ndarray,
    cfg: RobustPlannerConfig,
) -> tuple[float, np.ndarray]:
    costs = np.zeros(len(shadows), dtype=float)
    for scenario_idx, shadow in enumerate(shadows):
        restore_env(shadow, start_snapshots[scenario_idx])
        for _ in range(max(1, int(depth))):
            step_idx = int(shadow.current_idx)
            _, reward, done, info = shadow.step_mask(mask)
            loss = float(info.get("oracle_loss", np.nan))
            if not np.isfinite(loss):
                loss = -float(reward)
            costs[scenario_idx] += _scenario_step_cost(
                shadow,
                mask,
                info,
                loss,
                cfg.step_cost,
                candidate_masks,
                step_idx=step_idx,
            )
            if bool(done):
                break
    score, _, _ = robust_cost(
        costs,
        alpha=float(cfg.cvar_alpha),
        cvar_weight=float(cfg.cvar_weight),
    )
    return score, costs


def _scenario_task_error(
    env: WarmupSchedulingEnv,
    *,
    step_idx: int,
    cfg: MpcTeacherConfig,
) -> float:
    if float(cfg.task_error_weight) <= 0.0 or not cfg.task_error_columns:
        return 0.0
    if step_idx < 0 or step_idx >= len(env.truth_values):
        return 0.0
    if bool(cfg.task_error_event_only) and not bool(env.event_flags[step_idx]):
        return 0.0
    columns = tuple(str(name) for name in cfg.task_error_columns)
    available = [
        (position, env.state_index[name])
        for position, name in enumerate(columns)
        if name in env.state_index
    ]
    if not available:
        return 0.0
    indices = [idx for _, idx in available]
    observation = np.asarray(env.last_observation, dtype=float)[indices]
    truth = np.asarray(env.truth_values[step_idx], dtype=float)[indices]
    if cfg.task_error_scales is None:
        scales = np.ones(len(indices), dtype=float)
    else:
        raw = np.asarray(cfg.task_error_scales, dtype=float).reshape(-1)
        if raw.size != len(columns):
            scales = np.ones(len(indices), dtype=float)
        else:
            scales = np.asarray([raw[position] for position, _ in available], dtype=float)
    return float(np.mean(np.abs(observation - truth) / np.maximum(scales, 1.0e-12)))


def _extend_probabilities(probabilities: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(int(horizon), dtype=float)
    if values.size >= int(horizon):
        return values[: int(horizon)]
    return np.pad(values, (0, int(horizon) - values.size), mode="edge")
