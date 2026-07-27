#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
)
from forecast_cmdp.mpc_teacher import MpcTeacherConfig  # noqa: E402
from forecast_cmdp.probabilistic_world_model import (  # noqa: E402
    load_probabilistic_world_model,
)
from forecast_cmdp.rollout_world_model import load_rollout_world_model  # noqa: E402
from forecast_cmdp.protocol import final_objective, task_focus_metrics  # noqa: E402
from forecast_cmdp.robust_planner import (  # noqa: E402
    RobustPlannerConfig,
    RobustRecedingHorizonPolicy,
    robust_beam_search_plan,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.rollout import RolloutResult, run_policy_rollout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-first gate for causal probabilistic "
            "world-model MPC against the validation-selected static policy."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-steps", type=int, default=64)
    parser.add_argument("--validation-start-count", type=int, default=4)
    parser.add_argument("--final-start-count", type=int, default=4)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-branch", type=int, default=8)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--cvar-alpha", type=float, default=0.75)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--replan-interval", type=int, default=4)
    parser.add_argument("--support-top-k", type=int, default=16)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument(
        "--anchor-improvement-margin",
        type=float,
        default=None,
        help="Override the planner's static-anchor improvement margin.",
    )
    parser.add_argument(
        "--oracle-loss-weight",
        type=float,
        default=None,
        help="Override robust planner oracle-loss component weight.",
    )
    parser.add_argument(
        "--event-weight-alpha",
        type=float,
        default=None,
        help="Override robust planner event multiplier alpha.",
    )
    parser.add_argument(
        "--task-error-weight",
        type=float,
        default=None,
        help="Override robust planner task-error component weight.",
    )
    parser.add_argument(
        "--task-error-event-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether task-error component is event-only.",
    )
    parser.add_argument(
        "--saturated-coverage-bonus",
        type=float,
        default=None,
        help="Override robust planner saturated coverage bonus.",
    )
    parser.add_argument(
        "--candidate-prior-weight",
        type=float,
        default=None,
        help="Override robust planner candidate-prior weight.",
    )
    parser.add_argument(
        "--component-guard-min-task-margin",
        type=float,
        default=None,
        help=(
            "Optional online verifier: require mean predicted task-error "
            "component margin of raw dynamic sequence over repeated anchor."
        ),
    )
    parser.add_argument(
        "--component-guard-mode",
        choices=("sequence", "hold"),
        default="sequence",
        help=(
            "Evaluate component guard on the planned raw sequence or on the "
            "actual first action held for the replan interval."
        ),
    )
    parser.add_argument(
        "--component-guard-hold-steps",
        type=int,
        default=0,
        help=(
            "When component guard mode is hold, override the number of steps "
            "used to compare raw first-action hold against anchor hold. "
            "Default 0 uses replan_interval."
        ),
    )
    parser.add_argument(
        "--component-guard-min-task-q25",
        type=float,
        default=None,
        help=(
            "Optional online verifier: require q25 predicted task-error "
            "component margin of raw dynamic sequence over repeated anchor."
        ),
    )
    parser.add_argument(
        "--component-guard-min-total-margin",
        type=float,
        default=None,
        help=(
            "Optional online verifier: require mean predicted total component "
            "margin of raw dynamic sequence over repeated anchor."
        ),
    )
    parser.add_argument(
        "--component-guard-min-total-q25",
        type=float,
        default=None,
        help=(
            "Optional online verifier: require q25 predicted total component "
            "margin of raw dynamic sequence over repeated anchor."
        ),
    )
    parser.add_argument(
        "--write-traces",
        action="store_true",
        help="Write per-replan planner diagnostics and per-step mask traces.",
    )
    parser.add_argument("--min-mean-margin", type=float, default=0.0)
    parser.add_argument("--min-q25-margin", type=float, default=-0.002)
    parser.add_argument("--max-negative-starts", type=int, default=1)
    parser.add_argument(
        "--run-final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Final runs only when this flag is set and validation passes.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def source_truth_path(source_run: Path, manifest: dict[str, object]) -> Path:
    for name in (
        "truth_with_learned_event_forecast.csv",
        "truth_with_learned_continuous_forecast.csv",
    ):
        candidate = source_run / name
        if candidate.exists():
            return candidate
    return resolve_path(str(manifest["truth_csv"]))


def main() -> None:
    args = parse_args()
    source_run = resolve_path(args.source_run)
    output = resolve_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (source_run / "manifest.json").read_text(encoding="utf-8")
    )
    run_args = dict(manifest["run_args"])
    helpers = load_v2_helpers()
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(
        str(name) for name in helpers.REWARD_TARGET_COLUMNS
    )
    truth_path = source_truth_path(source_run, manifest)
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(
        resolve_path(str(manifest["sensor_cfg"]))
    )
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    normalization_bounds = tuple(
        int(value) for value in manifest["normalization_bounds"]
    )
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=normalization_bounds[0],
        end_idx=normalization_bounds[1],
    )
    if norm_mean is None or norm_std is None:
        raise ValueError("formal robust planning requires split-locked normalization")
    env_cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(run_args["lookback"]),
        episode_len=int(args.window_steps),
        seed=int(manifest["seed"]),
        freq_s=int(run_args["freq_s"]),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        lambda_warmup_abort=float(run_args["lambda_warmup_abort"]),
        lambda_switch=float(run_args["lambda_switch"]),
        event_reward_multiplier=float(run_args["event_reward_multiplier"]),
        energy_account=bool(run_args["energy_account"]),
        energy_capacity=float(run_args["energy_capacity"]),
        initial_energy=float(run_args["initial_energy"]),
        harvest_per_step=float(run_args["harvest_per_step"]),
        reserve_energy=float(run_args["reserve_energy"]),
        lambda_energy_deficit=float(run_args["lambda_energy_deficit"]),
        soc_soft_penalty_buffer=float(run_args["soc_soft_penalty_buffer"]),
        lambda_soc_soft_penalty=float(run_args["lambda_soc_soft_penalty"]),
        common_random_numbers=True,
    )
    oracle = load_archived_oracle(
        resolve_path(str(manifest["oracle_path"])),
        oracle_type=str(manifest["oracle_type"]),
        device=str(args.oracle_device),
    )
    world_model = load_world_model(resolve_path(args.world_model), device=str(args.model_device))
    if tuple(world_model.state_columns) != state_columns:
        raise ValueError("world-model and source state columns do not match")

    teacher_path = resolve_path(str(manifest["teacher_dataset"]))
    with np.load(teacher_path, allow_pickle=False) as teacher:
        all_masks = np.asarray(teacher["candidate_masks"], dtype=bool)
    selected_static = dict(manifest["selected_static"])
    anchor_idx = int(selected_static["action_idx"])
    anchor_mask = np.asarray(selected_static["mask"], dtype=bool)
    source_step_cfg = MpcTeacherConfig(**dict(manifest["teacher_cfg"]))
    support_indices = select_support(
        all_masks,
        source_step_cfg,
        anchor_idx=anchor_idx,
        top_k=int(args.support_top_k),
    )
    candidate_masks = all_masks[list(support_indices)]
    prior_costs = (
        tuple(
            float(source_step_cfg.candidate_prior_costs[index])
            for index in support_indices
        )
        if source_step_cfg.candidate_prior_costs is not None
        else None
    )
    step_cfg = replace(
        source_step_cfg,
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
        oracle_loss_weight=(
            float(args.oracle_loss_weight)
            if args.oracle_loss_weight is not None
            else float(source_step_cfg.oracle_loss_weight)
        ),
        event_weight_alpha=(
            float(args.event_weight_alpha)
            if args.event_weight_alpha is not None
            else float(source_step_cfg.event_weight_alpha)
        ),
        task_error_weight=(
            float(args.task_error_weight)
            if args.task_error_weight is not None
            else float(source_step_cfg.task_error_weight)
        ),
        task_error_event_only=(
            bool(args.task_error_event_only)
            if args.task_error_event_only is not None
            else bool(source_step_cfg.task_error_event_only)
        ),
        saturated_coverage_bonus=(
            float(args.saturated_coverage_bonus)
            if args.saturated_coverage_bonus is not None
            else float(source_step_cfg.saturated_coverage_bonus)
        ),
        candidate_prior_weight=(
            float(args.candidate_prior_weight)
            if args.candidate_prior_weight is not None
            else float(source_step_cfg.candidate_prior_weight)
        ),
        candidate_prior_costs=prior_costs,
        candidate_prefilter_top_k=0,
        anchor_mask=tuple(bool(value) for value in anchor_mask),
        anchor_regret_guard=True,
        anchor_improvement_margin=(
            float(args.anchor_improvement_margin)
            if args.anchor_improvement_margin is not None
            else float(source_step_cfg.anchor_improvement_margin)
        ),
    )
    event_probability_columns = tuple(
        str(name)
        for name in dict(manifest["forecast_cfg"]).get(
            "learned_event_probability_columns", ()
        )
    )
    planner_cfg = RobustPlannerConfig(
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
        n_scenarios=int(args.scenarios),
        cvar_alpha=float(args.cvar_alpha),
        cvar_weight=float(args.cvar_weight),
        seed=int(manifest["seed"]),
        replan_interval=int(args.replan_interval),
        event_probability_columns=event_probability_columns,
        step_cost=step_cfg,
        component_guard_mode=str(args.component_guard_mode),
        component_guard_hold_steps=int(args.component_guard_hold_steps),
        component_guard_min_task_margin=args.component_guard_min_task_margin,
        component_guard_min_task_q25=args.component_guard_min_task_q25,
        component_guard_min_total_margin=args.component_guard_min_total_margin,
        component_guard_min_total_q25=args.component_guard_min_total_q25,
    )

    validation_starts = tuple(
        int(value)
        for value in manifest["starts"]["validation"]["starts"][
            : max(1, int(args.validation_start_count))
        ]
    )
    validation_plan_rows: list[dict[str, object]] | None = (
        [] if bool(args.write_traces) else None
    )
    validation_step_rows: list[dict[str, object]] | None = (
        [] if bool(args.write_traces) else None
    )
    validation_rows = evaluate_split(
        split_name="validation",
        starts=validation_starts,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=oracle,
        world_model=world_model,
        candidate_masks=candidate_masks,
        anchor_mask=anchor_mask,
        planner_cfg=planner_cfg,
        run_args=run_args,
        state_columns=state_columns,
        window_steps=int(args.window_steps),
        support_indices=support_indices,
        trace_plan_rows=validation_plan_rows,
        trace_step_rows=validation_step_rows,
    )
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(output / "validation_paired.csv", index=False)
    if bool(args.write_traces):
        pd.DataFrame(validation_plan_rows or []).to_csv(
            output / "validation_plan_trace.csv",
            index=False,
        )
        pd.DataFrame(validation_step_rows or []).to_csv(
            output / "validation_step_trace.csv",
            index=False,
        )
    validation_gate = margin_gate(
        validation_table["margin"].to_numpy(dtype=float),
        min_mean=float(args.min_mean_margin),
        min_q25=float(args.min_q25_margin),
        max_negative=int(args.max_negative_starts),
    )

    final_rows: list[dict[str, object]] = []
    final_plan_rows: list[dict[str, object]] | None = (
        [] if bool(args.write_traces) else None
    )
    final_step_rows: list[dict[str, object]] | None = (
        [] if bool(args.write_traces) else None
    )
    final_status = "locked"
    if bool(args.run_final) and bool(validation_gate["pass"]):
        final_starts = tuple(
            int(value)
            for value in manifest["starts"]["final_test"]["starts"][
                : max(1, int(args.final_start_count))
            ]
        )
        final_rows = evaluate_split(
            split_name="final_test",
            starts=final_starts,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            env_cfg=env_cfg,
            oracle=oracle,
            world_model=world_model,
            candidate_masks=candidate_masks,
            anchor_mask=anchor_mask,
            planner_cfg=planner_cfg,
            run_args=run_args,
            state_columns=state_columns,
            window_steps=int(args.window_steps),
            trace_plan_rows=final_plan_rows,
            trace_step_rows=final_step_rows,
        )
        pd.DataFrame(final_rows).to_csv(output / "final_paired.csv", index=False)
        if bool(args.write_traces):
            pd.DataFrame(final_plan_rows or []).to_csv(
                output / "final_plan_trace.csv",
                index=False,
            )
            pd.DataFrame(final_step_rows or []).to_csv(
                output / "final_step_trace.csv",
                index=False,
            )
        final_status = "completed"
    elif bool(args.run_final):
        final_status = "blocked_by_validation_gate"

    summary = {
        "role": "causal_robust_planner_validation_gate",
        "source_run": str(source_run),
        "truth_csv": str(truth_path),
        "world_model": str(resolve_path(args.world_model)),
        "normalization_bounds": list(normalization_bounds),
        "support_indices": list(support_indices),
        "anchor_action_idx": anchor_idx,
        "anchor_mask": anchor_mask.astype(int).tolist(),
        "planner_config": {
            **asdict(planner_cfg),
            "step_cost": asdict(step_cfg),
        },
        "validation_gate": validation_gate,
        "final_status": final_status,
        "validation_or_final_used_for_world_model": False,
    }
    if final_rows:
        summary["final_result"] = margin_gate(
            np.asarray([row["margin"] for row in final_rows], dtype=float),
            min_mean=float(args.min_mean_margin),
            min_q25=float(args.min_q25_margin),
            max_negative=int(args.max_negative_starts),
        )
    (output / "robust_planner_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def select_support(
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig,
    *,
    anchor_idx: int,
    top_k: int,
) -> tuple[int, ...]:
    masks = np.asarray(candidate_masks, dtype=bool)
    if cfg.candidate_prior_costs is None:
        selected = list(range(min(max(1, int(top_k)), masks.shape[0])))
    else:
        costs = np.asarray(cfg.candidate_prior_costs, dtype=float)
        if costs.shape[0] != masks.shape[0]:
            raise ValueError("candidate prior cost width does not match masks")
        selected = np.argsort(
            np.where(np.isfinite(costs), costs, np.inf),
            kind="stable",
        )[: max(1, int(top_k))].astype(int).tolist()
    selected.append(int(anchor_idx))
    return tuple(dict.fromkeys(int(index) for index in selected))


def load_world_model(path: Path, *, device: str) -> object:
    try:
        return load_rollout_world_model(path, device=device)
    except ValueError:
        return load_probabilistic_world_model(path, device=device)


def optional_float(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


COMPONENT_TRACE_NAMES = (
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


def component_trace_fields(
    raw_components: dict[str, np.ndarray] | None,
    anchor_components: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    fields: dict[str, float] = {}
    for name in COMPONENT_TRACE_NAMES:
        raw_values = (
            np.asarray(raw_components.get(name), dtype=float).reshape(-1)
            if raw_components is not None and name in raw_components
            else np.asarray([], dtype=float)
        )
        anchor_values = (
            np.asarray(anchor_components.get(name), dtype=float).reshape(-1)
            if anchor_components is not None and name in anchor_components
            else np.asarray([], dtype=float)
        )
        raw_mean = float(np.mean(raw_values)) if raw_values.size else float("nan")
        anchor_mean = (
            float(np.mean(anchor_values)) if anchor_values.size else float("nan")
        )
        fields[f"raw_component_{name}_mean"] = raw_mean
        fields[f"anchor_component_{name}_mean"] = anchor_mean
        fields[f"predicted_anchor_minus_raw_component_{name}_mean"] = (
            anchor_mean - raw_mean
            if np.isfinite(anchor_mean) and np.isfinite(raw_mean)
            else float("nan")
        )
    return fields


def mask_bits(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return "".join("1" if bool(value) else "0" for value in values)


def mask_support_position(candidate_masks: np.ndarray, mask: np.ndarray) -> int:
    masks = np.asarray(candidate_masks, dtype=bool)
    target = np.asarray(mask, dtype=bool).reshape(1, -1)
    matches = np.flatnonzero(np.all(masks == target, axis=1))
    return int(matches[0]) if matches.size else -1


def support_global_index(support_indices: tuple[int, ...], position: int) -> int:
    pos = int(position)
    if pos < 0 or pos >= len(support_indices):
        return -1
    return int(support_indices[pos])


def append_step_trace_rows(
    rows: list[dict[str, object]],
    *,
    policy_name: str,
    split_name: str,
    start: int,
    result: RolloutResult,
    candidate_masks: np.ndarray,
    support_indices: tuple[int, ...],
    anchor_mask: np.ndarray,
    state_columns: tuple[str, ...],
    task_error_columns: tuple[str, ...],
    task_error_scales: tuple[float, ...],
) -> None:
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    state_index = {str(name): idx for idx, name in enumerate(state_columns)}
    scale_map = {
        str(name): float(task_error_scales[idx])
        for idx, name in enumerate(task_error_columns)
        if idx < len(task_error_scales)
    }
    for local_step, step_idx in enumerate(result.step_indices.astype(int).tolist()):
        mask = np.asarray(result.selected_masks[local_step], dtype=bool).reshape(-1)
        pos = mask_support_position(candidate_masks, mask)
        row: dict[str, object] = {
            "policy": str(policy_name),
            "split": split_name,
            "start": int(start),
            "relative_step": int(local_step),
            "step_idx": int(step_idx),
            "mask_support_pos": pos,
            "mask_global_idx": support_global_index(support_indices, pos),
            "mask_bits": mask_bits(mask),
            "is_anchor": bool(np.array_equal(mask, anchor)),
            "power": float(result.powers[local_step]),
            "peak": float(result.peaks[local_step]),
            "event": float(result.event_flags[local_step]),
            "oracle_loss": float(result.oracle_losses[local_step]),
            "reward": float(result.rewards[local_step]),
            "soc": float(result.soc[local_step]),
            "warmup_abort_delta": int(result.warmup_abort_deltas[local_step]),
            "energy_guard_dropped": int(result.energy_guard_dropped[local_step]),
        }
        for column in task_error_columns:
            idx = state_index.get(str(column))
            if idx is None:
                continue
            error = abs(
                float(result.observations[local_step, idx])
                - float(result.truth[local_step, idx])
            )
            scale = max(float(scale_map.get(str(column), 1.0)), 1.0e-12)
            safe = str(column).replace("/", "_")
            row[f"abs_error_{safe}"] = error
            row[f"norm_error_{safe}"] = error / scale
        rows.append(row)


class TracingRobustRecedingHorizonPolicy:
    def __init__(
        self,
        *,
        scenario_model: object,
        candidate_masks: np.ndarray,
        cfg: RobustPlannerConfig,
        support_indices: tuple[int, ...],
        split_name: str,
        start_idx: int,
        anchor_mask: np.ndarray,
        trace_rows: list[dict[str, object]],
    ) -> None:
        self.scenario_model = scenario_model
        self.candidate_masks = np.asarray(candidate_masks, dtype=bool)
        self.cfg = cfg
        self.support_indices = tuple(int(value) for value in support_indices)
        self.split_name = str(split_name)
        self.start_idx = int(start_idx)
        self.anchor_mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
        self.trace_rows = trace_rows
        self.name = "robust_world_model_mpc"
        self._cached_action: np.ndarray | None = None
        self._remaining_steps = 0
        self._replan_id = 0

    def reset(self) -> None:
        self._cached_action = None
        self._remaining_steps = 0
        self._replan_id = 0

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
        action = result.action.copy()
        raw_action = (
            np.asarray(result.raw_action, dtype=bool).reshape(-1)
            if result.raw_action is not None
            else action
        )
        action_pos = mask_support_position(self.candidate_masks, action)
        raw_pos = mask_support_position(self.candidate_masks, raw_action)
        row = {
                "split": self.split_name,
                "start": self.start_idx,
                "replan_id": self._replan_id,
                "current_idx": int(env.current_idx),
                "relative_step": int(env.current_idx) - self.start_idx,
                "action_support_pos": action_pos,
                "action_global_idx": support_global_index(
                    self.support_indices,
                    action_pos,
                ),
                "action_bits": mask_bits(action),
                "action_is_anchor": bool(np.array_equal(action, self.anchor_mask)),
                "raw_action_support_pos": raw_pos,
                "raw_action_global_idx": support_global_index(
                    self.support_indices,
                    raw_pos,
                ),
                "raw_action_bits": mask_bits(raw_action),
                "raw_action_is_anchor": bool(
                    np.array_equal(raw_action, self.anchor_mask)
                ),
                "anchor_guard_applied": bool(result.anchor_guard_applied),
                "component_guard_applied": bool(result.component_guard_applied),
                "expected_cost": float(result.expected_cost),
                "cvar_cost": float(result.cvar_cost),
                "robust_cost": float(result.robust_cost),
                "raw_expected_cost": optional_float(result.raw_expected_cost),
                "raw_cvar_cost": optional_float(result.raw_cvar_cost),
                "raw_robust_cost": optional_float(result.raw_robust_cost),
                "anchor_expected_cost": optional_float(result.anchor_expected_cost),
                "anchor_cvar_cost": optional_float(result.anchor_cvar_cost),
                "anchor_robust_cost": optional_float(result.anchor_robust_cost),
                "predicted_anchor_minus_raw": (
                    optional_float(result.anchor_robust_cost)
                    - optional_float(result.raw_robust_cost)
                    if result.anchor_robust_cost is not None
                    and result.raw_robust_cost is not None
                    else float("nan")
                ),
                "scenario_cost_mean": float(np.mean(result.scenario_costs)),
                "scenario_cost_std": float(np.std(result.scenario_costs)),
                "scenario_cost_q25": float(np.quantile(result.scenario_costs, 0.25)),
                "scenario_cost_q75": float(np.quantile(result.scenario_costs, 0.75)),
            }
        row.update(
            component_trace_fields(
                result.raw_component_costs,
                result.anchor_component_costs,
            )
        )
        for name, value in result.component_guard_stats.items():
            row[f"component_guard_{name}"] = float(value)
        self.trace_rows.append(row)
        self._cached_action = action.copy()
        self._remaining_steps = max(0, int(self.cfg.replan_interval) - 1)
        self._replan_id += 1
        return action

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        return np.where(self.act_mask(env), 1.0, -1.0)


def evaluate_split(
    *,
    split_name: str,
    starts: tuple[int, ...],
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    env_cfg: object,
    oracle: object,
    world_model: object,
    candidate_masks: np.ndarray,
    anchor_mask: np.ndarray,
    planner_cfg: RobustPlannerConfig,
    run_args: dict[str, object],
    state_columns: tuple[str, ...],
    window_steps: int,
    support_indices: tuple[int, ...] = (),
    trace_plan_rows: list[dict[str, object]] | None = None,
    trace_step_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, start in enumerate(starts):
        seed = int(env_cfg.seed) + 50_000 + position
        paired_cfg = replace(env_cfg, seed=seed, episode_len=int(window_steps))
        static_env = WarmupSchedulingEnv(
            truth,
            sensors,
            constraints,
            paired_cfg,
            oracle=oracle,
        )
        planner_env = WarmupSchedulingEnv(
            truth,
            sensors,
            constraints,
            paired_cfg,
            oracle=oracle,
        )
        static_result = run_policy_rollout(
            static_env,
            StaticMaskPolicy(
                tuple(bool(value) for value in anchor_mask),
                name="validation_selected_static",
            ),
            steps=int(window_steps),
            start_idx=int(start),
        )
        if trace_plan_rows is None:
            planner_policy = RobustRecedingHorizonPolicy(
                scenario_model=world_model,
                candidate_masks=candidate_masks,
                cfg=planner_cfg,
            )
        else:
            planner_policy = TracingRobustRecedingHorizonPolicy(
                scenario_model=world_model,
                candidate_masks=candidate_masks,
                cfg=planner_cfg,
                support_indices=support_indices,
                split_name=split_name,
                start_idx=int(start),
                anchor_mask=anchor_mask,
                trace_rows=trace_plan_rows,
            )
        planner_result = run_policy_rollout(
            planner_env,
            planner_policy,
            steps=int(window_steps),
            start_idx=int(start),
        )
        if trace_step_rows is not None:
            append_step_trace_rows(
                trace_step_rows,
                policy_name="validation_selected_static",
                split_name=split_name,
                start=int(start),
                result=static_result,
                candidate_masks=candidate_masks,
                support_indices=support_indices,
                anchor_mask=anchor_mask,
                state_columns=state_columns,
                task_error_columns=tuple(
                    str(name) for name in run_args["task_error_columns"]
                ),
                task_error_scales=tuple(
                    float(value) for value in run_args["task_error_scales"]
                ),
            )
            append_step_trace_rows(
                trace_step_rows,
                policy_name="robust_world_model_mpc",
                split_name=split_name,
                start=int(start),
                result=planner_result,
                candidate_masks=candidate_masks,
                support_indices=support_indices,
                anchor_mask=anchor_mask,
                state_columns=state_columns,
                task_error_columns=tuple(
                    str(name) for name in run_args["task_error_columns"]
                ),
                task_error_scales=tuple(
                    float(value) for value in run_args["task_error_scales"]
                ),
            )
        static_metrics = rollout_objective(
            static_result,
            run_args=run_args,
            state_columns=state_columns,
        )
        planner_metrics = rollout_objective(
            planner_result,
            run_args=run_args,
            state_columns=state_columns,
        )
        row = {
            "split": split_name,
            "start": int(start),
            "static_objective": float(static_metrics["objective"]),
            "planner_objective": float(planner_metrics["objective"]),
            "margin": float(
                static_metrics["objective"] - planner_metrics["objective"]
            ),
            "static_power_mean": float(static_metrics["power_mean"]),
            "planner_power_mean": float(planner_metrics["power_mean"]),
            "static_switch_rate": float(static_metrics["switch_rate"]),
            "planner_switch_rate": float(planner_metrics["switch_rate"]),
            "planner_dynamic_rate": float(
                np.mean(
                    np.any(
                        planner_result.selected_masks
                        != anchor_mask.reshape(1, -1),
                        axis=1,
                    )
                )
            ),
            "static_warmup_aborts": int(static_result.warmup_abort_count),
            "planner_warmup_aborts": int(planner_result.warmup_abort_count),
            "static_constraint_violations": int(
                static_metrics["constraint_violations"]
            ),
            "planner_constraint_violations": int(
                planner_metrics["constraint_violations"]
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def rollout_objective(
    result: RolloutResult,
    *,
    run_args: dict[str, object],
    state_columns: tuple[str, ...],
) -> dict[str, float | int]:
    finite_oracle = result.oracle_losses[np.isfinite(result.oracle_losses)]
    metrics: dict[str, object] = {
        "oracle_loss_mean": (
            float(np.mean(finite_oracle)) if finite_oracle.size else float("nan")
        ),
        "instant_mae": float(np.mean(np.abs(result.observations - result.truth))),
    }
    metrics.update(
        task_focus_metrics(
            result,
            state_columns=state_columns,
            task_error_columns=tuple(
                str(name) for name in run_args["task_error_columns"]
            ),
            task_error_scales=tuple(
                float(value) for value in run_args["task_error_scales"]
            ),
            event_only=bool(run_args["task_error_event_only"]),
        )
    )
    objective = final_objective(
        metrics,
        mode=str(run_args["objective_mode"]),
        task_error_weight=float(run_args["task_error_weight"]),
    )
    previous = np.vstack(
        [
            np.zeros((1, result.selected_masks.shape[1]), dtype=float),
            result.selected_masks[:-1],
        ]
    )
    switch_rate = float(
        np.mean(np.abs(result.selected_masks.astype(float) - previous))
    )
    violations = int(
        np.sum(result.powers > float(run_args["budget"]) + 1.0e-9)
        + np.sum(
            result.peaks
            > float(run_args["startup_peak_budget"]) + 1.0e-9
        )
    )
    return {
        "objective": float(objective),
        "power_mean": float(np.mean(result.powers)),
        "switch_rate": switch_rate,
        "constraint_violations": violations,
    }


def margin_gate(
    margins: np.ndarray,
    *,
    min_mean: float,
    min_q25: float,
    max_negative: int,
) -> dict[str, object]:
    values = np.asarray(margins, dtype=float).reshape(-1)
    mean = float(np.mean(values))
    q25 = float(np.quantile(values, 0.25))
    negative = int(np.sum(values < 0.0))
    return {
        "count": int(values.size),
        "mean_margin": mean,
        "q25_margin": q25,
        "min_margin": float(np.min(values)),
        "max_margin": float(np.max(values)),
        "negative_starts": negative,
        "pass": bool(
            mean > float(min_mean)
            and q25 >= float(min_q25)
            and negative <= int(max_negative)
        ),
    }


if __name__ == "__main__":
    main()
