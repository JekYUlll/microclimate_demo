#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
SCRIPTS_ROOT = V1_ROOT / "scripts"
sys.path.insert(0, str(V1_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    continue_policy_rollout,
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
)
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, restore_env, snapshot_env  # noqa: E402
from forecast_cmdp.robust_planner import RobustPlannerConfig, robust_beam_search_plan  # noqa: E402
from run_robust_planner_gate import (  # noqa: E402
    load_world_model,
    mask_bits,
    optional_float,
    resolve_path,
    select_support,
    source_truth_path,
    rollout_objective,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402


def runtime_effect_features(
    env: WarmupSchedulingEnv,
    *,
    raw_action: np.ndarray,
    anchor_mask: np.ndarray,
    event_probability_columns: tuple[str, ...],
    task_error_columns: tuple[str, ...],
) -> dict[str, float | int | str]:
    raw = np.asarray(raw_action, dtype=bool).reshape(-1)
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    previous = np.asarray(env.previous_action_mask, dtype=bool).reshape(-1)
    fields: dict[str, float | int | str] = {
        "current_idx": int(env.current_idx),
        "elapsed_steps": int(getattr(env, "elapsed_steps", 0)),
        "soc": float(getattr(env, "current_energy", 0.0)),
        "soc_ratio": float(env._soc_ratio()) if hasattr(env, "_soc_ratio") else float("nan"),
        "energy_deficit_steps": int(getattr(env, "energy_deficit_steps", 0)),
        "energy_deficit_total": float(getattr(env, "energy_deficit_total", 0.0)),
        "previous_action_bits": mask_bits(previous),
        "raw_anchor_hamming": float(np.mean(raw != anchor)),
        "raw_previous_hamming": float(np.mean(raw != previous)),
        "anchor_previous_hamming": float(np.mean(anchor != previous)),
        "raw_active_count": int(np.sum(raw)),
        "anchor_active_count": int(np.sum(anchor)),
    }
    for idx, sensor_id in enumerate(env.sensor_ids):
        runtime = env.runtimes[str(sensor_id)]
        fields[f"raw_sensor_{sensor_id}"] = int(raw[idx])
        fields[f"anchor_sensor_{sensor_id}"] = int(anchor[idx])
        fields[f"previous_sensor_{sensor_id}"] = int(previous[idx])
        fields[f"runtime_mode_{sensor_id}"] = int(runtime.mode)
        fields[f"runtime_warm_remaining_{sensor_id}"] = int(runtime.warm_remaining)
        fields[f"runtime_freshness_{sensor_id}"] = float(runtime.freshness(env.current_idx))
    for column in event_probability_columns:
        if column in env.truth_df.columns:
            fields[f"context_{column}"] = float(env.truth_df.iloc[int(env.current_idx)][column])
    for column in task_error_columns:
        if column not in env.state_index:
            continue
        idx = env.state_index[str(column)]
        safe = str(column).replace("/", "_")
        fields[f"last_obs_{safe}"] = float(env.last_observation[idx])
        fields[f"observed_mask_{safe}"] = float(env.observed_mask[idx])
        fields[f"history_mean_{safe}"] = float(np.mean(env.history[:, idx]))
        fields[f"history_std_{safe}"] = float(np.std(env.history[:, idx]))
        fields[f"mask_history_mean_{safe}"] = float(np.mean(env.mask_history[:, idx]))
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect paired true intervention effects for robust-planner raw "
            "dynamic deviations against the static anchor."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "final_test"), default="train")
    parser.add_argument("--start-count", type=int, default=4)
    parser.add_argument("--window-steps", type=int, default=64)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-branch", type=int, default=8)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--cvar-alpha", type=float, default=0.75)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--replan-interval", type=int, default=4)
    parser.add_argument("--support-top-k", type=int, default=16)
    parser.add_argument("--anchor-improvement-margin", type=float, default=None)
    parser.add_argument("--component-guard-min-task-margin", type=float, default=None)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args()


class PrefixThenAnchorPolicy:
    def __init__(
        self,
        *,
        prefix_mask: np.ndarray,
        anchor_mask: np.ndarray,
        prefix_steps: int,
        name: str,
    ) -> None:
        self.prefix_mask = np.asarray(prefix_mask, dtype=bool).reshape(-1)
        self.anchor_mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
        self.prefix_steps = max(0, int(prefix_steps))
        self.name = str(name)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        del env
        mask = self.prefix_mask if self._step < self.prefix_steps else self.anchor_mask
        self._step += 1
        return mask.copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        return np.where(self.act_mask(env), 1.0, -1.0)


def main() -> None:
    args = parse_args()
    source_run = resolve_path(args.source_run)
    output = resolve_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    run_args = dict(manifest["run_args"])
    helpers = load_v2_helpers()
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    truth_path = source_truth_path(source_run, manifest)
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(resolve_path(str(manifest["sensor_cfg"])))
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    normalization_bounds = tuple(int(value) for value in manifest["normalization_bounds"])
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=normalization_bounds[0],
        end_idx=normalization_bounds[1],
    )
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

    with np.load(resolve_path(str(manifest["teacher_dataset"])), allow_pickle=False) as teacher:
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
        tuple(float(source_step_cfg.candidate_prior_costs[index]) for index in support_indices)
        if source_step_cfg.candidate_prior_costs is not None
        else None
    )
    step_cfg = replace(
        source_step_cfg,
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
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
            "learned_event_probability_columns",
            (),
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
        component_guard_min_task_margin=args.component_guard_min_task_margin,
    )
    task_error_columns = tuple(str(name) for name in run_args["task_error_columns"])
    split_starts = tuple(
        int(value)
        for value in manifest["starts"][str(args.split)]["starts"][: max(1, int(args.start_count))]
    )

    rows: list[dict[str, object]] = []
    for position, start in enumerate(split_starts):
        seed = int(env_cfg.seed) + 70_000 + position
        paired_cfg = replace(env_cfg, seed=seed, episode_len=int(args.window_steps))
        env = WarmupSchedulingEnv(truth, sensors, constraints, paired_cfg, oracle=oracle)
        env.reset(start_idx=int(start))
        cached_action: np.ndarray | None = None
        remaining_hold = 0
        replan_id = 0
        for relative_step in range(int(args.window_steps)):
            if cached_action is None or remaining_hold <= 0:
                decision_snapshot = snapshot_env(env)
                plan = robust_beam_search_plan(env, world_model, candidate_masks, planner_cfg)
                raw_action = (
                    np.asarray(plan.raw_action, dtype=bool).reshape(-1)
                    if plan.raw_action is not None
                    else np.asarray(plan.action, dtype=bool).reshape(-1)
                )
                action = np.asarray(plan.action, dtype=bool).reshape(-1)
                if not np.array_equal(raw_action, anchor_mask):
                    feature_fields = runtime_effect_features(
                        env,
                        raw_action=raw_action,
                        anchor_mask=anchor_mask,
                        event_probability_columns=event_probability_columns,
                        task_error_columns=task_error_columns,
                    )
                    steps_remaining = int(args.window_steps) - int(relative_step)
                    prefix_steps = min(max(1, int(args.replan_interval)), steps_remaining)
                    branch_steps = steps_remaining
                    restore_env(env, decision_snapshot)
                    dynamic_result = continue_policy_rollout(
                        env,
                        PrefixThenAnchorPolicy(
                            prefix_mask=raw_action,
                            anchor_mask=anchor_mask,
                            prefix_steps=prefix_steps,
                            name="raw_dynamic_then_anchor",
                        ),
                        steps=branch_steps,
                    )
                    dynamic_metrics = rollout_objective(
                        dynamic_result,
                        run_args=run_args,
                        state_columns=state_columns,
                    )
                    restore_env(env, decision_snapshot)
                    anchor_result = continue_policy_rollout(
                        env,
                        PrefixThenAnchorPolicy(
                            prefix_mask=anchor_mask,
                            anchor_mask=anchor_mask,
                            prefix_steps=branch_steps,
                            name="anchor_continuation",
                        ),
                        steps=branch_steps,
                    )
                    anchor_metrics = rollout_objective(
                        anchor_result,
                        run_args=run_args,
                        state_columns=state_columns,
                    )
                    row = {
                            "split": str(args.split),
                            "start": int(start),
                            "relative_step": int(relative_step),
                            "replan_id": int(replan_id),
                            "raw_action_bits": mask_bits(raw_action),
                            "selected_action_bits": mask_bits(action),
                            "selected_is_anchor": bool(np.array_equal(action, anchor_mask)),
                            "anchor_guard_applied": bool(plan.anchor_guard_applied),
                            "component_guard_applied": bool(plan.component_guard_applied),
                            "predicted_anchor_minus_raw": (
                                optional_float(plan.anchor_robust_cost)
                                - optional_float(plan.raw_robust_cost)
                                if plan.anchor_robust_cost is not None
                                and plan.raw_robust_cost is not None
                                else float("nan")
                            ),
                            "component_task_margin_mean": float(
                                plan.component_guard_stats.get(
                                    "task_error_margin_mean",
                                    float("nan"),
                                )
                            ),
                            "component_total_margin_mean": float(
                                plan.component_guard_stats.get(
                                    "total_margin_mean",
                                    float("nan"),
                                )
                            ),
                            "prefix_steps": int(prefix_steps),
                            "branch_steps": int(branch_steps),
                            "anchor_objective": float(anchor_metrics["objective"]),
                            "dynamic_objective": float(dynamic_metrics["objective"]),
                            "effect_margin": float(
                                anchor_metrics["objective"] - dynamic_metrics["objective"]
                            ),
                            "anchor_oracle_loss_mean": float(anchor_metrics.get("oracle_loss_mean", float("nan"))),
                            "dynamic_oracle_loss_mean": float(dynamic_metrics.get("oracle_loss_mean", float("nan"))),
                        }
                    row.update(feature_fields)
                    rows.append(row)
                    restore_env(env, decision_snapshot)
                cached_action = action.copy()
                remaining_hold = max(0, int(args.replan_interval) - 1)
                replan_id += 1
            else:
                remaining_hold -= 1
            _, _, done, _ = env.step_mask(cached_action)
            if bool(done):
                break

    table = pd.DataFrame(rows)
    table.to_csv(output / "intervention_effects.csv", index=False)
    summary = {
        "rows": int(len(table)),
        "positive_effect_rows": int(np.sum(table["effect_margin"] > 0.0)) if len(table) else 0,
        "mean_effect_margin": float(table["effect_margin"].mean()) if len(table) else None,
        "q25_effect_margin": float(table["effect_margin"].quantile(0.25)) if len(table) else None,
        "source_run": str(source_run),
        "world_model": str(resolve_path(args.world_model)),
        "split": str(args.split),
        "start_count": int(args.start_count),
    }
    (output / "intervention_effects_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
