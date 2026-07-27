#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
    resolve_archive_path,
)
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, enumerate_action_masks  # noqa: E402
from forecast_cmdp.protocol import (  # noqa: E402
    choose_non_overlapping_starts,
    evaluate_policy_over_starts,
    evaluate_static_candidates,
    final_objective,
    normalized_candidate_costs,
    partition_bounds,
    rich_metrics,
    save_rollout,
    task_focus_metrics,
)
from v2.policies import StaticMaskPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration-only gate: validation-selected static vs MPC teacher, no deployable training."
    )
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--sensor-cfg", required=True)
    parser.add_argument("--oracle-path", required=True)
    parser.add_argument("--oracle-type", choices=["tcn", "linear", "none"], default="tcn")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.30, 0.45, 0.125, 0.125])
    parser.add_argument("--selection", choices=["event_rich", "event_transport_rich", "uniform"], default="uniform")
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--event-column", default="event_flag")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--static-selection-steps", type=int, default=256)
    parser.add_argument("--static-selection-rollouts", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--energy-capacity", type=float, default=70.0)
    parser.add_argument("--initial-energy", type=float, default=70.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-branch", type=int, default=8)
    parser.add_argument("--event-weight-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-lambda-warmup-abort", type=float, default=0.16)
    parser.add_argument("--teacher-lambda-switch", type=float, default=0.002)
    parser.add_argument("--teacher-lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--candidate-prior-weight", type=float, default=0.5)
    parser.add_argument("--candidate-prefilter-top-k", type=int, default=24)
    parser.add_argument("--anchor-regret-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-improvement-margin", type=float, default=0.002)
    parser.add_argument("--objective-mode", choices=["oracle", "task_composite"], default="task_composite")
    parser.add_argument(
        "--task-error-columns",
        nargs="*",
        default=[
            "snow_mass_flux_kg_m2_s",
            "snow_particle_mean_diameter_mm",
            "snow_particle_mean_velocity_ms",
        ],
    )
    parser.add_argument("--task-error-scales", nargs="*", type=float, default=[1.0e-4, 0.2, 5.0])
    parser.add_argument("--task-error-weight", type=float, default=0.2)
    parser.add_argument("--task-error-event-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.dry_run):
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_v2_helpers()
    truth_path = resolve_archive_path(args.truth_csv)
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(args.sensor_cfg)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    bounds = partition_bounds(len(truth), tuple(float(x) for x in args.split_ratios))
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=int(bounds["oracle_pretrain"][0]),
        end_idx=int(bounds["oracle_pretrain"][1]),
    )
    oracle = load_archived_oracle(args.oracle_path, oracle_type=str(args.oracle_type), device=str(args.oracle_device))
    constraints = make_constraints(
        max_active=int(args.max_active),
        budget=float(args.budget),
        startup_peak_budget=float(args.startup_peak_budget),
    )
    validation_cfg = make_common_env_config(
        args,
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        episode_len=int(args.static_selection_steps),
        seed=int(args.seed) + 10_000,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    eval_cfg = make_common_env_config(
        args,
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 20_000,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    starts = {
        "validation": choose_non_overlapping_starts(
            truth,
            bounds=bounds["validation"],
            window_steps=int(args.static_selection_steps),
            horizon=int(args.horizon),
            count=int(args.static_selection_rollouts),
            selection=str(args.selection),
            stride=int(args.selection_stride),
            event_column=str(args.event_column),
            seed=int(args.seed) + 202,
        ),
        "final_test": choose_non_overlapping_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=int(args.horizon),
            count=int(args.eval_rollouts),
            selection=str(args.selection),
            stride=int(args.selection_stride),
            event_column=str(args.event_column),
            seed=int(args.seed) + 303,
        ),
    }
    candidate_masks = enumerate_action_masks(len(sensors), max_active=int(args.max_active))
    validation_static_table = evaluate_static_candidates(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=validation_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        steps=int(args.static_selection_steps),
        start_indices=starts["validation"].starts,
        objective_mode=str(args.objective_mode),
        task_error_columns=tuple(str(x) for x in args.task_error_columns),
        task_error_scales=tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else None,
        task_error_event_only=bool(args.task_error_event_only),
        task_error_weight=float(args.task_error_weight),
    )
    validation_static_table.to_csv(out_dir / "validation_static_candidates.csv", index=False)
    selected_static = validation_static_table.iloc[0]
    selected_static_idx = int(selected_static["action_idx"])
    selected_static_mask = tuple(bool(x) for x in candidate_masks[selected_static_idx])
    candidate_prior_costs = normalized_candidate_costs(validation_static_table, n_actions=int(candidate_masks.shape[0]))
    teacher_cfg = MpcTeacherConfig(
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
        event_weight_alpha=float(args.event_weight_alpha),
        lambda_warmup_abort=float(args.teacher_lambda_warmup_abort),
        lambda_switch=float(args.teacher_lambda_switch),
        lambda_energy_deficit=float(args.teacher_lambda_energy_deficit),
        candidate_prior_weight=float(args.candidate_prior_weight),
        candidate_prior_costs=candidate_prior_costs if float(args.candidate_prior_weight) > 0.0 else None,
        candidate_prefilter_top_k=int(args.candidate_prefilter_top_k),
        anchor_mask=selected_static_mask,
        anchor_regret_guard=bool(args.anchor_regret_guard),
        anchor_improvement_margin=float(args.anchor_improvement_margin),
        task_error_weight=float(args.task_error_weight) if str(args.objective_mode) == "task_composite" else 0.0,
        task_error_columns=tuple(str(x) for x in args.task_error_columns),
        task_error_scales=tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else None,
        task_error_event_only=bool(args.task_error_event_only),
    )
    policies = [
        StaticMaskPolicy(mask=selected_static_mask, name="validation_selected_static"),
        MpcTeacherPolicy(candidate_masks=candidate_masks, cfg=teacher_cfg, name="mpc_teacher"),
    ]
    rows: list[dict[str, object]] = []
    rollout_summaries: list[dict[str, object]] = []
    for policy in policies:
        result, simple_metrics = evaluate_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(args.eval_steps),
            start_indices=starts["final_test"].starts,
        )
        metrics = rich_metrics(
            result,
            sensor_ids=sensor_ids,
            state_columns=state_columns,
            per_step_budget=float(args.budget),
            startup_peak_budget=float(args.startup_peak_budget),
        )
        metrics.update(
            task_focus_metrics(
                result,
                state_columns=state_columns,
                task_error_columns=tuple(str(x) for x in args.task_error_columns),
                task_error_scales=tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else None,
                event_only=bool(args.task_error_event_only),
            )
        )
        metrics.update({f"rollout_{key}": value for key, value in simple_metrics.items() if key not in metrics})
        metrics["objective_loss_mean"] = final_objective(
            metrics,
            mode=str(args.objective_mode),
            task_error_weight=float(args.task_error_weight),
        )
        rows.append(metrics)
        save_rollout(
            out_dir / f"rollout_{policy.name}.npz",
            result,
            sensor_ids=sensor_ids,
            state_columns=state_columns,
        )
        rollout_summaries.append(summarize_rollout(policy.name, result, sensor_ids=sensor_ids))
    metrics_df = pd.DataFrame(rows).sort_values(["objective_loss_mean", "power_mean", "warmup_abort_count"])
    metrics_df.to_csv(out_dir / "metrics_final.csv", index=False)
    summary = build_summary(args, metrics_df, selected_static, selected_static_mask, starts, rollout_summaries)
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "role": "v1_static_teacher_calibration_gate",
                "run_args": vars(args),
                "truth_csv": str(truth_path),
                "sensor_cfg": str(resolve_archive_path(args.sensor_cfg)),
                "bounds": {key: [int(x) for x in value] for key, value in bounds.items()},
                "starts": {
                    key: {"starts": [int(x) for x in value.starts], "diagnostics": value.diagnostics}
                    for key, value in starts.items()
                },
                "candidate_count": int(candidate_masks.shape[0]),
                "selected_static": {
                    "action_idx": selected_static_idx,
                    "mask": [int(x) for x in selected_static_mask],
                    "sensor_ids": str(selected_static.get("sensor_ids", "")),
                    "validation_objective": float(selected_static["objective_loss_mean"]),
                },
                "teacher_cfg": asdict(teacher_cfg),
                "calibration_summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def make_common_env_config(
    args: argparse.Namespace,
    *,
    state_columns: tuple[str, ...],
    reward_target_columns: tuple[str, ...],
    episode_len: int,
    seed: int,
    norm_mean: tuple[float, ...] | None,
    norm_std: tuple[float, ...] | None,
):
    return make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(args.lookback),
        episode_len=int(episode_len),
        seed=int(seed),
        freq_s=int(args.freq_s),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
        event_reward_multiplier=float(args.event_reward_multiplier),
        energy_account=True,
        energy_capacity=float(args.energy_capacity),
        initial_energy=float(args.initial_energy),
        harvest_per_step=float(args.harvest_per_step),
        reserve_energy=float(args.reserve_energy),
        lambda_energy_deficit=float(args.lambda_energy_deficit),
        soc_soft_penalty_buffer=0.0,
        lambda_soc_soft_penalty=0.0,
    )


def build_summary(
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    selected_static: pd.Series,
    selected_static_mask: tuple[bool, ...],
    starts: dict[str, object],
    rollout_summaries: list[dict[str, object]],
) -> dict[str, object]:
    static = metrics_df.loc[metrics_df["policy"] == "validation_selected_static"].iloc[0]
    teacher = metrics_df.loc[metrics_df["policy"] == "mpc_teacher"].iloc[0]
    static_obj = float(static["objective_loss_mean"])
    teacher_obj = float(teacher["objective_loss_mean"])
    margin = static_obj - teacher_obj
    return {
        "seed": int(args.seed),
        "sensor_cfg": str(args.sensor_cfg),
        "selection": str(args.selection),
        "budget": float(args.budget),
        "startup_peak_budget": float(args.startup_peak_budget),
        "energy_capacity": float(args.energy_capacity),
        "initial_energy": float(args.initial_energy),
        "harvest_per_step": float(args.harvest_per_step),
        "reserve_energy": float(args.reserve_energy),
        "selected_static_action_idx": int(selected_static["action_idx"]),
        "selected_static_mask": [int(x) for x in selected_static_mask],
        "selected_static_sensor_ids": str(selected_static.get("sensor_ids", "")),
        "static_objective": static_obj,
        "teacher_objective": teacher_obj,
        "teacher_margin": margin,
        "teacher_beats_static": bool(margin > 0.0),
        "static_power_mean": float(static["power_mean"]),
        "teacher_power_mean": float(teacher["power_mean"]),
        "static_soc_mean": row_value(static, "soc_mean"),
        "teacher_soc_mean": row_value(teacher, "soc_mean"),
        "validation_event_rate_mean": float(starts["validation"].diagnostics.get("selected_event_rate_mean", np.nan)),
        "final_event_rate_mean": float(starts["final_test"].diagnostics.get("selected_event_rate_mean", np.nan)),
        "rollout_summaries": rollout_summaries,
    }


def summarize_rollout(policy: str, result: object, *, sensor_ids: tuple[str, ...]) -> dict[str, object]:
    masks = np.asarray(result.selected_masks, dtype=int)
    powers = np.asarray(result.powers, dtype=float)
    soc = np.asarray(result.soc, dtype=float)
    event_flags = np.asarray(result.event_flags, dtype=float)
    unique, counts = np.unique(masks, axis=0, return_counts=True)
    order = np.argsort(-counts)[:8]
    top_masks = []
    for idx in order:
        mask = unique[int(idx)].astype(int)
        names = [sensor_ids[pos] for pos, active in enumerate(mask) if int(active)]
        top_masks.append({"count": int(counts[int(idx)]), "sensor_ids": "|".join(names)})
    return {
        "policy": str(policy),
        "event_rate": finite_mean(event_flags),
        "power_mean": finite_mean(powers),
        "soc_mean": finite_mean(soc),
        "soc_min": finite_min(soc),
        "soc_final": finite_last(soc),
        "unique_masks": int(unique.shape[0]),
        "top_masks": top_masks,
    }


def row_value(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    value = float(row[key])
    return value if np.isfinite(value) else None


def finite_mean(values: np.ndarray) -> float:
    arr = values[np.isfinite(values)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_min(values: np.ndarray) -> float:
    arr = values[np.isfinite(values)]
    return float(np.min(arr)) if arr.size else float("nan")


def finite_last(values: np.ndarray) -> float:
    arr = values[np.isfinite(values)]
    return float(arr[-1]) if arr.size else float("nan")


if __name__ == "__main__":
    main()
