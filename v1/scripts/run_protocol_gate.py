#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.archived_v2 import (
    load_archived_oracle,
    load_archived_sensor_specs,
    load_custom_ppo_policy,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
    resolve_archive_path,
)
from forecast_cmdp.dataset import collect_dagger_dataset, collect_teacher_dataset, concat_teacher_datasets
from forecast_cmdp.features import ForecastContextConfig
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, enumerate_action_masks
from forecast_cmdp.policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    save_bc_policy_checkpoint,
    train_bc_classifier,
)
from forecast_cmdp.protocol import (
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
from v2.policies import FullOpenUnconstrainedScorePolicy, default_policies, StaticMaskPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a v1 split-protocol gate for MPC teacher and BC policy.")
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--oracle-path", default=None)
    parser.add_argument("--oracle-type", choices=["tcn", "linear", "none"], default="tcn")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument(
        "--oracle-target-weight-mode",
        choices=["checkpoint", "event_transport", "primary_weather"],
        default="checkpoint",
        help="Override only the frozen oracle loss weights; predictor weights/checkpoint remain frozen.",
    )
    parser.add_argument("--custom-ppo-checkpoint", default=None)
    parser.add_argument("--out-dir", default="v1/artifacts/protocol_gate_seed41")

    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.30, 0.45, 0.125, 0.125])
    parser.add_argument("--selection", choices=["event_rich", "uniform"], default="event_rich")
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--event-column", default="event_flag")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--objective-mode", choices=["oracle", "task_composite"], default="oracle")
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
    parser.add_argument("--task-error-weight", type=float, default=0.0)
    parser.add_argument("--task-error-event-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--static-selection-steps", type=int, default=256)
    parser.add_argument("--static-selection-rollouts", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--eval-rollouts", type=int, default=4)

    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--energy-account", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--normalization-start-idx", type=int, default=None)
    parser.add_argument("--normalization-end-idx", type=int, default=None)

    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-branch", type=int, default=16)
    parser.add_argument("--event-weight-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-lambda-warmup-abort", type=float, default=0.16)
    parser.add_argument("--teacher-lambda-switch", type=float, default=0.002)
    parser.add_argument("--teacher-lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--saturated-loss-threshold", type=float, default=9.999)
    parser.add_argument("--saturated-coverage-bonus", type=float, default=0.25)
    parser.add_argument("--candidate-prior-weight", type=float, default=0.0)
    parser.add_argument("--candidate-prefilter-top-k", type=int, default=0)
    parser.add_argument(
        "--teacher-anchor-source",
        choices=["none", "train_best", "validation_best", "action_idx"],
        default="none",
    )
    parser.add_argument("--teacher-anchor-action-idx", type=int, default=None)
    parser.add_argument("--anchor-regret-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-improvement-margin", type=float, default=0.0)

    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-hidden-dim", type=int, default=128)
    parser.add_argument("--bc-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--bc-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bc-device", default="auto")
    parser.add_argument("--dagger-iters", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=None)
    parser.add_argument("--bc-fallback-source", choices=["none", "validation_static"], default="none")
    parser.add_argument(
        "--bc-logit-margin-grid",
        nargs="*",
        type=float,
        default=[-1000000000.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 1000000000.0],
    )
    parser.add_argument("--include-rule-baselines", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("loading archived helpers and inputs")
    helpers = load_v2_helpers()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_path = resolve_archive_path(args.truth_csv)
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(args.sensor_cfg)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    bounds = partition_bounds(len(truth), tuple(float(x) for x in args.split_ratios))

    norm_start = args.normalization_start_idx
    norm_end = args.normalization_end_idx
    if norm_start is None and norm_end is None:
        norm_start, norm_end = bounds["oracle_pretrain"]
    norm_mean, norm_std = normalization_stats(truth, state_columns, start_idx=norm_start, end_idx=norm_end)

    oracle = load_archived_oracle(args.oracle_path, oracle_type=str(args.oracle_type), device=str(args.oracle_device))
    oracle_target_weights = apply_oracle_target_weight_mode(
        oracle,
        reward_target_columns=reward_target_columns,
        mode=str(args.oracle_target_weight_mode),
    )
    log(f"loaded truth rows={len(truth)} sensors={len(sensors)} oracle_type={args.oracle_type}")
    constraints = make_constraints(
        max_active=int(args.max_active),
        budget=float(args.budget),
        startup_peak_budget=float(args.startup_peak_budget),
    )
    train_cfg = make_common_env_config(
        args,
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        episode_len=int(args.train_steps),
        seed=int(args.seed),
        norm_mean=norm_mean,
        norm_std=norm_std,
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
        "train": choose_non_overlapping_starts(
            truth,
            bounds=bounds["rl_train"],
            window_steps=int(args.train_steps),
            horizon=int(args.horizon),
            count=int(args.train_rollouts),
            selection=str(args.selection),
            stride=int(args.selection_stride),
            event_column=str(args.event_column),
            seed=int(args.seed) + 101,
        ),
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
    log(
        "selected starts: "
        f"train={list(starts['train'].starts)} "
        f"validation={list(starts['validation'].starts)} "
        f"final={list(starts['final_test'].starts)}"
    )

    candidate_masks = enumerate_action_masks(len(sensors), max_active=int(args.max_active))
    log(f"candidate masks={candidate_masks.shape[0]}")
    log("computing train-split static candidate prior")
    train_static_table = evaluate_static_candidates(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=train_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        steps=int(args.train_steps),
        start_indices=starts["train"].starts,
        objective_mode=str(args.objective_mode),
        task_error_columns=tuple(str(x) for x in args.task_error_columns),
        task_error_scales=tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else None,
        task_error_event_only=bool(args.task_error_event_only),
        task_error_weight=float(args.task_error_weight),
    )
    train_static_table.to_csv(out_dir / "train_static_candidates.csv", index=False)
    candidate_prior_costs = normalized_candidate_costs(train_static_table, n_actions=int(candidate_masks.shape[0]))
    log(
        "train prior best: "
        f"action={int(train_static_table.iloc[0]['action_idx'])} "
        f"objective={float(train_static_table.iloc[0]['objective_loss_mean']):.6f}"
    )
    log("selecting validation static candidate")
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

    teacher_anchor_idx, teacher_anchor_mask, teacher_anchor_ids = select_teacher_anchor(
        args,
        candidate_masks=candidate_masks,
        train_static_table=train_static_table,
        validation_static_table=validation_static_table,
    )
    if teacher_anchor_mask is not None:
        log(
            "teacher anchor: "
            f"source={args.teacher_anchor_source} action={teacher_anchor_idx} ids={teacher_anchor_ids}"
        )
    teacher_cfg = MpcTeacherConfig(
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
        event_weight_alpha=float(args.event_weight_alpha),
        lambda_warmup_abort=float(args.teacher_lambda_warmup_abort),
        lambda_switch=float(args.teacher_lambda_switch),
        lambda_energy_deficit=float(args.teacher_lambda_energy_deficit),
        saturated_loss_threshold=float(args.saturated_loss_threshold),
        saturated_coverage_bonus=float(args.saturated_coverage_bonus),
        candidate_prior_weight=float(args.candidate_prior_weight),
        candidate_prior_costs=candidate_prior_costs if float(args.candidate_prior_weight) > 0.0 else None,
        candidate_prefilter_top_k=int(args.candidate_prefilter_top_k),
        anchor_mask=teacher_anchor_mask,
        anchor_regret_guard=bool(args.anchor_regret_guard),
        anchor_improvement_margin=float(args.anchor_improvement_margin),
        task_error_weight=float(args.task_error_weight) if str(args.objective_mode) == "task_composite" else 0.0,
        task_error_columns=tuple(str(x) for x in args.task_error_columns),
        task_error_scales=tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else None,
        task_error_event_only=bool(args.task_error_event_only),
    )
    forecast_cfg = ForecastContextConfig(horizon=int(args.horizon), event_column=str(args.event_column), truth_future=False)

    train_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
    log("collecting MPC teacher dataset")
    teacher_dataset = collect_teacher_dataset(
        train_env,
        candidate_masks,
        start_indices=starts["train"].starts,
        steps_per_start=int(args.train_steps),
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    teacher_dataset_path = out_dir / "teacher_dataset.npz"
    teacher_dataset.save_npz(str(teacher_dataset_path))
    log(f"teacher dataset saved: samples={teacher_dataset.features.shape[0]} path={teacher_dataset_path}")

    bc_cfg = BCTrainingConfig(
        hidden_dim=int(args.bc_hidden_dim),
        epochs=int(args.bc_epochs),
        batch_size=int(args.bc_batch_size),
        learning_rate=float(args.bc_learning_rate),
        weight_decay=float(args.bc_weight_decay),
        seed=int(args.seed),
        device=str(args.bc_device),
    )
    bc_model, bc_history = train_bc_classifier(
        teacher_dataset.features,
        teacher_dataset.labels,
        teacher_dataset.action_masks,
        cfg=bc_cfg,
    )
    log(f"BC training complete: final_accuracy={bc_history['accuracy'][-1] if bc_history.get('accuracy') else float('nan')}")
    for dagger_iter in range(max(0, int(args.dagger_iters))):
        dagger_policy = ForecastAwareBCPolicy(
            model=bc_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            device=str(args.bc_device),
        )
        dagger_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        log(f"collecting DAgger dataset iter={dagger_iter + 1}")
        dagger_dataset = collect_dagger_dataset(
            dagger_env,
            candidate_masks,
            policy=dagger_policy,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.dagger_steps or args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
        )
        teacher_dataset = concat_teacher_datasets([teacher_dataset, dagger_dataset])
        teacher_dataset.save_npz(str(teacher_dataset_path))
        bc_model, bc_history = train_bc_classifier(
            teacher_dataset.features,
            teacher_dataset.labels,
            teacher_dataset.action_masks,
            cfg=bc_cfg,
        )
        log(
            "DAgger BC training complete: "
            f"iter={dagger_iter + 1} samples={teacher_dataset.features.shape[0]} "
            f"final_accuracy={bc_history['accuracy'][-1] if bc_history.get('accuracy') else float('nan')}"
        )
    bc_checkpoint = out_dir / "forecast_aware_bc.pt"
    save_bc_policy_checkpoint(
        bc_checkpoint,
        model=bc_model,
        candidate_masks=candidate_masks,
        forecast_cfg=forecast_cfg,
        train_cfg=bc_cfg,
        history=bc_history,
    )
    bc_fallback_margin = None
    if str(args.bc_fallback_source) == "validation_static":
        bc_fallback_margin, bc_validation_objective = calibrate_bc_fallback_margin(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=bc_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            fallback_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "BC fallback calibration: "
            f"margin={bc_fallback_margin} validation_objective={bc_validation_objective:.6f}"
        )
    policies = [
        StaticMaskPolicy(mask=selected_static_mask, name="validation_selected_static"),
        MpcTeacherPolicy(candidate_masks=candidate_masks, cfg=teacher_cfg, name="mpc_teacher"),
        ForecastAwareBCPolicy(
            model=bc_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            device=str(args.bc_device),
            fallback_mask=selected_static_mask if str(args.bc_fallback_source) == "validation_static" else None,
            min_logit_margin=bc_fallback_margin,
        ),
    ]
    if args.custom_ppo_checkpoint:
        log(f"loading custom PPO checkpoint: {args.custom_ppo_checkpoint}")
        policies.append(
            load_custom_ppo_policy(
                args.custom_ppo_checkpoint,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                env_cfg=eval_cfg,
                oracle=oracle,
                device=str(args.bc_device),
            )
        )
    if bool(args.include_rule_baselines):
        policies.append(FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors), name="full_open_unconstrained"))
        policies.extend(default_policies(len(sensors), seed=int(args.seed) + 404))

    rows: list[dict[str, object]] = []
    for policy in policies:
        log(f"evaluating final policy={policy.name}")
        policy_constraints = constraints
        if str(policy.name) == "full_open_unconstrained":
            policy_constraints = make_constraints(max_active=None, budget=None, startup_peak_budget=None)
        result, simple_metrics = evaluate_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=policy_constraints,
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
            per_step_budget=float(args.budget) if str(policy.name) != "full_open_unconstrained" else None,
            startup_peak_budget=float(args.startup_peak_budget) if str(policy.name) != "full_open_unconstrained" else None,
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

    metrics_df = pd.DataFrame(rows).sort_values(["objective_loss_mean", "power_mean", "warmup_abort_count"])
    metrics_df.to_csv(out_dir / "metrics_final.csv", index=False)

    static_final = metrics_df.loc[metrics_df["policy"] == "validation_selected_static"].iloc[0]
    static_objective = float(
        final_objective(
            static_final.to_dict(),
            mode=str(args.objective_mode),
            task_error_weight=float(args.task_error_weight),
        )
    )
    teacher_rows = metrics_df.loc[metrics_df["policy"] == "mpc_teacher"]
    teacher = teacher_rows.iloc[0] if not teacher_rows.empty else None
    deployable = metrics_df[metrics_df["policy"].isin(["forecast_aware_bc"])]
    best_deployable = deployable.sort_values("objective_loss_mean").iloc[0] if not deployable.empty else None
    gate_summary = {
        "objective_metric": str(args.objective_mode),
        "static_policy": "validation_selected_static",
        "validation_selected_static_action_idx": selected_static_idx,
        "validation_selected_static_mask": [int(x) for x in selected_static_mask],
        "validation_selected_static_objective": static_objective,
        "teacher_reference_policy": "mpc_teacher" if teacher is not None else None,
        "teacher_reference_objective": float(teacher["objective_loss_mean"]) if teacher is not None else None,
        "teacher_beats_static": bool(teacher is not None and float(teacher["objective_loss_mean"]) < static_objective),
        "best_deployable_policy": str(best_deployable["policy"]) if best_deployable is not None else None,
        "best_deployable_objective": float(best_deployable["objective_loss_mean"]) if best_deployable is not None else None,
        "gate_pass": bool(
            best_deployable is not None
            and float(best_deployable["objective_loss_mean"]) < static_objective
        ),
    }
    manifest = {
        "role": "v1_forecast_cmdp_protocol_gate",
        "truth_csv": str(truth_path),
        "sensor_cfg": str(resolve_archive_path(args.sensor_cfg)),
        "oracle_path": str(resolve_archive_path(args.oracle_path)) if args.oracle_path else None,
        "oracle_type": str(args.oracle_type),
        "oracle_target_weight_mode": str(args.oracle_target_weight_mode),
        "oracle_target_weights": oracle_target_weights,
        "objective_mode": str(args.objective_mode),
        "task_error_columns": [str(x) for x in args.task_error_columns],
        "task_error_scales": [float(x) for x in args.task_error_scales] if args.task_error_scales else None,
        "task_error_weight": float(args.task_error_weight),
        "task_error_event_only": bool(args.task_error_event_only),
        "bounds": {key: [int(x) for x in value] for key, value in bounds.items()},
        "normalization_bounds": [int(norm_start), int(norm_end)],
        "starts": {
            key: {"starts": [int(x) for x in value.starts], "diagnostics": value.diagnostics}
            for key, value in starts.items()
        },
        "candidate_count": int(candidate_masks.shape[0]),
        "teacher_dataset": str(teacher_dataset_path),
        "bc_checkpoint": str(bc_checkpoint),
        "selected_static": {
            "action_idx": selected_static_idx,
            "mask": [int(x) for x in selected_static_mask],
            "validation_objective": float(selected_static["objective_loss_mean"]),
            "sensor_ids": str(selected_static.get("sensor_ids", "")),
        },
        "teacher_anchor": {
            "source": str(args.teacher_anchor_source),
            "action_idx": teacher_anchor_idx,
            "mask": [int(x) for x in teacher_anchor_mask] if teacher_anchor_mask is not None else None,
            "sensor_ids": teacher_anchor_ids,
            "regret_guard": bool(args.anchor_regret_guard),
            "improvement_margin": float(args.anchor_improvement_margin),
        },
        "teacher_cfg": teacher_cfg.__dict__,
        "forecast_cfg": forecast_cfg.__dict__,
        "bc_cfg": bc_cfg.__dict__,
        "dagger": {
            "iters": int(args.dagger_iters),
            "steps": int(args.dagger_steps or args.train_steps),
            "final_samples": int(teacher_dataset.features.shape[0]),
        },
        "bc_fallback": {
            "source": str(args.bc_fallback_source),
            "logit_margin": bc_fallback_margin,
            "grid": [float(x) for x in args.bc_logit_margin_grid],
        },
        "gate_summary": gate_summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "gate_summary.json").write_text(json.dumps(gate_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("protocol gate complete")
    print(json.dumps({"out_dir": str(out_dir), **gate_summary}, ensure_ascii=False))


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
        energy_account=bool(args.energy_account),
        energy_capacity=float(args.energy_capacity),
        initial_energy=float(args.initial_energy),
        harvest_per_step=float(args.harvest_per_step),
        reserve_energy=float(args.reserve_energy),
        lambda_energy_deficit=float(args.lambda_energy_deficit),
        soc_soft_penalty_buffer=float(args.soc_soft_penalty_buffer),
        lambda_soc_soft_penalty=float(args.lambda_soc_soft_penalty),
    )


def build_env_for_dataset(truth, sensors, constraints, cfg, oracle):
    from v2.env import WarmupSchedulingEnv

    return WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)


def apply_oracle_target_weight_mode(
    oracle: object | None,
    *,
    reward_target_columns: tuple[str, ...],
    mode: str,
) -> list[float] | None:
    selected = str(mode)
    if oracle is None or selected == "checkpoint":
        return None
    if selected == "event_transport":
        by_name = {
            "air_temperature_c": 0.20,
            "snow_surface_temperature_c": 0.20,
            "wind_speed_ms": 1.00,
            "wind_dir_sin": 0.20,
            "wind_dir_cos": 0.20,
            "solar_radiation_wm2": 0.10,
            "snow_mass_flux_kg_m2_s": 12.0,
            "snow_particle_mean_diameter_mm": 6.0,
            "snow_particle_mean_velocity_ms": 6.0,
        }
    elif selected == "primary_weather":
        by_name = {
            "air_temperature_c": 2.0,
            "snow_surface_temperature_c": 2.0,
            "wind_speed_ms": 2.0,
            "wind_dir_sin": 0.5,
            "wind_dir_cos": 0.5,
            "solar_radiation_wm2": 0.1,
            "snow_mass_flux_kg_m2_s": 0.5,
            "snow_particle_mean_diameter_mm": 0.2,
            "snow_particle_mean_velocity_ms": 0.2,
        }
    else:
        raise ValueError(f"Unsupported oracle target weight mode: {mode}")
    weights = [float(by_name.get(str(name), 1.0)) for name in reward_target_columns]
    cfg = getattr(oracle, "cfg", None)
    if cfg is None:
        raise ValueError("Oracle object has no cfg; cannot override target weights")
    oracle.cfg = replace(cfg, target_weights=tuple(weights))
    log(f"oracle target weights overridden: mode={selected} weights={weights}")
    return weights


def select_teacher_anchor(
    args: argparse.Namespace,
    *,
    candidate_masks: np.ndarray,
    train_static_table: pd.DataFrame,
    validation_static_table: pd.DataFrame,
) -> tuple[int | None, tuple[bool, ...] | None, str | None]:
    source = str(args.teacher_anchor_source)
    if source == "none":
        return None, None, None
    if source == "action_idx":
        if args.teacher_anchor_action_idx is None:
            raise ValueError("--teacher-anchor-action-idx is required when --teacher-anchor-source action_idx")
        action_idx = int(args.teacher_anchor_action_idx)
        if action_idx < 0 or action_idx >= int(candidate_masks.shape[0]):
            raise ValueError(f"teacher anchor action_idx out of range: {action_idx}")
        row = None
        ids = ""
    elif source == "train_best":
        row = train_static_table.iloc[0]
        action_idx = int(row["action_idx"])
        ids = str(row.get("sensor_ids", ""))
    elif source == "validation_best":
        row = validation_static_table.iloc[0]
        action_idx = int(row["action_idx"])
        ids = str(row.get("sensor_ids", ""))
    else:
        raise ValueError(f"Unsupported teacher anchor source: {source}")
    mask = tuple(bool(x) for x in np.asarray(candidate_masks[action_idx], dtype=bool))
    if source == "action_idx":
        ids = f"action_idx_{action_idx}"
    return int(action_idx), mask, ids


def calibrate_bc_fallback_margin(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    model: object,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    fallback_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float]:
    rows: list[tuple[float, float, float, int]] = []
    grid = [float(x) for x in args.bc_logit_margin_grid] or [-1.0e9]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, margin in enumerate(grid):
        policy = ForecastAwareBCPolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            device=str(args.bc_device),
            fallback_mask=fallback_mask,
            min_logit_margin=float(margin),
            name=f"forecast_aware_bc_calib_{idx}",
        )
        result, _ = evaluate_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=policy,
            steps=int(args.static_selection_steps),
            start_indices=starts,
            seed_offset=80_000 + idx * 101,
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
        objective = final_objective(
            metrics,
            mode=str(args.objective_mode),
            task_error_weight=float(args.task_error_weight),
        )
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(margin), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_margin, _ = rows[0]
    return float(best_margin), float(best_objective)


def log(message: str) -> None:
    print(f"[run_protocol_gate] {message}", flush=True)


if __name__ == "__main__":
    main()
