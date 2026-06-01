#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
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
from forecast_cmdp.cost_policy import (
    ActionCostTrainingConfig,
    ForecastAwareAdvantageResidualPolicy,
    ForecastAwareCostPolicy,
    ForecastAwareEnsembleValuePolicy,
    ForecastAwareRolloutValuePolicy,
    ForecastAwareValueResidualPolicy,
    collect_anchor_advantage_dataset,
    collect_action_cost_dataset,
    collect_feature_transition_dataset,
    train_anchor_advantage_model,
    train_action_cost_ensemble,
    train_action_cost_model,
    train_feature_transition_model,
)
from forecast_cmdp.dataset import collect_dagger_dataset, collect_teacher_dataset, concat_teacher_datasets
from forecast_cmdp.event_forecaster import (
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    select_event_forecast_columns,
    train_event_forecaster,
)
from forecast_cmdp.features import ForecastContextConfig
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, enumerate_action_masks
from forecast_cmdp.policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    ForecastAwareContextualDutyPolicy,
    ForecastAwareCyclePolicy,
    ForecastAwareEventSupportCyclePolicy,
    ForecastAwareEventThresholdPolicy,
    ForecastAwareKNNPolicy,
    ForecastAwareMaskBCPolicy,
    ForecastAwareResidualBCPolicy,
    ForecastAwareTeacherRatePolicy,
    save_bc_policy_checkpoint,
    train_bc_classifier,
    train_deviation_gate,
    train_mask_bc,
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
from forecast_cmdp.selection import DEPLOYABLE_SELECTION_CRITERIA, choose_deployable_validation_row
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
    parser.add_argument("--forecast-truth-future", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learned-event-forecast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-forecast-lookback", type=int, default=8)
    parser.add_argument("--event-forecast-hidden-dim", type=int, default=128)
    parser.add_argument("--event-forecast-epochs", type=int, default=40)
    parser.add_argument("--event-forecast-batch-size", type=int, default=256)
    parser.add_argument("--event-forecast-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--event-forecast-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--event-forecast-feature-columns", nargs="*", default=[])
    parser.add_argument("--event-forecast-probability-prefix", default="learned_event_p")
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
    parser.add_argument("--bc-preserve-warming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-action-support-top-k", type=int, default=0)
    parser.add_argument("--bc-action-support-min-count", type=int, default=0)
    parser.add_argument("--bc-action-support-grid", nargs="*", type=int, default=[])
    parser.add_argument("--include-bc-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-residual-bc-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-bc-support-top-k", type=int, default=5)
    parser.add_argument("--residual-deviation-threshold-grid", nargs="*", type=float, default=[0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    parser.add_argument("--include-mask-bc-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-bc-required-rate", type=float, default=0.95)
    parser.add_argument("--mask-bc-anchor-bias", type=float, default=0.0)
    parser.add_argument("--include-event-threshold-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-threshold-support-top-k", type=int, default=4)
    parser.add_argument(
        "--event-threshold-grid",
        nargs="*",
        type=float,
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
    )
    parser.add_argument("--event-threshold-aggregation-grid", nargs="*", default=["max", "mean", "first"])
    parser.add_argument("--include-event-support-cycle-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-support-cycle-top-k", type=int, default=6)
    parser.add_argument(
        "--event-support-cycle-grid",
        nargs="*",
        type=float,
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
    )
    parser.add_argument("--event-support-cycle-aggregation-grid", nargs="*", default=["max", "mean", "first"])
    parser.add_argument("--event-support-cycle-period-grid", nargs="*", type=int, default=[1, 2, 4])
    parser.add_argument("--event-support-cycle-selection-grid", nargs="*", default=["time_cycle", "freshness"])
    parser.add_argument("--include-teacher-rate-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-rate-support-top-k", type=int, default=12)
    parser.add_argument("--teacher-rate-blend-grid", nargs="*", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--teacher-rate-freshness-grid", nargs="*", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--teacher-rate-power-grid", nargs="*", type=float, default=[0.0, 0.03, 0.08])
    parser.add_argument("--include-contextual-duty-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contextual-duty-support-top-k", type=int, default=16)
    parser.add_argument("--contextual-duty-blend-grid", nargs="*", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--contextual-duty-deficit-grid", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--contextual-duty-freshness-grid", nargs="*", type=float, default=[0.0, 0.1, 0.25])
    parser.add_argument("--contextual-duty-power-grid", nargs="*", type=float, default=[0.0, 0.03, 0.08])
    parser.add_argument("--include-teacher-cycle-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-cycle-max-lookahead", type=int, default=32)
    parser.add_argument("--include-knn-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--include-cost-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-value-residual-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-ensemble-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-advantage-residual-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-rollout-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--value-residual-support-top-k", type=int, default=5)
    parser.add_argument(
        "--value-residual-advantage-grid",
        nargs="*",
        type=float,
        default=[-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument("--ensemble-value-support-top-k", type=int, default=8)
    parser.add_argument("--ensemble-value-size", type=int, default=5)
    parser.add_argument("--ensemble-value-bootstrap-fraction", type=float, default=0.85)
    parser.add_argument("--ensemble-value-beta-grid", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--ensemble-value-advantage-grid", nargs="*", type=float, default=[-0.5, -0.2, 0.0, 0.1, 0.2, 0.5])
    parser.add_argument("--advantage-residual-support-top-k", type=int, default=6)
    parser.add_argument("--advantage-residual-support-grid", nargs="*", type=int, default=[])
    parser.add_argument(
        "--advantage-residual-grid",
        nargs="*",
        type=float,
        default=[-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0],
    )
    parser.add_argument("--rollout-value-support-top-k", type=int, default=8)
    parser.add_argument("--rollout-value-depth", type=int, default=2)
    parser.add_argument("--rollout-value-beam-width", type=int, default=4)
    parser.add_argument("--rollout-value-max-branch", type=int, default=6)
    parser.add_argument("--rollout-value-discount", type=float, default=0.95)
    parser.add_argument(
        "--rollout-value-advantage-grid",
        nargs="*",
        type=float,
        default=[-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument("--cost-epochs", type=int, default=50)
    parser.add_argument("--cost-hidden-dim", type=int, default=256)
    parser.add_argument("--deployable-selection", choices=["all_final", "validation"], default="all_final")
    parser.add_argument(
        "--deployable-selection-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="mean_objective",
    )
    parser.add_argument("--deployable-selection-min-mean-margin", type=float, default=0.0)
    parser.add_argument("--deployable-selection-min-start-margin", type=float, default=-1.0e9)
    parser.add_argument("--deployable-selection-max-negative-starts", type=int, default=1_000_000)
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
    learned_event_probability_columns: tuple[str, ...] = ()
    event_forecaster_summary: dict[str, object] | None = None
    if bool(args.learned_event_forecast):
        log("training split-compliant learned event forecaster")
        event_feature_columns = select_event_forecast_columns(
            truth,
            preferred_columns=tuple(str(x) for x in args.event_forecast_feature_columns) or state_columns,
            event_column=str(args.event_column),
        )
        event_forecast_cfg = EventForecasterTrainingConfig(
            horizon=int(args.horizon),
            lookback=int(args.event_forecast_lookback),
            event_column=str(args.event_column),
            hidden_dim=int(args.event_forecast_hidden_dim),
            epochs=int(args.event_forecast_epochs),
            batch_size=int(args.event_forecast_batch_size),
            learning_rate=float(args.event_forecast_learning_rate),
            weight_decay=float(args.event_forecast_weight_decay),
            seed=int(args.seed),
            device=str(args.bc_device),
            probability_prefix=str(args.event_forecast_probability_prefix),
            period_steps=max(1, int(round(86400.0 / max(float(args.freq_s), 1.0)))),
        )
        event_forecast_dataset = build_event_forecast_dataset(
            truth,
            bounds=(int(bounds["oracle_pretrain"][0]), int(bounds["rl_train"][1])),
            feature_columns=event_feature_columns,
            event_column=str(args.event_column),
            cfg=event_forecast_cfg,
        )
        event_forecaster = train_event_forecaster(event_forecast_dataset, event_forecast_cfg)
        truth, learned_event_probability_columns = augment_truth_with_event_forecasts(truth, event_forecaster)
        event_forecaster_summary = {
            "feature_columns": [str(x) for x in event_feature_columns],
            "probability_columns": [str(x) for x in learned_event_probability_columns],
            "train_bounds": [int(bounds["oracle_pretrain"][0]), int(bounds["rl_train"][1])],
            "history": event_forecaster.history,
            "final_loss": float(event_forecaster.history["loss"][-1])
            if event_forecaster.history.get("loss")
            else None,
            "final_brier": float(event_forecaster.history["brier"][-1])
            if event_forecaster.history.get("brier")
            else None,
        }
        log(
            "learned event forecaster complete: "
            f"columns={list(learned_event_probability_columns)} "
            f"final_brier={event_forecaster_summary['final_brier']}"
        )

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
    forecast_cfg = ForecastContextConfig(
        horizon=int(args.horizon),
        event_column=str(args.event_column),
        truth_future=bool(args.forecast_truth_future),
        learned_event_probability_columns=learned_event_probability_columns,
    )

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
            preserve_warming=bool(args.bc_preserve_warming),
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
    residual_gate_model = None
    residual_gate_history = None
    residual_threshold = None
    residual_support = None
    residual_validation_objective = None
    if bool(args.include_residual_bc_policy):
        log("training residual deviation gate")
        residual_gate_model, residual_gate_history = train_deviation_gate(
            teacher_dataset.features,
            teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            cfg=bc_cfg,
        )
        residual_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.residual_bc_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        residual_threshold, residual_validation_objective = calibrate_residual_threshold(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            bc_model=bc_model,
            gate_model=residual_gate_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=residual_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "residual BC calibration: "
            f"threshold={residual_threshold} validation_objective={residual_validation_objective:.6f} "
            f"support={list(residual_support) if residual_support is not None else None}"
        )
    mask_bc_model = None
    mask_bc_history = None
    mask_bc_required_indices: tuple[int, ...] = ()
    if bool(args.include_mask_bc_policy) or bool(args.include_contextual_duty_policy):
        log("training sensor-mask BC policy")
        mask_bc_model, mask_bc_history = train_mask_bc(
            teacher_dataset.features,
            teacher_dataset.labels,
            teacher_dataset.candidate_masks,
            cfg=bc_cfg,
        )
        label_masks = np.asarray(teacher_dataset.candidate_masks, dtype=bool)[
            np.asarray(teacher_dataset.labels, dtype=int)
        ]
        rates = np.mean(label_masks, axis=0) if label_masks.size else np.zeros(len(sensor_ids), dtype=float)
        threshold = float(args.mask_bc_required_rate)
        if threshold > 0.0:
            mask_bc_required_indices = tuple(int(idx) for idx in np.flatnonzero(rates >= threshold))
        log(
            "sensor-mask BC training complete: "
            f"final_sensor_accuracy={mask_bc_history['sensor_accuracy'][-1] if mask_bc_history and mask_bc_history.get('sensor_accuracy') else float('nan')} "
            f"required={[sensor_ids[idx] for idx in mask_bc_required_indices]}"
        )
    event_threshold_action_idx = None
    event_threshold_value = None
    event_threshold_aggregation = None
    event_threshold_validation_objective = None
    event_support_cycle_indices = None
    event_support_cycle_value = None
    event_support_cycle_aggregation = None
    event_support_cycle_period = None
    event_support_cycle_selection = None
    event_support_cycle_validation_objective = None
    teacher_rate_support = None
    teacher_rate_target_rates = None
    teacher_rate_blend = None
    teacher_rate_freshness_weight = None
    teacher_rate_power_weight = None
    teacher_rate_validation_objective = None
    contextual_duty_support = None
    contextual_duty_blend = None
    contextual_duty_deficit_weight = None
    contextual_duty_freshness_weight = None
    contextual_duty_power_weight = None
    contextual_duty_validation_objective = None
    if bool(args.include_event_threshold_policy):
        (
            event_threshold_action_idx,
            event_threshold_value,
            event_threshold_aggregation,
            event_threshold_validation_objective,
        ) = calibrate_event_threshold_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        event_ids = (
            "|".join(sensor_ids[idx] for idx in np.flatnonzero(candidate_masks[int(event_threshold_action_idx)]))
            if event_threshold_action_idx is not None
            else None
        )
        log(
            "event-threshold calibration: "
            f"action={event_threshold_action_idx} ids={event_ids} "
            f"aggregation={event_threshold_aggregation} threshold={event_threshold_value} "
            f"validation_objective={event_threshold_validation_objective:.6f}"
        )
    if bool(args.include_event_support_cycle_policy):
        (
            event_support_cycle_indices,
            event_support_cycle_value,
            event_support_cycle_aggregation,
            event_support_cycle_period,
            event_support_cycle_selection,
            event_support_cycle_validation_objective,
        ) = calibrate_event_support_cycle_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        cycle_ids = []
        if event_support_cycle_indices is not None:
            for action_idx in event_support_cycle_indices:
                cycle_ids.append("|".join(sensor_ids[idx] for idx in np.flatnonzero(candidate_masks[int(action_idx)])))
        log(
            "event-support-cycle calibration: "
            f"actions={event_support_cycle_indices} ids={cycle_ids} "
            f"aggregation={event_support_cycle_aggregation} threshold={event_support_cycle_value} "
            f"period={event_support_cycle_period} selection={event_support_cycle_selection} "
            f"validation_objective={event_support_cycle_validation_objective:.6f}"
        )
    if bool(args.include_teacher_rate_policy):
        (
            teacher_rate_support,
            teacher_rate_target_rates,
            teacher_rate_blend,
            teacher_rate_freshness_weight,
            teacher_rate_power_weight,
            teacher_rate_validation_objective,
        ) = calibrate_teacher_rate_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        rate_preview = (
            [float(x) for x in np.asarray(teacher_rate_target_rates, dtype=float).round(3).tolist()]
            if teacher_rate_target_rates is not None
            else None
        )
        log(
            "teacher-rate calibration: "
            f"support={list(teacher_rate_support) if teacher_rate_support is not None else None} "
            f"blend={teacher_rate_blend} freshness_weight={teacher_rate_freshness_weight} "
            f"power_weight={teacher_rate_power_weight} target_rates={rate_preview} "
            f"validation_objective={teacher_rate_validation_objective:.6f}"
        )
    if bool(args.include_contextual_duty_policy):
        if mask_bc_model is None:
            raise RuntimeError("contextual-duty policy requires the sensor-mask BC model")
        (
            contextual_duty_support,
            contextual_duty_blend,
            contextual_duty_deficit_weight,
            contextual_duty_freshness_weight,
            contextual_duty_power_weight,
            contextual_duty_validation_objective,
        ) = calibrate_contextual_duty_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=mask_bc_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "contextual-duty calibration: "
            f"support={list(contextual_duty_support) if contextual_duty_support is not None else None} "
            f"blend={contextual_duty_blend} "
            f"deficit_weight={contextual_duty_deficit_weight} "
            f"freshness_weight={contextual_duty_freshness_weight} "
            f"power_weight={contextual_duty_power_weight} "
            f"validation_objective={contextual_duty_validation_objective:.6f}"
        )
    cost_model = None
    cost_history = None
    cost_ensemble_models = None
    cost_ensemble_histories = None
    rollout_value_cost_model = None
    rollout_value_cost_history = None
    value_residual_support = None
    value_residual_threshold = None
    value_residual_validation_objective = None
    ensemble_value_support = None
    ensemble_value_threshold = None
    ensemble_value_beta = None
    ensemble_value_validation_objective = None
    advantage_residual_model = None
    advantage_residual_history = None
    advantage_residual_support = None
    advantage_residual_threshold = None
    advantage_residual_validation_objective = None
    rollout_value_transition_model = None
    rollout_value_transition_history = None
    rollout_value_support = None
    rollout_value_threshold = None
    rollout_value_validation_objective = None
    if (
        bool(args.include_cost_policy)
        or bool(args.include_value_residual_policy)
        or bool(args.include_ensemble_value_policy)
    ):
        log("collecting action-cost dataset")
        cost_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        cost_dataset = collect_action_cost_dataset(
            cost_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
        )
        log(f"action-cost dataset collected: rows={cost_dataset.inputs.shape[0]}")
        cost_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed),
            device=str(args.bc_device),
            ensemble_size=max(1, int(args.ensemble_value_size)) if bool(args.include_ensemble_value_policy) else 1,
            bootstrap_fraction=float(args.ensemble_value_bootstrap_fraction),
        )
        if bool(args.include_ensemble_value_policy):
            cost_ensemble_models, cost_ensemble_histories = train_action_cost_ensemble(cost_dataset, cost_train_cfg)
            cost_model = cost_ensemble_models[0]
            cost_history = cost_ensemble_histories[0]
            final_losses = [
                float(history["loss"][-1]) if history.get("loss") else float("nan")
                for history in cost_ensemble_histories
            ]
            log(
                "action-cost ensemble training complete: "
                f"members={len(cost_ensemble_models)} final_loss_mean={float(np.nanmean(final_losses)):.6f}"
            )
        else:
            cost_model, cost_history = train_action_cost_model(cost_dataset, cost_train_cfg)
            log(
                "action-cost model training complete: "
                f"final_loss={cost_history['loss'][-1] if cost_history and cost_history.get('loss') else float('nan')}"
            )
    if bool(args.include_advantage_residual_policy):
        log("collecting anchor-advantage dataset")
        advantage_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        advantage_dataset = collect_anchor_advantage_dataset(
            advantage_env,
            candidate_masks,
            anchor_mask=selected_static_mask,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
        )
        log(f"anchor-advantage dataset collected: rows={advantage_dataset.inputs.shape[0]}")
        advantage_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 17,
            device=str(args.bc_device),
        )
        advantage_residual_model, advantage_residual_history = train_anchor_advantage_model(
            advantage_dataset,
            advantage_train_cfg,
        )
        log(
            "anchor-advantage model training complete: "
            f"final_loss={advantage_residual_history['loss'][-1] if advantage_residual_history and advantage_residual_history.get('loss') else float('nan')}"
        )
        (
            selected_advantage_support_top_k,
            advantage_residual_support,
            advantage_residual_threshold,
            advantage_residual_validation_objective,
        ) = calibrate_advantage_residual_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=advantage_residual_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        args.advantage_residual_support_top_k = int(selected_advantage_support_top_k)
        log(
            "advantage-residual calibration: "
            f"top_k={selected_advantage_support_top_k} "
            f"threshold={advantage_residual_threshold} "
            f"validation_objective={advantage_residual_validation_objective:.6f} "
            f"support={list(advantage_residual_support) if advantage_residual_support is not None else None}"
        )
    if bool(args.include_value_residual_policy) and cost_model is not None:
        value_residual_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.value_residual_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        value_residual_threshold, value_residual_validation_objective = calibrate_value_residual_threshold(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=cost_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=value_residual_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "value-residual calibration: "
            f"threshold={value_residual_threshold} "
            f"validation_objective={value_residual_validation_objective:.6f} "
            f"support={list(value_residual_support) if value_residual_support is not None else None}"
        )
    if bool(args.include_rollout_value_policy):
        rollout_value_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.rollout_value_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        log("collecting raw action-cost dataset for rollout planner")
        rollout_cost_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        rollout_cost_dataset = collect_action_cost_dataset(
            rollout_cost_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            normalize_costs=False,
            allowed_action_indices=rollout_value_support,
            anchor_mask=selected_static_mask,
        )
        log(f"raw rollout action-cost dataset collected: rows={rollout_cost_dataset.inputs.shape[0]}")
        rollout_cost_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 23,
            device=str(args.bc_device),
        )
        rollout_value_cost_model, rollout_value_cost_history = train_action_cost_model(
            rollout_cost_dataset,
            rollout_cost_train_cfg,
        )
        log(
            "raw rollout action-cost model training complete: "
            f"final_loss={rollout_value_cost_history['loss'][-1] if rollout_value_cost_history and rollout_value_cost_history.get('loss') else float('nan')}"
        )
        log("collecting feature-transition dataset")
        transition_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        transition_dataset = collect_feature_transition_dataset(
            transition_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            allowed_action_indices=rollout_value_support,
            anchor_mask=selected_static_mask,
        )
        log(f"feature-transition dataset collected: rows={transition_dataset.inputs.shape[0]}")
        transition_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 29,
            device=str(args.bc_device),
        )
        rollout_value_transition_model, rollout_value_transition_history = train_feature_transition_model(
            transition_dataset,
            transition_train_cfg,
        )
        log(
            "feature-transition model training complete: "
            f"final_loss={rollout_value_transition_history['loss'][-1] if rollout_value_transition_history and rollout_value_transition_history.get('loss') else float('nan')}"
        )
        rollout_value_threshold, rollout_value_validation_objective = calibrate_rollout_value_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            cost_model=rollout_value_cost_model,
            transition_model=rollout_value_transition_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=rollout_value_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "rollout-value calibration: "
            f"threshold={rollout_value_threshold} "
            f"validation_objective={rollout_value_validation_objective:.6f} "
            f"support={list(rollout_value_support) if rollout_value_support is not None else None}"
        )
    if bool(args.include_ensemble_value_policy) and cost_ensemble_models is not None:
        ensemble_value_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.ensemble_value_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        (
            ensemble_value_beta,
            ensemble_value_threshold,
            ensemble_value_validation_objective,
        ) = calibrate_ensemble_value_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            models=cost_ensemble_models,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=ensemble_value_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "ensemble-value calibration: "
            f"beta={ensemble_value_beta} threshold={ensemble_value_threshold} "
            f"validation_objective={ensemble_value_validation_objective:.6f} "
            f"support={list(ensemble_value_support) if ensemble_value_support is not None else None}"
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
    bc_action_support_top_k = int(args.bc_action_support_top_k)
    bc_action_support_validation_objective = None
    if args.bc_action_support_grid:
        (
            bc_action_support_top_k,
            bc_action_support_validation_objective,
        ) = calibrate_bc_action_support(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=bc_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "BC action-support calibration: "
            f"top_k={bc_action_support_top_k} validation_objective={bc_action_support_validation_objective:.6f}"
        )
    bc_action_support = action_support_from_labels(
        teacher_dataset.labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=int(bc_action_support_top_k),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=selected_static_idx,
    )
    if bc_action_support is not None:
        log(f"BC action support enabled: n={len(bc_action_support)} indices={list(bc_action_support)}")
    policies = [
        StaticMaskPolicy(mask=selected_static_mask, name="validation_selected_static"),
        MpcTeacherPolicy(candidate_masks=candidate_masks, cfg=teacher_cfg, name="mpc_teacher"),
    ]
    if bool(args.include_bc_policy):
        policies.append(
            ForecastAwareBCPolicy(
                model=bc_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                fallback_mask=selected_static_mask if str(args.bc_fallback_source) == "validation_static" else None,
                allowed_action_indices=bc_action_support,
                min_logit_margin=bc_fallback_margin,
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_residual_bc_policy) and residual_gate_model is not None:
        policies.append(
            ForecastAwareResidualBCPolicy(
                bc_model=bc_model,
                gate_model=residual_gate_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=residual_support,
                deviate_threshold=float(residual_threshold if residual_threshold is not None else 0.5),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_knn_policy):
        policies.append(
            ForecastAwareKNNPolicy(
                features=teacher_dataset.features,
                labels=teacher_dataset.labels,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                k=int(args.knn_k),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_mask_bc_policy) and mask_bc_model is not None:
        policies.append(
            ForecastAwareMaskBCPolicy(
                model=mask_bc_model,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                preserve_warming=bool(args.bc_preserve_warming),
                required_sensor_indices=mask_bc_required_indices,
                anchor_mask=selected_static_mask,
                anchor_bias=float(args.mask_bc_anchor_bias),
            )
        )
    if bool(args.include_event_threshold_policy) and event_threshold_action_idx is not None:
        policies.append(
            ForecastAwareEventThresholdPolicy(
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                event_action_idx=int(event_threshold_action_idx),
                threshold=float(event_threshold_value if event_threshold_value is not None else 1.0),
                aggregation=str(event_threshold_aggregation or "max"),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_event_support_cycle_policy) and event_support_cycle_indices is not None:
        policies.append(
            ForecastAwareEventSupportCyclePolicy(
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                event_action_indices=tuple(int(x) for x in event_support_cycle_indices),
                threshold=float(event_support_cycle_value if event_support_cycle_value is not None else 1.0),
                aggregation=str(event_support_cycle_aggregation or "max"),
                cycle_period=int(event_support_cycle_period if event_support_cycle_period is not None else 1),
                selection_mode=str(event_support_cycle_selection or "time_cycle"),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_teacher_rate_policy) and teacher_rate_target_rates is not None:
        policies.append(
            ForecastAwareTeacherRatePolicy(
                candidate_masks=candidate_masks,
                target_rates=np.asarray(teacher_rate_target_rates, dtype=float),
                allowed_action_indices=teacher_rate_support,
                freshness_weight=float(teacher_rate_freshness_weight or 0.0),
                power_weight=float(teacher_rate_power_weight or 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
                anchor_mask=selected_static_mask,
            )
        )
    if bool(args.include_contextual_duty_policy) and mask_bc_model is not None:
        policies.append(
            ForecastAwareContextualDutyPolicy(
                model=mask_bc_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                allowed_action_indices=contextual_duty_support,
                anchor_mask=selected_static_mask,
                blend=float(contextual_duty_blend if contextual_duty_blend is not None else 1.0),
                deficit_weight=float(
                    contextual_duty_deficit_weight if contextual_duty_deficit_weight is not None else 1.0
                ),
                freshness_weight=float(
                    contextual_duty_freshness_weight if contextual_duty_freshness_weight is not None else 0.0
                ),
                power_weight=float(contextual_duty_power_weight if contextual_duty_power_weight is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_teacher_cycle_policy):
        policies.append(
            ForecastAwareCyclePolicy(
                labels=teacher_dataset.labels,
                candidate_masks=candidate_masks,
                preserve_warming=bool(args.bc_preserve_warming),
                max_lookahead=int(args.teacher_cycle_max_lookahead),
                name="forecast_aware_teacher_cycle",
            )
        )
    if bool(args.include_cost_policy) and cost_model is not None:
        policies.append(
            ForecastAwareCostPolicy(
                model=cost_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                allowed_action_indices=bc_action_support,
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_value_residual_policy) and cost_model is not None:
        policies.append(
            ForecastAwareValueResidualPolicy(
                model=cost_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=value_residual_support,
                advantage_threshold=float(value_residual_threshold if value_residual_threshold is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_rollout_value_policy)
        and rollout_value_cost_model is not None
        and rollout_value_transition_model is not None
    ):
        policies.append(
            ForecastAwareRolloutValuePolicy(
                cost_model=rollout_value_cost_model,
                transition_model=rollout_value_transition_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=rollout_value_support,
                advantage_threshold=float(rollout_value_threshold if rollout_value_threshold is not None else 0.0),
                planning_depth=int(args.rollout_value_depth),
                beam_width=int(args.rollout_value_beam_width),
                max_branch=int(args.rollout_value_max_branch),
                discount=float(args.rollout_value_discount),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_ensemble_value_policy) and cost_ensemble_models is not None:
        policies.append(
            ForecastAwareEnsembleValuePolicy(
                models=cost_ensemble_models,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=ensemble_value_support,
                advantage_threshold=float(ensemble_value_threshold if ensemble_value_threshold is not None else 0.0),
                uncertainty_beta=float(ensemble_value_beta if ensemble_value_beta is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_advantage_residual_policy) and advantage_residual_model is not None:
        policies.append(
            ForecastAwareAdvantageResidualPolicy(
                model=advantage_residual_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=advantage_residual_support,
                advantage_threshold=float(
                    advantage_residual_threshold if advantage_residual_threshold is not None else 0.0
                ),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
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

    deployable_validation_rows: list[dict[str, object]] = []
    selected_deployable_name = None
    if str(args.deployable_selection) == "validation":
        policies, selected_deployable_name, deployable_validation_rows = select_deployables_for_final(
            args,
            policies=policies,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            state_columns=state_columns,
            sensor_ids=sensor_ids,
            starts=starts["validation"].starts,
        )
        log(f"validation-selected deployable policy={selected_deployable_name}")

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
    deployable = metrics_df[
        metrics_df["policy"].isin(
            [
                "forecast_aware_bc",
                "forecast_aware_knn",
                "forecast_aware_cost",
                "forecast_aware_mask_bc",
                "forecast_aware_event_threshold",
                "forecast_aware_event_support_cycle",
                "forecast_aware_teacher_rate",
                "forecast_aware_contextual_duty",
                "forecast_aware_teacher_cycle",
                "forecast_aware_residual_bc",
                "forecast_aware_value_residual",
                "forecast_aware_rollout_value",
                "forecast_aware_ensemble_value",
                "forecast_aware_advantage_residual",
            ]
        )
    ]
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
        "seed": int(args.seed),
        "truth_csv": str(truth_path),
        "sensor_cfg": str(resolve_archive_path(args.sensor_cfg)),
        "oracle_path": str(resolve_archive_path(args.oracle_path)) if args.oracle_path else None,
        "oracle_type": str(args.oracle_type),
        "custom_ppo_checkpoint": str(resolve_archive_path(args.custom_ppo_checkpoint))
        if args.custom_ppo_checkpoint
        else None,
        "oracle_target_weight_mode": str(args.oracle_target_weight_mode),
        "oracle_target_weights": oracle_target_weights,
        "objective_mode": str(args.objective_mode),
        "learned_event_forecast": event_forecaster_summary,
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
        "bc_preserve_warming": bool(args.bc_preserve_warming),
        "bc_policy_included": bool(args.include_bc_policy),
        "residual_bc_policy": {
            "included": bool(args.include_residual_bc_policy),
            "support_top_k": int(args.residual_bc_support_top_k),
            "support_indices": [int(x) for x in residual_support] if residual_support is not None else None,
            "threshold": residual_threshold,
            "threshold_grid": [float(x) for x in args.residual_deviation_threshold_grid],
            "validation_objective": residual_validation_objective,
            "gate_history": residual_gate_history,
        },
        "bc_action_support": {
            "top_k": int(bc_action_support_top_k),
            "min_count": int(args.bc_action_support_min_count),
            "grid": [int(x) for x in args.bc_action_support_grid],
            "validation_objective": bc_action_support_validation_objective,
            "indices": [int(x) for x in bc_action_support] if bc_action_support is not None else None,
        },
        "mask_bc_policy": {
            "included": bool(args.include_mask_bc_policy),
            "required_rate": float(args.mask_bc_required_rate),
            "required_indices": [int(x) for x in mask_bc_required_indices],
            "required_sensor_ids": [sensor_ids[int(x)] for x in mask_bc_required_indices],
            "anchor_bias": float(args.mask_bc_anchor_bias),
            "history": mask_bc_history,
        },
        "event_threshold_policy": {
            "included": bool(args.include_event_threshold_policy),
            "support_top_k": int(args.event_threshold_support_top_k),
            "action_idx": event_threshold_action_idx,
            "threshold": event_threshold_value,
            "aggregation": event_threshold_aggregation,
            "threshold_grid": [float(x) for x in args.event_threshold_grid],
            "aggregation_grid": [str(x) for x in args.event_threshold_aggregation_grid],
            "validation_objective": event_threshold_validation_objective,
        },
        "event_support_cycle_policy": {
            "included": bool(args.include_event_support_cycle_policy),
            "support_top_k": int(args.event_support_cycle_top_k),
            "action_indices": [int(x) for x in event_support_cycle_indices]
            if event_support_cycle_indices is not None
            else None,
            "threshold": event_support_cycle_value,
            "aggregation": event_support_cycle_aggregation,
            "cycle_period": event_support_cycle_period,
            "selection_mode": event_support_cycle_selection,
            "threshold_grid": [float(x) for x in args.event_support_cycle_grid],
            "aggregation_grid": [str(x) for x in args.event_support_cycle_aggregation_grid],
            "period_grid": [int(x) for x in args.event_support_cycle_period_grid],
            "selection_grid": [str(x) for x in args.event_support_cycle_selection_grid],
            "validation_objective": event_support_cycle_validation_objective,
        },
        "teacher_rate_policy": {
            "included": bool(args.include_teacher_rate_policy),
            "support_top_k": int(args.teacher_rate_support_top_k),
            "action_indices": [int(x) for x in teacher_rate_support] if teacher_rate_support is not None else None,
            "target_rates": [float(x) for x in np.asarray(teacher_rate_target_rates, dtype=float).reshape(-1)]
            if teacher_rate_target_rates is not None
            else None,
            "blend": teacher_rate_blend,
            "freshness_weight": teacher_rate_freshness_weight,
            "power_weight": teacher_rate_power_weight,
            "blend_grid": [float(x) for x in args.teacher_rate_blend_grid],
            "freshness_grid": [float(x) for x in args.teacher_rate_freshness_grid],
            "power_grid": [float(x) for x in args.teacher_rate_power_grid],
            "validation_objective": teacher_rate_validation_objective,
        },
        "contextual_duty_policy": {
            "included": bool(args.include_contextual_duty_policy),
            "support_top_k": int(args.contextual_duty_support_top_k),
            "action_indices": [int(x) for x in contextual_duty_support]
            if contextual_duty_support is not None
            else None,
            "blend": contextual_duty_blend,
            "deficit_weight": contextual_duty_deficit_weight,
            "freshness_weight": contextual_duty_freshness_weight,
            "power_weight": contextual_duty_power_weight,
            "blend_grid": [float(x) for x in args.contextual_duty_blend_grid],
            "deficit_grid": [float(x) for x in args.contextual_duty_deficit_grid],
            "freshness_grid": [float(x) for x in args.contextual_duty_freshness_grid],
            "power_grid": [float(x) for x in args.contextual_duty_power_grid],
            "validation_objective": contextual_duty_validation_objective,
            "mask_bc_history": mask_bc_history,
        },
        "teacher_cycle_policy": {
            "included": bool(args.include_teacher_cycle_policy),
            "max_lookahead": int(args.teacher_cycle_max_lookahead),
        },
        "deployable_selection": {
            "mode": str(args.deployable_selection),
            "criterion": str(args.deployable_selection_criterion),
            "min_mean_margin": float(args.deployable_selection_min_mean_margin),
            "min_start_margin": float(args.deployable_selection_min_start_margin),
            "max_negative_starts": int(args.deployable_selection_max_negative_starts),
            "selected_policy": selected_deployable_name,
            "validation_rows": deployable_validation_rows,
        },
        "knn_policy": {
            "included": bool(args.include_knn_policy),
            "k": int(args.knn_k),
        },
        "cost_policy": {
            "included": bool(args.include_cost_policy),
            "epochs": int(args.cost_epochs),
            "hidden_dim": int(args.cost_hidden_dim),
            "history": cost_history,
        },
        "value_residual_policy": {
            "included": bool(args.include_value_residual_policy),
            "support_top_k": int(args.value_residual_support_top_k),
            "support_indices": [int(x) for x in value_residual_support]
            if value_residual_support is not None
            else None,
            "advantage_threshold": value_residual_threshold,
            "advantage_grid": [float(x) for x in args.value_residual_advantage_grid],
            "validation_objective": value_residual_validation_objective,
        },
        "rollout_value_policy": {
            "included": bool(args.include_rollout_value_policy),
            "support_top_k": int(args.rollout_value_support_top_k),
            "support_indices": [int(x) for x in rollout_value_support]
            if rollout_value_support is not None
            else None,
            "planning_depth": int(args.rollout_value_depth),
            "beam_width": int(args.rollout_value_beam_width),
            "max_branch": int(args.rollout_value_max_branch),
            "discount": float(args.rollout_value_discount),
            "advantage_threshold": rollout_value_threshold,
            "advantage_grid": [float(x) for x in args.rollout_value_advantage_grid],
            "validation_objective": rollout_value_validation_objective,
            "cost_target": "raw_teacher_rollout_cost",
            "cost_history": rollout_value_cost_history,
            "transition_history": rollout_value_transition_history,
        },
        "ensemble_value_policy": {
            "included": bool(args.include_ensemble_value_policy),
            "support_top_k": int(args.ensemble_value_support_top_k),
            "support_indices": [int(x) for x in ensemble_value_support]
            if ensemble_value_support is not None
            else None,
            "ensemble_size": int(args.ensemble_value_size),
            "bootstrap_fraction": float(args.ensemble_value_bootstrap_fraction),
            "selected_uncertainty_beta": ensemble_value_beta,
            "selected_advantage_threshold": ensemble_value_threshold,
            "beta_grid": [float(x) for x in args.ensemble_value_beta_grid],
            "advantage_grid": [float(x) for x in args.ensemble_value_advantage_grid],
            "validation_objective": ensemble_value_validation_objective,
            "histories": cost_ensemble_histories,
        },
        "advantage_residual_policy": {
            "included": bool(args.include_advantage_residual_policy),
            "support_top_k": int(args.advantage_residual_support_top_k),
            "support_grid": [int(x) for x in args.advantage_residual_support_grid],
            "support_indices": [int(x) for x in advantage_residual_support]
            if advantage_residual_support is not None
            else None,
            "advantage_threshold": advantage_residual_threshold,
            "advantage_grid": [float(x) for x in args.advantage_residual_grid],
            "validation_objective": advantage_residual_validation_objective,
            "history": advantage_residual_history,
        },
        "bc_fallback": {
            "source": str(args.bc_fallback_source),
            "logit_margin": bc_fallback_margin,
            "grid": [float(x) for x in args.bc_logit_margin_grid],
        },
        "run_args": vars(args),
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
            preserve_warming=bool(args.bc_preserve_warming),
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


def calibrate_bc_action_support(
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
    labels: np.ndarray,
    anchor_idx: int | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int, float]:
    rows: list[tuple[float, float, int, int]] = []
    grid = [int(x) for x in args.bc_action_support_grid]
    if not grid:
        grid = [int(args.bc_action_support_top_k)]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, top_k in enumerate(grid):
        support = action_support_from_labels(
            labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=max(0, int(top_k)),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=anchor_idx,
        )
        policy = ForecastAwareBCPolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            device=str(args.bc_device),
            allowed_action_indices=support,
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_bc_support_calib_{idx}",
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
            seed_offset=90_000 + idx * 101,
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
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), max(0, int(top_k)), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_top_k, _ = rows[0]
    return int(best_top_k), float(best_objective)


def calibrate_residual_threshold(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    bc_model: object,
    gate_model: object,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float]:
    rows: list[tuple[float, float, float, int]] = []
    grid = [float(x) for x in args.residual_deviation_threshold_grid] or [0.5]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareResidualBCPolicy(
            bc_model=bc_model,
            gate_model=gate_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            deviate_threshold=float(threshold),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_residual_bc_calib_{idx}",
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
            seed_offset=120_000 + idx * 101,
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
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(threshold), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_threshold, _ = rows[0]
    return float(best_threshold), float(best_objective)


def calibrate_value_residual_threshold(
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
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float]:
    rows: list[tuple[float, float, float, int]] = []
    grid = [float(x) for x in args.value_residual_advantage_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareValueResidualPolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            advantage_threshold=float(threshold),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_value_residual_calib_{idx}",
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
            seed_offset=130_000 + idx * 101,
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
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(threshold), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_threshold, _ = rows[0]
    return float(best_threshold), float(best_objective)


def calibrate_rollout_value_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    cost_model: object,
    transition_model: object,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float]:
    rows: list[tuple[float, float, float, int]] = []
    grid = [float(x) for x in args.rollout_value_advantage_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareRolloutValuePolicy(
            cost_model=cost_model,
            transition_model=transition_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            advantage_threshold=float(threshold),
            planning_depth=int(args.rollout_value_depth),
            beam_width=int(args.rollout_value_beam_width),
            max_branch=int(args.rollout_value_max_branch),
            discount=float(args.rollout_value_discount),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_rollout_value_calib_{idx}",
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
            seed_offset=145_000 + idx * 101,
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
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(threshold), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_threshold, _ = rows[0]
    return float(best_threshold), float(best_objective)


def calibrate_advantage_residual_threshold(
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
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float]:
    rows: list[tuple[float, float, float, int]] = []
    grid = [float(x) for x in args.advantage_residual_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareAdvantageResidualPolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            advantage_threshold=float(threshold),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_advantage_residual_calib_{idx}",
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
            seed_offset=150_000 + idx * 101,
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
        rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(threshold), idx))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    best_objective, _, best_threshold, _ = rows[0]
    return float(best_threshold), float(best_objective)


def calibrate_advantage_residual_policy(
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
    anchor_mask: tuple[bool, ...],
    labels: np.ndarray,
    anchor_idx: int | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int, tuple[int, ...] | None, float, float]:
    rows: list[tuple[float, float, int, float, int]] = []
    top_k_grid = [int(x) for x in args.advantage_residual_support_grid]
    if not top_k_grid:
        top_k_grid = [int(args.advantage_residual_support_top_k)]
    threshold_grid = [float(x) for x in args.advantage_residual_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for top_k in top_k_grid:
        support = action_support_from_labels(
            labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=max(0, int(top_k)),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=anchor_idx,
        )
        for threshold in threshold_grid:
            policy = ForecastAwareAdvantageResidualPolicy(
                model=model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=anchor_mask,
                device=str(args.bc_device),
                allowed_action_indices=support,
                advantage_threshold=float(threshold),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_advantage_residual_calib_{combo_idx}",
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
                seed_offset=160_000 + combo_idx * 101,
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
            rows.append(
                (
                    float(objective),
                    float(metrics.get("power_mean", np.nan)),
                    max(0, int(top_k)),
                    float(threshold),
                    combo_idx,
                )
            )
            combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[4]))
    best_objective, _, best_top_k, best_threshold, _ = rows[0]
    best_support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(0, int(best_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    return int(best_top_k), best_support, float(best_threshold), float(best_objective)


def calibrate_ensemble_value_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    models: list[object],
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float, float]:
    rows: list[tuple[float, float, float, float, int]] = []
    beta_grid = [float(x) for x in args.ensemble_value_beta_grid] or [0.0]
    threshold_grid = [float(x) for x in args.ensemble_value_advantage_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for beta in beta_grid:
        for threshold in threshold_grid:
            policy = ForecastAwareEnsembleValuePolicy(
                models=models,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=anchor_mask,
                device=str(args.bc_device),
                allowed_action_indices=allowed_action_indices,
                advantage_threshold=float(threshold),
                uncertainty_beta=float(beta),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_ensemble_value_calib_{combo_idx}",
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
                seed_offset=140_000 + combo_idx * 101,
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
            rows.append((float(objective), float(metrics.get("power_mean", np.nan)), float(beta), float(threshold), combo_idx))
            combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[4]))
    best_objective, _, best_beta, best_threshold, _ = rows[0]
    return float(best_beta), float(best_threshold), float(best_objective)


def calibrate_event_threshold_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    labels: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int | None, float | None, str | None, float]:
    support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(1, int(args.event_threshold_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=None,
    )
    if support is None:
        return None, None, None, float("inf")
    event_actions = [int(idx) for idx in support if anchor_idx is None or int(idx) != int(anchor_idx)]
    if not event_actions and anchor_idx is not None:
        event_actions = [int(anchor_idx)]
    thresholds = [float(x) for x in args.event_threshold_grid] or [0.5]
    aggregations = [str(x) for x in args.event_threshold_aggregation_grid] or ["max"]
    rows: list[tuple[float, float, int, float, str, int]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for action_idx in event_actions:
        for aggregation in aggregations:
            if aggregation not in {"max", "mean", "first"}:
                raise ValueError(f"Unsupported event-threshold aggregation: {aggregation}")
            for threshold in thresholds:
                policy = ForecastAwareEventThresholdPolicy(
                    candidate_masks=candidate_masks,
                    forecast_cfg=forecast_cfg,
                    anchor_mask=anchor_mask,
                    event_action_idx=int(action_idx),
                    threshold=float(threshold),
                    aggregation=str(aggregation),
                    preserve_warming=bool(args.bc_preserve_warming),
                    name=f"forecast_aware_event_threshold_calib_{combo_idx}",
                )
                metrics, objective = evaluate_validation_policy_metrics(
                    args,
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=cfg,
                    oracle=oracle,
                    policy=policy,
                    state_columns=state_columns,
                    sensor_ids=sensor_ids,
                    starts=starts,
                    seed_offset=170_000 + combo_idx * 101,
                )
                rows.append(
                    (
                        float(objective),
                        float(metrics.get("power_mean", np.nan)),
                        int(action_idx),
                        float(threshold),
                        str(aggregation),
                        combo_idx,
                    )
                )
                combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[5]))
    best_objective, _, best_action_idx, best_threshold, best_aggregation, _ = rows[0]
    return int(best_action_idx), float(best_threshold), str(best_aggregation), float(best_objective)


def calibrate_event_support_cycle_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    labels: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, float | None, str | None, int | None, str | None, float]:
    support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(1, int(args.event_support_cycle_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=None,
    )
    if support is None:
        return None, None, None, None, None, float("inf")
    event_actions = tuple(int(idx) for idx in support if anchor_idx is None or int(idx) != int(anchor_idx))
    if not event_actions and anchor_idx is not None:
        event_actions = (int(anchor_idx),)
    thresholds = [float(x) for x in args.event_support_cycle_grid] or [0.5]
    aggregations = [str(x) for x in args.event_support_cycle_aggregation_grid] or ["max"]
    periods = [max(1, int(x)) for x in args.event_support_cycle_period_grid] or [1]
    selection_modes = [str(x) for x in args.event_support_cycle_selection_grid] or ["time_cycle"]
    rows: list[tuple[float, float, float, str, int, str, int]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for aggregation in aggregations:
        if aggregation not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported event-support-cycle aggregation: {aggregation}")
        for selection_mode in selection_modes:
            if selection_mode not in {"time_cycle", "freshness"}:
                raise ValueError(f"Unsupported event-support-cycle selection mode: {selection_mode}")
            for period in periods:
                for threshold in thresholds:
                    policy = ForecastAwareEventSupportCyclePolicy(
                        candidate_masks=candidate_masks,
                        forecast_cfg=forecast_cfg,
                        anchor_mask=anchor_mask,
                        event_action_indices=event_actions,
                        threshold=float(threshold),
                        aggregation=str(aggregation),
                        cycle_period=int(period),
                        selection_mode=str(selection_mode),
                        preserve_warming=bool(args.bc_preserve_warming),
                        name=f"forecast_aware_event_support_cycle_calib_{combo_idx}",
                    )
                    metrics, objective = evaluate_validation_policy_metrics(
                        args,
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=cfg,
                        oracle=oracle,
                        policy=policy,
                        state_columns=state_columns,
                        sensor_ids=sensor_ids,
                        starts=starts,
                        seed_offset=180_000 + combo_idx * 101,
                    )
                    rows.append(
                        (
                            float(objective),
                            float(metrics.get("power_mean", np.nan)),
                            float(threshold),
                            str(aggregation),
                            int(period),
                            str(selection_mode),
                            combo_idx,
                        )
                    )
                    combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[6]))
    best_objective, _, best_threshold, best_aggregation, best_period, best_selection, _ = rows[0]
    return (
        event_actions,
        float(best_threshold),
        str(best_aggregation),
        int(best_period),
        str(best_selection),
        float(best_objective),
    )


def calibrate_teacher_rate_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    labels: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, np.ndarray | None, float | None, float | None, float | None, float]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return None, None, None, None, None, float("inf")
    masks = np.asarray(candidate_masks, dtype=bool)
    if np.any(labels_arr < 0) or np.any(labels_arr >= masks.shape[0]):
        raise ValueError("teacher-rate calibration received labels outside candidate mask range")
    teacher_rates = np.mean(masks[labels_arr].astype(float), axis=0)
    anchor_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.teacher_rate_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    blends = [float(x) for x in args.teacher_rate_blend_grid] or [1.0]
    freshness_weights = [float(x) for x in args.teacher_rate_freshness_grid] or [0.0]
    power_weights = [float(x) for x in args.teacher_rate_power_grid] or [0.0]
    rows: list[tuple[float, float, float, float, float, int, np.ndarray]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for blend in blends:
        blend = float(np.clip(blend, 0.0, 1.0))
        target_rates = np.clip((1.0 - blend) * anchor_rates + blend * teacher_rates, 0.0, 1.0)
        for freshness_weight in freshness_weights:
            for power_weight in power_weights:
                policy = ForecastAwareTeacherRatePolicy(
                    candidate_masks=masks,
                    target_rates=target_rates,
                    allowed_action_indices=support,
                    freshness_weight=float(freshness_weight),
                    power_weight=float(power_weight),
                    preserve_warming=bool(args.bc_preserve_warming),
                    anchor_mask=anchor_mask,
                    name=f"forecast_aware_teacher_rate_calib_{combo_idx}",
                )
                metrics, objective = evaluate_validation_policy_metrics(
                    args,
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=cfg,
                    oracle=oracle,
                    policy=policy,
                    state_columns=state_columns,
                    sensor_ids=sensor_ids,
                    starts=starts,
                    seed_offset=190_000 + combo_idx * 101,
                )
                rows.append(
                    (
                        float(objective),
                        float(metrics.get("power_mean", np.nan)),
                        float(blend),
                        float(freshness_weight),
                        float(power_weight),
                        combo_idx,
                        target_rates.astype(float).copy(),
                    )
                )
                combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[5]))
    best_objective, _, best_blend, best_freshness, best_power, _, best_rates = rows[0]
    return (
        tuple(int(x) for x in support),
        np.asarray(best_rates, dtype=float),
        float(best_blend),
        float(best_freshness),
        float(best_power),
        float(best_objective),
    )


def calibrate_contextual_duty_policy(
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
    labels: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, float | None, float | None, float | None, float | None, float]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return None, None, None, None, None, float("inf")
    masks = np.asarray(candidate_masks, dtype=bool)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.contextual_duty_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    blends = [float(x) for x in args.contextual_duty_blend_grid] or [1.0]
    deficit_weights = [float(x) for x in args.contextual_duty_deficit_grid] or [1.0]
    freshness_weights = [float(x) for x in args.contextual_duty_freshness_grid] or [0.0]
    power_weights = [float(x) for x in args.contextual_duty_power_grid] or [0.0]
    rows: list[tuple[float, float, float, float, float, float, int]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    combo_idx = 0
    for blend in blends:
        for deficit_weight in deficit_weights:
            for freshness_weight in freshness_weights:
                for power_weight in power_weights:
                    policy = ForecastAwareContextualDutyPolicy(
                        model=model,
                        candidate_masks=masks,
                        forecast_cfg=forecast_cfg,
                        device=str(args.bc_device),
                        allowed_action_indices=support,
                        anchor_mask=anchor_mask,
                        blend=float(blend),
                        deficit_weight=float(deficit_weight),
                        freshness_weight=float(freshness_weight),
                        power_weight=float(power_weight),
                        preserve_warming=bool(args.bc_preserve_warming),
                        name=f"forecast_aware_contextual_duty_calib_{combo_idx}",
                    )
                    metrics, objective = evaluate_validation_policy_metrics(
                        args,
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=cfg,
                        oracle=oracle,
                        policy=policy,
                        state_columns=state_columns,
                        sensor_ids=sensor_ids,
                        starts=starts,
                        seed_offset=210_000 + combo_idx * 101,
                    )
                    rows.append(
                        (
                            float(objective),
                            float(metrics.get("power_mean", np.nan)),
                            float(blend),
                            float(deficit_weight),
                            float(freshness_weight),
                            float(power_weight),
                            combo_idx,
                        )
                    )
                    combo_idx += 1
    rows.sort(key=lambda item: (item[0], item[1], item[6]))
    best_objective, _, best_blend, best_deficit, best_freshness, best_power, _ = rows[0]
    return (
        tuple(int(x) for x in support),
        float(best_blend),
        float(best_deficit),
        float(best_freshness),
        float(best_power),
        float(best_objective),
    )


def select_deployables_for_final(
    args: argparse.Namespace,
    *,
    policies: list[object],
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    state_columns: tuple[str, ...],
    sensor_ids: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[list[object], str | None, list[dict[str, object]]]:
    deployable_names = {
        "forecast_aware_bc",
        "forecast_aware_knn",
        "forecast_aware_cost",
        "forecast_aware_mask_bc",
        "forecast_aware_event_threshold",
        "forecast_aware_event_support_cycle",
        "forecast_aware_teacher_rate",
        "forecast_aware_contextual_duty",
        "forecast_aware_teacher_cycle",
        "forecast_aware_residual_bc",
        "forecast_aware_value_residual",
        "forecast_aware_rollout_value",
        "forecast_aware_ensemble_value",
        "forecast_aware_advantage_residual",
    }
    fixed = [policy for policy in policies if str(policy.name) not in deployable_names]
    candidates = [policy for policy in policies if str(policy.name) in deployable_names]
    if not candidates:
        return policies, None, []
    rows: list[dict[str, object]] = []
    static_start_objectives: list[float] = []
    if str(args.deployable_selection_criterion) == "static_margin_guard":
        static_candidates = [policy for policy in fixed if str(policy.name) == "validation_selected_static"]
        if not static_candidates:
            raise ValueError("static_margin_guard selection requires validation_selected_static in policy list")
        static_policy = static_candidates[0]
        for start_idx, start in enumerate(starts):
            metrics, objective = evaluate_validation_policy_metrics(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=static_policy,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=(int(start),),
                seed_offset=100_000 + int(start_idx) * 101,
            )
            del metrics
            static_start_objectives.append(float(objective))
    for idx, policy in enumerate(candidates):
        metrics, objective = evaluate_validation_policy_metrics(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=policy,
            state_columns=state_columns,
            sensor_ids=sensor_ids,
            starts=starts,
            seed_offset=110_000 + idx * 101,
        )
        row = {
            "policy": str(policy.name),
            "objective": float(objective),
            "power_mean": float(metrics.get("power_mean", np.nan)),
            "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
        }
        if static_start_objectives:
            candidate_start_objectives: list[float] = []
            for start_idx, start in enumerate(starts):
                _, start_objective = evaluate_validation_policy_metrics(
                    args,
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=cfg,
                    oracle=oracle,
                    policy=policy,
                    state_columns=state_columns,
                    sensor_ids=sensor_ids,
                    starts=(int(start),),
                    seed_offset=100_000 + int(start_idx) * 101,
                )
                candidate_start_objectives.append(float(start_objective))
            margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                candidate_start_objectives,
                dtype=float,
            )
            row.update(
                {
                    "static_start_objectives": [float(x) for x in static_start_objectives],
                    "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                    "objective_margin_mean": float(np.mean(margins)),
                    "objective_margin_min": float(np.min(margins)),
                    "negative_start_count": int(np.sum(margins < 0.0)),
                }
            )
        rows.append(row)
    selected_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.deployable_selection_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
    )
    selected_name = str(selected_row["policy"])
    selected = [policy for policy in candidates if str(policy.name) == selected_name][0]
    return [*fixed, selected], selected_name, rows


def evaluate_validation_policy_metrics(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    policy: object,
    state_columns: tuple[str, ...],
    sensor_ids: tuple[str, ...],
    starts: tuple[int, ...],
    seed_offset: int,
) -> tuple[dict[str, object], float]:
    result, _ = evaluate_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=cfg,
        oracle=oracle,
        policy=policy,
        steps=int(args.static_selection_steps),
        start_indices=starts,
        seed_offset=int(seed_offset),
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
    return metrics, float(objective)


def action_support_from_labels(
    labels: np.ndarray,
    *,
    n_actions: int,
    top_k: int,
    min_count: int,
    anchor_idx: int | None,
) -> tuple[int, ...] | None:
    top_k = int(top_k)
    min_count = int(min_count)
    if top_k <= 0 and min_count <= 0:
        return None
    values = np.asarray(labels, dtype=int).reshape(-1)
    values = values[(values >= 0) & (values < int(n_actions))]
    if values.size == 0:
        return None
    counts = np.bincount(values, minlength=int(n_actions))
    selected: set[int] = set()
    if top_k > 0:
        positive = np.flatnonzero(counts > 0)
        order = sorted((int(idx) for idx in positive), key=lambda idx: (-int(counts[idx]), idx))
        selected.update(order[: min(top_k, len(order))])
    if min_count > 0:
        selected.update(int(idx) for idx in np.flatnonzero(counts >= min_count))
    if anchor_idx is not None and 0 <= int(anchor_idx) < int(n_actions):
        selected.add(int(anchor_idx))
    if not selected:
        return None
    return tuple(sorted(selected))


def log(message: str) -> None:
    print(f"[run_protocol_gate] {message}", flush=True)


if __name__ == "__main__":
    main()
