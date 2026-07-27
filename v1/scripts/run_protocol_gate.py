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
    ForecastAwareCostKNNPolicy,
    ForecastAwareCostPolicy,
    ForecastAwareEnsembleValuePolicy,
    ForecastAwareRecurrentAdvantagePolicy,
    ForecastAwareRecurrentValuePolicy,
    ForecastAwareRolloutValuePolicy,
    ForecastAwareSequenceValuePolicy,
    ForecastAwareValueResidualPolicy,
    collect_anchor_advantage_dataset,
    collect_action_cost_dataset,
    collect_executed_outcome_datasets,
    collect_feature_transition_dataset,
    collect_sequence_value_dataset,
    concat_action_cost_datasets,
    concat_feature_transition_datasets,
    concat_recurrent_action_cost_datasets,
    collect_recurrent_action_cost_dataset,
    collect_recurrent_anchor_advantage_dataset,
    train_anchor_advantage_model,
    train_action_cost_ensemble,
    train_action_cost_model,
    train_feature_transition_model,
    train_recurrent_action_cost_model,
    train_recurrent_anchor_advantage_model,
    train_sequence_value_model,
)
from forecast_cmdp.continuous_forecaster import (
    ContinuousForecasterTrainingConfig,
    augment_truth_with_continuous_forecasts,
    build_continuous_forecast_dataset,
    select_continuous_forecast_columns,
    train_continuous_forecaster,
)
from forecast_cmdp.dataset import collect_dagger_dataset, collect_teacher_dataset, concat_teacher_datasets
from forecast_cmdp.event_forecaster import (
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    select_event_forecast_columns,
    train_event_forecaster,
)
from forecast_cmdp.features import ForecastContextConfig, append_event_forecast, build_event_forecast
from forecast_cmdp.mpc_teacher import (
    MpcTeacherConfig,
    MpcTeacherPolicy,
    beam_search_first_action_costs,
    beam_search_teacher_action,
    enumerate_action_masks,
)
from forecast_cmdp.policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    ForecastAwareContextualDutyPolicy,
    ForecastAwareCyclePolicy,
    ForecastAwareEventSupportCyclePolicy,
    ForecastAwareEventThresholdPolicy,
    ForecastAwareKNNPolicy,
    ForecastAwareMacroOptionPolicy,
    ForecastAwareMaskBCPolicy,
    ForecastAwareOptionPlannerPolicy,
    ForecastAwareProxyMPCPolicy,
    ForecastAwareResidualBCPolicy,
    ForecastAwareRuntimeRiskGuardPolicy,
    ForecastAwareSequenceMaskPolicy,
    ForecastAwareTeacherRatePolicy,
    ForecastAwareTeacherImprovementGatePolicy,
    ForecastAwareUtilityPlannerPolicy,
    ForecastAwareWindowCandidatePolicy,
    ForecastAwareWindowEligibilityPolicy,
    ValidationCyclicDwellPolicy,
    save_bc_policy_checkpoint,
    train_bc_classifier,
    train_binary_gate,
    train_deviation_gate,
    train_mask_bc,
    train_sequence_mask_bc,
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


DEPLOYABLE_POLICY_NAMES = frozenset(
    {
        "forecast_aware_bc",
        "forecast_aware_knn",
        "forecast_aware_cost",
        "forecast_aware_cost_knn",
        "forecast_aware_mask_bc",
        "forecast_aware_event_threshold",
        "forecast_aware_event_support_cycle",
        "forecast_aware_option_planner",
        "forecast_aware_proxy_mpc",
        "forecast_aware_macro_option",
        "forecast_aware_teacher_improvement_gate",
        "forecast_aware_utility_planner",
        "forecast_aware_runtime_risk_guard",
        "forecast_aware_window_candidate",
        "forecast_aware_window_eligibility",
        "forecast_aware_teacher_rate",
        "forecast_aware_contextual_duty",
        "forecast_aware_sequence_mask",
        "forecast_aware_teacher_cycle",
        "forecast_aware_residual_bc",
        "forecast_aware_value_residual",
        "forecast_aware_rollout_value",
        "forecast_aware_sequence_value",
        "forecast_aware_recurrent_value",
        "forecast_aware_recurrent_advantage",
        "forecast_aware_ensemble_value",
        "forecast_aware_advantage_residual",
    }
)


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
    parser.add_argument("--selection", choices=["event_rich", "event_transport_rich", "uniform"], default="event_rich")
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--event-column", default="event_flag")
    parser.add_argument("--forecast-truth-future", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forecast-continuous-truth-future", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forecast-continuous-columns", nargs="*", default=[])
    parser.add_argument("--forecast-continuous-scales", nargs="*", type=float, default=[])
    parser.add_argument("--learned-event-forecast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-forecast-lookback", type=int, default=8)
    parser.add_argument("--event-forecast-hidden-dim", type=int, default=128)
    parser.add_argument("--event-forecast-epochs", type=int, default=40)
    parser.add_argument("--event-forecast-batch-size", type=int, default=256)
    parser.add_argument("--event-forecast-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--event-forecast-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--event-forecast-feature-columns", nargs="*", default=[])
    parser.add_argument("--event-forecast-probability-prefix", default="learned_event_p")
    parser.add_argument("--learned-continuous-forecast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continuous-forecast-lookback", type=int, default=8)
    parser.add_argument("--continuous-forecast-hidden-dim", type=int, default=128)
    parser.add_argument("--continuous-forecast-epochs", type=int, default=40)
    parser.add_argument("--continuous-forecast-batch-size", type=int, default=256)
    parser.add_argument("--continuous-forecast-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--continuous-forecast-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--continuous-forecast-feature-columns", nargs="*", default=[])
    parser.add_argument("--continuous-forecast-target-columns", nargs="*", default=[])
    parser.add_argument("--continuous-forecast-prediction-prefix", default="learned_cont")
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
        "--event-threshold-calibration-criterion",
        choices=DEPLOYABLE_SELECTION_CRITERIA,
        default="mean_objective",
    )
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
    parser.add_argument("--include-option-planner-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--option-planner-support-top-k", type=int, default=16)
    parser.add_argument(
        "--option-planner-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_guard",
    )
    parser.add_argument("--option-planner-threshold-grid", nargs="*", type=float, default=[0.35, 0.5, 0.65])
    parser.add_argument("--option-planner-aggregation-grid", nargs="*", default=["max", "mean"])
    parser.add_argument("--option-planner-min-dwell-grid", nargs="*", type=int, default=[2, 4])
    parser.add_argument("--option-planner-cooldown-grid", nargs="*", type=int, default=[0, 2])
    parser.add_argument("--option-planner-target-rate-grid", nargs="*", type=float, default=[1.0])
    parser.add_argument("--option-planner-rate-balance-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--option-planner-freshness-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--option-planner-transport-grid", nargs="*", type=float, default=[0.0, 0.3])
    parser.add_argument("--option-planner-power-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--option-planner-switch-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--option-planner-min-soc-grid", nargs="*", type=float, default=[0.0, 0.25])
    parser.add_argument("--include-macro-option-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--macro-option-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--macro-option-segment-grid", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--macro-option-snippet-stride", type=int, default=1)
    parser.add_argument("--macro-option-k-grid", nargs="*", type=int, default=[1, 4, 8])
    parser.add_argument("--macro-option-threshold-grid", nargs="*", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--macro-option-aggregation-grid", nargs="*", default=["mean"])
    parser.add_argument("--macro-option-distance-weighting-grid", nargs="*", default=["inverse", "uniform"])
    parser.add_argument("--macro-option-refresh-grid", nargs="*", type=int, default=[0])
    parser.add_argument("--macro-option-max-lookahead", type=int, default=4)
    parser.add_argument("--include-teacher-improvement-gate-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-improvement-gate-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-improvement-gate-epochs", type=int, default=40)
    parser.add_argument("--teacher-improvement-gate-label-margin", type=float, default=0.0)
    parser.add_argument(
        "--teacher-improvement-gate-threshold-grid",
        nargs="*",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
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
    parser.add_argument(
        "--contextual-duty-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="mean_objective",
    )
    parser.add_argument("--include-utility-planner-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--utility-planner-support-top-k", type=int, default=16)
    parser.add_argument("--utility-planner-event-weight-grid", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--utility-planner-magnitude-weight-grid", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--utility-planner-variability-weight-grid", nargs="*", type=float, default=[0.0, 0.5])
    parser.add_argument("--utility-planner-freshness-grid", nargs="*", type=float, default=[0.0, 0.25])
    parser.add_argument("--utility-planner-target-rate-grid", nargs="*", type=float, default=[0.0, 0.5])
    parser.add_argument("--utility-planner-anchor-bias-grid", nargs="*", type=float, default=[0.0, 0.1])
    parser.add_argument("--utility-planner-power-grid", nargs="*", type=float, default=[0.0, 0.03, 0.08])
    parser.add_argument("--utility-planner-switch-grid", nargs="*", type=float, default=[0.0, 0.03])
    parser.add_argument("--utility-planner-min-soc-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--utility-planner-dwell-grid", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--utility-planner-aggregation-grid", nargs="*", default=["max", "mean"])
    parser.add_argument(
        "--utility-planner-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--include-proxy-mpc-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proxy-mpc-support-top-k", type=int, default=16)
    parser.add_argument("--proxy-mpc-event-weight-grid", nargs="*", type=float, default=[0.5, 1.5])
    parser.add_argument("--proxy-mpc-magnitude-weight-grid", nargs="*", type=float, default=[1.0])
    parser.add_argument("--proxy-mpc-variability-weight-grid", nargs="*", type=float, default=[0.5])
    parser.add_argument("--proxy-mpc-freshness-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--proxy-mpc-target-rate-grid", nargs="*", type=float, default=[0.0, 0.5])
    parser.add_argument("--proxy-mpc-anchor-bias-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--proxy-mpc-power-grid", nargs="*", type=float, default=[0.03])
    parser.add_argument("--proxy-mpc-switch-grid", nargs="*", type=float, default=[0.03])
    parser.add_argument("--proxy-mpc-min-soc-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--proxy-mpc-dwell-grid", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--proxy-mpc-aggregation-grid", nargs="*", default=["max"])
    parser.add_argument("--proxy-mpc-depth-grid", nargs="*", type=int, default=[2, 3])
    parser.add_argument("--proxy-mpc-beam-width-grid", nargs="*", type=int, default=[4])
    parser.add_argument("--proxy-mpc-max-branch-grid", nargs="*", type=int, default=[8])
    parser.add_argument("--proxy-mpc-age-weight-grid", nargs="*", type=float, default=[0.25, 0.75])
    parser.add_argument("--proxy-mpc-anchor-improvement-grid", nargs="*", type=float, default=[0.0, 0.02])
    parser.add_argument(
        "--proxy-mpc-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--include-sequence-mask-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequence-mask-support-top-k", type=int, default=16)
    parser.add_argument("--sequence-mask-anchor-bias-grid", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--sequence-mask-power-grid", nargs="*", type=float, default=[0.0, 0.03, 0.08])
    parser.add_argument(
        "--sequence-mask-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_guard",
    )
    parser.add_argument("--include-teacher-cycle-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-cycle-max-lookahead", type=int, default=32)
    parser.add_argument("--include-validation-cyclic-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-cyclic-top-k", type=int, default=4)
    parser.add_argument("--validation-cyclic-dwell-grid", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--validation-cyclic-preserve-warming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-runtime-risk-guard-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--runtime-risk-guard-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--runtime-risk-threshold-grid", nargs="*", type=float, default=[0.5, 0.8, 1.1])
    parser.add_argument("--runtime-risk-window-grid", nargs="*", type=int, default=[8, 16])
    parser.add_argument("--runtime-risk-aggregation-grid", nargs="*", default=["max"])
    parser.add_argument("--runtime-risk-event-weight-grid", nargs="*", type=float, default=[1.0])
    parser.add_argument("--runtime-risk-freshness-weight-grid", nargs="*", type=float, default=[0.0, 0.25])
    parser.add_argument("--runtime-risk-transport-weight-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--runtime-risk-soc-weight-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--runtime-risk-min-soc-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--include-window-eligibility-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--window-eligibility-dynamic-grid", nargs="*", default=["option"])
    parser.add_argument("--window-eligibility-support-top-k", type=int, default=16)
    parser.add_argument("--window-eligibility-samples-per-start", type=int, default=4)
    parser.add_argument("--window-eligibility-max-train-windows", type=int, default=96)
    parser.add_argument("--window-eligibility-window-grid", nargs="*", type=int, default=[16])
    parser.add_argument("--window-eligibility-k-grid", nargs="*", type=int, default=[3, 5])
    parser.add_argument("--window-eligibility-margin-grid", nargs="*", type=float, default=[0.0, 0.005, 0.01])
    parser.add_argument("--window-eligibility-blend-grid", nargs="*", type=float, default=[1.0])
    parser.add_argument("--window-eligibility-min-dwell-grid", nargs="*", type=int, default=[2])
    parser.add_argument("--window-eligibility-freshness-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--window-eligibility-transport-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--window-eligibility-power-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--window-eligibility-switch-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--window-eligibility-min-soc-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--window-eligibility-distance-weighting-grid", nargs="*", default=["inverse"])
    parser.add_argument("--window-eligibility-macro-k-grid", nargs="*", type=int, default=[4])
    parser.add_argument("--window-eligibility-macro-snippet-stride", type=int, default=1)
    parser.add_argument("--window-eligibility-macro-max-lookahead", type=int, default=8)
    parser.add_argument(
        "--window-eligibility-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--include-window-candidate-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--window-candidate-family-grid", nargs="*", default=["option", "macro", "rate"])
    parser.add_argument("--window-candidate-support-top-k", type=int, default=16)
    parser.add_argument("--window-candidate-samples-per-start", type=int, default=4)
    parser.add_argument("--window-candidate-max-train-windows", type=int, default=96)
    parser.add_argument("--window-candidate-window-grid", nargs="*", type=int, default=[16])
    parser.add_argument("--window-candidate-k-grid", nargs="*", type=int, default=[3, 5])
    parser.add_argument("--window-candidate-margin-grid", nargs="*", type=float, default=[0.0, 0.005, 0.01])
    parser.add_argument("--window-candidate-quantile-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--window-candidate-distance-weighting-grid", nargs="*", default=["inverse"])
    parser.add_argument("--window-candidate-min-neighbors", type=int, default=1)
    parser.add_argument("--window-candidate-max-candidates", type=int, default=12)
    parser.add_argument("--window-candidate-option-blend-grid", nargs="*", type=float, default=[1.0])
    parser.add_argument("--window-candidate-option-min-dwell-grid", nargs="*", type=int, default=[2])
    parser.add_argument("--window-candidate-option-freshness-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--window-candidate-option-transport-grid", nargs="*", type=float, default=[0.25])
    parser.add_argument("--window-candidate-option-power-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--window-candidate-option-switch-grid", nargs="*", type=float, default=[0.05])
    parser.add_argument("--window-candidate-min-soc-grid", nargs="*", type=float, default=[0.0])
    parser.add_argument("--window-candidate-macro-k-grid", nargs="*", type=int, default=[4])
    parser.add_argument("--window-candidate-macro-snippet-stride", type=int, default=1)
    parser.add_argument("--window-candidate-macro-max-lookahead", type=int, default=8)
    parser.add_argument("--window-candidate-rate-blend-grid", nargs="*", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--window-candidate-rate-freshness-grid", nargs="*", type=float, default=[0.0, 0.25])
    parser.add_argument("--window-candidate-rate-power-grid", nargs="*", type=float, default=[0.0, 0.03])
    parser.add_argument(
        "--window-candidate-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument(
        "--window-candidate-full-rollout-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Select window-candidate hyperparameters using full validation rollouts "
            "rather than only the local window length."
        ),
    )
    parser.add_argument("--include-knn-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--include-cost-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-cost-knn-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cost-knn-support-top-k", type=int, default=16)
    parser.add_argument("--cost-knn-k-grid", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--cost-knn-advantage-grid", nargs="*", type=float, default=[0.0, 0.01, 0.025, 0.05])
    parser.add_argument("--cost-knn-distance-weighting-grid", nargs="*", default=["inverse", "uniform"])
    parser.add_argument(
        "--cost-knn-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_risk",
    )
    parser.add_argument("--include-value-residual-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-ensemble-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-advantage-residual-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-rollout-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-sequence-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-recurrent-value-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-recurrent-advantage-policy", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--rollout-value-cost-target", choices=["teacher_beam", "executed_step"], default="teacher_beam")
    parser.add_argument("--rollout-value-random-rollouts", type=int, default=0)
    parser.add_argument("--rollout-value-depth", type=int, default=2)
    parser.add_argument("--rollout-value-beam-width", type=int, default=4)
    parser.add_argument("--rollout-value-max-branch", type=int, default=6)
    parser.add_argument("--rollout-value-discount", type=float, default=0.95)
    parser.add_argument("--rollout-value-self-iters", type=int, default=0)
    parser.add_argument("--rollout-value-self-steps", type=int, default=0)
    parser.add_argument("--rollout-value-self-threshold", type=float, default=0.0)
    parser.add_argument(
        "--rollout-value-advantage-grid",
        nargs="*",
        type=float,
        default=[-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument("--sequence-value-segment-len", type=int, default=8)
    parser.add_argument("--sequence-value-snippet-stride", type=int, default=4)
    parser.add_argument("--sequence-value-negatives-per-state", type=int, default=3)
    parser.add_argument("--sequence-value-max-rows", type=int, default=4096)
    parser.add_argument("--sequence-value-top-k-sequences", type=int, default=128)
    parser.add_argument("--sequence-value-augment-bank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequence-value-static-top-k", type=int, default=0)
    parser.add_argument("--sequence-value-cycle-support-top-k", type=int, default=0)
    parser.add_argument("--sequence-value-cycle-dwell-grid", nargs="*", type=int, default=[1, 2, 4])
    parser.add_argument("--sequence-value-cycle-max-sequences", type=int, default=512)
    parser.add_argument(
        "--sequence-value-advantage-grid",
        nargs="*",
        type=float,
        default=[-0.05, 0.0, 0.01, 0.025, 0.05, 0.1],
    )
    parser.add_argument("--recurrent-value-support-top-k", type=int, default=16)
    parser.add_argument("--recurrent-value-rank-weight", type=float, default=0.0)
    parser.add_argument("--recurrent-value-cost-dagger-iters", type=int, default=0)
    parser.add_argument("--recurrent-value-cost-dagger-threshold", type=float, default=0.0)
    parser.add_argument(
        "--recurrent-value-cost-dagger-steps",
        type=int,
        default=0,
        help="Steps per train start for recurrent cost-DAgger; 0 reuses --train-steps.",
    )
    parser.add_argument(
        "--recurrent-value-advantage-grid",
        nargs="*",
        type=float,
        default=[-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--recurrent-value-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_guard",
    )
    parser.add_argument("--recurrent-advantage-support-top-k", type=int, default=16)
    parser.add_argument("--recurrent-advantage-rank-weight", type=float, default=0.5)
    parser.add_argument(
        "--recurrent-advantage-grid",
        nargs="*",
        type=float,
        default=[-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0],
    )
    parser.add_argument(
        "--recurrent-advantage-calibration-criterion",
        choices=list(DEPLOYABLE_SELECTION_CRITERIA),
        default="static_margin_guard",
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
    parser.add_argument(
        "--deployable-selection-require-guard-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using static_margin_guard validation selection, fall back to the static anchor if no deployable passes the guard.",
    )
    parser.add_argument(
        "--deployable-selection-require-positive-center",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using static_margin_risk validation selection, fall back to the static anchor unless the selected deployable has positive mean/median static margin.",
    )
    parser.add_argument(
        "--deployable-selection-require-risk-band",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using static_margin_risk validation selection, also require q25 and negative-start risk-band checks.",
    )
    parser.add_argument("--deployable-selection-risk-min-q25-margin", type=float, default=-1.0e9)
    parser.add_argument("--deployable-selection-risk-max-negative-starts", type=int, default=1_000_000)
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
        augmented_truth_path = out_dir / "truth_with_learned_event_forecast.csv"
        truth.to_csv(augmented_truth_path, index=False)
        event_forecaster_summary = {
            "feature_columns": [str(x) for x in event_feature_columns],
            "probability_columns": [str(x) for x in learned_event_probability_columns],
            "augmented_truth_csv": str(augmented_truth_path),
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
    learned_continuous_prediction_columns: tuple[str, ...] = ()
    continuous_forecaster_summary: dict[str, object] | None = None
    forecast_continuous_columns = tuple(str(x) for x in args.forecast_continuous_columns)
    forecast_continuous_scales = tuple(float(x) for x in args.forecast_continuous_scales)
    if bool(args.learned_continuous_forecast):
        log("training split-compliant learned continuous forecaster")
        continuous_target_columns = tuple(str(x) for x in args.continuous_forecast_target_columns)
        if not continuous_target_columns:
            continuous_target_columns = forecast_continuous_columns
        if not continuous_target_columns:
            continuous_target_columns = tuple(str(x) for x in args.task_error_columns)
        if not continuous_target_columns:
            raise ValueError("--learned-continuous-forecast requires target columns")
        continuous_feature_candidates = tuple(str(x) for x in args.continuous_forecast_feature_columns)
        if not continuous_feature_candidates:
            continuous_feature_candidates = tuple([*state_columns, *learned_event_probability_columns])
        continuous_feature_columns = select_continuous_forecast_columns(
            truth,
            preferred_columns=continuous_feature_candidates,
        )
        continuous_forecast_cfg = ContinuousForecasterTrainingConfig(
            horizon=int(args.horizon),
            lookback=int(args.continuous_forecast_lookback),
            target_columns=continuous_target_columns,
            hidden_dim=int(args.continuous_forecast_hidden_dim),
            epochs=int(args.continuous_forecast_epochs),
            batch_size=int(args.continuous_forecast_batch_size),
            learning_rate=float(args.continuous_forecast_learning_rate),
            weight_decay=float(args.continuous_forecast_weight_decay),
            seed=int(args.seed),
            device=str(args.bc_device),
            prediction_prefix=str(args.continuous_forecast_prediction_prefix),
            period_steps=max(1, int(round(86400.0 / max(float(args.freq_s), 1.0)))),
        )
        continuous_forecast_dataset = build_continuous_forecast_dataset(
            truth,
            bounds=(int(bounds["oracle_pretrain"][0]), int(bounds["rl_train"][1])),
            feature_columns=continuous_feature_columns,
            target_columns=continuous_target_columns,
            cfg=continuous_forecast_cfg,
        )
        continuous_forecaster = train_continuous_forecaster(
            continuous_forecast_dataset,
            continuous_forecast_cfg,
        )
        truth, learned_continuous_prediction_columns = augment_truth_with_continuous_forecasts(
            truth,
            continuous_forecaster,
        )
        augmented_truth_path = out_dir / "truth_with_learned_continuous_forecast.csv"
        truth.to_csv(augmented_truth_path, index=False)
        if not forecast_continuous_columns:
            forecast_continuous_columns = continuous_target_columns
        if not forecast_continuous_scales:
            task_columns = tuple(str(x) for x in args.task_error_columns)
            task_scales = tuple(float(x) for x in args.task_error_scales) if args.task_error_scales else ()
            if forecast_continuous_columns == task_columns and len(task_scales) == len(forecast_continuous_columns):
                forecast_continuous_scales = task_scales
        continuous_forecaster_summary = {
            "feature_columns": [str(x) for x in continuous_feature_columns],
            "target_columns": [str(x) for x in continuous_target_columns],
            "prediction_columns": [str(x) for x in learned_continuous_prediction_columns],
            "prediction_prefix": str(args.continuous_forecast_prediction_prefix),
            "augmented_truth_csv": str(augmented_truth_path),
            "train_bounds": [int(bounds["oracle_pretrain"][0]), int(bounds["rl_train"][1])],
            "history": continuous_forecaster.history,
            "final_loss": float(continuous_forecaster.history["loss"][-1])
            if continuous_forecaster.history.get("loss")
            else None,
            "final_rmse": float(continuous_forecaster.history["rmse"][-1])
            if continuous_forecaster.history.get("rmse")
            else None,
        }
        log(
            "learned continuous forecaster complete: "
            f"targets={list(continuous_target_columns)} "
            f"columns={len(learned_continuous_prediction_columns)} "
            f"final_rmse={continuous_forecaster_summary['final_rmse']}"
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
        continuous_columns=forecast_continuous_columns,
        continuous_scales=forecast_continuous_scales,
        continuous_truth_future=bool(args.forecast_continuous_truth_future),
        learned_continuous_prefix=str(args.continuous_forecast_prediction_prefix),
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
    sequence_mask_model = None
    sequence_mask_history = None
    if bool(args.include_sequence_mask_policy):
        log("training sequence-mask BC policy")
        sequence_mask_model, sequence_mask_history = train_sequence_mask_bc(
            teacher_dataset.features,
            teacher_dataset.labels,
            teacher_dataset.candidate_masks,
            teacher_dataset.step_indices,
            cfg=bc_cfg,
        )
        log(
            "sequence-mask BC training complete: "
            f"final_sensor_accuracy={sequence_mask_history['sensor_accuracy'][-1] if sequence_mask_history and sequence_mask_history.get('sensor_accuracy') else float('nan')} "
            f"final_exact_match={sequence_mask_history['exact_match'][-1] if sequence_mask_history and sequence_mask_history.get('exact_match') else float('nan')}"
        )
    event_threshold_action_idx = None
    event_threshold_value = None
    event_threshold_aggregation = None
    event_threshold_validation_objective = None
    event_threshold_calibration_row = None
    event_support_cycle_indices = None
    event_support_cycle_value = None
    event_support_cycle_aggregation = None
    event_support_cycle_period = None
    event_support_cycle_selection = None
    event_support_cycle_validation_objective = None
    option_planner_support = None
    option_planner_target_rates = None
    option_planner_threshold = None
    option_planner_aggregation = None
    option_planner_min_dwell = None
    option_planner_cooldown = None
    option_planner_target_rate_weight = None
    option_planner_rate_balance_weight = None
    option_planner_freshness_weight = None
    option_planner_transport_weight = None
    option_planner_power_weight = None
    option_planner_switch_weight = None
    option_planner_min_soc = None
    option_planner_validation_objective = None
    option_planner_calibration_row = None
    macro_option_segment_len = None
    macro_option_k = None
    macro_option_threshold = None
    macro_option_aggregation = None
    macro_option_distance_weighting = None
    macro_option_refresh_interval = None
    macro_option_validation_objective = None
    macro_option_calibration_row = None
    macro_option_candidate_enabled = True
    improvement_gate_model = None
    improvement_gate_mean = None
    improvement_gate_std = None
    improvement_gate_history = None
    improvement_gate_segment_len = None
    improvement_gate_k = None
    improvement_gate_threshold = None
    improvement_gate_aggregation = None
    improvement_gate_distance_weighting = None
    improvement_gate_refresh_interval = None
    improvement_gate_validation_objective = None
    improvement_gate_calibration_row = None
    improvement_gate_candidate_enabled = True
    runtime_risk_threshold = None
    runtime_risk_aggregation = None
    runtime_risk_window_steps = None
    runtime_risk_event_weight = None
    runtime_risk_freshness_weight = None
    runtime_risk_transport_weight = None
    runtime_risk_soc_weight = None
    runtime_risk_min_soc = None
    runtime_risk_validation_objective = None
    runtime_risk_calibration_row = None
    window_eligibility_support = None
    window_eligibility_dynamic_family = None
    window_eligibility_macro_k = None
    window_eligibility_target_rates = None
    window_eligibility_features = None
    window_eligibility_margins = None
    window_eligibility_window_steps = None
    window_eligibility_k = None
    window_eligibility_margin_threshold = None
    window_eligibility_blend = None
    window_eligibility_min_dwell = None
    window_eligibility_freshness_weight = None
    window_eligibility_transport_weight = None
    window_eligibility_power_weight = None
    window_eligibility_switch_weight = None
    window_eligibility_min_soc = None
    window_eligibility_distance_weighting = None
    window_eligibility_validation_objective = None
    window_eligibility_calibration_row = None
    window_eligibility_candidate_enabled = True
    window_candidate_support = None
    window_candidate_specs = None
    window_candidate_features = None
    window_candidate_margins = None
    window_candidate_ids = None
    window_candidate_window_steps = None
    window_candidate_k = None
    window_candidate_margin_threshold = None
    window_candidate_score_quantile = None
    window_candidate_distance_weighting = None
    window_candidate_min_soc = None
    window_candidate_validation_objective = None
    window_candidate_calibration_row = None
    window_candidate_candidate_enabled = True
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
    contextual_duty_calibration_row = None
    utility_planner_support = None
    utility_planner_target_rates = None
    utility_planner_event_weight = None
    utility_planner_magnitude_weight = None
    utility_planner_variability_weight = None
    utility_planner_freshness_weight = None
    utility_planner_target_rate_weight = None
    utility_planner_anchor_bias = None
    utility_planner_power_weight = None
    utility_planner_switch_weight = None
    utility_planner_min_soc = None
    utility_planner_min_dwell = None
    utility_planner_aggregation = None
    utility_planner_validation_objective = None
    utility_planner_calibration_row = None
    utility_planner_candidate_enabled = True
    proxy_mpc_support = None
    proxy_mpc_target_rates = None
    proxy_mpc_event_weight = None
    proxy_mpc_magnitude_weight = None
    proxy_mpc_variability_weight = None
    proxy_mpc_freshness_weight = None
    proxy_mpc_target_rate_weight = None
    proxy_mpc_anchor_bias = None
    proxy_mpc_power_weight = None
    proxy_mpc_switch_weight = None
    proxy_mpc_min_soc = None
    proxy_mpc_min_dwell = None
    proxy_mpc_aggregation = None
    proxy_mpc_planning_depth = None
    proxy_mpc_beam_width = None
    proxy_mpc_max_branch = None
    proxy_mpc_age_weight = None
    proxy_mpc_anchor_improvement_threshold = None
    proxy_mpc_validation_objective = None
    proxy_mpc_calibration_row = None
    proxy_mpc_candidate_enabled = True
    sequence_mask_support = None
    sequence_mask_anchor_bias = None
    sequence_mask_power_weight = None
    sequence_mask_validation_objective = None
    sequence_mask_calibration_row = None
    validation_cyclic_action_indices = None
    validation_cyclic_dwell = None
    validation_cyclic_validation_objective = None
    validation_cyclic_calibration_rows = []
    if bool(args.include_validation_cyclic_policy):
        (
            validation_cyclic_action_indices,
            validation_cyclic_dwell,
            validation_cyclic_validation_objective,
            validation_cyclic_calibration_rows,
        ) = calibrate_validation_cyclic_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            validation_static_table=validation_static_table,
            selected_static_idx=selected_static_idx,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        validation_cyclic_ids = [
            "|".join(sensor_ids[idx] for idx in np.flatnonzero(candidate_masks[int(action_idx)]))
            for action_idx in validation_cyclic_action_indices
        ] if validation_cyclic_action_indices is not None else []
        log(
            "validation-cyclic calibration: "
            f"actions={list(validation_cyclic_action_indices) if validation_cyclic_action_indices is not None else None} "
            f"ids={validation_cyclic_ids} dwell={validation_cyclic_dwell} "
            f"validation_objective={validation_cyclic_validation_objective:.6f}"
        )
        pd.DataFrame(validation_cyclic_calibration_rows).to_csv(
            out_dir / "validation_cyclic_calibration.csv",
            index=False,
        )
    if bool(args.include_event_threshold_policy):
        (
            event_threshold_action_idx,
            event_threshold_value,
            event_threshold_aggregation,
            event_threshold_validation_objective,
            event_threshold_calibration_row,
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
    if bool(args.include_option_planner_policy) or bool(args.include_runtime_risk_guard_policy):
        (
            option_planner_support,
            option_planner_target_rates,
            option_planner_threshold,
            option_planner_aggregation,
            option_planner_min_dwell,
            option_planner_cooldown,
            option_planner_target_rate_weight,
            option_planner_rate_balance_weight,
            option_planner_freshness_weight,
            option_planner_transport_weight,
            option_planner_power_weight,
            option_planner_switch_weight,
            option_planner_min_soc,
            option_planner_validation_objective,
            option_planner_calibration_row,
        ) = calibrate_option_planner_policy(
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
        option_ids = []
        if option_planner_support is not None:
            for action_idx in option_planner_support:
                option_ids.append("|".join(sensor_ids[idx] for idx in np.flatnonzero(candidate_masks[int(action_idx)])))
        log(
            "option-planner calibration: "
            f"actions={option_planner_support} ids={option_ids} "
            f"aggregation={option_planner_aggregation} threshold={option_planner_threshold} "
            f"min_dwell={option_planner_min_dwell} cooldown={option_planner_cooldown} "
            f"target_rate_weight={option_planner_target_rate_weight} "
            f"rate_balance_weight={option_planner_rate_balance_weight} "
            f"freshness_weight={option_planner_freshness_weight} "
            f"transport_weight={option_planner_transport_weight} power_weight={option_planner_power_weight} "
            f"switch_weight={option_planner_switch_weight} min_soc={option_planner_min_soc} "
            f"validation_objective={option_planner_validation_objective:.6f} "
            f"criterion={args.option_planner_calibration_criterion}"
        )
    if bool(args.include_macro_option_policy):
        (
            macro_option_segment_len,
            macro_option_k,
            macro_option_threshold,
            macro_option_aggregation,
            macro_option_distance_weighting,
            macro_option_refresh_interval,
            macro_option_validation_objective,
            macro_option_calibration_row,
        ) = calibrate_macro_option_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            features=teacher_dataset.features,
            labels=teacher_dataset.labels,
            step_indices=teacher_dataset.step_indices,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "macro-option calibration: "
            f"segment_len={macro_option_segment_len} k={macro_option_k} "
            f"threshold={macro_option_threshold} aggregation={macro_option_aggregation} "
            f"distance_weighting={macro_option_distance_weighting} "
            f"refresh_interval={macro_option_refresh_interval} "
            f"validation_objective={macro_option_validation_objective:.6f} "
            f"criterion={args.macro_option_calibration_criterion}"
        )
    if bool(args.include_teacher_improvement_gate_policy):
        log("collecting teacher-improvement gate labels")
        improvement_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        gate_features, gate_labels, gate_margins = collect_teacher_improvement_gate_dataset(
            improvement_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            anchor_idx=int(selected_static_idx),
            label_margin=float(args.teacher_improvement_gate_label_margin),
        )
        pd.DataFrame(
            {
                "label": gate_labels.astype(float),
                "margin": gate_margins.astype(float),
            }
        ).to_csv(out_dir / "teacher_improvement_gate_training_labels.csv", index=False)
        gate_cfg = BCTrainingConfig(
            hidden_dim=int(args.teacher_improvement_gate_hidden_dim),
            epochs=int(args.teacher_improvement_gate_epochs),
            batch_size=int(args.bc_batch_size),
            learning_rate=float(args.bc_learning_rate),
            weight_decay=float(args.bc_weight_decay),
            seed=int(args.seed) + 17,
            device=str(args.bc_device),
        )
        improvement_gate_model, improvement_gate_history, improvement_gate_mean, improvement_gate_std = train_binary_gate(
            gate_features,
            gate_labels,
            cfg=gate_cfg,
        )
        log(
            "teacher-improvement gate trained: "
            f"positive_rate={float(np.mean(gate_labels)):.4f} "
            f"final_accuracy={improvement_gate_history['accuracy'][-1] if improvement_gate_history.get('accuracy') else float('nan')}"
        )
        (
            improvement_gate_segment_len,
            improvement_gate_k,
            improvement_gate_threshold,
            improvement_gate_aggregation,
            improvement_gate_distance_weighting,
            improvement_gate_refresh_interval,
            improvement_gate_validation_objective,
            improvement_gate_calibration_row,
        ) = calibrate_teacher_improvement_gate_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            gate_model=improvement_gate_model,
            feature_mean=improvement_gate_mean,
            feature_std=improvement_gate_std,
            features=teacher_dataset.features,
            labels=teacher_dataset.labels,
            step_indices=teacher_dataset.step_indices,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "teacher-improvement gate calibration: "
            f"segment_len={improvement_gate_segment_len} k={improvement_gate_k} "
            f"threshold={improvement_gate_threshold} aggregation={improvement_gate_aggregation} "
            f"distance_weighting={improvement_gate_distance_weighting} "
            f"refresh_interval={improvement_gate_refresh_interval} "
            f"validation_objective={improvement_gate_validation_objective:.6f}"
        )
    if bool(args.include_runtime_risk_guard_policy) and option_planner_support is not None:
        (
            runtime_risk_threshold,
            runtime_risk_aggregation,
            runtime_risk_window_steps,
            runtime_risk_event_weight,
            runtime_risk_freshness_weight,
            runtime_risk_transport_weight,
            runtime_risk_soc_weight,
            runtime_risk_min_soc,
            runtime_risk_validation_objective,
            runtime_risk_calibration_row,
        ) = calibrate_runtime_risk_guard_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            option_action_indices=tuple(int(x) for x in option_planner_support),
            option_target_rates=option_planner_target_rates,
            option_threshold=float(option_planner_threshold if option_planner_threshold is not None else 1.0),
            option_aggregation=str(option_planner_aggregation or "max"),
            option_min_dwell=int(option_planner_min_dwell if option_planner_min_dwell is not None else 1),
            option_cooldown=int(option_planner_cooldown if option_planner_cooldown is not None else 0),
            option_target_rate_weight=float(
                option_planner_target_rate_weight if option_planner_target_rate_weight is not None else 1.0
            ),
            option_rate_balance_weight=float(
                option_planner_rate_balance_weight if option_planner_rate_balance_weight is not None else 0.0
            ),
            option_freshness_weight=float(
                option_planner_freshness_weight if option_planner_freshness_weight is not None else 0.25
            ),
            option_transport_weight=float(
                option_planner_transport_weight if option_planner_transport_weight is not None else 0.25
            ),
            option_power_weight=float(option_planner_power_weight if option_planner_power_weight is not None else 0.05),
            option_switch_weight=float(
                option_planner_switch_weight if option_planner_switch_weight is not None else 0.05
            ),
            option_min_soc=float(option_planner_min_soc if option_planner_min_soc is not None else 0.0),
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "runtime-risk guard calibration: "
            f"threshold={runtime_risk_threshold} aggregation={runtime_risk_aggregation} "
            f"window_steps={runtime_risk_window_steps} event_weight={runtime_risk_event_weight} "
            f"freshness_weight={runtime_risk_freshness_weight} "
            f"transport_weight={runtime_risk_transport_weight} soc_weight={runtime_risk_soc_weight} "
            f"min_soc={runtime_risk_min_soc} validation_objective={runtime_risk_validation_objective:.6f} "
            f"criterion={args.runtime_risk_guard_calibration_criterion}"
        )
    if bool(args.include_window_eligibility_policy):
        (
            window_eligibility_support,
            window_eligibility_dynamic_family,
            window_eligibility_macro_k,
            window_eligibility_target_rates,
            window_eligibility_features,
            window_eligibility_margins,
            window_eligibility_window_steps,
            window_eligibility_k,
            window_eligibility_margin_threshold,
            window_eligibility_blend,
            window_eligibility_min_dwell,
            window_eligibility_freshness_weight,
            window_eligibility_transport_weight,
            window_eligibility_power_weight,
            window_eligibility_switch_weight,
            window_eligibility_min_soc,
            window_eligibility_distance_weighting,
            window_eligibility_validation_objective,
            window_eligibility_calibration_row,
        ) = calibrate_window_eligibility_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            train_cfg=train_cfg,
            validation_cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            features=teacher_dataset.features,
            labels=teacher_dataset.labels,
            step_indices=teacher_dataset.step_indices,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            train_starts=starts["train"].starts,
            validation_starts=starts["validation"].starts,
        )
        if window_eligibility_calibration_row is None:
            window_eligibility_candidate_enabled = False
            log("window-eligibility candidate disabled: no validation row passed deployable selection guard")
        log(
            "window-eligibility calibration: "
            f"family={window_eligibility_dynamic_family} macro_k={window_eligibility_macro_k} "
            f"support={list(window_eligibility_support) if window_eligibility_support is not None else None} "
            f"window={window_eligibility_window_steps} k={window_eligibility_k} "
            f"threshold={window_eligibility_margin_threshold} blend={window_eligibility_blend} "
            f"min_dwell={window_eligibility_min_dwell} "
            f"freshness_weight={window_eligibility_freshness_weight} "
            f"transport_weight={window_eligibility_transport_weight} "
            f"power_weight={window_eligibility_power_weight} "
            f"switch_weight={window_eligibility_switch_weight} "
            f"min_soc={window_eligibility_min_soc} "
            f"distance_weighting={window_eligibility_distance_weighting} "
            f"validation_objective={window_eligibility_validation_objective:.6f}"
        )
    if bool(args.include_window_candidate_policy):
        (
            window_candidate_support,
            window_candidate_specs,
            window_candidate_features,
            window_candidate_margins,
            window_candidate_ids,
            window_candidate_window_steps,
            window_candidate_k,
            window_candidate_margin_threshold,
            window_candidate_score_quantile,
            window_candidate_distance_weighting,
            window_candidate_min_soc,
            window_candidate_validation_objective,
            window_candidate_calibration_row,
        ) = calibrate_window_candidate_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            train_cfg=train_cfg,
            validation_cfg=validation_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            features=teacher_dataset.features,
            labels=teacher_dataset.labels,
            step_indices=teacher_dataset.step_indices,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            train_starts=starts["train"].starts,
            validation_starts=starts["validation"].starts,
        )
        if window_candidate_calibration_row is None:
            window_candidate_candidate_enabled = False
            log("window-candidate candidate disabled: no validation row passed deployable selection guard")
        log(
            "window-candidate calibration: "
            f"support={list(window_candidate_support) if window_candidate_support is not None else None} "
            f"candidates={len(window_candidate_specs) if window_candidate_specs is not None else 0} "
            f"window={window_candidate_window_steps} k={window_candidate_k} "
            f"threshold={window_candidate_margin_threshold} "
            f"quantile={window_candidate_score_quantile} "
            f"distance_weighting={window_candidate_distance_weighting} "
            f"validation_objective={window_candidate_validation_objective:.6f}"
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
            contextual_duty_calibration_row,
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
            f"validation_objective={contextual_duty_validation_objective:.6f} "
            f"criterion={args.contextual_duty_calibration_criterion}"
        )
    if bool(args.include_utility_planner_policy):
        (
            utility_planner_support,
            utility_planner_target_rates,
            utility_planner_event_weight,
            utility_planner_magnitude_weight,
            utility_planner_variability_weight,
            utility_planner_freshness_weight,
            utility_planner_target_rate_weight,
            utility_planner_anchor_bias,
            utility_planner_power_weight,
            utility_planner_switch_weight,
            utility_planner_min_soc,
            utility_planner_min_dwell,
            utility_planner_aggregation,
            utility_planner_validation_objective,
            utility_planner_calibration_row,
        ) = calibrate_utility_planner_policy(
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
        if utility_planner_calibration_row is None:
            utility_planner_candidate_enabled = False
            log("utility-planner candidate disabled: no validation row passed deployable selection guard")
        log(
            "utility-planner calibration: "
            f"support={list(utility_planner_support) if utility_planner_support is not None else None} "
            f"event_weight={utility_planner_event_weight} "
            f"magnitude_weight={utility_planner_magnitude_weight} "
            f"variability_weight={utility_planner_variability_weight} "
            f"freshness_weight={utility_planner_freshness_weight} "
            f"target_rate_weight={utility_planner_target_rate_weight} "
            f"anchor_bias={utility_planner_anchor_bias} "
            f"power_weight={utility_planner_power_weight} "
            f"switch_weight={utility_planner_switch_weight} "
            f"min_soc={utility_planner_min_soc} "
            f"min_dwell={utility_planner_min_dwell} "
            f"aggregation={utility_planner_aggregation} "
            f"validation_objective={utility_planner_validation_objective:.6f} "
            f"criterion={args.utility_planner_calibration_criterion}"
        )
    if bool(args.include_proxy_mpc_policy):
        (
            proxy_mpc_support,
            proxy_mpc_target_rates,
            proxy_mpc_event_weight,
            proxy_mpc_magnitude_weight,
            proxy_mpc_variability_weight,
            proxy_mpc_freshness_weight,
            proxy_mpc_target_rate_weight,
            proxy_mpc_anchor_bias,
            proxy_mpc_power_weight,
            proxy_mpc_switch_weight,
            proxy_mpc_min_soc,
            proxy_mpc_min_dwell,
            proxy_mpc_aggregation,
            proxy_mpc_planning_depth,
            proxy_mpc_beam_width,
            proxy_mpc_max_branch,
            proxy_mpc_age_weight,
            proxy_mpc_anchor_improvement_threshold,
            proxy_mpc_validation_objective,
            proxy_mpc_calibration_row,
        ) = calibrate_proxy_mpc_policy(
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
        if proxy_mpc_calibration_row is None:
            proxy_mpc_candidate_enabled = False
            log("proxy-MPC candidate disabled: no validation row passed deployable selection guard")
        log(
            "proxy-MPC calibration: "
            f"support={list(proxy_mpc_support) if proxy_mpc_support is not None else None} "
            f"event_weight={proxy_mpc_event_weight} "
            f"magnitude_weight={proxy_mpc_magnitude_weight} "
            f"variability_weight={proxy_mpc_variability_weight} "
            f"freshness_weight={proxy_mpc_freshness_weight} "
            f"target_rate_weight={proxy_mpc_target_rate_weight} "
            f"anchor_bias={proxy_mpc_anchor_bias} "
            f"power_weight={proxy_mpc_power_weight} "
            f"switch_weight={proxy_mpc_switch_weight} "
            f"min_soc={proxy_mpc_min_soc} "
            f"min_dwell={proxy_mpc_min_dwell} "
            f"aggregation={proxy_mpc_aggregation} "
            f"depth={proxy_mpc_planning_depth} "
            f"beam_width={proxy_mpc_beam_width} "
            f"max_branch={proxy_mpc_max_branch} "
            f"age_weight={proxy_mpc_age_weight} "
            f"anchor_improvement={proxy_mpc_anchor_improvement_threshold} "
            f"validation_objective={proxy_mpc_validation_objective:.6f} "
            f"criterion={args.proxy_mpc_calibration_criterion}"
        )
    if bool(args.include_sequence_mask_policy):
        if sequence_mask_model is None:
            raise RuntimeError("sequence-mask policy requires a trained sequence mask model")
        (
            sequence_mask_support,
            sequence_mask_anchor_bias,
            sequence_mask_power_weight,
            sequence_mask_validation_objective,
            sequence_mask_calibration_row,
        ) = calibrate_sequence_mask_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=sequence_mask_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            labels=teacher_dataset.labels,
            anchor_idx=selected_static_idx,
            anchor_mask=selected_static_mask,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "sequence-mask calibration: "
            f"support={list(sequence_mask_support) if sequence_mask_support is not None else None} "
            f"anchor_bias={sequence_mask_anchor_bias} "
            f"power_weight={sequence_mask_power_weight} "
            f"validation_objective={sequence_mask_validation_objective:.6f} "
            f"criterion={args.sequence_mask_calibration_criterion}"
        )
    cost_model = None
    cost_history = None
    cost_ensemble_models = None
    cost_ensemble_histories = None
    rollout_value_cost_model = None
    rollout_value_cost_history = None
    rollout_value_candidate_enabled = True
    recurrent_value_model = None
    recurrent_value_history = None
    recurrent_value_support = None
    recurrent_value_threshold = None
    recurrent_value_validation_objective = None
    recurrent_value_calibration_row = None
    recurrent_value_candidate_enabled = True
    cost_knn_dataset = None
    cost_knn_support = None
    cost_knn_k = None
    cost_knn_advantage_threshold = None
    cost_knn_distance_weighting = None
    cost_knn_validation_objective = None
    cost_knn_calibration_row = None
    cost_knn_candidate_enabled = True
    recurrent_advantage_model = None
    recurrent_advantage_history = None
    recurrent_advantage_support = None
    recurrent_advantage_threshold = None
    recurrent_advantage_validation_objective = None
    recurrent_advantage_calibration_row = None
    recurrent_advantage_candidate_enabled = True
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
    advantage_residual_calibration_row = None
    advantage_residual_candidate_enabled = True
    rollout_value_transition_model = None
    rollout_value_transition_history = None
    rollout_value_support = None
    rollout_value_threshold = None
    rollout_value_validation_objective = None
    rollout_value_calibration_row = None
    sequence_value_model = None
    sequence_value_history = None
    sequence_value_dataset = None
    sequence_value_threshold = None
    sequence_value_validation_objective = None
    sequence_value_calibration_row = None
    sequence_value_candidate_enabled = True
    sequence_value_extra_bank = None
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
    if bool(args.include_cost_knn_policy):
        cost_knn_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.cost_knn_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        log("collecting cost-KNN teacher cost memory")
        cost_knn_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        cost_knn_dataset = collect_recurrent_action_cost_dataset(
            cost_knn_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            allowed_action_indices=cost_knn_support,
            anchor_mask=selected_static_mask,
        )
        log(f"cost-KNN memory collected: rows={cost_knn_dataset.features.shape[0]}")
        (
            cost_knn_k,
            cost_knn_advantage_threshold,
            cost_knn_distance_weighting,
            cost_knn_validation_objective,
            cost_knn_calibration_row,
        ) = calibrate_cost_knn_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            dataset=cost_knn_dataset,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=cost_knn_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "cost-KNN calibration: "
            f"k={cost_knn_k} threshold={cost_knn_advantage_threshold} "
            f"distance_weighting={cost_knn_distance_weighting} "
            f"validation_objective={cost_knn_validation_objective:.6f} "
            f"support={list(cost_knn_support) if cost_knn_support is not None else None} "
            f"criterion={args.cost_knn_calibration_criterion}"
        )
        if (
            str(args.cost_knn_calibration_criterion) == "static_margin_guard"
            and float(args.deployable_selection_min_mean_margin) > 0.0
            and cost_knn_calibration_row is not None
            and not bool(cost_knn_calibration_row.get("static_margin_guard_pass", False))
        ):
            cost_knn_candidate_enabled = False
            log("cost-KNN candidate disabled: calibration failed positive static-margin guard")
    if bool(args.include_recurrent_value_policy):
        recurrent_value_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.recurrent_value_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        log("collecting recurrent action-cost dataset")
        recurrent_cost_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        recurrent_cost_dataset = collect_recurrent_action_cost_dataset(
            recurrent_cost_env,
            candidate_masks,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            allowed_action_indices=recurrent_value_support,
            anchor_mask=selected_static_mask,
        )
        log(f"recurrent action-cost dataset collected: rows={recurrent_cost_dataset.features.shape[0]}")
        recurrent_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 31,
            device=str(args.bc_device),
            rank_weight=float(args.recurrent_value_rank_weight),
        )
        recurrent_value_model, recurrent_value_history = train_recurrent_action_cost_model(
            recurrent_cost_dataset,
            recurrent_train_cfg,
        )
        log(
            "recurrent action-cost model training complete: "
            f"final_loss={recurrent_value_history['loss'][-1] if recurrent_value_history and recurrent_value_history.get('loss') else float('nan')} "
            f"final_best_action_accuracy={recurrent_value_history['best_action_accuracy'][-1] if recurrent_value_history and recurrent_value_history.get('best_action_accuracy') else float('nan')}"
        )
        for cost_dagger_iter in range(max(0, int(args.recurrent_value_cost_dagger_iters))):
            log(f"collecting recurrent cost-DAgger dataset iter={cost_dagger_iter + 1}")
            cost_dagger_policy = ForecastAwareRecurrentValuePolicy(
                model=recurrent_value_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=recurrent_value_support,
                advantage_threshold=float(args.recurrent_value_cost_dagger_threshold),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_recurrent_value_cost_dagger_{cost_dagger_iter + 1}",
            )
            recurrent_cost_dagger_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
            recurrent_cost_dagger_dataset = collect_recurrent_action_cost_dataset(
                recurrent_cost_dagger_env,
                candidate_masks,
                start_indices=starts["train"].starts,
                steps_per_start=int(args.recurrent_value_cost_dagger_steps or args.train_steps),
                teacher_cfg=teacher_cfg,
                forecast_cfg=forecast_cfg,
                allowed_action_indices=recurrent_value_support,
                anchor_mask=selected_static_mask,
                rollout_policy=cost_dagger_policy,
            )
            recurrent_cost_dataset = concat_recurrent_action_cost_datasets(
                [recurrent_cost_dataset, recurrent_cost_dagger_dataset]
            )
            log(
                "recurrent cost-DAgger dataset merged: "
                f"iter={cost_dagger_iter + 1} rows={recurrent_cost_dataset.features.shape[0]}"
            )
            recurrent_value_model, recurrent_value_history = train_recurrent_action_cost_model(
                recurrent_cost_dataset,
                recurrent_train_cfg,
            )
            log(
                "recurrent cost-DAgger model training complete: "
                f"iter={cost_dagger_iter + 1} "
                f"final_loss={recurrent_value_history['loss'][-1] if recurrent_value_history and recurrent_value_history.get('loss') else float('nan')} "
                f"final_best_action_accuracy={recurrent_value_history['best_action_accuracy'][-1] if recurrent_value_history and recurrent_value_history.get('best_action_accuracy') else float('nan')}"
            )
        (
            recurrent_value_threshold,
            recurrent_value_validation_objective,
            recurrent_value_calibration_row,
        ) = calibrate_recurrent_value_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=recurrent_value_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=recurrent_value_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "recurrent-value calibration: "
            f"threshold={recurrent_value_threshold} "
            f"validation_objective={recurrent_value_validation_objective:.6f} "
            f"support={list(recurrent_value_support) if recurrent_value_support is not None else None} "
            f"criterion={args.recurrent_value_calibration_criterion}"
        )
        if (
            str(args.recurrent_value_calibration_criterion) == "static_margin_guard"
            and float(args.deployable_selection_min_mean_margin) > 0.0
            and recurrent_value_calibration_row is not None
            and not bool(recurrent_value_calibration_row.get("static_margin_guard_pass", False))
        ):
            recurrent_value_candidate_enabled = False
            log("recurrent-value candidate disabled: calibration failed positive static-margin guard")
    if bool(args.include_recurrent_advantage_policy):
        recurrent_advantage_support = action_support_from_labels(
            teacher_dataset.labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(args.recurrent_advantage_support_top_k),
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=selected_static_idx,
        )
        log("collecting recurrent anchor-advantage dataset")
        recurrent_advantage_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        recurrent_advantage_dataset = collect_recurrent_anchor_advantage_dataset(
            recurrent_advantage_env,
            candidate_masks,
            anchor_mask=selected_static_mask,
            start_indices=starts["train"].starts,
            steps_per_start=int(args.train_steps),
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            allowed_action_indices=recurrent_advantage_support,
        )
        log(
            "recurrent anchor-advantage dataset collected: "
            f"rows={recurrent_advantage_dataset.features.shape[0]}"
        )
        recurrent_advantage_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 47,
            device=str(args.bc_device),
            rank_weight=float(args.recurrent_advantage_rank_weight),
        )
        recurrent_advantage_model, recurrent_advantage_history = train_recurrent_anchor_advantage_model(
            recurrent_advantage_dataset,
            recurrent_advantage_train_cfg,
        )
        log(
            "recurrent anchor-advantage model training complete: "
            f"final_loss={recurrent_advantage_history['loss'][-1] if recurrent_advantage_history and recurrent_advantage_history.get('loss') else float('nan')} "
            f"final_best_action_accuracy={recurrent_advantage_history['best_action_accuracy'][-1] if recurrent_advantage_history and recurrent_advantage_history.get('best_action_accuracy') else float('nan')}"
        )
        (
            recurrent_advantage_threshold,
            recurrent_advantage_validation_objective,
            recurrent_advantage_calibration_row,
        ) = calibrate_recurrent_advantage_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=recurrent_advantage_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            allowed_action_indices=recurrent_advantage_support,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "recurrent-advantage calibration: "
            f"threshold={recurrent_advantage_threshold} "
            f"validation_objective={recurrent_advantage_validation_objective:.6f} "
            f"support={list(recurrent_advantage_support) if recurrent_advantage_support is not None else None} "
            f"criterion={args.recurrent_advantage_calibration_criterion}"
        )
        if (
            str(args.recurrent_advantage_calibration_criterion) == "static_margin_guard"
            and float(args.deployable_selection_min_mean_margin) > 0.0
            and recurrent_advantage_calibration_row is not None
            and not bool(recurrent_advantage_calibration_row.get("static_margin_guard_pass", False))
        ):
            recurrent_advantage_candidate_enabled = False
            log("recurrent-advantage candidate disabled: calibration failed positive static-margin guard")
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
            advantage_residual_calibration_row,
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
            f"support={list(advantage_residual_support) if advantage_residual_support is not None else None} "
            f"row={advantage_residual_calibration_row}"
        )
        if advantage_residual_calibration_row is None:
            advantage_residual_candidate_enabled = False
            log("advantage-residual candidate disabled: no validation row passed deployable selection guard")
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
        if str(args.rollout_value_cost_target) == "executed_step":
            log("collecting executed-step twin datasets for rollout planner")
            twin_cost_datasets = []
            twin_transition_datasets = []
            twin_sources = [
                (
                    "static_anchor",
                    StaticMaskPolicy(tuple(bool(x) for x in selected_static_mask), name="twin_collect_static_anchor"),
                    int(args.seed) + 31,
                ),
                (
                    "mpc_teacher",
                    MpcTeacherPolicy(candidate_masks=candidate_masks, cfg=teacher_cfg, name="twin_collect_mpc_teacher"),
                    int(args.seed) + 37,
                ),
            ]
            for source_name, source_policy, source_seed in twin_sources:
                source_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
                cost_part, transition_part = collect_executed_outcome_datasets(
                    source_env,
                    candidate_masks,
                    start_indices=starts["train"].starts,
                    steps_per_start=int(args.train_steps),
                    teacher_cfg=teacher_cfg,
                    forecast_cfg=forecast_cfg,
                    rollout_policy=source_policy,
                    allowed_action_indices=rollout_value_support,
                    anchor_mask=selected_static_mask,
                    seed=int(source_seed),
                )
                twin_cost_datasets.append(cost_part)
                twin_transition_datasets.append(transition_part)
                log(
                    "executed-step twin rows: "
                    f"source={source_name} rows={cost_part.inputs.shape[0]}"
                )
            for random_iter in range(max(0, int(args.rollout_value_random_rollouts))):
                random_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
                cost_part, transition_part = collect_executed_outcome_datasets(
                    random_env,
                    candidate_masks,
                    start_indices=starts["train"].starts,
                    steps_per_start=int(args.train_steps),
                    teacher_cfg=teacher_cfg,
                    forecast_cfg=forecast_cfg,
                    rollout_policy=None,
                    allowed_action_indices=rollout_value_support,
                    anchor_mask=selected_static_mask,
                    seed=int(args.seed) + 41 + 997 * int(random_iter),
                )
                twin_cost_datasets.append(cost_part)
                twin_transition_datasets.append(transition_part)
                log(
                    "executed-step twin rows: "
                    f"source=random_{random_iter + 1} rows={cost_part.inputs.shape[0]}"
                )
            rollout_cost_dataset = concat_action_cost_datasets(twin_cost_datasets)
            transition_dataset = concat_feature_transition_datasets(twin_transition_datasets)
            log(
                "executed-step twin datasets collected: "
                f"cost_rows={rollout_cost_dataset.inputs.shape[0]} "
                f"transition_rows={transition_dataset.inputs.shape[0]}"
            )
        else:
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
        for self_iter in range(max(0, int(args.rollout_value_self_iters))):
            self_steps = int(args.rollout_value_self_steps) if int(args.rollout_value_self_steps) > 0 else int(args.train_steps)
            log(
                "collecting rollout-value self-distribution datasets: "
                f"iter={self_iter + 1} steps={self_steps}"
            )
            self_policy = ForecastAwareRolloutValuePolicy(
                cost_model=rollout_value_cost_model,
                transition_model=rollout_value_transition_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=rollout_value_support,
                advantage_threshold=float(args.rollout_value_self_threshold),
                planning_depth=int(args.rollout_value_depth),
                beam_width=int(args.rollout_value_beam_width),
                max_branch=int(args.rollout_value_max_branch),
                discount=float(args.rollout_value_discount),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_rollout_value_self_collect_{self_iter + 1}",
            )
            if str(args.rollout_value_cost_target) == "executed_step":
                self_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
                self_cost_dataset, self_transition_dataset = collect_executed_outcome_datasets(
                    self_env,
                    candidate_masks,
                    start_indices=starts["train"].starts,
                    steps_per_start=self_steps,
                    teacher_cfg=teacher_cfg,
                    forecast_cfg=forecast_cfg,
                    rollout_policy=self_policy,
                    allowed_action_indices=rollout_value_support,
                    anchor_mask=selected_static_mask,
                    seed=int(args.seed) + 53 + 997 * int(self_iter),
                )
            else:
                self_cost_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
                self_cost_dataset = collect_action_cost_dataset(
                    self_cost_env,
                    candidate_masks,
                    start_indices=starts["train"].starts,
                    steps_per_start=self_steps,
                    teacher_cfg=teacher_cfg,
                    forecast_cfg=forecast_cfg,
                    normalize_costs=False,
                    allowed_action_indices=rollout_value_support,
                    anchor_mask=selected_static_mask,
                    rollout_policy=self_policy,
                )
                self_transition_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
                self_transition_dataset = collect_feature_transition_dataset(
                    self_transition_env,
                    candidate_masks,
                    start_indices=starts["train"].starts,
                    steps_per_start=self_steps,
                    teacher_cfg=teacher_cfg,
                    forecast_cfg=forecast_cfg,
                    allowed_action_indices=rollout_value_support,
                    anchor_mask=selected_static_mask,
                    rollout_policy=self_policy,
                )
            rollout_cost_dataset = concat_action_cost_datasets([rollout_cost_dataset, self_cost_dataset])
            transition_dataset = concat_feature_transition_datasets([transition_dataset, self_transition_dataset])
            log(
                "rollout-value self-distribution rows: "
                f"iter={self_iter + 1} cost_rows={self_cost_dataset.inputs.shape[0]} "
                f"transition_rows={self_transition_dataset.inputs.shape[0]} "
                f"combined_cost_rows={rollout_cost_dataset.inputs.shape[0]} "
                f"combined_transition_rows={transition_dataset.inputs.shape[0]}"
            )
            rollout_cost_train_cfg = ActionCostTrainingConfig(
                hidden_dim=int(args.cost_hidden_dim),
                epochs=int(args.cost_epochs),
                batch_size=512,
                seed=int(args.seed) + 23 + 211 * (self_iter + 1),
                device=str(args.bc_device),
            )
            rollout_value_cost_model, rollout_value_cost_history = train_action_cost_model(
                rollout_cost_dataset,
                rollout_cost_train_cfg,
            )
            transition_train_cfg = ActionCostTrainingConfig(
                hidden_dim=int(args.cost_hidden_dim),
                epochs=int(args.cost_epochs),
                batch_size=512,
                seed=int(args.seed) + 29 + 211 * (self_iter + 1),
                device=str(args.bc_device),
            )
            rollout_value_transition_model, rollout_value_transition_history = train_feature_transition_model(
                transition_dataset,
                transition_train_cfg,
            )
            log(
                "rollout-value self-distribution retrain complete: "
                f"iter={self_iter + 1} "
                f"cost_loss={rollout_value_cost_history['loss'][-1] if rollout_value_cost_history and rollout_value_cost_history.get('loss') else float('nan')} "
                f"transition_loss={rollout_value_transition_history['loss'][-1] if rollout_value_transition_history and rollout_value_transition_history.get('loss') else float('nan')}"
            )
        (
            rollout_value_threshold,
            rollout_value_validation_objective,
            rollout_value_calibration_row,
        ) = calibrate_rollout_value_policy(
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
            f"support={list(rollout_value_support) if rollout_value_support is not None else None} "
            f"row={rollout_value_calibration_row}"
        )
        if rollout_value_calibration_row is None:
            rollout_value_candidate_enabled = False
            log("rollout-value candidate disabled: no validation row passed deployable selection guard")
    if bool(args.include_sequence_value_policy):
        log("collecting sequence-value dataset")
        sequence_value_env = build_env_for_dataset(truth, sensors, constraints, train_cfg, oracle)
        if bool(args.sequence_value_augment_bank):
            sequence_value_extra_bank = build_augmented_sequence_value_bank(
                labels=teacher_dataset.labels,
                candidate_masks=candidate_masks,
                anchor_idx=selected_static_idx,
                train_static_table=train_static_table,
                sequence_len=int(args.sequence_value_segment_len),
                static_top_k=int(args.sequence_value_static_top_k),
                support_top_k=int(args.sequence_value_cycle_support_top_k),
                dwell_grid=tuple(int(x) for x in args.sequence_value_cycle_dwell_grid),
                max_sequences=int(args.sequence_value_cycle_max_sequences),
            )
            log(
                "sequence-value augmented bank: "
                f"extra_rows={int(sequence_value_extra_bank.shape[0]) if sequence_value_extra_bank is not None else 0}"
            )
        sequence_value_dataset = collect_sequence_value_dataset(
            sequence_value_env,
            candidate_masks,
            features=teacher_dataset.features,
            labels=teacher_dataset.labels,
            step_indices=teacher_dataset.step_indices,
            teacher_cfg=teacher_cfg,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            sequence_len=int(args.sequence_value_segment_len),
            snippet_stride=int(args.sequence_value_snippet_stride),
            negatives_per_state=int(args.sequence_value_negatives_per_state),
            max_rows=int(args.sequence_value_max_rows),
            extra_sequence_bank=sequence_value_extra_bank,
            seed=int(args.seed) + 31,
        )
        log(
            "sequence-value dataset collected: "
            f"rows={sequence_value_dataset.inputs.shape[0]} "
            f"bank={sequence_value_dataset.sequence_bank.shape[0]} "
            f"positive_rate={float(np.mean(sequence_value_dataset.advantages > 0.0)):.4f}"
        )
        sequence_value_train_cfg = ActionCostTrainingConfig(
            hidden_dim=int(args.cost_hidden_dim),
            epochs=int(args.cost_epochs),
            batch_size=512,
            seed=int(args.seed) + 37,
            device=str(args.bc_device),
        )
        sequence_value_model, sequence_value_history = train_sequence_value_model(
            sequence_value_dataset,
            sequence_value_train_cfg,
        )
        log(
            "sequence-value model training complete: "
            f"final_loss={sequence_value_history['loss'][-1] if sequence_value_history and sequence_value_history.get('loss') else float('nan')}"
        )
        (
            sequence_value_threshold,
            sequence_value_validation_objective,
            sequence_value_calibration_row,
        ) = calibrate_sequence_value_policy(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=validation_cfg,
            oracle=oracle,
            model=sequence_value_model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            sequence_bank=sequence_value_dataset.sequence_bank,
            state_columns=state_columns,
            starts=starts["validation"].starts,
        )
        log(
            "sequence-value calibration: "
            f"threshold={sequence_value_threshold} "
            f"validation_objective={sequence_value_validation_objective:.6f} "
            f"row={sequence_value_calibration_row}"
        )
        if sequence_value_calibration_row is None:
            sequence_value_candidate_enabled = False
            log("sequence-value candidate disabled: no validation row passed deployable selection guard")
        elif (
            sequence_value_calibration_row is not None
            and float(args.deployable_selection_min_mean_margin) > 0.0
            and not bool(sequence_value_calibration_row.get("static_margin_guard_pass", False))
            and str(args.deployable_selection_criterion) == "static_margin_guard"
        ):
            sequence_value_candidate_enabled = False
            log("sequence-value candidate disabled: calibration failed positive static-margin guard")
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
    if bool(args.include_option_planner_policy) and option_planner_support is not None:
        policies.append(
            ForecastAwareOptionPlannerPolicy(
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                option_action_indices=tuple(int(x) for x in option_planner_support),
                target_rates=np.asarray(option_planner_target_rates, dtype=float)
                if option_planner_target_rates is not None
                else None,
                threshold=float(option_planner_threshold if option_planner_threshold is not None else 1.0),
                aggregation=str(option_planner_aggregation or "max"),
                min_dwell=int(option_planner_min_dwell if option_planner_min_dwell is not None else 1),
                cooldown=int(option_planner_cooldown if option_planner_cooldown is not None else 0),
                target_rate_weight=float(
                    option_planner_target_rate_weight if option_planner_target_rate_weight is not None else 1.0
                ),
                rate_balance_weight=float(
                    option_planner_rate_balance_weight if option_planner_rate_balance_weight is not None else 0.0
                ),
                freshness_weight=float(
                    option_planner_freshness_weight if option_planner_freshness_weight is not None else 0.25
                ),
                transport_weight=float(
                    option_planner_transport_weight if option_planner_transport_weight is not None else 0.25
                ),
                power_weight=float(option_planner_power_weight if option_planner_power_weight is not None else 0.05),
                switch_weight=float(
                    option_planner_switch_weight if option_planner_switch_weight is not None else 0.05
                ),
                min_soc=float(option_planner_min_soc if option_planner_min_soc is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_macro_option_policy)
        and macro_option_segment_len is not None
        and bool(macro_option_candidate_enabled)
    ):
        policies.append(
            ForecastAwareMacroOptionPolicy(
                features=np.asarray(teacher_dataset.features, dtype=np.float32),
                labels=np.asarray(teacher_dataset.labels, dtype=np.int64),
                candidate_masks=candidate_masks,
                step_indices=np.asarray(teacher_dataset.step_indices, dtype=np.int64),
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                segment_len=int(macro_option_segment_len),
                snippet_stride=int(args.macro_option_snippet_stride),
                k=int(macro_option_k if macro_option_k is not None else 4),
                event_threshold=float(macro_option_threshold if macro_option_threshold is not None else 1.0),
                aggregation=str(macro_option_aggregation or "max"),
                distance_weighting=str(macro_option_distance_weighting or "inverse"),
                refresh_interval=int(macro_option_refresh_interval if macro_option_refresh_interval is not None else 0),
                max_lookahead=int(args.macro_option_max_lookahead),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_teacher_improvement_gate_policy)
        and improvement_gate_model is not None
        and improvement_gate_segment_len is not None
        and bool(improvement_gate_candidate_enabled)
    ):
        improvement_dynamic_policy = ForecastAwareMacroOptionPolicy(
            features=np.asarray(teacher_dataset.features, dtype=np.float32),
            labels=np.asarray(teacher_dataset.labels, dtype=np.int64),
            candidate_masks=candidate_masks,
            step_indices=np.asarray(teacher_dataset.step_indices, dtype=np.int64),
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            segment_len=int(improvement_gate_segment_len),
            snippet_stride=int(args.macro_option_snippet_stride),
            k=int(improvement_gate_k if improvement_gate_k is not None else 4),
            event_threshold=0.0,
            aggregation=str(improvement_gate_aggregation or "mean"),
            distance_weighting=str(improvement_gate_distance_weighting or "inverse"),
            refresh_interval=int(
                improvement_gate_refresh_interval if improvement_gate_refresh_interval is not None else 0
            ),
            max_lookahead=int(args.macro_option_max_lookahead),
            preserve_warming=bool(args.bc_preserve_warming),
            name="forecast_aware_teacher_improvement_inner_macro_option",
        )
        policies.append(
            ForecastAwareTeacherImprovementGatePolicy(
                gate_model=improvement_gate_model,
                dynamic_policy=improvement_dynamic_policy,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                feature_mean=np.asarray(improvement_gate_mean, dtype=np.float32),
                feature_std=np.asarray(improvement_gate_std, dtype=np.float32),
                threshold=float(improvement_gate_threshold if improvement_gate_threshold is not None else 0.6),
                preserve_warming=bool(args.bc_preserve_warming),
                device=str(args.bc_device),
            )
        )
    if (
        bool(args.include_runtime_risk_guard_policy)
        and option_planner_support is not None
        and runtime_risk_threshold is not None
    ):
        runtime_dynamic_policy = ForecastAwareOptionPlannerPolicy(
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            option_action_indices=tuple(int(x) for x in option_planner_support),
            target_rates=np.asarray(option_planner_target_rates, dtype=float)
            if option_planner_target_rates is not None
            else None,
            threshold=float(option_planner_threshold if option_planner_threshold is not None else 1.0),
            aggregation=str(option_planner_aggregation or "max"),
            min_dwell=int(option_planner_min_dwell if option_planner_min_dwell is not None else 1),
            cooldown=int(option_planner_cooldown if option_planner_cooldown is not None else 0),
            target_rate_weight=float(
                option_planner_target_rate_weight if option_planner_target_rate_weight is not None else 1.0
            ),
            rate_balance_weight=float(
                option_planner_rate_balance_weight if option_planner_rate_balance_weight is not None else 0.0
            ),
            freshness_weight=float(
                option_planner_freshness_weight if option_planner_freshness_weight is not None else 0.25
            ),
            transport_weight=float(
                option_planner_transport_weight if option_planner_transport_weight is not None else 0.25
            ),
            power_weight=float(option_planner_power_weight if option_planner_power_weight is not None else 0.05),
            switch_weight=float(
                option_planner_switch_weight if option_planner_switch_weight is not None else 0.05
            ),
            min_soc=float(option_planner_min_soc if option_planner_min_soc is not None else 0.0),
            preserve_warming=bool(args.bc_preserve_warming),
            name="forecast_aware_runtime_inner_option_planner",
        )
        policies.append(
            ForecastAwareRuntimeRiskGuardPolicy(
                dynamic_policy=runtime_dynamic_policy,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                threshold=float(runtime_risk_threshold),
                aggregation=str(runtime_risk_aggregation or "max"),
                window_steps=int(runtime_risk_window_steps if runtime_risk_window_steps is not None else 8),
                event_weight=float(runtime_risk_event_weight if runtime_risk_event_weight is not None else 1.0),
                freshness_weight=float(
                    runtime_risk_freshness_weight if runtime_risk_freshness_weight is not None else 0.25
                ),
                transport_weight=float(
                    runtime_risk_transport_weight if runtime_risk_transport_weight is not None else 0.25
                ),
                soc_weight=float(runtime_risk_soc_weight if runtime_risk_soc_weight is not None else 0.0),
                min_soc=float(runtime_risk_min_soc if runtime_risk_min_soc is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_window_eligibility_policy)
        and bool(window_eligibility_candidate_enabled)
        and window_eligibility_features is not None
        and window_eligibility_margins is not None
        and window_eligibility_target_rates is not None
        and window_eligibility_support is not None
    ):
        window_dynamic_policy = make_window_eligibility_inner_policy(
            dynamic_family=str(window_eligibility_dynamic_family or "option"),
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=selected_static_mask,
            support=tuple(int(x) for x in window_eligibility_support),
            target_rates=np.asarray(window_eligibility_target_rates, dtype=float),
            min_dwell=int(window_eligibility_min_dwell if window_eligibility_min_dwell is not None else 2),
            freshness_weight=float(
                window_eligibility_freshness_weight if window_eligibility_freshness_weight is not None else 0.25
            ),
            transport_weight=float(
                window_eligibility_transport_weight if window_eligibility_transport_weight is not None else 0.25
            ),
            power_weight=float(window_eligibility_power_weight if window_eligibility_power_weight is not None else 0.05),
            switch_weight=float(
                window_eligibility_switch_weight if window_eligibility_switch_weight is not None else 0.05
            ),
            min_soc=float(window_eligibility_min_soc if window_eligibility_min_soc is not None else 0.0),
            macro_features=np.asarray(teacher_dataset.features, dtype=np.float32),
            macro_labels=np.asarray(teacher_dataset.labels, dtype=np.int64),
            macro_step_indices=np.asarray(teacher_dataset.step_indices, dtype=np.int64),
            macro_k=int(window_eligibility_macro_k if window_eligibility_macro_k is not None else 4),
            macro_snippet_stride=int(args.window_eligibility_macro_snippet_stride),
            macro_max_lookahead=min(
                int(window_eligibility_window_steps if window_eligibility_window_steps is not None else 16),
                max(1, int(args.window_eligibility_macro_max_lookahead)),
            ),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_window_eligibility_inner_{window_eligibility_dynamic_family or 'option'}",
        )
        policies.append(
            ForecastAwareWindowEligibilityPolicy(
                memory_features=np.asarray(window_eligibility_features, dtype=np.float32),
                memory_margins=np.asarray(window_eligibility_margins, dtype=float),
                dynamic_policy=window_dynamic_policy,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                k=int(window_eligibility_k if window_eligibility_k is not None else 3),
                margin_threshold=float(
                    window_eligibility_margin_threshold
                    if window_eligibility_margin_threshold is not None
                    else 0.0
                ),
                window_steps=int(window_eligibility_window_steps if window_eligibility_window_steps is not None else 16),
                distance_weighting=str(window_eligibility_distance_weighting or "inverse"),
                min_soc=float(window_eligibility_min_soc if window_eligibility_min_soc is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_window_candidate_policy)
        and bool(window_candidate_candidate_enabled)
        and window_candidate_specs is not None
        and window_candidate_features is not None
        and window_candidate_margins is not None
        and window_candidate_ids is not None
        and window_candidate_support is not None
    ):
        candidate_policies = tuple(
            make_window_candidate_inner_policy(
                args=args,
                spec=spec,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                support=tuple(int(x) for x in window_candidate_support),
                features=np.asarray(teacher_dataset.features, dtype=np.float32),
                labels=np.asarray(teacher_dataset.labels, dtype=np.int64),
                step_indices=np.asarray(teacher_dataset.step_indices, dtype=np.int64),
                window_steps=int(window_candidate_window_steps if window_candidate_window_steps is not None else 16),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_window_candidate_inner_{candidate_id}_{spec.get('family', 'option')}",
            )
            for candidate_id, spec in enumerate(window_candidate_specs)
        )
        policies.append(
            ForecastAwareWindowCandidatePolicy(
                memory_features=np.asarray(window_candidate_features, dtype=np.float32),
                memory_margins=np.asarray(window_candidate_margins, dtype=float),
                memory_candidate_ids=np.asarray(window_candidate_ids, dtype=int),
                candidate_policies=candidate_policies,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                k=int(window_candidate_k if window_candidate_k is not None else 5),
                margin_threshold=float(
                    window_candidate_margin_threshold if window_candidate_margin_threshold is not None else 0.0
                ),
                score_quantile=float(
                    window_candidate_score_quantile if window_candidate_score_quantile is not None else 0.25
                ),
                window_steps=int(window_candidate_window_steps if window_candidate_window_steps is not None else 16),
                distance_weighting=str(window_candidate_distance_weighting or "inverse"),
                min_soc=float(window_candidate_min_soc if window_candidate_min_soc is not None else 0.0),
                min_candidate_neighbors=int(args.window_candidate_min_neighbors),
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
    if (
        bool(args.include_utility_planner_policy)
        and bool(utility_planner_candidate_enabled)
        and utility_planner_support is not None
        and utility_planner_target_rates is not None
    ):
        policies.append(
            ForecastAwareUtilityPlannerPolicy(
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                allowed_action_indices=tuple(int(x) for x in utility_planner_support),
                target_rates=np.asarray(utility_planner_target_rates, dtype=float),
                event_weight=float(utility_planner_event_weight if utility_planner_event_weight is not None else 1.0),
                magnitude_weight=float(
                    utility_planner_magnitude_weight if utility_planner_magnitude_weight is not None else 1.0
                ),
                variability_weight=float(
                    utility_planner_variability_weight if utility_planner_variability_weight is not None else 0.5
                ),
                freshness_weight=float(
                    utility_planner_freshness_weight if utility_planner_freshness_weight is not None else 0.0
                ),
                target_rate_weight=float(
                    utility_planner_target_rate_weight if utility_planner_target_rate_weight is not None else 0.0
                ),
                anchor_bias=float(utility_planner_anchor_bias if utility_planner_anchor_bias is not None else 0.0),
                power_weight=float(utility_planner_power_weight if utility_planner_power_weight is not None else 0.0),
                switch_weight=float(utility_planner_switch_weight if utility_planner_switch_weight is not None else 0.0),
                min_soc=float(utility_planner_min_soc if utility_planner_min_soc is not None else 0.0),
                min_dwell=int(utility_planner_min_dwell if utility_planner_min_dwell is not None else 1),
                aggregation=str(utility_planner_aggregation or "max"),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_proxy_mpc_policy)
        and bool(proxy_mpc_candidate_enabled)
        and proxy_mpc_support is not None
        and proxy_mpc_target_rates is not None
    ):
        policies.append(
            ForecastAwareProxyMPCPolicy(
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                allowed_action_indices=tuple(int(x) for x in proxy_mpc_support),
                target_rates=np.asarray(proxy_mpc_target_rates, dtype=float),
                event_weight=float(proxy_mpc_event_weight if proxy_mpc_event_weight is not None else 1.0),
                magnitude_weight=float(proxy_mpc_magnitude_weight if proxy_mpc_magnitude_weight is not None else 1.0),
                variability_weight=float(
                    proxy_mpc_variability_weight if proxy_mpc_variability_weight is not None else 0.5
                ),
                freshness_weight=float(proxy_mpc_freshness_weight if proxy_mpc_freshness_weight is not None else 0.0),
                target_rate_weight=float(
                    proxy_mpc_target_rate_weight if proxy_mpc_target_rate_weight is not None else 0.0
                ),
                anchor_bias=float(proxy_mpc_anchor_bias if proxy_mpc_anchor_bias is not None else 0.0),
                power_weight=float(proxy_mpc_power_weight if proxy_mpc_power_weight is not None else 0.0),
                switch_weight=float(proxy_mpc_switch_weight if proxy_mpc_switch_weight is not None else 0.0),
                min_soc=float(proxy_mpc_min_soc if proxy_mpc_min_soc is not None else 0.0),
                min_dwell=int(proxy_mpc_min_dwell if proxy_mpc_min_dwell is not None else 1),
                aggregation=str(proxy_mpc_aggregation or "max"),
                planning_depth=int(proxy_mpc_planning_depth if proxy_mpc_planning_depth is not None else 3),
                beam_width=int(proxy_mpc_beam_width if proxy_mpc_beam_width is not None else 4),
                max_branch=int(proxy_mpc_max_branch if proxy_mpc_max_branch is not None else 8),
                age_weight=float(proxy_mpc_age_weight if proxy_mpc_age_weight is not None else 0.5),
                anchor_improvement_threshold=float(
                    proxy_mpc_anchor_improvement_threshold
                    if proxy_mpc_anchor_improvement_threshold is not None
                    else 0.0
                ),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if bool(args.include_sequence_mask_policy) and sequence_mask_model is not None:
        policies.append(
            ForecastAwareSequenceMaskPolicy(
                model=sequence_mask_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                allowed_action_indices=sequence_mask_support,
                anchor_mask=selected_static_mask,
                anchor_bias=float(sequence_mask_anchor_bias if sequence_mask_anchor_bias is not None else 0.0),
                power_weight=float(sequence_mask_power_weight if sequence_mask_power_weight is not None else 0.0),
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
    if bool(args.include_validation_cyclic_policy) and validation_cyclic_action_indices is not None:
        policies.append(
            ValidationCyclicDwellPolicy(
                candidate_masks=candidate_masks,
                action_indices=tuple(int(x) for x in validation_cyclic_action_indices),
                dwell_steps=int(validation_cyclic_dwell if validation_cyclic_dwell is not None else 4),
                fallback_action_idx=int(selected_static_idx),
                preserve_warming=bool(args.validation_cyclic_preserve_warming),
                name="validation_cyclic_dwell",
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
    if (
        bool(args.include_cost_knn_policy)
        and cost_knn_dataset is not None
        and cost_knn_k is not None
        and bool(cost_knn_candidate_enabled)
    ):
        policies.append(
            ForecastAwareCostKNNPolicy(
                features=np.asarray(cost_knn_dataset.features, dtype=np.float32),
                costs=np.asarray(cost_knn_dataset.costs, dtype=np.float32),
                action_masks=np.asarray(cost_knn_dataset.action_masks, dtype=bool),
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                allowed_action_indices=cost_knn_support,
                k=int(cost_knn_k),
                advantage_threshold=float(
                    cost_knn_advantage_threshold if cost_knn_advantage_threshold is not None else 0.0
                ),
                distance_weighting=str(cost_knn_distance_weighting or "inverse"),
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
        and bool(rollout_value_candidate_enabled)
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
    if (
        bool(args.include_sequence_value_policy)
        and sequence_value_model is not None
        and sequence_value_dataset is not None
        and bool(sequence_value_candidate_enabled)
    ):
        policies.append(
            ForecastAwareSequenceValuePolicy(
                model=sequence_value_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                sequence_bank=np.asarray(sequence_value_dataset.sequence_bank, dtype=np.int64),
                device=str(args.bc_device),
                advantage_threshold=float(sequence_value_threshold if sequence_value_threshold is not None else 0.0),
                top_k_sequences=int(args.sequence_value_top_k_sequences),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_recurrent_value_policy)
        and recurrent_value_model is not None
        and bool(recurrent_value_candidate_enabled)
    ):
        policies.append(
            ForecastAwareRecurrentValuePolicy(
                model=recurrent_value_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=recurrent_value_support,
                advantage_threshold=float(recurrent_value_threshold if recurrent_value_threshold is not None else 0.0),
                preserve_warming=bool(args.bc_preserve_warming),
            )
        )
    if (
        bool(args.include_recurrent_advantage_policy)
        and recurrent_advantage_model is not None
        and bool(recurrent_advantage_candidate_enabled)
    ):
        policies.append(
            ForecastAwareRecurrentAdvantagePolicy(
                model=recurrent_advantage_model,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=selected_static_mask,
                device=str(args.bc_device),
                allowed_action_indices=recurrent_advantage_support,
                advantage_threshold=float(
                    recurrent_advantage_threshold if recurrent_advantage_threshold is not None else 0.0
                ),
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
    if (
        bool(args.include_advantage_residual_policy)
        and advantage_residual_model is not None
        and bool(advantage_residual_candidate_enabled)
    ):
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
    deployable = metrics_df[metrics_df["policy"].isin(DEPLOYABLE_POLICY_NAMES)]
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
        "learned_continuous_forecast": continuous_forecaster_summary,
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
            "calibration_criterion": str(args.event_threshold_calibration_criterion),
            "action_idx": event_threshold_action_idx,
            "threshold": event_threshold_value,
            "aggregation": event_threshold_aggregation,
            "threshold_grid": [float(x) for x in args.event_threshold_grid],
            "aggregation_grid": [str(x) for x in args.event_threshold_aggregation_grid],
            "calibration_row": event_threshold_calibration_row,
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
        "option_planner_policy": {
            "included": bool(args.include_option_planner_policy),
            "support_top_k": int(args.option_planner_support_top_k),
            "action_indices": [int(x) for x in option_planner_support]
            if option_planner_support is not None
            else None,
            "target_rates": [float(x) for x in np.asarray(option_planner_target_rates, dtype=float).reshape(-1)]
            if option_planner_target_rates is not None
            else None,
            "threshold": option_planner_threshold,
            "aggregation": option_planner_aggregation,
            "min_dwell": option_planner_min_dwell,
            "cooldown": option_planner_cooldown,
            "target_rate_weight": option_planner_target_rate_weight,
            "rate_balance_weight": option_planner_rate_balance_weight,
            "freshness_weight": option_planner_freshness_weight,
            "transport_weight": option_planner_transport_weight,
            "power_weight": option_planner_power_weight,
            "switch_weight": option_planner_switch_weight,
            "min_soc": option_planner_min_soc,
            "threshold_grid": [float(x) for x in args.option_planner_threshold_grid],
            "aggregation_grid": [str(x) for x in args.option_planner_aggregation_grid],
            "min_dwell_grid": [int(x) for x in args.option_planner_min_dwell_grid],
            "cooldown_grid": [int(x) for x in args.option_planner_cooldown_grid],
            "target_rate_grid": [float(x) for x in args.option_planner_target_rate_grid],
            "rate_balance_grid": [float(x) for x in args.option_planner_rate_balance_grid],
            "freshness_grid": [float(x) for x in args.option_planner_freshness_grid],
            "transport_grid": [float(x) for x in args.option_planner_transport_grid],
            "power_grid": [float(x) for x in args.option_planner_power_grid],
            "switch_grid": [float(x) for x in args.option_planner_switch_grid],
            "min_soc_grid": [float(x) for x in args.option_planner_min_soc_grid],
            "calibration_criterion": str(args.option_planner_calibration_criterion),
            "calibration_row": option_planner_calibration_row,
            "validation_objective": option_planner_validation_objective,
        },
        "macro_option_policy": {
            "included": bool(args.include_macro_option_policy),
            "segment_len": macro_option_segment_len,
            "segment_grid": [int(x) for x in args.macro_option_segment_grid],
            "snippet_stride": int(args.macro_option_snippet_stride),
            "k": macro_option_k,
            "k_grid": [int(x) for x in args.macro_option_k_grid],
            "event_threshold": macro_option_threshold,
            "threshold_grid": [float(x) for x in args.macro_option_threshold_grid],
            "aggregation": macro_option_aggregation,
            "aggregation_grid": [str(x) for x in args.macro_option_aggregation_grid],
            "distance_weighting": macro_option_distance_weighting,
            "distance_weighting_grid": [str(x) for x in args.macro_option_distance_weighting_grid],
            "refresh_interval": macro_option_refresh_interval,
            "refresh_grid": [int(x) for x in args.macro_option_refresh_grid],
            "max_lookahead": int(args.macro_option_max_lookahead),
            "calibration_criterion": str(args.macro_option_calibration_criterion),
            "calibration_row": macro_option_calibration_row,
            "candidate_enabled": bool(macro_option_candidate_enabled),
            "validation_objective": macro_option_validation_objective,
            "teacher_rows": int(teacher_dataset.features.shape[0]),
        },
        "teacher_improvement_gate_policy": {
            "included": bool(args.include_teacher_improvement_gate_policy),
            "segment_len": improvement_gate_segment_len,
            "segment_grid": [int(x) for x in args.macro_option_segment_grid],
            "snippet_stride": int(args.macro_option_snippet_stride),
            "k": improvement_gate_k,
            "k_grid": [int(x) for x in args.macro_option_k_grid],
            "gate_threshold": improvement_gate_threshold,
            "gate_threshold_grid": [float(x) for x in args.teacher_improvement_gate_threshold_grid],
            "label_margin": float(args.teacher_improvement_gate_label_margin),
            "hidden_dim": int(args.teacher_improvement_gate_hidden_dim),
            "epochs": int(args.teacher_improvement_gate_epochs),
            "training_positive_rate": (
                float(improvement_gate_history["positive_rate"][0])
                if isinstance(improvement_gate_history, dict)
                and improvement_gate_history.get("positive_rate")
                else None
            ),
            "training_final_accuracy": (
                float(improvement_gate_history["accuracy"][-1])
                if isinstance(improvement_gate_history, dict)
                and improvement_gate_history.get("accuracy")
                else None
            ),
            "aggregation": improvement_gate_aggregation,
            "aggregation_grid": [str(x) for x in args.macro_option_aggregation_grid],
            "distance_weighting": improvement_gate_distance_weighting,
            "distance_weighting_grid": [str(x) for x in args.macro_option_distance_weighting_grid],
            "refresh_interval": improvement_gate_refresh_interval,
            "refresh_grid": [int(x) for x in args.macro_option_refresh_grid],
            "max_lookahead": int(args.macro_option_max_lookahead),
            "calibration_criterion": str(args.macro_option_calibration_criterion),
            "calibration_row": improvement_gate_calibration_row,
            "candidate_enabled": bool(improvement_gate_candidate_enabled),
            "validation_objective": improvement_gate_validation_objective,
            "teacher_rows": int(teacher_dataset.features.shape[0]),
        },
        "runtime_risk_guard_policy": {
            "included": bool(args.include_runtime_risk_guard_policy),
            "calibration_criterion": str(args.runtime_risk_guard_calibration_criterion),
            "threshold": runtime_risk_threshold,
            "aggregation": runtime_risk_aggregation,
            "window_steps": runtime_risk_window_steps,
            "event_weight": runtime_risk_event_weight,
            "freshness_weight": runtime_risk_freshness_weight,
            "transport_weight": runtime_risk_transport_weight,
            "soc_weight": runtime_risk_soc_weight,
            "min_soc": runtime_risk_min_soc,
            "threshold_grid": [float(x) for x in args.runtime_risk_threshold_grid],
            "aggregation_grid": [str(x) for x in args.runtime_risk_aggregation_grid],
            "window_grid": [int(x) for x in args.runtime_risk_window_grid],
            "event_weight_grid": [float(x) for x in args.runtime_risk_event_weight_grid],
            "freshness_weight_grid": [float(x) for x in args.runtime_risk_freshness_weight_grid],
            "transport_weight_grid": [float(x) for x in args.runtime_risk_transport_weight_grid],
            "soc_weight_grid": [float(x) for x in args.runtime_risk_soc_weight_grid],
            "min_soc_grid": [float(x) for x in args.runtime_risk_min_soc_grid],
            "calibration_row": runtime_risk_calibration_row,
            "validation_objective": runtime_risk_validation_objective,
        },
        "window_eligibility_policy": {
            "included": bool(args.include_window_eligibility_policy),
            "dynamic_grid": [str(x) for x in args.window_eligibility_dynamic_grid],
            "dynamic_family": window_eligibility_dynamic_family,
            "macro_k": window_eligibility_macro_k,
            "macro_k_grid": [int(x) for x in args.window_eligibility_macro_k_grid],
            "macro_snippet_stride": int(args.window_eligibility_macro_snippet_stride),
            "macro_max_lookahead": int(args.window_eligibility_macro_max_lookahead),
            "support_top_k": int(args.window_eligibility_support_top_k),
            "support_indices": [int(x) for x in window_eligibility_support]
            if window_eligibility_support is not None
            else None,
            "target_rates": [float(x) for x in np.asarray(window_eligibility_target_rates, dtype=float).reshape(-1)]
            if window_eligibility_target_rates is not None
            else None,
            "window_steps": window_eligibility_window_steps,
            "k": window_eligibility_k,
            "margin_threshold": window_eligibility_margin_threshold,
            "blend": window_eligibility_blend,
            "min_dwell": window_eligibility_min_dwell,
            "freshness_weight": window_eligibility_freshness_weight,
            "transport_weight": window_eligibility_transport_weight,
            "power_weight": window_eligibility_power_weight,
            "switch_weight": window_eligibility_switch_weight,
            "min_soc": window_eligibility_min_soc,
            "distance_weighting": window_eligibility_distance_weighting,
            "window_grid": [int(x) for x in args.window_eligibility_window_grid],
            "k_grid": [int(x) for x in args.window_eligibility_k_grid],
            "margin_grid": [float(x) for x in args.window_eligibility_margin_grid],
            "blend_grid": [float(x) for x in args.window_eligibility_blend_grid],
            "min_dwell_grid": [int(x) for x in args.window_eligibility_min_dwell_grid],
            "freshness_grid": [float(x) for x in args.window_eligibility_freshness_grid],
            "transport_grid": [float(x) for x in args.window_eligibility_transport_grid],
            "power_grid": [float(x) for x in args.window_eligibility_power_grid],
            "switch_grid": [float(x) for x in args.window_eligibility_switch_grid],
            "min_soc_grid": [float(x) for x in args.window_eligibility_min_soc_grid],
            "distance_weighting_grid": [str(x) for x in args.window_eligibility_distance_weighting_grid],
            "samples_per_start": int(args.window_eligibility_samples_per_start),
            "max_train_windows": int(args.window_eligibility_max_train_windows),
            "calibration_criterion": str(args.window_eligibility_calibration_criterion),
            "calibration_row": window_eligibility_calibration_row,
            "candidate_enabled": bool(window_eligibility_candidate_enabled),
            "validation_objective": window_eligibility_validation_objective,
            "memory_rows": int(np.asarray(window_eligibility_features).shape[0])
            if window_eligibility_features is not None
            else 0,
            "memory_margin_mean": float(np.mean(np.asarray(window_eligibility_margins, dtype=float)))
            if window_eligibility_margins is not None
            else None,
            "memory_positive_rate": float(np.mean(np.asarray(window_eligibility_margins, dtype=float) > 0.0))
            if window_eligibility_margins is not None
            else None,
        },
        "window_candidate_policy": {
            "included": bool(args.include_window_candidate_policy),
            "family_grid": [str(x) for x in args.window_candidate_family_grid],
            "support_top_k": int(args.window_candidate_support_top_k),
            "support_indices": [int(x) for x in window_candidate_support]
            if window_candidate_support is not None
            else None,
            "candidate_count": len(window_candidate_specs) if window_candidate_specs is not None else 0,
            "window_steps": window_candidate_window_steps,
            "k": window_candidate_k,
            "margin_threshold": window_candidate_margin_threshold,
            "score_quantile": window_candidate_score_quantile,
            "distance_weighting": window_candidate_distance_weighting,
            "min_neighbors": int(args.window_candidate_min_neighbors),
            "max_candidates": int(args.window_candidate_max_candidates),
            "window_grid": [int(x) for x in args.window_candidate_window_grid],
            "k_grid": [int(x) for x in args.window_candidate_k_grid],
            "margin_grid": [float(x) for x in args.window_candidate_margin_grid],
            "quantile_grid": [float(x) for x in args.window_candidate_quantile_grid],
            "distance_weighting_grid": [str(x) for x in args.window_candidate_distance_weighting_grid],
            "full_rollout_calibration": bool(args.window_candidate_full_rollout_calibration),
            "samples_per_start": int(args.window_candidate_samples_per_start),
            "max_train_windows": int(args.window_candidate_max_train_windows),
            "calibration_criterion": str(args.window_candidate_calibration_criterion),
            "calibration_row": window_candidate_calibration_row,
            "candidate_enabled": bool(window_candidate_candidate_enabled),
            "validation_objective": window_candidate_validation_objective,
            "memory_rows": int(np.asarray(window_candidate_features).shape[0])
            if window_candidate_features is not None
            else 0,
            "memory_margin_mean": float(np.mean(np.asarray(window_candidate_margins, dtype=float)))
            if window_candidate_margins is not None
            else None,
            "memory_positive_rate": float(np.mean(np.asarray(window_candidate_margins, dtype=float) > 0.0))
            if window_candidate_margins is not None
            else None,
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
            "calibration_criterion": str(args.contextual_duty_calibration_criterion),
            "calibration_row": contextual_duty_calibration_row,
            "validation_objective": contextual_duty_validation_objective,
            "mask_bc_history": mask_bc_history,
        },
        "utility_planner_policy": {
            "included": bool(args.include_utility_planner_policy),
            "support_top_k": int(args.utility_planner_support_top_k),
            "action_indices": [int(x) for x in utility_planner_support]
            if utility_planner_support is not None
            else None,
            "target_rates": [float(x) for x in np.asarray(utility_planner_target_rates, dtype=float).reshape(-1)]
            if utility_planner_target_rates is not None
            else None,
            "event_weight": utility_planner_event_weight,
            "magnitude_weight": utility_planner_magnitude_weight,
            "variability_weight": utility_planner_variability_weight,
            "freshness_weight": utility_planner_freshness_weight,
            "target_rate_weight": utility_planner_target_rate_weight,
            "anchor_bias": utility_planner_anchor_bias,
            "power_weight": utility_planner_power_weight,
            "switch_weight": utility_planner_switch_weight,
            "min_soc": utility_planner_min_soc,
            "min_dwell": utility_planner_min_dwell,
            "aggregation": utility_planner_aggregation,
            "event_weight_grid": [float(x) for x in args.utility_planner_event_weight_grid],
            "magnitude_weight_grid": [float(x) for x in args.utility_planner_magnitude_weight_grid],
            "variability_weight_grid": [float(x) for x in args.utility_planner_variability_weight_grid],
            "freshness_grid": [float(x) for x in args.utility_planner_freshness_grid],
            "target_rate_grid": [float(x) for x in args.utility_planner_target_rate_grid],
            "anchor_bias_grid": [float(x) for x in args.utility_planner_anchor_bias_grid],
            "power_grid": [float(x) for x in args.utility_planner_power_grid],
            "switch_grid": [float(x) for x in args.utility_planner_switch_grid],
            "min_soc_grid": [float(x) for x in args.utility_planner_min_soc_grid],
            "dwell_grid": [int(x) for x in args.utility_planner_dwell_grid],
            "aggregation_grid": [str(x) for x in args.utility_planner_aggregation_grid],
            "calibration_criterion": str(args.utility_planner_calibration_criterion),
            "calibration_row": utility_planner_calibration_row,
            "candidate_enabled": bool(utility_planner_candidate_enabled),
            "validation_objective": utility_planner_validation_objective,
        },
        "proxy_mpc_policy": {
            "included": bool(args.include_proxy_mpc_policy),
            "support_top_k": int(args.proxy_mpc_support_top_k),
            "action_indices": [int(x) for x in proxy_mpc_support] if proxy_mpc_support is not None else None,
            "target_rates": [float(x) for x in np.asarray(proxy_mpc_target_rates, dtype=float).reshape(-1)]
            if proxy_mpc_target_rates is not None
            else None,
            "event_weight": proxy_mpc_event_weight,
            "magnitude_weight": proxy_mpc_magnitude_weight,
            "variability_weight": proxy_mpc_variability_weight,
            "freshness_weight": proxy_mpc_freshness_weight,
            "target_rate_weight": proxy_mpc_target_rate_weight,
            "anchor_bias": proxy_mpc_anchor_bias,
            "power_weight": proxy_mpc_power_weight,
            "switch_weight": proxy_mpc_switch_weight,
            "min_soc": proxy_mpc_min_soc,
            "min_dwell": proxy_mpc_min_dwell,
            "aggregation": proxy_mpc_aggregation,
            "planning_depth": proxy_mpc_planning_depth,
            "beam_width": proxy_mpc_beam_width,
            "max_branch": proxy_mpc_max_branch,
            "age_weight": proxy_mpc_age_weight,
            "anchor_improvement": proxy_mpc_anchor_improvement_threshold,
            "event_weight_grid": [float(x) for x in args.proxy_mpc_event_weight_grid],
            "magnitude_weight_grid": [float(x) for x in args.proxy_mpc_magnitude_weight_grid],
            "variability_weight_grid": [float(x) for x in args.proxy_mpc_variability_weight_grid],
            "freshness_grid": [float(x) for x in args.proxy_mpc_freshness_grid],
            "target_rate_grid": [float(x) for x in args.proxy_mpc_target_rate_grid],
            "anchor_bias_grid": [float(x) for x in args.proxy_mpc_anchor_bias_grid],
            "power_grid": [float(x) for x in args.proxy_mpc_power_grid],
            "switch_grid": [float(x) for x in args.proxy_mpc_switch_grid],
            "min_soc_grid": [float(x) for x in args.proxy_mpc_min_soc_grid],
            "dwell_grid": [int(x) for x in args.proxy_mpc_dwell_grid],
            "aggregation_grid": [str(x) for x in args.proxy_mpc_aggregation_grid],
            "depth_grid": [int(x) for x in args.proxy_mpc_depth_grid],
            "beam_width_grid": [int(x) for x in args.proxy_mpc_beam_width_grid],
            "max_branch_grid": [int(x) for x in args.proxy_mpc_max_branch_grid],
            "age_weight_grid": [float(x) for x in args.proxy_mpc_age_weight_grid],
            "anchor_improvement_grid": [float(x) for x in args.proxy_mpc_anchor_improvement_grid],
            "calibration_criterion": str(args.proxy_mpc_calibration_criterion),
            "calibration_row": proxy_mpc_calibration_row,
            "candidate_enabled": bool(proxy_mpc_candidate_enabled),
            "validation_objective": proxy_mpc_validation_objective,
        },
        "sequence_mask_policy": {
            "included": bool(args.include_sequence_mask_policy),
            "support_top_k": int(args.sequence_mask_support_top_k),
            "action_indices": [int(x) for x in sequence_mask_support]
            if sequence_mask_support is not None
            else None,
            "anchor_bias": sequence_mask_anchor_bias,
            "power_weight": sequence_mask_power_weight,
            "anchor_bias_grid": [float(x) for x in args.sequence_mask_anchor_bias_grid],
            "power_grid": [float(x) for x in args.sequence_mask_power_grid],
            "calibration_criterion": str(args.sequence_mask_calibration_criterion),
            "calibration_row": sequence_mask_calibration_row,
            "validation_objective": sequence_mask_validation_objective,
            "history": sequence_mask_history,
        },
        "teacher_cycle_policy": {
            "included": bool(args.include_teacher_cycle_policy),
            "max_lookahead": int(args.teacher_cycle_max_lookahead),
        },
        "validation_cyclic_policy": {
            "included": bool(args.include_validation_cyclic_policy),
            "top_k": int(args.validation_cyclic_top_k),
            "dwell_grid": [int(x) for x in args.validation_cyclic_dwell_grid],
            "preserve_warming": bool(args.validation_cyclic_preserve_warming),
            "action_indices": [int(x) for x in validation_cyclic_action_indices]
            if validation_cyclic_action_indices is not None
            else None,
            "dwell_steps": validation_cyclic_dwell,
            "validation_objective": validation_cyclic_validation_objective,
            "calibration_rows": validation_cyclic_calibration_rows,
        },
        "deployable_selection": {
            "mode": str(args.deployable_selection),
            "criterion": str(args.deployable_selection_criterion),
            "min_mean_margin": float(args.deployable_selection_min_mean_margin),
            "min_start_margin": float(args.deployable_selection_min_start_margin),
            "max_negative_starts": int(args.deployable_selection_max_negative_starts),
            "require_guard_pass": bool(args.deployable_selection_require_guard_pass),
            "require_positive_center": bool(args.deployable_selection_require_positive_center),
            "require_risk_band": bool(args.deployable_selection_require_risk_band),
            "risk_min_q25_margin": float(args.deployable_selection_risk_min_q25_margin),
            "risk_max_negative_starts": int(args.deployable_selection_risk_max_negative_starts),
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
        "cost_knn_policy": {
            "included": bool(args.include_cost_knn_policy),
            "support_top_k": int(args.cost_knn_support_top_k),
            "support_indices": [int(x) for x in cost_knn_support]
            if cost_knn_support is not None
            else None,
            "k": cost_knn_k,
            "k_grid": [int(x) for x in args.cost_knn_k_grid],
            "advantage_threshold": cost_knn_advantage_threshold,
            "advantage_grid": [float(x) for x in args.cost_knn_advantage_grid],
            "distance_weighting": cost_knn_distance_weighting,
            "distance_weighting_grid": [str(x) for x in args.cost_knn_distance_weighting_grid],
            "calibration_criterion": str(args.cost_knn_calibration_criterion),
            "calibration_row": cost_knn_calibration_row,
            "candidate_enabled": bool(cost_knn_candidate_enabled),
            "validation_objective": cost_knn_validation_objective,
            "memory_rows": int(cost_knn_dataset.features.shape[0]) if cost_knn_dataset is not None else 0,
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
            "self_iters": int(args.rollout_value_self_iters),
            "self_steps": int(args.rollout_value_self_steps),
            "self_threshold": float(args.rollout_value_self_threshold),
            "random_rollouts": int(args.rollout_value_random_rollouts),
            "advantage_threshold": rollout_value_threshold,
            "advantage_grid": [float(x) for x in args.rollout_value_advantage_grid],
            "validation_objective": rollout_value_validation_objective,
            "calibration_row": rollout_value_calibration_row,
            "candidate_enabled": bool(rollout_value_candidate_enabled),
            "cost_target": str(args.rollout_value_cost_target),
            "cost_history": rollout_value_cost_history,
            "transition_history": rollout_value_transition_history,
        },
        "sequence_value_policy": {
            "included": bool(args.include_sequence_value_policy),
            "segment_len": int(args.sequence_value_segment_len),
            "snippet_stride": int(args.sequence_value_snippet_stride),
            "negatives_per_state": int(args.sequence_value_negatives_per_state),
            "max_rows": int(args.sequence_value_max_rows),
            "top_k_sequences": int(args.sequence_value_top_k_sequences),
            "augment_bank": bool(args.sequence_value_augment_bank),
            "static_top_k": int(args.sequence_value_static_top_k),
            "cycle_support_top_k": int(args.sequence_value_cycle_support_top_k),
            "cycle_dwell_grid": [int(x) for x in args.sequence_value_cycle_dwell_grid],
            "extra_bank_rows": int(sequence_value_extra_bank.shape[0])
            if sequence_value_extra_bank is not None
            else 0,
            "advantage_threshold": sequence_value_threshold,
            "advantage_grid": [float(x) for x in args.sequence_value_advantage_grid],
            "validation_objective": sequence_value_validation_objective,
            "calibration_row": sequence_value_calibration_row,
            "candidate_enabled": bool(sequence_value_candidate_enabled),
            "history": sequence_value_history,
            "rows": int(sequence_value_dataset.inputs.shape[0]) if sequence_value_dataset is not None else 0,
            "sequence_bank_rows": int(sequence_value_dataset.sequence_bank.shape[0])
            if sequence_value_dataset is not None
            else 0,
        },
        "recurrent_value_policy": {
            "included": bool(args.include_recurrent_value_policy),
            "support_top_k": int(args.recurrent_value_support_top_k),
            "support_indices": [int(x) for x in recurrent_value_support]
            if recurrent_value_support is not None
            else None,
            "advantage_threshold": recurrent_value_threshold,
            "advantage_grid": [float(x) for x in args.recurrent_value_advantage_grid],
            "rank_weight": float(args.recurrent_value_rank_weight),
            "cost_dagger_iters": int(args.recurrent_value_cost_dagger_iters),
            "cost_dagger_threshold": float(args.recurrent_value_cost_dagger_threshold),
            "cost_dagger_steps": int(args.recurrent_value_cost_dagger_steps or args.train_steps),
            "calibration_criterion": str(args.recurrent_value_calibration_criterion),
            "calibration_row": recurrent_value_calibration_row,
            "candidate_enabled": bool(recurrent_value_candidate_enabled),
            "validation_objective": recurrent_value_validation_objective,
            "history": recurrent_value_history,
        },
        "recurrent_advantage_policy": {
            "included": bool(args.include_recurrent_advantage_policy),
            "support_top_k": int(args.recurrent_advantage_support_top_k),
            "support_indices": [int(x) for x in recurrent_advantage_support]
            if recurrent_advantage_support is not None
            else None,
            "advantage_threshold": recurrent_advantage_threshold,
            "advantage_grid": [float(x) for x in args.recurrent_advantage_grid],
            "rank_weight": float(args.recurrent_advantage_rank_weight),
            "calibration_criterion": str(args.recurrent_advantage_calibration_criterion),
            "calibration_row": recurrent_advantage_calibration_row,
            "candidate_enabled": bool(recurrent_advantage_candidate_enabled),
            "validation_objective": recurrent_advantage_validation_objective,
            "history": recurrent_advantage_history,
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
            "calibration_row": advantage_residual_calibration_row,
            "candidate_enabled": bool(advantage_residual_candidate_enabled),
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


def calibrate_validation_cyclic_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    validation_static_table: pd.DataFrame,
    selected_static_idx: int,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[tuple[int, ...], int, float, list[dict[str, object]]]:
    sort_columns = [
        column
        for column in ("objective_loss_mean", "power_mean", "warmup_abort_count")
        if column in validation_static_table.columns
    ]
    static_table = validation_static_table.sort_values(sort_columns) if sort_columns else validation_static_table
    action_indices: list[int] = []
    top_k = max(1, int(args.validation_cyclic_top_k))
    for _, row in static_table.iterrows():
        action_idx = int(row["action_idx"])
        if action_idx not in action_indices:
            action_indices.append(action_idx)
        if len(action_indices) >= top_k:
            break
    if int(selected_static_idx) not in action_indices:
        action_indices.insert(0, int(selected_static_idx))
    if not action_indices:
        raise ValueError("validation cyclic calibration found no static candidates")

    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_key: tuple[float, float, int, int] | None = None
    dwell_grid = [max(1, int(x)) for x in (args.validation_cyclic_dwell_grid or [4])]
    for grid_idx, dwell in enumerate(dwell_grid):
        policy = ValidationCyclicDwellPolicy(
            candidate_masks=candidate_masks,
            action_indices=tuple(action_indices),
            dwell_steps=int(dwell),
            fallback_action_idx=int(selected_static_idx),
            preserve_warming=bool(args.validation_cyclic_preserve_warming),
            name=f"validation_cyclic_dwell_{dwell}",
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
            seed_offset=76_000 + grid_idx * 101,
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
        selected_masks = np.asarray(result.selected_masks, dtype=bool)
        diffs = np.abs(np.diff(selected_masks.astype(np.int8), axis=0)) if selected_masks.shape[0] > 1 else np.zeros((0, selected_masks.shape[1]), dtype=np.int8)
        step_toggles = diffs.sum(axis=1) if diffs.size else np.zeros(0, dtype=int)
        row = {
            "policy": "validation_cyclic_dwell",
            "dwell_steps": int(dwell),
            "action_indices": "|".join(str(int(x)) for x in action_indices),
            "objective_loss_mean": float(objective),
            "oracle_loss_mean": float(metrics.get("oracle_loss_mean", np.nan)),
            "task_error_mean": float(metrics.get("task_error_mean", np.nan)),
            "power_mean": float(metrics.get("power_mean", np.nan)),
            "active_count_mean": float(np.mean(selected_masks.sum(axis=1))) if selected_masks.size else float("nan"),
            "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
            "switch_any_frac": float(np.mean(step_toggles > 0)) if step_toggles.size else 0.0,
            "switch_ge2_frac": float(np.mean(step_toggles >= 2)) if step_toggles.size else 0.0,
            "switch_ge3_frac": float(np.mean(step_toggles >= 3)) if step_toggles.size else 0.0,
            "preserve_warming": bool(args.validation_cyclic_preserve_warming),
        }
        rows.append(row)
        key = (
            float(row["objective_loss_mean"]),
            float(row["power_mean"]),
            int(row["warmup_abort_count"]),
            int(dwell),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_row = row

    if best_row is None:
        raise RuntimeError("validation cyclic calibration produced no rows")
    return tuple(int(x) for x in action_indices), int(best_row["dwell_steps"]), float(best_row["objective_loss_mean"]), rows


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
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
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
) -> tuple[float, float, dict[str, object] | None]:
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="rollout_value_calibration_static")
    static_start_objectives: list[float] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
            seed_offset=146_000 + int(start_idx) * 101,
        )
        static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    grid = [float(x) for x in args.rollout_value_advantage_grid] or [0.0]
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
        candidate_start_objectives: list[float] = []
        power_values: list[float] = []
        warmup_abort_count = 0
        for start_idx, start in enumerate(starts):
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
                starts=(int(start),),
                seed_offset=145_000 + idx * 101 + int(start_idx),
            )
            candidate_start_objectives.append(float(objective))
            power_values.append(float(metrics.get("power_mean", np.nan)))
            warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
        margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(candidate_start_objectives, dtype=float)
        rows.append(
            {
                "policy": "forecast_aware_rollout_value",
                "combo_idx": int(idx),
                "advantage_threshold": float(threshold),
                "objective": float(np.mean(candidate_start_objectives)),
                "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                "warmup_abort_count": int(warmup_abort_count),
                "objective_margin_mean": float(np.mean(margins)),
                "objective_margin_min": float(np.min(margins)),
                "objective_margin_q25": float(np.quantile(margins, 0.25)),
                "negative_start_count": int(np.sum(margins < 0.0)),
                "static_start_objectives": [float(x) for x in static_start_objectives],
                "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
            }
        )
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "rollout_value_calibration.csv", index=False)
    except Exception as exc:
        log(f"rollout-value calibration CSV save skipped: {exc}")
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.deployable_selection_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_guard_pass=bool(args.deployable_selection_require_guard_pass),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    if best_row is None:
        return 1.0e9, float(np.mean(static_start_objectives)), None
    return float(best_row["advantage_threshold"]), float(best_row["objective"]), dict(best_row)


def build_augmented_sequence_value_bank(
    *,
    labels: np.ndarray,
    candidate_masks: np.ndarray,
    anchor_idx: int,
    train_static_table: pd.DataFrame,
    sequence_len: int,
    static_top_k: int,
    support_top_k: int,
    dwell_grid: tuple[int, ...],
    max_sequences: int,
) -> np.ndarray:
    masks = np.asarray(candidate_masks, dtype=bool)
    seq_len = max(1, int(sequence_len))
    n_actions = int(masks.shape[0])
    anchor = int(np.clip(int(anchor_idx), 0, max(n_actions - 1, 0)))
    selected: list[int] = [anchor]
    if int(static_top_k) > 0 and "action_idx" in train_static_table.columns:
        for value in train_static_table["action_idx"].head(max(0, int(static_top_k))).tolist():
            idx = int(value)
            if 0 <= idx < n_actions:
                selected.append(idx)
    support = action_support_from_labels(
        labels,
        n_actions=n_actions,
        top_k=max(0, int(support_top_k)),
        min_count=0,
        anchor_idx=anchor,
    )
    if support is not None:
        selected.extend(int(x) for x in support)
    selected = _unique_ints(selected, limit=n_actions)
    rows: list[np.ndarray] = []
    for action_idx in selected:
        rows.append(np.full(seq_len, int(action_idx), dtype=np.int64))
    cycle_actions = [idx for idx in selected if idx != anchor] or [anchor]
    dwell_values = [max(1, int(x)) for x in dwell_grid] or [1]
    for dwell in dwell_values:
        orders: list[list[int]] = []
        full_order = [anchor, *cycle_actions]
        if len(full_order) > 1:
            for shift in range(len(full_order)):
                orders.append(full_order[shift:] + full_order[:shift])
        for action_idx in cycle_actions:
            if int(action_idx) != anchor:
                orders.append([anchor, int(action_idx)])
                orders.append([int(action_idx), anchor])
        for order in orders:
            rows.append(_cycle_sequence(order, seq_len=seq_len, dwell=dwell))
            if int(max_sequences) > 0 and len(rows) >= int(max_sequences):
                break
        if int(max_sequences) > 0 and len(rows) >= int(max_sequences):
            break
    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for row in rows:
        seq = np.clip(np.asarray(row, dtype=np.int64).reshape(-1), 0, max(n_actions - 1, 0))
        if seq.shape[0] < seq_len:
            seq = np.pad(seq, (0, seq_len - seq.shape[0]), constant_values=anchor)
        seq = seq[:seq_len].astype(np.int64)
        key = tuple(int(x) for x in seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seq)
        if int(max_sequences) > 0 and len(unique) >= int(max_sequences):
            break
    if not unique:
        return np.zeros((0, seq_len), dtype=np.int64)
    return np.vstack(unique).astype(np.int64)


def _cycle_sequence(order: list[int], *, seq_len: int, dwell: int) -> np.ndarray:
    if not order:
        return np.zeros(max(1, int(seq_len)), dtype=np.int64)
    values: list[int] = []
    pos = 0
    while len(values) < int(seq_len):
        values.extend([int(order[pos % len(order)])] * max(1, int(dwell)))
        pos += 1
    return np.asarray(values[: int(seq_len)], dtype=np.int64)


def _unique_ints(values: list[int], *, limit: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        idx = int(value)
        if idx < 0 or idx >= int(limit) or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def calibrate_sequence_value_policy(
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
    sequence_bank: np.ndarray,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, float, dict[str, object] | None]:
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="sequence_value_calibration_static")
    static_start_objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
        static_start_objectives.append(float(objective))
    rows: list[dict[str, object]] = []
    grid = [float(x) for x in args.sequence_value_advantage_grid] or [0.0]
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareSequenceValuePolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            sequence_bank=np.asarray(sequence_bank, dtype=np.int64),
            device=str(args.bc_device),
            advantage_threshold=float(threshold),
            top_k_sequences=int(args.sequence_value_top_k_sequences),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_sequence_value_calib_{idx}",
        )
        candidate_start_objectives: list[float] = []
        power_values: list[float] = []
        warmup_abort_count = 0
        for start_idx, start in enumerate(starts):
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
                starts=(int(start),),
                seed_offset=100_000 + int(start_idx) * 101,
            )
            candidate_start_objectives.append(float(objective))
            power_values.append(float(metrics.get("power_mean", np.nan)))
            warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
        margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(candidate_start_objectives, dtype=float)
        row: dict[str, object] = {
            "policy": "forecast_aware_sequence_value",
            "combo_idx": int(idx),
            "advantage_threshold": float(threshold),
            "objective": float(np.mean(candidate_start_objectives)),
            "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
            "warmup_abort_count": int(warmup_abort_count),
            "objective_margin_mean": float(np.mean(margins)),
            "objective_margin_min": float(np.min(margins)),
            "objective_margin_q25": float(np.quantile(margins, 0.25)),
            "negative_start_count": int(np.sum(margins < 0.0)),
            "static_start_objectives": [float(x) for x in static_start_objectives],
            "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
        }
        rows.append(row)
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "sequence_value_calibration.csv", index=False)
    except Exception as exc:
        log(f"sequence-value calibration CSV save skipped: {exc}")
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.deployable_selection_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_guard_pass=bool(args.deployable_selection_require_guard_pass),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    if best_row is None:
        return 1.0e9, float(np.mean(static_start_objectives)), None
    return float(best_row["advantage_threshold"]), float(best_row["objective"]), dict(best_row)


def calibrate_cost_knn_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    dataset: object,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    allowed_action_indices: tuple[int, ...] | None,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int, float, str, float, dict[str, object] | None]:
    k_grid = [max(1, int(x)) for x in args.cost_knn_k_grid] or [16]
    thresholds = [float(x) for x in args.cost_knn_advantage_grid] or [0.0]
    weightings = [str(x) for x in args.cost_knn_distance_weighting_grid] or ["inverse"]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.cost_knn_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="cost_knn_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    rows: list[dict[str, object]] = []
    combo_idx = 0
    for weighting in weightings:
        if weighting not in {"uniform", "inverse"}:
            raise ValueError(f"Unsupported cost-KNN distance weighting: {weighting}")
        for k in k_grid:
            for threshold in thresholds:
                policy = ForecastAwareCostKNNPolicy(
                    features=np.asarray(dataset.features, dtype=np.float32),
                    costs=np.asarray(dataset.costs, dtype=np.float32),
                    action_masks=np.asarray(dataset.action_masks, dtype=bool),
                    candidate_masks=candidate_masks,
                    forecast_cfg=forecast_cfg,
                    anchor_mask=anchor_mask,
                    allowed_action_indices=allowed_action_indices,
                    k=int(k),
                    advantage_threshold=float(threshold),
                    distance_weighting=str(weighting),
                    preserve_warming=bool(args.bc_preserve_warming),
                    name=f"forecast_aware_cost_knn_calib_{combo_idx}",
                )
                if static_start_objectives:
                    candidate_start_objectives: list[float] = []
                    power_values: list[float] = []
                    warmup_abort_count = 0
                    for start_idx, start in enumerate(starts):
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
                            starts=(int(start),),
                            seed_offset=100_000 + int(start_idx) * 101,
                        )
                        candidate_start_objectives.append(float(objective))
                        power_values.append(float(metrics.get("power_mean", np.nan)))
                        warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                    margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                        candidate_start_objectives,
                        dtype=float,
                    )
                    row: dict[str, object] = {
                        "policy": "forecast_aware_cost_knn",
                        "objective": float(np.mean(candidate_start_objectives)),
                        "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                        "warmup_abort_count": int(warmup_abort_count),
                        "objective_margin_mean": float(np.mean(margins)),
                        "objective_margin_min": float(np.min(margins)),
                        "negative_start_count": int(np.sum(margins < 0.0)),
                        "static_start_objectives": [float(x) for x in static_start_objectives],
                        "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                    }
                else:
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
                        seed_offset=165_000 + combo_idx * 101,
                    )
                    row = {
                        "policy": "forecast_aware_cost_knn",
                        "objective": float(objective),
                        "power_mean": float(metrics.get("power_mean", np.nan)),
                        "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                    }
                row.update(
                    {
                        "combo_idx": int(combo_idx),
                        "k": int(k),
                        "advantage_threshold": float(threshold),
                        "distance_weighting": str(weighting),
                    }
                )
                rows.append(row)
                combo_idx += 1
    if not rows:
        return 16, 0.0, "inverse", float("inf"), None
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "cost_knn_calibration.csv", index=False)
    except Exception as exc:
        log(f"cost-KNN calibration CSV save skipped: {exc}")
    if str(args.cost_knn_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.cost_knn_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
            require_positive_center=bool(args.deployable_selection_require_positive_center),
            require_risk_band=bool(getattr(args, "deployable_selection_require_risk_band", False)),
            risk_min_q25_margin=float(getattr(args, "deployable_selection_risk_min_q25_margin", -1.0e9)),
            risk_max_negative_starts=int(getattr(args, "deployable_selection_risk_max_negative_starts", 1_000_000)),
        )
        if best_row is None:
            best_row = sorted(
                rows,
                key=lambda row: (
                    int(row.get("negative_start_count", 1_000_000)),
                    -float(row.get("objective_margin_mean", -1.0e9)),
                    float(row.get("objective", float("inf"))),
                    int(row.get("combo_idx", 0)),
                ),
            )[0]
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        int(best_row["k"]),
        float(best_row["advantage_threshold"]),
        str(best_row["distance_weighting"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_recurrent_value_policy(
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
) -> tuple[float, float, dict[str, object] | None]:
    grid = [float(x) for x in args.recurrent_value_advantage_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.recurrent_value_calibration_criterion) == "static_margin_guard":
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="recurrent_value_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    rows: list[dict[str, object]] = []
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareRecurrentValuePolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            advantage_threshold=float(threshold),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_recurrent_value_calib_{idx}",
        )
        if static_start_objectives:
            candidate_start_objectives: list[float] = []
            power_values: list[float] = []
            warmup_abort_count = 0
            for start_idx, start in enumerate(starts):
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
                    starts=(int(start),),
                    seed_offset=100_000 + int(start_idx) * 101,
                )
                candidate_start_objectives.append(float(objective))
                power_values.append(float(metrics.get("power_mean", np.nan)))
                warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
            margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                candidate_start_objectives,
                dtype=float,
            )
            row = {
                "policy": "forecast_aware_recurrent_value",
                "objective": float(np.mean(candidate_start_objectives)),
                "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                "warmup_abort_count": int(warmup_abort_count),
                "objective_margin_mean": float(np.mean(margins)),
                "objective_margin_min": float(np.min(margins)),
                "negative_start_count": int(np.sum(margins < 0.0)),
                "static_start_objectives": [float(x) for x in static_start_objectives],
                "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
            }
        else:
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
                seed_offset=155_000 + idx * 101,
            )
            row = {
                "policy": "forecast_aware_recurrent_value",
                "objective": float(objective),
                "power_mean": float(metrics.get("power_mean", np.nan)),
                "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
            }
        row.update({"threshold": float(threshold), "combo_idx": int(idx)})
        rows.append(row)
    if static_start_objectives:
        best_row = choose_deployable_validation_row(
            rows,
            criterion="static_margin_guard",
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
    )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return float(best_row["threshold"]), float(best_row["objective"]), dict(best_row)


def calibrate_recurrent_advantage_policy(
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
) -> tuple[float, float, dict[str, object] | None]:
    grid = [float(x) for x in args.recurrent_advantage_grid] or [0.0]
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.recurrent_advantage_calibration_criterion) == "static_margin_guard":
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="recurrent_advantage_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    rows: list[dict[str, object]] = []
    for idx, threshold in enumerate(grid):
        policy = ForecastAwareRecurrentAdvantagePolicy(
            model=model,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            device=str(args.bc_device),
            allowed_action_indices=allowed_action_indices,
            advantage_threshold=float(threshold),
            preserve_warming=bool(args.bc_preserve_warming),
            name=f"forecast_aware_recurrent_advantage_calib_{idx}",
        )
        if static_start_objectives:
            candidate_start_objectives: list[float] = []
            power_values: list[float] = []
            warmup_abort_count = 0
            for start_idx, start in enumerate(starts):
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
                    starts=(int(start),),
                    seed_offset=100_000 + int(start_idx) * 101,
                )
                candidate_start_objectives.append(float(objective))
                power_values.append(float(metrics.get("power_mean", np.nan)))
                warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
            margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                candidate_start_objectives,
                dtype=float,
            )
            row = {
                "policy": "forecast_aware_recurrent_advantage",
                "objective": float(np.mean(candidate_start_objectives)),
                "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                "warmup_abort_count": int(warmup_abort_count),
                "objective_margin_mean": float(np.mean(margins)),
                "objective_margin_min": float(np.min(margins)),
                "negative_start_count": int(np.sum(margins < 0.0)),
                "static_start_objectives": [float(x) for x in static_start_objectives],
                "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
            }
        else:
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
                seed_offset=165_000 + idx * 101,
            )
            row = {
                "policy": "forecast_aware_recurrent_advantage",
                "objective": float(objective),
                "power_mean": float(metrics.get("power_mean", np.nan)),
                "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
            }
        row.update({"threshold": float(threshold), "combo_idx": int(idx)})
        rows.append(row)
    if static_start_objectives:
        best_row = choose_deployable_validation_row(
            rows,
            criterion="static_margin_guard",
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
        )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return float(best_row["threshold"]), float(best_row["objective"]), dict(best_row)


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
) -> tuple[int, tuple[int, ...] | None, float, float, dict[str, object] | None]:
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="advantage_residual_calibration_static")
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
            seed_offset=166_000 + int(start_idx) * 101,
        )
        static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    top_k_grid = [int(x) for x in args.advantage_residual_support_grid]
    if not top_k_grid:
        top_k_grid = [int(args.advantage_residual_support_top_k)]
    threshold_grid = [float(x) for x in args.advantage_residual_grid] or [0.0]
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
            candidate_start_objectives: list[float] = []
            power_values: list[float] = []
            warmup_abort_count = 0
            for start_idx, start in enumerate(starts):
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
                    starts=(int(start),),
                    seed_offset=160_000 + combo_idx * 101 + int(start_idx),
                )
                candidate_start_objectives.append(float(objective))
                power_values.append(float(metrics.get("power_mean", np.nan)))
                warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
            margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(candidate_start_objectives, dtype=float)
            rows.append(
                {
                    "policy": "forecast_aware_advantage_residual",
                    "combo_idx": int(combo_idx),
                    "support_top_k": max(0, int(top_k)),
                    "advantage_threshold": float(threshold),
                    "objective": float(np.mean(candidate_start_objectives)),
                    "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                    "warmup_abort_count": int(warmup_abort_count),
                    "objective_margin_mean": float(np.mean(margins)),
                    "objective_margin_min": float(np.min(margins)),
                    "objective_margin_q25": float(np.quantile(margins, 0.25)),
                    "negative_start_count": int(np.sum(margins < 0.0)),
                    "static_start_objectives": [float(x) for x in static_start_objectives],
                    "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                }
            )
            combo_idx += 1
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "advantage_residual_calibration.csv", index=False)
    except Exception as exc:
        log(f"advantage-residual calibration CSV save skipped: {exc}")
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.deployable_selection_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_guard_pass=bool(args.deployable_selection_require_guard_pass),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    if best_row is None:
        fallback_top_k = max(0, int(top_k_grid[0]))
        fallback_support = action_support_from_labels(
            labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=fallback_top_k,
            min_count=int(args.bc_action_support_min_count),
            anchor_idx=anchor_idx,
        )
        return fallback_top_k, fallback_support, 1.0e9, float(np.mean(static_start_objectives)), None
    best_top_k = int(best_row["support_top_k"])
    best_threshold = float(best_row["advantage_threshold"])
    best_objective = float(best_row["objective"])
    best_support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(0, int(best_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    return int(best_top_k), best_support, float(best_threshold), float(best_objective), dict(best_row)


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
) -> tuple[int | None, float | None, str | None, float, dict[str, object] | None]:
    support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(1, int(args.event_threshold_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=None,
    )
    if support is None:
        return None, None, None, float("inf"), None
    event_actions = [int(idx) for idx in support if anchor_idx is None or int(idx) != int(anchor_idx)]
    if not event_actions and anchor_idx is not None:
        event_actions = [int(anchor_idx)]
    thresholds = [float(x) for x in args.event_threshold_grid] or [0.5]
    aggregations = [str(x) for x in args.event_threshold_aggregation_grid] or ["max"]
    total_combos = len(event_actions) * len(aggregations) * len(thresholds)
    log(
        "event-threshold calibration grid: "
        f"actions={len(event_actions)} aggregations={len(aggregations)} "
        f"thresholds={len(thresholds)} combos={total_combos} starts={len(starts)}"
    )
    rows: list[dict[str, object]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.event_threshold_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="event_threshold_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    combo_idx = 0
    for action_idx in event_actions:
        for aggregation in aggregations:
            if aggregation not in {"max", "mean", "first"}:
                raise ValueError(f"Unsupported event-threshold aggregation: {aggregation}")
            for threshold in thresholds:
                if combo_idx == 0 or combo_idx % 10 == 0:
                    log(f"event-threshold calibration progress: combo={combo_idx + 1}/{total_combos}")
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
                row: dict[str, object] = {
                    "policy": "forecast_aware_event_threshold",
                    "objective": float(objective),
                    "power_mean": float(metrics.get("power_mean", np.nan)),
                    "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                    "action_idx": int(action_idx),
                    "threshold": float(threshold),
                    "aggregation": str(aggregation),
                    "combo_idx": int(combo_idx),
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
                combo_idx += 1
    if not rows:
        return None, None, None, float("inf"), None
    log(f"event-threshold calibration complete: evaluated={len(rows)}")
    if str(args.event_threshold_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.event_threshold_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
        )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        int(best_row["action_idx"]),
        float(best_row["threshold"]),
        str(best_row["aggregation"]),
        float(best_row["objective"]),
        dict(best_row),
    )


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
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
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


def calibrate_option_planner_policy(
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
) -> tuple[
    tuple[int, ...] | None,
    np.ndarray | None,
    float | None,
    str | None,
    int | None,
    int | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float,
    dict[str, object] | None,
]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    masks = np.asarray(candidate_masks, dtype=bool)
    if labels_arr.size == 0:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    labels_arr = labels_arr[(labels_arr >= 0) & (labels_arr < int(masks.shape[0]))]
    if labels_arr.size == 0:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.option_planner_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=None,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    options = tuple(int(idx) for idx in support if anchor_idx is None or int(idx) != int(anchor_idx))
    if not options:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    target_rates = np.mean(masks[labels_arr].astype(float), axis=0)
    thresholds = [float(x) for x in args.option_planner_threshold_grid] or [0.5]
    aggregations = [str(x) for x in args.option_planner_aggregation_grid] or ["max"]
    dwells = [max(1, int(x)) for x in args.option_planner_min_dwell_grid] or [1]
    cooldowns = [max(0, int(x)) for x in args.option_planner_cooldown_grid] or [0]
    target_weights = [float(x) for x in args.option_planner_target_rate_grid] or [1.0]
    rate_balance_weights = [float(x) for x in args.option_planner_rate_balance_grid] or [0.0]
    freshness_weights = [float(x) for x in args.option_planner_freshness_grid] or [0.25]
    transport_weights = [float(x) for x in args.option_planner_transport_grid] or [0.25]
    power_weights = [float(x) for x in args.option_planner_power_grid] or [0.05]
    switch_weights = [float(x) for x in args.option_planner_switch_grid] or [0.05]
    min_socs = [float(x) for x in args.option_planner_min_soc_grid] or [0.0]
    total_combos = (
        len(thresholds)
        * len(aggregations)
        * len(dwells)
        * len(cooldowns)
        * len(target_weights)
        * len(rate_balance_weights)
        * len(freshness_weights)
        * len(transport_weights)
        * len(power_weights)
        * len(switch_weights)
        * len(min_socs)
    )
    log(
        "option-planner calibration grid: "
        f"options={len(options)} combos={total_combos} starts={len(starts)}"
    )
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.option_planner_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="option_planner_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    rows: list[dict[str, object]] = []
    combo_idx = 0
    for aggregation in aggregations:
        if aggregation not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported option-planner aggregation: {aggregation}")
        for threshold in thresholds:
            for dwell in dwells:
                for cooldown in cooldowns:
                    for target_weight in target_weights:
                        for rate_balance_weight in rate_balance_weights:
                            for freshness_weight in freshness_weights:
                                for transport_weight in transport_weights:
                                    for power_weight in power_weights:
                                        for switch_weight in switch_weights:
                                            for min_soc in min_socs:
                                                if combo_idx == 0 or combo_idx % 25 == 0:
                                                    log(
                                                        "option-planner calibration progress: "
                                                        f"combo={combo_idx + 1}/{total_combos}"
                                                    )
                                                policy = ForecastAwareOptionPlannerPolicy(
                                                    candidate_masks=masks,
                                                    forecast_cfg=forecast_cfg,
                                                    anchor_mask=anchor_mask,
                                                    option_action_indices=options,
                                                    target_rates=target_rates,
                                                    threshold=float(threshold),
                                                    aggregation=str(aggregation),
                                                    min_dwell=int(dwell),
                                                    cooldown=int(cooldown),
                                                    target_rate_weight=float(target_weight),
                                                    rate_balance_weight=float(rate_balance_weight),
                                                    freshness_weight=float(freshness_weight),
                                                    transport_weight=float(transport_weight),
                                                    power_weight=float(power_weight),
                                                    switch_weight=float(switch_weight),
                                                    min_soc=float(min_soc),
                                                    preserve_warming=bool(args.bc_preserve_warming),
                                                    name=f"forecast_aware_option_planner_calib_{combo_idx}",
                                                )
                                                if static_start_objectives:
                                                    candidate_start_objectives: list[float] = []
                                                    power_values: list[float] = []
                                                    warmup_abort_count = 0
                                                    for start_idx, start in enumerate(starts):
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
                                                            starts=(int(start),),
                                                            seed_offset=100_000 + int(start_idx) * 101,
                                                        )
                                                        candidate_start_objectives.append(float(objective))
                                                        power_values.append(float(metrics.get("power_mean", np.nan)))
                                                        warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                                                    margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                                        candidate_start_objectives,
                                                        dtype=float,
                                                    )
                                                    row: dict[str, object] = {
                                                        "policy": "forecast_aware_option_planner",
                                                        "objective": float(np.mean(candidate_start_objectives)),
                                                        "power_mean": float(np.nanmean(power_values))
                                                        if power_values
                                                        else float("nan"),
                                                        "warmup_abort_count": int(warmup_abort_count),
                                                        "objective_margin_mean": float(np.mean(margins)),
                                                        "objective_margin_min": float(np.min(margins)),
                                                        "negative_start_count": int(np.sum(margins < 0.0)),
                                                        "static_start_objectives": [
                                                            float(x) for x in static_start_objectives
                                                        ],
                                                        "candidate_start_objectives": [
                                                            float(x) for x in candidate_start_objectives
                                                        ],
                                                    }
                                                else:
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
                                                        seed_offset=220_000 + combo_idx * 101,
                                                    )
                                                    row = {
                                                        "policy": "forecast_aware_option_planner",
                                                        "objective": float(objective),
                                                        "power_mean": float(metrics.get("power_mean", np.nan)),
                                                        "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                                                    }
                                                row.update(
                                                    {
                                                        "combo_idx": int(combo_idx),
                                                        "threshold": float(threshold),
                                                        "aggregation": str(aggregation),
                                                        "min_dwell": int(dwell),
                                                        "cooldown": int(cooldown),
                                                        "target_rate_weight": float(target_weight),
                                                        "rate_balance_weight": float(rate_balance_weight),
                                                        "freshness_weight": float(freshness_weight),
                                                        "transport_weight": float(transport_weight),
                                                        "power_weight": float(power_weight),
                                                        "switch_weight": float(switch_weight),
                                                        "min_soc": float(min_soc),
                                                    }
                                                )
                                                rows.append(row)
                                                combo_idx += 1
    if not rows:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "option_planner_calibration.csv", index=False)
    except Exception as exc:
        log(f"option-planner calibration CSV save skipped: {exc}")
    if str(args.option_planner_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.option_planner_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
        )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        tuple(int(x) for x in options),
        np.asarray(target_rates, dtype=float),
        float(best_row["threshold"]),
        str(best_row["aggregation"]),
        int(best_row["min_dwell"]),
        int(best_row["cooldown"]),
        float(best_row["target_rate_weight"]),
        float(best_row["rate_balance_weight"]),
        float(best_row["freshness_weight"]),
        float(best_row["transport_weight"]),
        float(best_row["power_weight"]),
        float(best_row["switch_weight"]),
        float(best_row["min_soc"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_macro_option_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int, int, float, str, str, int, float, dict[str, object] | None]:
    segment_grid = [max(1, int(x)) for x in args.macro_option_segment_grid] or [8]
    k_grid = [max(1, int(x)) for x in args.macro_option_k_grid] or [4]
    thresholds = [float(x) for x in args.macro_option_threshold_grid] or [0.5]
    aggregations = [str(x) for x in args.macro_option_aggregation_grid] or ["max"]
    weightings = [str(x) for x in args.macro_option_distance_weighting_grid] or ["inverse"]
    refresh_grid = [max(0, int(x)) for x in args.macro_option_refresh_grid] or [0]
    total = (
        len(segment_grid)
        * len(k_grid)
        * len(thresholds)
        * len(aggregations)
        * len(weightings)
        * len(refresh_grid)
    )
    log(f"macro-option calibration grid: combos={total} starts={len(starts)}")
    masks = np.asarray(candidate_masks, dtype=bool)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.macro_option_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="macro_option_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    combo_idx = 0
    for aggregation in aggregations:
        if aggregation not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported macro-option aggregation: {aggregation}")
        for weighting in weightings:
            if weighting not in {"uniform", "inverse"}:
                raise ValueError(f"Unsupported macro-option distance weighting: {weighting}")
            for segment_len in segment_grid:
                for k in k_grid:
                    for threshold in thresholds:
                        for refresh_interval in refresh_grid:
                            if combo_idx == 0 or combo_idx % 25 == 0:
                                log(
                                    "macro-option calibration progress: "
                                    f"combo={combo_idx + 1}/{total}"
                                )
                            policy = ForecastAwareMacroOptionPolicy(
                                features=np.asarray(features, dtype=np.float32),
                                labels=np.asarray(labels, dtype=np.int64),
                                candidate_masks=masks,
                                step_indices=np.asarray(step_indices, dtype=np.int64),
                                forecast_cfg=forecast_cfg,
                                anchor_mask=anchor_mask,
                                segment_len=int(segment_len),
                                snippet_stride=int(args.macro_option_snippet_stride),
                                k=int(k),
                                event_threshold=float(threshold),
                                aggregation=str(aggregation),
                                distance_weighting=str(weighting),
                                refresh_interval=int(refresh_interval),
                                max_lookahead=int(args.macro_option_max_lookahead),
                                preserve_warming=bool(args.bc_preserve_warming),
                                name=f"forecast_aware_macro_option_calib_{combo_idx}",
                            )
                            if static_start_objectives:
                                candidate_start_objectives: list[float] = []
                                power_values: list[float] = []
                                warmup_abort_count = 0
                                for start_idx, start in enumerate(starts):
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
                                        starts=(int(start),),
                                        seed_offset=100_000 + int(start_idx) * 101,
                                    )
                                    candidate_start_objectives.append(float(objective))
                                    power_values.append(float(metrics.get("power_mean", np.nan)))
                                    warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                                margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                    candidate_start_objectives,
                                    dtype=float,
                                )
                                row: dict[str, object] = {
                                    "policy": "forecast_aware_macro_option",
                                    "objective": float(np.mean(candidate_start_objectives)),
                                    "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                                    "warmup_abort_count": int(warmup_abort_count),
                                    "objective_margin_mean": float(np.mean(margins)),
                                    "objective_margin_min": float(np.min(margins)),
                                    "negative_start_count": int(np.sum(margins < 0.0)),
                                    "static_start_objectives": [float(x) for x in static_start_objectives],
                                    "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                                }
                            else:
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
                                row = {
                                    "policy": "forecast_aware_macro_option",
                                    "objective": float(objective),
                                    "power_mean": float(metrics.get("power_mean", np.nan)),
                                    "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                                }
                            row.update(
                                {
                                    "combo_idx": int(combo_idx),
                                    "segment_len": int(segment_len),
                                    "k": int(k),
                                    "event_threshold": float(threshold),
                                    "aggregation": str(aggregation),
                                    "distance_weighting": str(weighting),
                                    "refresh_interval": int(refresh_interval),
                                }
                            )
                            rows.append(row)
                            combo_idx += 1
    if not rows:
        return 8, 4, 1.0, "max", "inverse", 0, float("inf"), None
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "macro_option_calibration.csv", index=False)
    except Exception as exc:
        log(f"macro-option calibration CSV save skipped: {exc}")
    if str(args.macro_option_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.macro_option_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
            require_positive_center=bool(args.deployable_selection_require_positive_center),
            require_risk_band=bool(args.deployable_selection_require_risk_band),
            risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
            risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
        )
        if best_row is None:
            best_row = sorted(
                rows,
                key=lambda row: (
                    int(row.get("negative_start_count", 1_000_000)),
                    -float(row.get("objective_margin_mean", -1.0e9)),
                    float(row.get("objective", float("inf"))),
                    int(row.get("combo_idx", 0)),
                ),
            )[0]
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("warmup_abort_count", 0)),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        int(best_row["segment_len"]),
        int(best_row["k"]),
        float(best_row["event_threshold"]),
        str(best_row["aggregation"]),
        str(best_row["distance_weighting"]),
        int(best_row["refresh_interval"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def collect_teacher_improvement_gate_dataset(
    env: object,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    anchor_idx: int,
    label_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    margins: list[float] = []
    masks = np.asarray(candidate_masks, dtype=bool)
    for start in start_indices:
        env.reset(start_idx=int(start))
        for _ in range(int(steps_per_start)):
            state = env._state().astype(np.float32)
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(state, forecast).astype(np.float32)
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            anchor_cost = float(costs[int(anchor_idx)]) if 0 <= int(anchor_idx) < int(costs.shape[0]) else float("inf")
            finite = costs[np.isfinite(costs)]
            best_cost = float(np.min(finite)) if finite.size else float("inf")
            margin = float(anchor_cost - best_cost) if np.isfinite(anchor_cost) and np.isfinite(best_cost) else 0.0
            features.append(feature)
            labels.append(float(margin > float(label_margin)))
            margins.append(float(margin))
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not features:
        raise ValueError("No teacher-improvement gate samples collected")
    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(margins, dtype=np.float32),
    )


def calibrate_teacher_improvement_gate_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    gate_model: object,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[int, int, float, str, str, int, float, dict[str, object] | None]:
    segment_grid = [max(1, int(x)) for x in args.macro_option_segment_grid] or [8]
    k_grid = [max(1, int(x)) for x in args.macro_option_k_grid] or [4]
    gate_thresholds = [float(x) for x in args.teacher_improvement_gate_threshold_grid] or [0.6]
    aggregations = [str(x) for x in args.macro_option_aggregation_grid] or ["mean"]
    weightings = [str(x) for x in args.macro_option_distance_weighting_grid] or ["inverse"]
    refresh_grid = [max(0, int(x)) for x in args.macro_option_refresh_grid] or [0]
    total = (
        len(segment_grid)
        * len(k_grid)
        * len(gate_thresholds)
        * len(aggregations)
        * len(weightings)
        * len(refresh_grid)
    )
    log(f"teacher-improvement gate calibration grid: combos={total} starts={len(starts)}")
    masks = np.asarray(candidate_masks, dtype=bool)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="teacher_improvement_gate_static")
    static_start_objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
        static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    combo_idx = 0
    for aggregation in aggregations:
        if aggregation not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported macro-option aggregation: {aggregation}")
        for weighting in weightings:
            if weighting not in {"uniform", "inverse"}:
                raise ValueError(f"Unsupported macro-option distance weighting: {weighting}")
            for segment_len in segment_grid:
                for k in k_grid:
                    for threshold in gate_thresholds:
                        for refresh_interval in refresh_grid:
                            if combo_idx == 0 or combo_idx % 25 == 0:
                                log(
                                    "teacher-improvement gate calibration progress: "
                                    f"combo={combo_idx + 1}/{total}"
                                )
                            dynamic = ForecastAwareMacroOptionPolicy(
                                features=np.asarray(features, dtype=np.float32),
                                labels=np.asarray(labels, dtype=np.int64),
                                candidate_masks=masks,
                                step_indices=np.asarray(step_indices, dtype=np.int64),
                                forecast_cfg=forecast_cfg,
                                anchor_mask=anchor_mask,
                                segment_len=int(segment_len),
                                snippet_stride=int(args.macro_option_snippet_stride),
                                k=int(k),
                                event_threshold=0.0,
                                aggregation=str(aggregation),
                                distance_weighting=str(weighting),
                                refresh_interval=int(refresh_interval),
                                max_lookahead=int(args.macro_option_max_lookahead),
                                preserve_warming=bool(args.bc_preserve_warming),
                                name=f"teacher_improvement_gate_dynamic_{combo_idx}",
                            )
                            policy = ForecastAwareTeacherImprovementGatePolicy(
                                gate_model=gate_model,
                                dynamic_policy=dynamic,
                                forecast_cfg=forecast_cfg,
                                anchor_mask=anchor_mask,
                                feature_mean=np.asarray(feature_mean, dtype=np.float32),
                                feature_std=np.asarray(feature_std, dtype=np.float32),
                                threshold=float(threshold),
                                preserve_warming=bool(args.bc_preserve_warming),
                                device=str(args.bc_device),
                                name=f"forecast_aware_teacher_improvement_gate_calib_{combo_idx}",
                            )
                            candidate_start_objectives: list[float] = []
                            power_values: list[float] = []
                            warmup_abort_count = 0
                            for start_idx, start in enumerate(starts):
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
                                    starts=(int(start),),
                                    seed_offset=100_000 + int(start_idx) * 101,
                                )
                                candidate_start_objectives.append(float(objective))
                                power_values.append(float(metrics.get("power_mean", np.nan)))
                                warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                            margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                candidate_start_objectives,
                                dtype=float,
                            )
                            rows.append(
                                {
                                    "policy": "forecast_aware_teacher_improvement_gate",
                                    "objective": float(np.mean(candidate_start_objectives)),
                                    "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                                    "warmup_abort_count": int(warmup_abort_count),
                                    "objective_margin_mean": float(np.mean(margins)),
                                    "objective_margin_min": float(np.min(margins)),
                                    "negative_start_count": int(np.sum(margins < 0.0)),
                                    "static_start_objectives": [float(x) for x in static_start_objectives],
                                    "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                                    "combo_idx": int(combo_idx),
                                    "segment_len": int(segment_len),
                                    "k": int(k),
                                    "gate_threshold": float(threshold),
                                    "aggregation": str(aggregation),
                                    "distance_weighting": str(weighting),
                                    "refresh_interval": int(refresh_interval),
                                }
                            )
                            combo_idx += 1
    if not rows:
        return 8, 4, 1.0, "mean", "inverse", 0, float("inf"), None
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "teacher_improvement_gate_calibration.csv", index=False)
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.deployable_selection_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    if best_row is None:
        best_row = sorted(
            rows,
            key=lambda row: (
                int(row.get("negative_start_count", 1_000_000)),
                -float(row.get("objective_margin_mean", -1.0e9)),
                float(row.get("objective", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        int(best_row["segment_len"]),
        int(best_row["k"]),
        float(best_row["gate_threshold"]),
        str(best_row["aggregation"]),
        str(best_row["distance_weighting"]),
        int(best_row["refresh_interval"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_runtime_risk_guard_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    option_action_indices: tuple[int, ...],
    option_target_rates: np.ndarray | None,
    option_threshold: float,
    option_aggregation: str,
    option_min_dwell: int,
    option_cooldown: int,
    option_target_rate_weight: float,
    option_rate_balance_weight: float,
    option_freshness_weight: float,
    option_transport_weight: float,
    option_power_weight: float,
    option_switch_weight: float,
    option_min_soc: float,
    state_columns: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[float, str, int, float, float, float, float, float, float, dict[str, object] | None]:
    thresholds = [float(x) for x in args.runtime_risk_threshold_grid] or [0.8]
    aggregations = [str(x) for x in args.runtime_risk_aggregation_grid] or ["max"]
    windows = [max(1, int(x)) for x in args.runtime_risk_window_grid] or [8]
    event_weights = [float(x) for x in args.runtime_risk_event_weight_grid] or [1.0]
    freshness_weights = [float(x) for x in args.runtime_risk_freshness_weight_grid] or [0.25]
    transport_weights = [float(x) for x in args.runtime_risk_transport_weight_grid] or [0.25]
    soc_weights = [float(x) for x in args.runtime_risk_soc_weight_grid] or [0.0]
    min_socs = [float(x) for x in args.runtime_risk_min_soc_grid] or [0.0]
    total = (
        len(thresholds)
        * len(aggregations)
        * len(windows)
        * len(event_weights)
        * len(freshness_weights)
        * len(transport_weights)
        * len(soc_weights)
        * len(min_socs)
    )
    log(f"runtime-risk guard calibration grid: combos={total} starts={len(starts)}")
    masks = np.asarray(candidate_masks, dtype=bool)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.runtime_risk_guard_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="runtime_risk_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    combo_idx = 0
    for aggregation in aggregations:
        if aggregation not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported runtime-risk aggregation: {aggregation}")
        for threshold in thresholds:
            for window_steps in windows:
                for event_weight in event_weights:
                    for freshness_weight in freshness_weights:
                        for transport_weight in transport_weights:
                            for soc_weight in soc_weights:
                                for min_soc in min_socs:
                                    if combo_idx == 0 or combo_idx % 25 == 0:
                                        log(
                                            "runtime-risk guard calibration progress: "
                                            f"combo={combo_idx + 1}/{total}"
                                        )
                                    dynamic_policy = ForecastAwareOptionPlannerPolicy(
                                        candidate_masks=masks,
                                        forecast_cfg=forecast_cfg,
                                        anchor_mask=anchor_mask,
                                        option_action_indices=tuple(int(x) for x in option_action_indices),
                                        target_rates=option_target_rates,
                                        threshold=float(option_threshold),
                                        aggregation=str(option_aggregation),
                                        min_dwell=int(option_min_dwell),
                                        cooldown=int(option_cooldown),
                                        target_rate_weight=float(option_target_rate_weight),
                                        rate_balance_weight=float(option_rate_balance_weight),
                                        freshness_weight=float(option_freshness_weight),
                                        transport_weight=float(option_transport_weight),
                                        power_weight=float(option_power_weight),
                                        switch_weight=float(option_switch_weight),
                                        min_soc=float(option_min_soc),
                                        preserve_warming=bool(args.bc_preserve_warming),
                                        name=f"runtime_risk_inner_option_{combo_idx}",
                                    )
                                    policy = ForecastAwareRuntimeRiskGuardPolicy(
                                        dynamic_policy=dynamic_policy,
                                        forecast_cfg=forecast_cfg,
                                        anchor_mask=anchor_mask,
                                        threshold=float(threshold),
                                        aggregation=str(aggregation),
                                        window_steps=int(window_steps),
                                        event_weight=float(event_weight),
                                        freshness_weight=float(freshness_weight),
                                        transport_weight=float(transport_weight),
                                        soc_weight=float(soc_weight),
                                        min_soc=float(min_soc),
                                        preserve_warming=bool(args.bc_preserve_warming),
                                        name=f"forecast_aware_runtime_risk_guard_calib_{combo_idx}",
                                    )
                                    if static_start_objectives:
                                        candidate_start_objectives: list[float] = []
                                        power_values: list[float] = []
                                        warmup_abort_count = 0
                                        for start_idx, start in enumerate(starts):
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
                                                starts=(int(start),),
                                                seed_offset=100_000 + int(start_idx) * 101,
                                            )
                                            candidate_start_objectives.append(float(objective))
                                            power_values.append(float(metrics.get("power_mean", np.nan)))
                                            warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                                        margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                            candidate_start_objectives,
                                            dtype=float,
                                        )
                                        row: dict[str, object] = {
                                            "policy": "forecast_aware_runtime_risk_guard",
                                            "objective": float(np.mean(candidate_start_objectives)),
                                            "power_mean": float(np.nanmean(power_values))
                                            if power_values
                                            else float("nan"),
                                            "warmup_abort_count": int(warmup_abort_count),
                                            "objective_margin_mean": float(np.mean(margins)),
                                            "objective_margin_min": float(np.min(margins)),
                                            "negative_start_count": int(np.sum(margins < 0.0)),
                                            "static_start_objectives": [float(x) for x in static_start_objectives],
                                            "candidate_start_objectives": [
                                                float(x) for x in candidate_start_objectives
                                            ],
                                        }
                                    else:
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
                                            seed_offset=240_000 + combo_idx * 101,
                                        )
                                        row = {
                                            "policy": "forecast_aware_runtime_risk_guard",
                                            "objective": float(objective),
                                            "power_mean": float(metrics.get("power_mean", np.nan)),
                                            "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                                        }
                                    row.update(
                                        {
                                            "combo_idx": int(combo_idx),
                                            "threshold": float(threshold),
                                            "aggregation": str(aggregation),
                                            "window_steps": int(window_steps),
                                            "event_weight": float(event_weight),
                                            "freshness_weight": float(freshness_weight),
                                            "transport_weight": float(transport_weight),
                                            "soc_weight": float(soc_weight),
                                            "min_soc": float(min_soc),
                                        }
                                    )
                                    rows.append(row)
                                    combo_idx += 1
    if not rows:
        return (0.8, "max", 8, 1.0, 0.25, 0.25, 0.0, 0.0, float("inf"), None)
    try:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "runtime_risk_guard_calibration.csv", index=False)
    except Exception as exc:
        log(f"runtime-risk guard calibration CSV save skipped: {exc}")
    if str(args.runtime_risk_guard_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.runtime_risk_guard_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
            require_positive_center=bool(args.deployable_selection_require_positive_center),
            require_risk_band=bool(getattr(args, "deployable_selection_require_risk_band", False)),
            risk_min_q25_margin=float(getattr(args, "deployable_selection_risk_min_q25_margin", -1.0e9)),
            risk_max_negative_starts=int(getattr(args, "deployable_selection_risk_max_negative_starts", 1_000_000)),
        )
        if best_row is None:
            best_row = sorted(
                rows,
                key=lambda row: (
                    int(row.get("negative_start_count", 1_000_000)),
                    -float(row.get("objective_margin_mean", -1.0e9)),
                    float(row.get("objective", float("inf"))),
                    int(row.get("combo_idx", 0)),
                ),
            )[0]
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        float(best_row["threshold"]),
        str(best_row["aggregation"]),
        int(best_row["window_steps"]),
        float(best_row["event_weight"]),
        float(best_row["freshness_weight"]),
        float(best_row["transport_weight"]),
        float(best_row["soc_weight"]),
        float(best_row["min_soc"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def build_window_eligibility_feature(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    forecast_cfg: ForecastContextConfig,
    start: int,
) -> np.ndarray:
    env = build_env_for_dataset(truth, sensors, constraints, cfg, oracle)
    env.reset(start_idx=int(start))
    state = env._state().astype(np.float32)
    forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
    return append_event_forecast(state, forecast).astype(np.float32)


def window_eligibility_sample_starts(
    base_starts: tuple[int, ...],
    *,
    train_steps: int,
    window_steps: int,
    samples_per_start: int,
    max_windows: int,
) -> tuple[int, ...]:
    samples: list[int] = []
    max_offset = max(0, int(train_steps) - int(window_steps))
    count = max(1, int(samples_per_start))
    if count == 1 or max_offset <= 0:
        offsets = [0]
    else:
        offsets = sorted({int(round(x)) for x in np.linspace(0, max_offset, num=count)})
    for start in base_starts:
        for offset in offsets:
            samples.append(int(start) + int(offset))
    if int(max_windows) > 0 and len(samples) > int(max_windows):
        positions = np.linspace(0, len(samples) - 1, num=int(max_windows))
        samples = [samples[int(round(pos))] for pos in positions]
    return tuple(dict.fromkeys(int(x) for x in samples))


def evaluate_policy_start_objectives(
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
    steps: int,
    seed_offset: int,
) -> list[float]:
    objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        result, _ = evaluate_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=policy,
            steps=int(steps),
            start_indices=(int(start),),
            seed_offset=int(seed_offset) + int(start_idx) * 101,
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
        objectives.append(
            float(
                final_objective(
                    metrics,
                    mode=str(args.objective_mode),
                    task_error_weight=float(args.task_error_weight),
                )
            )
        )
    return objectives


def window_eligibility_target_rates(
    *,
    labels: np.ndarray,
    candidate_masks: np.ndarray,
    anchor_mask: tuple[bool, ...],
    blend: float,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    masks = np.asarray(candidate_masks, dtype=bool)
    labels_arr = labels_arr[(labels_arr >= 0) & (labels_arr < int(masks.shape[0]))]
    if labels_arr.size:
        teacher_rates = np.mean(masks[labels_arr].astype(float), axis=0)
    else:
        teacher_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
    anchor_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
    weight = float(np.clip(float(blend), 0.0, 1.0))
    return np.clip((1.0 - weight) * anchor_rates + weight * teacher_rates, 0.0, 1.0)


def make_window_eligibility_inner_policy(
    *,
    dynamic_family: str,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    support: tuple[int, ...] | None,
    target_rates: np.ndarray,
    min_dwell: int,
    freshness_weight: float,
    transport_weight: float,
    power_weight: float,
    switch_weight: float,
    min_soc: float,
    macro_features: np.ndarray | None = None,
    macro_labels: np.ndarray | None = None,
    macro_step_indices: np.ndarray | None = None,
    macro_k: int = 4,
    macro_snippet_stride: int = 1,
    macro_max_lookahead: int = 8,
    preserve_warming: bool,
    name: str,
) -> object:
    family = str(dynamic_family)
    if family == "macro":
        if macro_features is None or macro_labels is None or macro_step_indices is None:
            raise ValueError("macro window-eligibility inner requires teacher features, labels, and step indices")
        return ForecastAwareMacroOptionPolicy(
            features=np.asarray(macro_features, dtype=np.float32),
            labels=np.asarray(macro_labels, dtype=np.int64),
            candidate_masks=candidate_masks,
            step_indices=np.asarray(macro_step_indices, dtype=np.int64),
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            segment_len=max(1, int(macro_max_lookahead)),
            snippet_stride=max(1, int(macro_snippet_stride)),
            k=max(1, int(macro_k)),
            event_threshold=0.0,
            aggregation="mean",
            distance_weighting="inverse",
            refresh_interval=0,
            max_lookahead=max(1, int(macro_max_lookahead)),
            preserve_warming=bool(preserve_warming),
            name=str(name),
        )
    if family != "option":
        raise ValueError(f"Unsupported window-eligibility dynamic family: {family}")
    return ForecastAwareOptionPlannerPolicy(
        candidate_masks=candidate_masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor_mask,
        option_action_indices=tuple(int(x) for x in support) if support is not None else (),
        target_rates=np.asarray(target_rates, dtype=float),
        threshold=0.0,
        aggregation="max",
        min_dwell=int(min_dwell),
        cooldown=0,
        target_rate_weight=1.0,
        rate_balance_weight=0.0,
        freshness_weight=float(freshness_weight),
        transport_weight=float(transport_weight),
        power_weight=float(power_weight),
        switch_weight=float(switch_weight),
        min_soc=float(min_soc),
        preserve_warming=bool(preserve_warming),
        name=str(name),
    )


def build_window_candidate_specs(
    args: argparse.Namespace,
    *,
    labels: np.ndarray,
    candidate_masks: np.ndarray,
    anchor_mask: tuple[bool, ...],
    support: tuple[int, ...],
    window_steps: int,
) -> list[dict[str, object]]:
    families = [str(x) for x in args.window_candidate_family_grid] or ["option"]
    supported = {"option", "macro", "rate"}
    unknown = sorted(set(families) - supported)
    if unknown:
        raise ValueError(f"Unsupported window-candidate families: {unknown}")
    specs: list[dict[str, object]] = []
    anchor_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
    for family in families:
        if family == "option":
            for blend in [float(x) for x in args.window_candidate_option_blend_grid] or [1.0]:
                target_rates = window_eligibility_target_rates(
                    labels=labels,
                    candidate_masks=candidate_masks,
                    anchor_mask=anchor_mask,
                    blend=float(blend),
                )
                for min_dwell in [max(1, int(x)) for x in args.window_candidate_option_min_dwell_grid] or [2]:
                    for freshness_weight in [float(x) for x in args.window_candidate_option_freshness_grid] or [0.25]:
                        for transport_weight in [float(x) for x in args.window_candidate_option_transport_grid] or [0.25]:
                            for power_weight in [float(x) for x in args.window_candidate_option_power_grid] or [0.05]:
                                for switch_weight in [float(x) for x in args.window_candidate_option_switch_grid] or [0.05]:
                                    for min_soc in [float(x) for x in args.window_candidate_min_soc_grid] or [0.0]:
                                        specs.append(
                                            {
                                                "family": "option",
                                                "target_rates": target_rates.astype(float).copy(),
                                                "blend": float(blend),
                                                "min_dwell": int(min_dwell),
                                                "freshness_weight": float(freshness_weight),
                                                "transport_weight": float(transport_weight),
                                                "power_weight": float(power_weight),
                                                "switch_weight": float(switch_weight),
                                                "min_soc": float(min_soc),
                                                "macro_k": 0,
                                                "rate_freshness_weight": 0.0,
                                                "rate_power_weight": 0.0,
                                            }
                                        )
        elif family == "macro":
            for macro_k in [max(1, int(x)) for x in args.window_candidate_macro_k_grid] or [4]:
                for min_soc in [float(x) for x in args.window_candidate_min_soc_grid] or [0.0]:
                    specs.append(
                        {
                            "family": "macro",
                            "target_rates": anchor_rates.copy(),
                            "blend": float("nan"),
                            "min_dwell": 1,
                            "freshness_weight": 0.0,
                            "transport_weight": 0.0,
                            "power_weight": 0.0,
                            "switch_weight": 0.0,
                            "min_soc": float(min_soc),
                            "macro_k": int(macro_k),
                            "rate_freshness_weight": 0.0,
                            "rate_power_weight": 0.0,
                        }
                    )
        elif family == "rate":
            for blend in [float(x) for x in args.window_candidate_rate_blend_grid] or [1.0]:
                target_rates = window_eligibility_target_rates(
                    labels=labels,
                    candidate_masks=candidate_masks,
                    anchor_mask=anchor_mask,
                    blend=float(blend),
                )
                for freshness_weight in [float(x) for x in args.window_candidate_rate_freshness_grid] or [0.0]:
                    for power_weight in [float(x) for x in args.window_candidate_rate_power_grid] or [0.0]:
                        for min_soc in [float(x) for x in args.window_candidate_min_soc_grid] or [0.0]:
                            specs.append(
                                {
                                    "family": "rate",
                                    "target_rates": target_rates.astype(float).copy(),
                                    "blend": float(blend),
                                    "min_dwell": 1,
                                    "freshness_weight": 0.0,
                                    "transport_weight": 0.0,
                                    "power_weight": 0.0,
                                    "switch_weight": 0.0,
                                    "min_soc": float(min_soc),
                                    "macro_k": 0,
                                    "rate_freshness_weight": float(freshness_weight),
                                    "rate_power_weight": float(power_weight),
                                }
                            )
    max_candidates = int(args.window_candidate_max_candidates)
    if max_candidates > 0 and len(specs) > max_candidates:
        specs = specs[:max_candidates]
    return specs


def make_window_candidate_inner_policy(
    *,
    args: argparse.Namespace,
    spec: dict[str, object],
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: tuple[bool, ...],
    support: tuple[int, ...],
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    window_steps: int,
    preserve_warming: bool,
    name: str,
) -> object:
    family = str(spec.get("family", "option"))
    if family in {"option", "macro"}:
        return make_window_eligibility_inner_policy(
            dynamic_family=family,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
            anchor_mask=anchor_mask,
            support=support,
            target_rates=np.asarray(spec.get("target_rates"), dtype=float),
            min_dwell=int(spec.get("min_dwell", 2)),
            freshness_weight=float(spec.get("freshness_weight", 0.25)),
            transport_weight=float(spec.get("transport_weight", 0.25)),
            power_weight=float(spec.get("power_weight", 0.05)),
            switch_weight=float(spec.get("switch_weight", 0.05)),
            min_soc=float(spec.get("min_soc", 0.0)),
            macro_features=np.asarray(features, dtype=np.float32),
            macro_labels=np.asarray(labels, dtype=np.int64),
            macro_step_indices=np.asarray(step_indices, dtype=np.int64),
            macro_k=int(spec.get("macro_k", 4)),
            macro_snippet_stride=int(args.window_candidate_macro_snippet_stride),
            macro_max_lookahead=min(
                int(window_steps),
                max(1, int(args.window_candidate_macro_max_lookahead)),
            ),
            preserve_warming=bool(preserve_warming),
            name=str(name),
        )
    if family == "rate":
        return ForecastAwareTeacherRatePolicy(
            candidate_masks=candidate_masks,
            target_rates=np.asarray(spec.get("target_rates"), dtype=float),
            allowed_action_indices=support,
            freshness_weight=float(spec.get("rate_freshness_weight", 0.0)),
            power_weight=float(spec.get("rate_power_weight", 0.0)),
            preserve_warming=bool(preserve_warming),
            anchor_mask=anchor_mask,
            name=str(name),
        )
    raise ValueError(f"Unsupported window-candidate family: {family}")


def _window_candidate_spec_for_csv(spec: dict[str, object], *, candidate_id: int) -> dict[str, object]:
    out = {k: v for k, v in spec.items() if k != "target_rates"}
    out["candidate_id"] = int(candidate_id)
    out["target_rates"] = json.dumps([float(x) for x in np.asarray(spec.get("target_rates"), dtype=float).reshape(-1)])
    return out


def calibrate_window_candidate_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    train_cfg: object,
    validation_cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    train_starts: tuple[int, ...],
    validation_starts: tuple[int, ...],
) -> tuple[
    tuple[int, ...] | None,
    list[dict[str, object]] | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int | None,
    int | None,
    float | None,
    float | None,
    str | None,
    float | None,
    float,
    dict[str, object] | None,
]:
    support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(1, int(args.window_candidate_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        return (None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    window_grid = [max(1, int(x)) for x in args.window_candidate_window_grid] or [16]
    k_grid = [max(1, int(x)) for x in args.window_candidate_k_grid] or [5]
    threshold_grid = [float(x) for x in args.window_candidate_margin_grid] or [0.0]
    quantile_grid = [float(np.clip(float(x), 0.0, 1.0)) for x in args.window_candidate_quantile_grid] or [0.25]
    weighting_grid = [str(x) for x in args.window_candidate_distance_weighting_grid] or ["inverse"]
    full_rollout_calibration = bool(getattr(args, "window_candidate_full_rollout_calibration", False))
    log(
        "window-candidate calibration grid: "
        f"families={list(args.window_candidate_family_grid)} windows={window_grid} "
        f"k={k_grid} thresholds={threshold_grid} quantiles={quantile_grid} "
        f"full_rollout_calibration={full_rollout_calibration}"
    )

    static_train_cache: dict[int, list[float]] = {}
    static_validation_cache: dict[int, list[float]] = {}
    static_full_validation_start_objectives: list[float] | None = None
    feature_cache: dict[int, np.ndarray] = {}
    spec_cache: dict[int, list[dict[str, object]]] = {}
    memory_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    training_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    row_idx = 0
    for window_steps in window_grid:
        train_window_starts = window_eligibility_sample_starts(
            train_starts,
            train_steps=int(args.train_steps),
            window_steps=int(window_steps),
            samples_per_start=int(args.window_candidate_samples_per_start),
            max_windows=int(args.window_candidate_max_train_windows),
        )
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="window_candidate_train_static")
        static_train_cache[int(window_steps)] = evaluate_policy_start_objectives(
            args,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=train_cfg,
            oracle=oracle,
            policy=static_policy,
            state_columns=state_columns,
            sensor_ids=sensor_ids,
            starts=train_window_starts,
            steps=int(window_steps),
            seed_offset=410_000,
        )
        if full_rollout_calibration:
            if static_full_validation_start_objectives is None:
                static_full_validation_start_objectives = []
                for start_idx, start in enumerate(validation_starts):
                    _, objective = evaluate_validation_policy_metrics(
                        args,
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=validation_cfg,
                        oracle=oracle,
                        policy=static_policy,
                        state_columns=state_columns,
                        sensor_ids=sensor_ids,
                        starts=(int(start),),
                        seed_offset=100_000 + int(start_idx) * 101,
                    )
                    static_full_validation_start_objectives.append(float(objective))
        else:
            static_validation_cache[int(window_steps)] = evaluate_policy_start_objectives(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=validation_cfg,
                oracle=oracle,
                policy=static_policy,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=validation_starts,
                steps=int(window_steps),
                seed_offset=420_000,
            )
        feature_cache[int(window_steps)] = np.vstack(
            [
                build_window_eligibility_feature(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=train_cfg,
                    oracle=oracle,
                    forecast_cfg=forecast_cfg,
                    start=int(start),
                )
                for start in train_window_starts
            ]
        ).astype(np.float32)
        specs = build_window_candidate_specs(
            args,
            labels=labels,
            candidate_masks=candidate_masks,
            anchor_mask=anchor_mask,
            support=tuple(int(x) for x in support),
            window_steps=int(window_steps),
        )
        if not specs:
            continue
        spec_cache[int(window_steps)] = specs
        memory_features: list[np.ndarray] = []
        memory_margins: list[np.ndarray] = []
        memory_candidate_ids: list[np.ndarray] = []
        for candidate_id, spec in enumerate(specs):
            log(
                "window-candidate train replay: "
                f"window={window_steps} candidate={candidate_id + 1}/{len(specs)} family={spec.get('family')}"
            )
            policy = make_window_candidate_inner_policy(
                args=args,
                spec=spec,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=anchor_mask,
                support=tuple(int(x) for x in support),
                features=np.asarray(features, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                step_indices=np.asarray(step_indices, dtype=np.int64),
                window_steps=int(window_steps),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"window_candidate_train_{candidate_id}",
            )
            candidate_objectives = evaluate_policy_start_objectives(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=train_cfg,
                oracle=oracle,
                policy=policy,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=train_window_starts,
                steps=int(window_steps),
                seed_offset=430_000 + int(candidate_id) * 997,
            )
            margins = np.asarray(static_train_cache[int(window_steps)], dtype=float) - np.asarray(
                candidate_objectives,
                dtype=float,
            )
            memory_features.append(feature_cache[int(window_steps)])
            memory_margins.append(margins.astype(float))
            memory_candidate_ids.append(np.full(margins.shape[0], int(candidate_id), dtype=int))
            for sample_idx, start in enumerate(train_window_starts):
                row = {
                    "window_steps": int(window_steps),
                    "start": int(start),
                    "static_objective": float(static_train_cache[int(window_steps)][sample_idx]),
                    "candidate_objective": float(candidate_objectives[sample_idx]),
                    "margin": float(margins[sample_idx]),
                }
                row.update(_window_candidate_spec_for_csv(spec, candidate_id=int(candidate_id)))
                training_rows.append(row)
        memory_cache[int(window_steps)] = (
            np.vstack(memory_features).astype(np.float32),
            np.concatenate(memory_margins).astype(float),
            np.concatenate(memory_candidate_ids).astype(int),
        )
        for k in k_grid:
            for threshold in threshold_grid:
                for quantile in quantile_grid:
                    for weighting in weighting_grid:
                        candidate_policies = tuple(
                            make_window_candidate_inner_policy(
                                args=args,
                                spec=spec,
                                candidate_masks=candidate_masks,
                                forecast_cfg=forecast_cfg,
                                anchor_mask=anchor_mask,
                                support=tuple(int(x) for x in support),
                                features=np.asarray(features, dtype=np.float32),
                                labels=np.asarray(labels, dtype=np.int64),
                                step_indices=np.asarray(step_indices, dtype=np.int64),
                                window_steps=int(window_steps),
                                preserve_warming=bool(args.bc_preserve_warming),
                                name=f"window_candidate_val_{row_idx}_{candidate_id}",
                            )
                            for candidate_id, spec in enumerate(specs)
                        )
                        mem_features, mem_margins, mem_ids = memory_cache[int(window_steps)]
                        policy = ForecastAwareWindowCandidatePolicy(
                            memory_features=mem_features,
                            memory_margins=mem_margins,
                            memory_candidate_ids=mem_ids,
                            candidate_policies=candidate_policies,
                            forecast_cfg=forecast_cfg,
                            anchor_mask=anchor_mask,
                            k=int(k),
                            margin_threshold=float(threshold),
                            score_quantile=float(quantile),
                            window_steps=int(window_steps),
                            distance_weighting=str(weighting),
                            min_soc=0.0,
                            min_candidate_neighbors=int(args.window_candidate_min_neighbors),
                            preserve_warming=bool(args.bc_preserve_warming),
                            name=f"forecast_aware_window_candidate_calib_{row_idx}",
                        )
                        if full_rollout_calibration:
                            if static_full_validation_start_objectives is None:
                                raise RuntimeError("full-rollout static validation cache was not initialized")
                            candidate_start_objectives = []
                            power_values: list[float] = []
                            warmup_abort_count = 0
                            for start_idx, start in enumerate(validation_starts):
                                metrics, objective = evaluate_validation_policy_metrics(
                                    args,
                                    truth=truth,
                                    sensors=sensors,
                                    constraints=constraints,
                                    cfg=validation_cfg,
                                    oracle=oracle,
                                    policy=policy,
                                    state_columns=state_columns,
                                    sensor_ids=sensor_ids,
                                    starts=(int(start),),
                                    seed_offset=100_000 + int(start_idx) * 101,
                                )
                                candidate_start_objectives.append(float(objective))
                                power_values.append(float(metrics.get("power_mean", np.nan)))
                                warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                            static_reference = static_full_validation_start_objectives
                            calibration_steps = int(args.static_selection_steps)
                            power_mean = float(np.nanmean(power_values)) if power_values else float("nan")
                        else:
                            candidate_start_objectives = evaluate_policy_start_objectives(
                                args,
                                truth=truth,
                                sensors=sensors,
                                constraints=constraints,
                                cfg=validation_cfg,
                                oracle=oracle,
                                policy=policy,
                                state_columns=state_columns,
                                sensor_ids=sensor_ids,
                                starts=validation_starts,
                                steps=int(window_steps),
                                seed_offset=440_000 + row_idx * 997,
                            )
                            static_reference = static_validation_cache[int(window_steps)]
                            calibration_steps = int(window_steps)
                            power_mean = float("nan")
                            warmup_abort_count = 0
                        margins = np.asarray(static_reference, dtype=float) - np.asarray(
                            candidate_start_objectives,
                            dtype=float,
                        )
                        rows.append(
                            {
                                "policy": "forecast_aware_window_candidate",
                                "objective": float(np.mean(candidate_start_objectives)),
                                "power_mean": float(power_mean),
                                "warmup_abort_count": int(warmup_abort_count),
                                "objective_margin_mean": float(np.mean(margins)),
                                "objective_margin_min": float(np.min(margins)),
                                "objective_margin_q25": float(np.quantile(margins, 0.25)),
                                "negative_start_count": int(np.sum(margins < 0.0)),
                                "static_start_objectives": [float(x) for x in static_reference],
                                "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                                "row_idx": int(row_idx),
                                "window_steps": int(window_steps),
                                "calibration_steps": int(calibration_steps),
                                "full_rollout_calibration": bool(full_rollout_calibration),
                                "k": int(k),
                                "margin_threshold": float(threshold),
                                "score_quantile": float(quantile),
                                "distance_weighting": str(weighting),
                                "candidate_count": int(len(specs)),
                                "memory_rows": int(mem_features.shape[0]),
                                "train_margin_mean": float(np.mean(mem_margins)),
                                "train_margin_min": float(np.min(mem_margins)),
                                "train_positive_rate": float(np.mean(mem_margins > 0.0)),
                            }
                        )
                        row_idx += 1
    if training_rows:
        pd.DataFrame(training_rows).to_csv(Path(args.out_dir) / "window_candidate_training_windows.csv", index=False)
    if not rows:
        return (support, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.window_candidate_calibration_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "window_candidate_calibration.csv", index=False)
    if best_row is None:
        return (support, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    selected_window_steps = int(best_row["window_steps"])
    mem_features, mem_margins, mem_ids = memory_cache[selected_window_steps]
    payload = dict(best_row)
    payload["support_indices"] = [int(x) for x in support]
    payload["candidate_specs"] = [
        {
            **{k: v for k, v in spec.items() if k != "target_rates"},
            "target_rates": [float(x) for x in np.asarray(spec.get("target_rates"), dtype=float).reshape(-1)],
        }
        for spec in spec_cache[selected_window_steps]
    ]
    return (
        support,
        spec_cache[selected_window_steps],
        mem_features,
        mem_margins,
        mem_ids,
        int(selected_window_steps),
        int(best_row["k"]),
        float(best_row["margin_threshold"]),
        float(best_row["score_quantile"]),
        str(best_row["distance_weighting"]),
        0.0,
        float(best_row["objective"]),
        payload,
    )


def calibrate_window_eligibility_policy(
    args: argparse.Namespace,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    train_cfg: object,
    validation_cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    features: np.ndarray,
    labels: np.ndarray,
    step_indices: np.ndarray,
    anchor_idx: int | None,
    anchor_mask: tuple[bool, ...],
    state_columns: tuple[str, ...],
    train_starts: tuple[int, ...],
    validation_starts: tuple[int, ...],
) -> tuple[
    tuple[int, ...] | None,
    str | None,
    int | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int | None,
    int | None,
    float | None,
    float | None,
    int | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    str | None,
    float,
    dict[str, object] | None,
]:
    support = action_support_from_labels(
        labels,
        n_actions=int(candidate_masks.shape[0]),
        top_k=max(1, int(args.window_eligibility_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )

    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    window_grid = [max(1, int(x)) for x in args.window_eligibility_window_grid] or [16]
    k_grid = [max(1, int(x)) for x in args.window_eligibility_k_grid] or [3]
    margin_grid = [float(x) for x in args.window_eligibility_margin_grid] or [0.0]
    dynamic_grid = [str(x) for x in args.window_eligibility_dynamic_grid] or ["option"]
    macro_k_grid = [max(1, int(x)) for x in args.window_eligibility_macro_k_grid] or [4]
    blend_grid = [float(x) for x in args.window_eligibility_blend_grid] or [1.0]
    dwell_grid = [max(1, int(x)) for x in args.window_eligibility_min_dwell_grid] or [2]
    freshness_grid = [float(x) for x in args.window_eligibility_freshness_grid] or [0.25]
    transport_grid = [float(x) for x in args.window_eligibility_transport_grid] or [0.25]
    power_grid = [float(x) for x in args.window_eligibility_power_grid] or [0.05]
    switch_grid = [float(x) for x in args.window_eligibility_switch_grid] or [0.05]
    soc_grid = [float(x) for x in args.window_eligibility_min_soc_grid] or [0.0]
    weighting_grid = [str(x) for x in args.window_eligibility_distance_weighting_grid] or ["inverse"]
    for family in dynamic_grid:
        if family not in {"option", "macro"}:
            raise ValueError(f"Unsupported window-eligibility dynamic family: {family}")
    option_inner = (
        len(window_grid)
        * len(blend_grid)
        * len(dwell_grid)
        * len(freshness_grid)
        * len(transport_grid)
        * len(power_grid)
        * len(switch_grid)
        * len(soc_grid)
        if "option" in set(dynamic_grid)
        else 0
    )
    macro_inner = len(window_grid) * len(macro_k_grid) if "macro" in set(dynamic_grid) else 0
    total_inner = int(option_inner + macro_inner)
    total_rows = total_inner * len(k_grid) * len(margin_grid) * len(weighting_grid)
    log(
        "window-eligibility calibration grid: "
        f"inner={total_inner} rows={total_rows} train_base={len(train_starts)} validation={len(validation_starts)}"
    )

    static_train_cache: dict[int, list[float]] = {}
    static_validation_cache: dict[int, list[float]] = {}
    feature_cache: dict[int, np.ndarray] = {}
    training_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    selected_payload: dict[str, object] | None = None
    combo_idx = 0
    row_idx = 0
    for window_steps in window_grid:
        train_window_starts = window_eligibility_sample_starts(
            train_starts,
            train_steps=int(args.train_steps),
            window_steps=int(window_steps),
            samples_per_start=int(args.window_eligibility_samples_per_start),
            max_windows=int(args.window_eligibility_max_train_windows),
        )
        if int(window_steps) not in static_train_cache:
            static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="window_eligibility_train_static")
            static_train_cache[int(window_steps)] = evaluate_policy_start_objectives(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=train_cfg,
                oracle=oracle,
                policy=static_policy,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=train_window_starts,
                steps=int(window_steps),
                seed_offset=310_000,
            )
            static_validation_cache[int(window_steps)] = evaluate_policy_start_objectives(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=validation_cfg,
                oracle=oracle,
                policy=static_policy,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=validation_starts,
                steps=int(window_steps),
                seed_offset=320_000,
            )
        if int(window_steps) not in feature_cache:
            feature_cache[int(window_steps)] = np.vstack(
                [
                    build_window_eligibility_feature(
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=train_cfg,
                        oracle=oracle,
                        forecast_cfg=forecast_cfg,
                        start=int(start),
                    )
                    for start in train_window_starts
                ]
            ).astype(np.float32)

        inner_rows: list[dict[str, object]] = []
        if "option" in set(dynamic_grid):
            for blend in blend_grid:
                target_rates = window_eligibility_target_rates(
                    labels=labels,
                    candidate_masks=candidate_masks,
                    anchor_mask=anchor_mask,
                    blend=float(blend),
                )
                for min_dwell in dwell_grid:
                    for freshness_weight in freshness_grid:
                        for transport_weight in transport_grid:
                            for power_weight in power_grid:
                                for switch_weight in switch_grid:
                                    for min_soc in soc_grid:
                                        inner_rows.append(
                                            {
                                                "dynamic_family": "option",
                                                "target_rates": target_rates,
                                                "blend": float(blend),
                                                "min_dwell": int(min_dwell),
                                                "freshness_weight": float(freshness_weight),
                                                "transport_weight": float(transport_weight),
                                                "power_weight": float(power_weight),
                                                "switch_weight": float(switch_weight),
                                                "min_soc": float(min_soc),
                                                "macro_k": 0,
                                            }
                                        )
        if "macro" in set(dynamic_grid):
            anchor_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
            for macro_k in macro_k_grid:
                inner_rows.append(
                    {
                        "dynamic_family": "macro",
                        "target_rates": anchor_rates,
                        "blend": float("nan"),
                        "min_dwell": 1,
                        "freshness_weight": 0.0,
                        "transport_weight": 0.0,
                        "power_weight": 0.0,
                        "switch_weight": 0.0,
                        "min_soc": 0.0,
                        "macro_k": int(macro_k),
                    }
                )
        for inner in inner_rows:
            if combo_idx == 0 or combo_idx % 10 == 0:
                log(
                    "window-eligibility inner progress: "
                    f"combo={combo_idx + 1}/{total_inner} family={inner['dynamic_family']}"
                )
            dynamic_train = make_window_eligibility_inner_policy(
                dynamic_family=str(inner["dynamic_family"]),
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=anchor_mask,
                support=support,
                target_rates=np.asarray(inner["target_rates"], dtype=float),
                min_dwell=int(inner["min_dwell"]),
                freshness_weight=float(inner["freshness_weight"]),
                transport_weight=float(inner["transport_weight"]),
                power_weight=float(inner["power_weight"]),
                switch_weight=float(inner["switch_weight"]),
                min_soc=float(inner["min_soc"]),
                macro_features=np.asarray(features, dtype=np.float32),
                macro_labels=np.asarray(labels, dtype=np.int64),
                macro_step_indices=np.asarray(step_indices, dtype=np.int64),
                macro_k=int(inner["macro_k"]),
                macro_snippet_stride=int(args.window_eligibility_macro_snippet_stride),
                macro_max_lookahead=min(int(window_steps), max(1, int(args.window_eligibility_macro_max_lookahead))),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"window_eligibility_train_inner_{combo_idx}",
            )
            dynamic_train_objectives = evaluate_policy_start_objectives(
                args,
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=train_cfg,
                oracle=oracle,
                policy=dynamic_train,
                state_columns=state_columns,
                sensor_ids=sensor_ids,
                starts=train_window_starts,
                steps=int(window_steps),
                seed_offset=330_000 + combo_idx * 997,
            )
            train_margins = np.asarray(static_train_cache[int(window_steps)], dtype=float) - np.asarray(
                dynamic_train_objectives,
                dtype=float,
            )
            memory_features = feature_cache[int(window_steps)]
            for sample_idx, start in enumerate(train_window_starts):
                training_rows.append(
                    {
                        "combo_idx": int(combo_idx),
                        "dynamic_family": str(inner["dynamic_family"]),
                        "window_steps": int(window_steps),
                        "start": int(start),
                        "static_objective": float(static_train_cache[int(window_steps)][sample_idx]),
                        "dynamic_objective": float(dynamic_train_objectives[sample_idx]),
                        "margin": float(train_margins[sample_idx]),
                        "blend": float(inner["blend"]),
                        "min_dwell": int(inner["min_dwell"]),
                        "freshness_weight": float(inner["freshness_weight"]),
                        "transport_weight": float(inner["transport_weight"]),
                        "power_weight": float(inner["power_weight"]),
                        "switch_weight": float(inner["switch_weight"]),
                        "min_soc": float(inner["min_soc"]),
                        "macro_k": int(inner["macro_k"]),
                    }
                )
            for k in k_grid:
                for margin_threshold in margin_grid:
                    for weighting in weighting_grid:
                        dynamic_validation = make_window_eligibility_inner_policy(
                            dynamic_family=str(inner["dynamic_family"]),
                            candidate_masks=candidate_masks,
                            forecast_cfg=forecast_cfg,
                            anchor_mask=anchor_mask,
                            support=support,
                            target_rates=np.asarray(inner["target_rates"], dtype=float),
                            min_dwell=int(inner["min_dwell"]),
                            freshness_weight=float(inner["freshness_weight"]),
                            transport_weight=float(inner["transport_weight"]),
                            power_weight=float(inner["power_weight"]),
                            switch_weight=float(inner["switch_weight"]),
                            min_soc=float(inner["min_soc"]),
                            macro_features=np.asarray(features, dtype=np.float32),
                            macro_labels=np.asarray(labels, dtype=np.int64),
                            macro_step_indices=np.asarray(step_indices, dtype=np.int64),
                            macro_k=int(inner["macro_k"]),
                            macro_snippet_stride=int(args.window_eligibility_macro_snippet_stride),
                            macro_max_lookahead=min(
                                int(window_steps),
                                max(1, int(args.window_eligibility_macro_max_lookahead)),
                            ),
                            preserve_warming=bool(args.bc_preserve_warming),
                            name=f"window_eligibility_val_inner_{row_idx}",
                        )
                        policy = ForecastAwareWindowEligibilityPolicy(
                            memory_features=memory_features,
                            memory_margins=train_margins,
                            dynamic_policy=dynamic_validation,
                            forecast_cfg=forecast_cfg,
                            anchor_mask=anchor_mask,
                            k=int(k),
                            margin_threshold=float(margin_threshold),
                            window_steps=int(window_steps),
                            distance_weighting=str(weighting),
                            min_soc=float(inner["min_soc"]),
                            preserve_warming=bool(args.bc_preserve_warming),
                            name=f"forecast_aware_window_eligibility_calib_{row_idx}",
                        )
                        candidate_start_objectives = evaluate_policy_start_objectives(
                            args,
                            truth=truth,
                            sensors=sensors,
                            constraints=constraints,
                            cfg=validation_cfg,
                            oracle=oracle,
                            policy=policy,
                            state_columns=state_columns,
                            sensor_ids=sensor_ids,
                            starts=validation_starts,
                            steps=int(window_steps),
                            seed_offset=340_000 + row_idx * 997,
                        )
                        margins = np.asarray(
                            static_validation_cache[int(window_steps)],
                            dtype=float,
                        ) - np.asarray(candidate_start_objectives, dtype=float)
                        row = {
                            "policy": "forecast_aware_window_eligibility",
                            "objective": float(np.mean(candidate_start_objectives)),
                            "power_mean": float("nan"),
                            "warmup_abort_count": 0,
                            "objective_margin_mean": float(np.mean(margins)),
                            "objective_margin_min": float(np.min(margins)),
                            "negative_start_count": int(np.sum(margins < 0.0)),
                            "static_start_objectives": [
                                float(x) for x in static_validation_cache[int(window_steps)]
                            ],
                            "candidate_start_objectives": [
                                float(x) for x in candidate_start_objectives
                            ],
                            "combo_idx": int(combo_idx),
                            "row_idx": int(row_idx),
                            "dynamic_family": str(inner["dynamic_family"]),
                            "window_steps": int(window_steps),
                            "k": int(k),
                            "margin_threshold": float(margin_threshold),
                            "distance_weighting": str(weighting),
                            "blend": float(inner["blend"]),
                            "min_dwell": int(inner["min_dwell"]),
                            "freshness_weight": float(inner["freshness_weight"]),
                            "transport_weight": float(inner["transport_weight"]),
                            "power_weight": float(inner["power_weight"]),
                            "switch_weight": float(inner["switch_weight"]),
                            "min_soc": float(inner["min_soc"]),
                            "macro_k": int(inner["macro_k"]),
                            "train_margin_mean": float(np.mean(train_margins)),
                            "train_margin_min": float(np.min(train_margins)),
                            "train_positive_rate": float(np.mean(train_margins > 0.0)),
                        }
                        rows.append(row)
                        row_idx += 1
            combo_idx += 1
    if training_rows:
        pd.DataFrame(training_rows).to_csv(Path(args.out_dir) / "window_eligibility_training_windows.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / "window_eligibility_calibration.csv", index=False)
    if not rows:
        return (
            support,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )

    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.window_eligibility_calibration_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    if best_row is None:
        return (
            support,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )

    selected_window_steps = int(best_row["window_steps"])
    selected_combo = int(best_row["combo_idx"])
    selected_training = [row for row in training_rows if int(row["combo_idx"]) == selected_combo]
    selected_margins = np.asarray([float(row["margin"]) for row in selected_training], dtype=float)
    selected_features = feature_cache[selected_window_steps]
    if selected_margins.shape[0] != selected_features.shape[0]:
        raise ValueError("selected window-eligibility feature/margin rows are misaligned")
    if str(best_row.get("dynamic_family", "option")) == "macro":
        selected_target_rates = np.asarray(anchor_mask, dtype=float).reshape(-1)
    else:
        selected_target_rates = window_eligibility_target_rates(
            labels=labels,
            candidate_masks=candidate_masks,
            anchor_mask=anchor_mask,
            blend=float(best_row["blend"]),
        )
    selected_payload = dict(best_row)
    selected_payload["support_indices"] = [int(x) for x in support]
    selected_payload["target_rates"] = [float(x) for x in selected_target_rates.reshape(-1)]
    return (
        support,
        str(best_row.get("dynamic_family", "option")),
        int(best_row.get("macro_k", 0)),
        selected_target_rates,
        selected_features,
        selected_margins,
        int(best_row["window_steps"]),
        int(best_row["k"]),
        float(best_row["margin_threshold"]),
        float(best_row["blend"]),
        int(best_row["min_dwell"]),
        float(best_row["freshness_weight"]),
        float(best_row["transport_weight"]),
        float(best_row["power_weight"]),
        float(best_row["switch_weight"]),
        float(best_row["min_soc"]),
        str(best_row["distance_weighting"]),
        float(best_row["objective"]),
        selected_payload,
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
) -> tuple[tuple[int, ...] | None, float | None, float | None, float | None, float | None, float, dict[str, object] | None]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return None, None, None, None, None, float("inf"), None
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
    rows: list[dict[str, object]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.contextual_duty_calibration_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="contextual_duty_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
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
                    if static_start_objectives:
                        candidate_start_objectives: list[float] = []
                        power_values: list[float] = []
                        warmup_abort_count = 0
                        for start_idx, start in enumerate(starts):
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
                                starts=(int(start),),
                                seed_offset=100_000 + int(start_idx) * 101,
                            )
                            candidate_start_objectives.append(float(objective))
                            power_values.append(float(metrics.get("power_mean", np.nan)))
                            warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                        margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                            candidate_start_objectives,
                            dtype=float,
                        )
                        row = {
                            "policy": "forecast_aware_contextual_duty",
                            "objective": float(np.mean(candidate_start_objectives)),
                            "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                            "warmup_abort_count": int(warmup_abort_count),
                            "objective_margin_mean": float(np.mean(margins)),
                            "objective_margin_min": float(np.min(margins)),
                            "negative_start_count": int(np.sum(margins < 0.0)),
                            "static_start_objectives": [float(x) for x in static_start_objectives],
                            "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                        }
                    else:
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
                        row = {
                            "policy": "forecast_aware_contextual_duty",
                            "objective": float(objective),
                            "power_mean": float(metrics.get("power_mean", np.nan)),
                            "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                        }
                    row.update(
                        {
                            "combo_idx": int(combo_idx),
                            "blend": float(blend),
                            "deficit_weight": float(deficit_weight),
                            "freshness_weight": float(freshness_weight),
                            "power_weight": float(power_weight),
                        }
                    )
                    rows.append(row)
                    combo_idx += 1
    if static_start_objectives:
        best_row = choose_deployable_validation_row(
            rows,
            criterion=str(args.contextual_duty_calibration_criterion),
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
        )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        tuple(int(x) for x in support),
        float(best_row["blend"]),
        float(best_row["deficit_weight"]),
        float(best_row["freshness_weight"]),
        float(best_row["power_weight"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_utility_planner_policy(
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
) -> tuple[
    tuple[int, ...] | None,
    np.ndarray | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int | None,
    str | None,
    float,
    dict[str, object] | None,
]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    masks = np.asarray(candidate_masks, dtype=bool)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.utility_planner_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    target_rates = np.mean(masks[labels_arr].astype(float), axis=0)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="utility_planner_calibration_static")
    static_start_objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
        static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    combo_idx = 0
    for event_weight in [float(x) for x in args.utility_planner_event_weight_grid] or [1.0]:
        for magnitude_weight in [float(x) for x in args.utility_planner_magnitude_weight_grid] or [1.0]:
            for variability_weight in [float(x) for x in args.utility_planner_variability_weight_grid] or [0.5]:
                for freshness_weight in [float(x) for x in args.utility_planner_freshness_grid] or [0.0]:
                    for target_rate_weight in [float(x) for x in args.utility_planner_target_rate_grid] or [0.0]:
                        for anchor_bias in [float(x) for x in args.utility_planner_anchor_bias_grid] or [0.0]:
                            for power_weight in [float(x) for x in args.utility_planner_power_grid] or [0.0]:
                                for switch_weight in [float(x) for x in args.utility_planner_switch_grid] or [0.0]:
                                    for min_soc in [float(x) for x in args.utility_planner_min_soc_grid] or [0.0]:
                                        for min_dwell in [max(1, int(x)) for x in args.utility_planner_dwell_grid] or [1]:
                                            for aggregation in [str(x) for x in args.utility_planner_aggregation_grid] or ["max"]:
                                                policy = ForecastAwareUtilityPlannerPolicy(
                                                    candidate_masks=masks,
                                                    forecast_cfg=forecast_cfg,
                                                    anchor_mask=anchor_mask,
                                                    allowed_action_indices=support,
                                                    target_rates=target_rates,
                                                    event_weight=float(event_weight),
                                                    magnitude_weight=float(magnitude_weight),
                                                    variability_weight=float(variability_weight),
                                                    freshness_weight=float(freshness_weight),
                                                    target_rate_weight=float(target_rate_weight),
                                                    anchor_bias=float(anchor_bias),
                                                    power_weight=float(power_weight),
                                                    switch_weight=float(switch_weight),
                                                    min_soc=float(min_soc),
                                                    min_dwell=int(min_dwell),
                                                    aggregation=str(aggregation),
                                                    preserve_warming=bool(args.bc_preserve_warming),
                                                    name=f"forecast_aware_utility_planner_calib_{combo_idx}",
                                                )
                                                candidate_start_objectives: list[float] = []
                                                power_values: list[float] = []
                                                warmup_abort_count = 0
                                                for start_idx, start in enumerate(starts):
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
                                                        starts=(int(start),),
                                                        seed_offset=100_000 + int(start_idx) * 101,
                                                    )
                                                    candidate_start_objectives.append(float(objective))
                                                    power_values.append(float(metrics.get("power_mean", np.nan)))
                                                    warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                                                margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                                    candidate_start_objectives,
                                                    dtype=float,
                                                )
                                                rows.append(
                                                    {
                                                        "policy": "forecast_aware_utility_planner",
                                                        "objective": float(np.mean(candidate_start_objectives)),
                                                        "power_mean": float(np.nanmean(power_values))
                                                        if power_values
                                                        else float("nan"),
                                                        "warmup_abort_count": int(warmup_abort_count),
                                                        "objective_margin_mean": float(np.mean(margins)),
                                                        "objective_margin_min": float(np.min(margins)),
                                                        "negative_start_count": int(np.sum(margins < 0.0)),
                                                        "static_start_objectives": [float(x) for x in static_start_objectives],
                                                        "candidate_start_objectives": [
                                                            float(x) for x in candidate_start_objectives
                                                        ],
                                                        "combo_idx": int(combo_idx),
                                                        "event_weight": float(event_weight),
                                                        "magnitude_weight": float(magnitude_weight),
                                                        "variability_weight": float(variability_weight),
                                                        "freshness_weight": float(freshness_weight),
                                                        "target_rate_weight": float(target_rate_weight),
                                                        "anchor_bias": float(anchor_bias),
                                                        "power_weight": float(power_weight),
                                                        "switch_weight": float(switch_weight),
                                                        "min_soc": float(min_soc),
                                                        "min_dwell": int(min_dwell),
                                                        "aggregation": str(aggregation),
                                                    }
                                                )
                                                combo_idx += 1
    if not rows:
        return (support, target_rates, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.utility_planner_calibration_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "utility_planner_calibration.csv", index=False)
    if best_row is None:
        return (support, target_rates, None, None, None, None, None, None, None, None, None, None, None, float("inf"), None)
    return (
        tuple(int(x) for x in support),
        np.asarray(target_rates, dtype=float),
        float(best_row["event_weight"]),
        float(best_row["magnitude_weight"]),
        float(best_row["variability_weight"]),
        float(best_row["freshness_weight"]),
        float(best_row["target_rate_weight"]),
        float(best_row["anchor_bias"]),
        float(best_row["power_weight"]),
        float(best_row["switch_weight"]),
        float(best_row["min_soc"]),
        int(best_row["min_dwell"]),
        str(best_row["aggregation"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_proxy_mpc_policy(
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
) -> tuple[
    tuple[int, ...] | None,
    np.ndarray | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int | None,
    str | None,
    int | None,
    int | None,
    int | None,
    float | None,
    float | None,
    float,
    dict[str, object] | None,
]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )
    masks = np.asarray(candidate_masks, dtype=bool)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.proxy_mpc_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    target_rates = np.mean(masks[labels_arr].astype(float), axis=0)
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="proxy_mpc_calibration_static")
    static_start_objectives: list[float] = []
    for start_idx, start in enumerate(starts):
        _, objective = evaluate_validation_policy_metrics(
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
            seed_offset=110_000 + int(start_idx) * 101,
        )
        static_start_objectives.append(float(objective))

    rows: list[dict[str, object]] = []
    combo_idx = 0
    for event_weight in [float(x) for x in args.proxy_mpc_event_weight_grid] or [1.0]:
        for magnitude_weight in [float(x) for x in args.proxy_mpc_magnitude_weight_grid] or [1.0]:
            for variability_weight in [float(x) for x in args.proxy_mpc_variability_weight_grid] or [0.5]:
                for freshness_weight in [float(x) for x in args.proxy_mpc_freshness_grid] or [0.0]:
                    for target_rate_weight in [float(x) for x in args.proxy_mpc_target_rate_grid] or [0.0]:
                        for anchor_bias in [float(x) for x in args.proxy_mpc_anchor_bias_grid] or [0.0]:
                            for power_weight in [float(x) for x in args.proxy_mpc_power_grid] or [0.0]:
                                for switch_weight in [float(x) for x in args.proxy_mpc_switch_grid] or [0.0]:
                                    for min_soc in [float(x) for x in args.proxy_mpc_min_soc_grid] or [0.0]:
                                        for min_dwell in [max(1, int(x)) for x in args.proxy_mpc_dwell_grid] or [1]:
                                            for aggregation in [str(x) for x in args.proxy_mpc_aggregation_grid] or ["max"]:
                                                for planning_depth in [max(1, int(x)) for x in args.proxy_mpc_depth_grid] or [3]:
                                                    for beam_width in [max(1, int(x)) for x in args.proxy_mpc_beam_width_grid] or [4]:
                                                        for max_branch in [max(1, int(x)) for x in args.proxy_mpc_max_branch_grid] or [8]:
                                                            for age_weight in [float(x) for x in args.proxy_mpc_age_weight_grid] or [0.5]:
                                                                for anchor_improvement in [float(x) for x in args.proxy_mpc_anchor_improvement_grid] or [0.0]:
                                                                    policy = ForecastAwareProxyMPCPolicy(
                                                                        candidate_masks=masks,
                                                                        forecast_cfg=forecast_cfg,
                                                                        anchor_mask=anchor_mask,
                                                                        allowed_action_indices=support,
                                                                        target_rates=target_rates,
                                                                        event_weight=float(event_weight),
                                                                        magnitude_weight=float(magnitude_weight),
                                                                        variability_weight=float(variability_weight),
                                                                        freshness_weight=float(freshness_weight),
                                                                        target_rate_weight=float(target_rate_weight),
                                                                        anchor_bias=float(anchor_bias),
                                                                        power_weight=float(power_weight),
                                                                        switch_weight=float(switch_weight),
                                                                        min_soc=float(min_soc),
                                                                        min_dwell=int(min_dwell),
                                                                        aggregation=str(aggregation),
                                                                        planning_depth=int(planning_depth),
                                                                        beam_width=int(beam_width),
                                                                        max_branch=int(max_branch),
                                                                        age_weight=float(age_weight),
                                                                        anchor_improvement_threshold=float(anchor_improvement),
                                                                        preserve_warming=bool(args.bc_preserve_warming),
                                                                        name=f"forecast_aware_proxy_mpc_calib_{combo_idx}",
                                                                    )
                                                                    candidate_start_objectives: list[float] = []
                                                                    power_values: list[float] = []
                                                                    warmup_abort_count = 0
                                                                    for start_idx, start in enumerate(starts):
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
                                                                            starts=(int(start),),
                                                                            seed_offset=110_000 + int(start_idx) * 101,
                                                                        )
                                                                        candidate_start_objectives.append(float(objective))
                                                                        power_values.append(float(metrics.get("power_mean", np.nan)))
                                                                        warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                                                                    margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                                                                        candidate_start_objectives,
                                                                        dtype=float,
                                                                    )
                                                                    rows.append(
                                                                        {
                                                                            "policy": "forecast_aware_proxy_mpc",
                                                                            "objective": float(np.mean(candidate_start_objectives)),
                                                                            "power_mean": float(np.nanmean(power_values))
                                                                            if power_values
                                                                            else float("nan"),
                                                                            "warmup_abort_count": int(warmup_abort_count),
                                                                            "objective_margin_mean": float(np.mean(margins)),
                                                                            "objective_margin_min": float(np.min(margins)),
                                                                            "negative_start_count": int(np.sum(margins < 0.0)),
                                                                            "static_start_objectives": [
                                                                                float(x) for x in static_start_objectives
                                                                            ],
                                                                            "candidate_start_objectives": [
                                                                                float(x) for x in candidate_start_objectives
                                                                            ],
                                                                            "combo_idx": int(combo_idx),
                                                                            "event_weight": float(event_weight),
                                                                            "magnitude_weight": float(magnitude_weight),
                                                                            "variability_weight": float(variability_weight),
                                                                            "freshness_weight": float(freshness_weight),
                                                                            "target_rate_weight": float(target_rate_weight),
                                                                            "anchor_bias": float(anchor_bias),
                                                                            "power_weight": float(power_weight),
                                                                            "switch_weight": float(switch_weight),
                                                                            "min_soc": float(min_soc),
                                                                            "min_dwell": int(min_dwell),
                                                                            "aggregation": str(aggregation),
                                                                            "planning_depth": int(planning_depth),
                                                                            "beam_width": int(beam_width),
                                                                            "max_branch": int(max_branch),
                                                                            "age_weight": float(age_weight),
                                                                            "anchor_improvement": float(anchor_improvement),
                                                                        }
                                                                    )
                                                                    combo_idx += 1
    if not rows:
        return (
            support,
            target_rates,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )
    best_row = choose_deployable_validation_row(
        rows,
        criterion=str(args.proxy_mpc_calibration_criterion),
        min_mean_margin=float(args.deployable_selection_min_mean_margin),
        min_start_margin=float(args.deployable_selection_min_start_margin),
        max_negative_starts=int(args.deployable_selection_max_negative_starts),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(args.deployable_selection_require_risk_band),
        risk_min_q25_margin=float(args.deployable_selection_risk_min_q25_margin),
        risk_max_negative_starts=int(args.deployable_selection_risk_max_negative_starts),
    )
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "proxy_mpc_calibration.csv", index=False)
    if best_row is None:
        return (
            support,
            target_rates,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            float("inf"),
            None,
        )
    return (
        tuple(int(x) for x in support),
        np.asarray(target_rates, dtype=float),
        float(best_row["event_weight"]),
        float(best_row["magnitude_weight"]),
        float(best_row["variability_weight"]),
        float(best_row["freshness_weight"]),
        float(best_row["target_rate_weight"]),
        float(best_row["anchor_bias"]),
        float(best_row["power_weight"]),
        float(best_row["switch_weight"]),
        float(best_row["min_soc"]),
        int(best_row["min_dwell"]),
        str(best_row["aggregation"]),
        int(best_row["planning_depth"]),
        int(best_row["beam_width"]),
        int(best_row["max_branch"]),
        float(best_row["age_weight"]),
        float(best_row["anchor_improvement"]),
        float(best_row["objective"]),
        dict(best_row),
    )


def calibrate_sequence_mask_policy(
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
) -> tuple[tuple[int, ...] | None, float | None, float | None, float, dict[str, object] | None]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if labels_arr.size == 0:
        return None, None, None, float("inf"), None
    masks = np.asarray(candidate_masks, dtype=bool)
    support = action_support_from_labels(
        labels_arr,
        n_actions=int(masks.shape[0]),
        top_k=max(1, int(args.sequence_mask_support_top_k)),
        min_count=int(args.bc_action_support_min_count),
        anchor_idx=anchor_idx,
    )
    if support is None:
        support = tuple(int(idx) for idx in np.unique(labels_arr))
    anchor_biases = [float(x) for x in args.sequence_mask_anchor_bias_grid] or [0.0]
    power_weights = [float(x) for x in args.sequence_mask_power_grid] or [0.0]
    rows: list[dict[str, object]] = []
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    static_start_objectives: list[float] = []
    if str(args.sequence_mask_calibration_criterion) == "static_margin_guard":
        static_policy = StaticMaskPolicy(tuple(bool(x) for x in anchor_mask), name="sequence_mask_calibration_static")
        for start_idx, start in enumerate(starts):
            _, objective = evaluate_validation_policy_metrics(
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
            static_start_objectives.append(float(objective))
    combo_idx = 0
    for anchor_bias in anchor_biases:
        for power_weight in power_weights:
            policy = ForecastAwareSequenceMaskPolicy(
                model=model,
                candidate_masks=masks,
                forecast_cfg=forecast_cfg,
                device=str(args.bc_device),
                allowed_action_indices=support,
                anchor_mask=anchor_mask,
                anchor_bias=float(anchor_bias),
                power_weight=float(power_weight),
                preserve_warming=bool(args.bc_preserve_warming),
                name=f"forecast_aware_sequence_mask_calib_{combo_idx}",
            )
            if static_start_objectives:
                candidate_start_objectives: list[float] = []
                power_values: list[float] = []
                warmup_abort_count = 0
                for start_idx, start in enumerate(starts):
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
                        starts=(int(start),),
                        seed_offset=100_000 + int(start_idx) * 101,
                    )
                    candidate_start_objectives.append(float(objective))
                    power_values.append(float(metrics.get("power_mean", np.nan)))
                    warmup_abort_count += int(metrics.get("warmup_abort_count", 0))
                margins = np.asarray(static_start_objectives, dtype=float) - np.asarray(
                    candidate_start_objectives,
                    dtype=float,
                )
                row = {
                    "policy": "forecast_aware_sequence_mask",
                    "objective": float(np.mean(candidate_start_objectives)),
                    "power_mean": float(np.nanmean(power_values)) if power_values else float("nan"),
                    "warmup_abort_count": int(warmup_abort_count),
                    "objective_margin_mean": float(np.mean(margins)),
                    "objective_margin_min": float(np.min(margins)),
                    "negative_start_count": int(np.sum(margins < 0.0)),
                    "static_start_objectives": [float(x) for x in static_start_objectives],
                    "candidate_start_objectives": [float(x) for x in candidate_start_objectives],
                }
            else:
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
                    seed_offset=230_000 + combo_idx * 101,
                )
                row = {
                    "policy": "forecast_aware_sequence_mask",
                    "objective": float(objective),
                    "power_mean": float(metrics.get("power_mean", np.nan)),
                    "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
                }
            row.update(
                {
                    "combo_idx": int(combo_idx),
                    "anchor_bias": float(anchor_bias),
                    "power_weight": float(power_weight),
                }
            )
            rows.append(row)
            combo_idx += 1
    if static_start_objectives:
        best_row = choose_deployable_validation_row(
            rows,
            criterion="static_margin_guard",
            min_mean_margin=float(args.deployable_selection_min_mean_margin),
            min_start_margin=float(args.deployable_selection_min_start_margin),
            max_negative_starts=int(args.deployable_selection_max_negative_starts),
        )
    else:
        best_row = sorted(
            rows,
            key=lambda row: (
                float(row.get("objective", float("inf"))),
                float(row.get("power_mean", float("inf"))),
                int(row.get("combo_idx", 0)),
            ),
        )[0]
    return (
        tuple(int(x) for x in support),
        float(best_row["anchor_bias"]),
        float(best_row["power_weight"]),
        float(best_row["objective"]),
        dict(best_row),
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
    deployable_names = DEPLOYABLE_POLICY_NAMES
    fixed = [policy for policy in policies if str(policy.name) not in deployable_names]
    candidates = [policy for policy in policies if str(policy.name) in deployable_names]
    if not candidates:
        return policies, None, []
    rows: list[dict[str, object]] = []
    static_start_objectives: list[float] = []
    if str(args.deployable_selection_criterion) in {"static_margin_guard", "static_margin_risk"}:
        static_candidates = [policy for policy in fixed if str(policy.name) == "validation_selected_static"]
        if not static_candidates:
            raise ValueError("static-margin validation selection requires validation_selected_static in policy list")
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
        require_guard_pass=bool(args.deployable_selection_require_guard_pass),
        require_positive_center=bool(args.deployable_selection_require_positive_center),
        require_risk_band=bool(getattr(args, "deployable_selection_require_risk_band", False)),
        risk_min_q25_margin=float(getattr(args, "deployable_selection_risk_min_q25_margin", -1.0e9)),
        risk_max_negative_starts=int(getattr(args, "deployable_selection_risk_max_negative_starts", 1_000_000)),
    )
    if selected_row is None:
        return fixed, None, rows
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
