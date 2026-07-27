from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.features import (
    ForecastContextConfig,
    append_event_forecast,
    build_event_forecast,
    event_forecast_feature_names,
    sensor_timing_features,
)
from forecast_cmdp.event_forecaster import (
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    select_event_forecast_columns,
    train_event_forecaster,
)
from forecast_cmdp.continuous_forecaster import (
    ContinuousForecasterTrainingConfig,
    augment_truth_with_continuous_forecasts,
    build_continuous_forecast_dataset,
    select_continuous_forecast_columns,
    train_continuous_forecaster,
    predict_continuous_from_history,
)
from forecast_cmdp.probabilistic_world_model import (
    load_probabilistic_world_model,
    ProbabilisticWorldModelTrainingConfig,
    save_probabilistic_world_model,
    train_probabilistic_world_model,
)
from forecast_cmdp.rollout_world_model import (
    RolloutWorldModelTrainingConfig,
    build_rollout_world_model_dataset,
    load_rollout_world_model,
    save_rollout_world_model,
    train_rollout_world_model,
)
from forecast_cmdp.archived_v2 import continue_policy_rollout
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
    RecurrentActionCostDataset,
    SequenceValueDataset,
    collect_anchor_advantage_dataset,
    collect_action_cost_dataset,
    collect_executed_outcome_datasets,
    collect_feature_transition_dataset,
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
from forecast_cmdp.dataset import TeacherDataset, collect_dagger_dataset, collect_teacher_dataset, concat_teacher_datasets
from forecast_cmdp.mpc_teacher import (
    MpcTeacherConfig,
    beam_search_teacher_action,
    enumerate_action_masks,
    restore_env,
    snapshot_env,
)
from forecast_cmdp.robust_planner import (
    FixedScenarioModel,
    RobustPlannerConfig,
    RobustRecedingHorizonPolicy,
    build_causal_world_model_context,
    build_scenario_environment,
    robust_beam_search_plan,
    robust_cost,
)
from forecast_cmdp.mean_risk_policy import (
    ForecastAwareMeanRiskControllerPolicy,
    ForecastAwareResidualRiskControllerPolicy,
    RecedingForecastAwareMeanRiskControllerPolicy,
    anchor_neighborhood_support,
    build_residual_risk_feature,
    build_window_risk_feature,
    causal_history_summary,
    causal_residual_history_summary,
    causal_window_agent_state,
    residual_boundary_state,
    residual_action_controller_specs,
    valid_anchor_residual_action,
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
    ForecastAwareRuntimeRiskGuardPolicy,
    ForecastAwareSequenceMaskPolicy,
    ForecastAwareTeacherRatePolicy,
    ForecastAwareUtilityPlannerPolicy,
    ForecastAwareWindowCandidatePolicy,
    ValidationCyclicDwellPolicy,
    load_bc_policy_checkpoint,
    save_bc_policy_checkpoint,
    train_bc_classifier,
    train_mask_bc,
    train_sequence_mask_bc,
)
from forecast_cmdp.protocol import choose_non_overlapping_starts, task_focus_metrics
from forecast_cmdp.reuse import ensure_archive_src
from forecast_cmdp.selection import choose_deployable_validation_row
from forecast_cmdp.window_risk import (
    ControllerSpec,
    WindowOutcome,
    WindowRiskRecord,
    assign_balanced_anchors,
    audit_feature_names,
    build_window_risk_dataset,
    collect_paired_window_risk_dataset,
    filter_exact_anchor_boundaries,
    refresh_window_risk_features,
    select_train_anchor_bank,
    split_train_risk_starts,
    static_candidate_margin,
)
from forecast_cmdp.window_risk_model import (
    WindowRiskModelBundle,
    WindowRiskTrainingConfig,
    one_sided_conformal_correction,
    risk_bin_diagnostics,
    train_window_risk_models,
)

ensure_archive_src()

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.rollout import RolloutResult  # noqa: E402
from v2.rollout import run_policy_rollout  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


def test_static_candidate_margin_is_positive_for_lower_candidate_objective():
    assert np.isclose(static_candidate_margin(1.25, 1.10), 0.15)
    assert np.isclose(static_candidate_margin(1.10, 1.25), -0.15)


def test_exact_anchor_boundary_filter_removes_projected_prefix_rows():
    names = (
        "residual_boundary_previous_mask_a",
        "residual_boundary_previous_mask_b",
        "residual_anchor_mask_a",
        "residual_anchor_mask_b",
    )

    def record(start: int, feature: tuple[float, ...]) -> WindowRiskRecord:
        return WindowRiskRecord(
            seed=41,
            split_name="risk_fit",
            start=start,
            anchor_action_idx=0,
            anchor_mask=(True, False),
            controller_id="residual_action_001",
            controller_config={"target_action_idx": 1},
            paired_seed_offset=start,
            static_objective=1.0,
            candidate_objective=0.9,
            margin=0.1,
            power_mean=0.5,
            warmup_abort_count=0,
            constraint_violation_count=0,
            feature_vector=feature,
            feature_names=names,
        )

    dataset = build_window_risk_dataset(
        [
            record(100, (1.0, 0.0, 1.0, 0.0)),
            record(200, (1.0, 1.0, 1.0, 0.0)),
        ]
    )
    filtered, summary = filter_exact_anchor_boundaries(dataset)
    assert filtered.starts.tolist() == [100]
    assert summary == {
        "input_rows": 2,
        "exact_rows": 1,
        "dropped_rows": 1,
    }


def test_train_risk_start_split_is_non_overlapping_and_blocked():
    truth = pd.DataFrame(
        {
            "event_flag": np.zeros(4096, dtype=np.int8),
        }
    )
    split = split_train_risk_starts(
        truth,
        bounds=(0, len(truth)),
        window_steps=128,
        horizon=3,
        fit_count=8,
        calibration_count=4,
        selection="uniform",
        stride=1,
        event_column="event_flag",
        seed=41,
    )
    combined = (*split.fit, *split.calibration)
    assert len(split.fit) == 8
    assert len(split.calibration) == 4
    assert split.diagnostics["chronological_blocked"] is True
    assert max(split.fit) < min(split.calibration)
    assert all(right - left >= 132 for left, right in zip(combined, combined[1:]))


def test_train_anchor_bank_and_balanced_assignment():
    table = pd.DataFrame(
        {
            "action_idx": [4, 2, 3, 1],
            "objective_loss_mean": [0.8, 0.5, 0.6, 0.7],
            "power_mean": [0.5, 0.4, 0.3, 0.2],
            "warmup_abort_count": [0, 0, 0, 0],
        }
    )
    bank = select_train_anchor_bank(table, top_k=3)
    assignments = assign_balanced_anchors(
        starts=(100, 400, 700),
        anchor_bank=bank,
        anchors_per_start=2,
    )
    assert bank == (2, 3, 1)
    assert all(anchors[0] == 2 for anchors in assignments.values())
    assert {anchors[1] for anchors in assignments.values()} == {1, 3}
    rotated = assign_balanced_anchors(
        starts=range(8),
        anchor_bank=(0, 1, 2, 3),
        anchors_per_start=2,
        always_include_best=False,
    )
    counts = {
        anchor: sum(anchor in assigned for assigned in rotated.values())
        for anchor in range(4)
    }
    assert counts == {0: 4, 1: 4, 2: 4, 3: 4}


def test_window_risk_feature_audit_rejects_future_or_realized_features():
    allowed = audit_feature_names(("soc", "event_probability_h1", "anchor_power"))
    blocked = audit_feature_names(("soc", "truth_future_flux", "candidate_realized_duty"))
    duplicated = audit_feature_names(("soc", "soc"))
    assert allowed["pass"] is True
    assert blocked["pass"] is False
    assert blocked["blocked_features"] == [
        "candidate_realized_duty",
        "truth_future_flux",
    ]
    assert duplicated["pass"] is False


def test_paired_window_collection_uses_common_seeds_and_resumes(tmp_path):
    calls: list[tuple[str, int, int, int, str]] = []
    controllers = (
        ControllerSpec("proxy_a", {"depth": 2.0}),
        ControllerSpec("proxy_b", {"depth": 3.0}),
    )

    def static_evaluator(start: int, anchor_idx: int, seed_offset: int) -> WindowOutcome:
        calls.append(("static", start, anchor_idx, seed_offset, ""))
        return WindowOutcome(objective=2.0, power_mean=0.5)

    def candidate_evaluator(
        start: int,
        anchor_idx: int,
        controller: ControllerSpec,
        seed_offset: int,
    ) -> WindowOutcome:
        calls.append(("candidate", start, anchor_idx, seed_offset, controller.controller_id))
        objective = 1.5 if controller.controller_id == "proxy_a" else 2.25
        return WindowOutcome(objective=objective, power_mean=0.6)

    def feature_builder(
        start: int,
        anchor_idx: int,
        controller: ControllerSpec,
        seed_offset: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        return (
            np.asarray(
                [start, anchor_idx, controller.parameters["depth"], seed_offset],
                dtype=float,
            ),
            (
                "start_phase",
                "anchor_action_idx",
                "controller_depth",
                "paired_seed_offset",
            ),
        )

    kwargs = {
        "out_dir": tmp_path,
        "seed": 41,
        "split_name": "risk_fit",
        "starts": (100, 500),
        "anchor_assignments": {100: (1,), 500: (1,)},
        "anchor_masks": {1: (True, False)},
        "controllers": controllers,
        "static_evaluator": static_evaluator,
        "candidate_evaluator": candidate_evaluator,
        "feature_builder": feature_builder,
    }
    first = collect_paired_window_risk_dataset(**kwargs)
    first_call_count = len(calls)
    second = collect_paired_window_risk_dataset(**kwargs)

    assert first.features.shape == (4, 4)
    assert first.margins.tolist() == [0.5, -0.25, 0.5, -0.25]
    assert first.negative_labels.tolist() == [0, 1, 0, 1]
    assert sum(call[0] == "static" for call in calls) == 2
    assert sum(call[0] == "candidate" for call in calls) == 4
    assert len(calls) == first_call_count
    assert second.margins.tolist() == first.margins.tolist()
    for candidate_call in (call for call in calls if call[0] == "candidate"):
        assert any(
            static_call[1:4] == candidate_call[1:4]
            for static_call in calls
            if static_call[0] == "static"
        )
    assert (tmp_path / "window_risk_rows.jsonl").exists()
    assert (tmp_path / "window_risk_rows.csv").exists()
    assert (tmp_path / "window_risk_dataset.npz").exists()
    assert (tmp_path / "window_risk_feature_schema.json").exists()
    assert (tmp_path / "window_risk_collection_manifest.json").exists()
    refreshed = refresh_window_risk_features(
        tmp_path,
        feature_builder=lambda start, anchor_idx, controller, seed_offset: (
            np.asarray(
                [
                    start,
                    anchor_idx,
                    controller.parameters["depth"],
                    seed_offset,
                    1.0,
                ]
            ),
            (
                "start_phase",
                "anchor_action_idx",
                "controller_depth",
                "paired_seed_offset",
                "history_signal",
            ),
        ),
    )
    assert refreshed.features.shape == (4, 5)
    assert refreshed.margins.tolist() == first.margins.tolist()
    assert (tmp_path / "window_risk_rows_pre_refresh.jsonl").exists()


def test_paired_window_collection_filters_anchor_controller_pairs(tmp_path):
    controllers = (
        ControllerSpec("keep", {"depth": 1.0}),
        ControllerSpec("skip", {"depth": 2.0}),
    )

    dataset = collect_paired_window_risk_dataset(
        out_dir=tmp_path,
        seed=41,
        split_name="risk_fit",
        starts=(100,),
        anchor_assignments={100: (1,)},
        anchor_masks={1: (True, False)},
        controllers=controllers,
        static_evaluator=lambda start, anchor_idx, seed_offset: WindowOutcome(
            objective=2.0
        ),
        candidate_evaluator=lambda start, anchor_idx, controller, seed_offset: WindowOutcome(
            objective=1.5
        ),
        feature_builder=lambda start, anchor_idx, controller, seed_offset: (
            np.asarray([controller.parameters["depth"]], dtype=float),
            ("controller_depth",),
        ),
        controller_filter=lambda anchor_idx, controller: (
            controller.controller_id == "keep"
        ),
    )

    assert dataset.controller_ids == ("keep",)
    assert dataset.margins.tolist() == [0.5]


def test_window_risk_pilot_controller_grid_and_causal_feature_schema():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location("run_window_risk_pilot_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    controllers = module.balanced_proxy_controller_specs(16)
    assert len(controllers) == 16
    assert {controller.parameters["event_weight"] for controller in controllers} == {0.5, 1.5}
    assert {controller.parameters["target_rate_weight"] for controller in controllers} == {0.0, 0.5}
    assert {controller.parameters["min_dwell"] for controller in controllers} == {1, 2}
    assert {controller.parameters["planning_depth"] for controller in controllers} == {2, 3}
    assert {controller.parameters["age_weight"] for controller in controllers} == {0.25, 0.75}
    guarded = module.anchor_guard_controller_specs(16)
    assert len(guarded) == 16
    assert {
        controller.parameters["anchor_improvement_threshold"]
        for controller in guarded
    } == {0.0, 0.01, 0.02, 0.04}
    assert len(
        {
            controller.controller_id.split("_t", 1)[0]
            for controller in guarded
        }
    ) == 4
    neighborhoods = module.anchor_neighborhood_controller_specs(16)
    assert len(neighborhoods) == 16
    assert {
        controller.parameters["max_anchor_hamming"]
        for controller in neighborhoods
    } == {1, 2, 3, 4}
    assert all(
        controller.parameters["anchor_improvement_threshold"] == 0.0
        for controller in neighborhoods
    )

    base_env = make_env()
    candidate_masks = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    feature, names = module.build_feature_vector(
        truth=base_env.truth_df,
        sensors=base_env.sensor_specs,
        constraints=base_env.projector.constraints,
        cfg=base_env.cfg,
        oracle=base_env.oracle,
        forecast_cfg=ForecastContextConfig(horizon=3),
        candidate_masks=candidate_masks,
        anchor_idx=0,
        controller=controllers[0],
        support=(0, 1),
        target_rates=np.asarray([1.0, 0.5, 0.5]),
        preserve_warming=True,
        start=8,
    )
    assert feature.shape == (len(names),)
    assert np.all(np.isfinite(feature))
    assert audit_feature_names(names)["pass"] is True


def test_window_risk_teacher_support_requires_core_forecast_sensor():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location("run_window_risk_support_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    masks = np.asarray(
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    support = module.action_support_from_teacher(
        np.asarray([0, 0, 1, 1, 2, 3]),
        n_actions=4,
        top_k=4,
        anchor_bank=(1, 2),
        candidate_masks=masks,
        required_sensor_indices=(0,),
    )
    assert support == (1, 2)
    assert all(masks[action_idx, 0] for action_idx in support)


def test_window_risk_teacher_support_uses_fit_windows_only():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location("run_window_risk_teacher_scope_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    labels, steps = module.teacher_rows_within_windows(
        np.asarray([1, 2, 3, 4, 5]),
        np.asarray([100, 120, 200, 220, 300]),
        starts=(110, 210),
        window_steps=20,
    )
    assert labels.tolist() == [2, 4]
    assert steps.tolist() == [120, 220]


def test_window_risk_models_train_calibrate_and_persist(tmp_path):
    def make_dataset(starts: range, split_name: str):
        records = []
        for start in starts:
            regime = np.sin(float(start) * 0.37)
            for controller_idx in range(4):
                controller_effect = 0.018 * float(controller_idx - 1)
                margin = 0.035 * regime + controller_effect
                records.append(
                    WindowRiskRecord(
                        seed=41,
                        split_name=split_name,
                        start=int(start),
                        anchor_action_idx=0,
                        anchor_mask=(True, False),
                        controller_id=f"controller_{controller_idx}",
                        controller_config={"controller_idx": controller_idx},
                        paired_seed_offset=1000 + int(start),
                        static_objective=1.0,
                        candidate_objective=1.0 - margin,
                        margin=margin,
                        power_mean=0.5,
                        warmup_abort_count=0,
                        constraint_violation_count=0,
                        feature_vector=(
                            float(regime),
                            float(controller_idx),
                            float(regime * controller_idx),
                        ),
                        feature_names=("regime", "controller_idx", "regime_controller"),
                    )
                )
        return build_window_risk_dataset(records)

    fit = make_dataset(range(32), "risk_fit")
    calibration = make_dataset(range(40, 52), "risk_calibration")
    bundle, metrics = train_window_risk_models(
        fit,
        calibration,
        cfg=WindowRiskTrainingConfig(
            n_estimators=30,
            learning_rate=0.08,
            min_samples_leaf=4,
            seed=41,
        ),
        out_dir=tmp_path,
    )
    prediction = bundle.predict(calibration.features[:3])
    assert prediction["mean_margin"].shape == (3,)
    assert prediction["risk_lower_bound"].shape == (3,)
    assert np.all(prediction["risk_lower_bound"] <= prediction["q25_margin"] + 1.0e-12)
    assert metrics["fit_independent_starts"] == 32
    assert metrics["calibration_independent_starts"] == 12
    assert (tmp_path / "window_risk_model.joblib").exists()
    assert (tmp_path / "window_risk_model_metrics.json").exists()
    loaded = WindowRiskModelBundle.load(tmp_path / "window_risk_model.joblib")
    assert np.allclose(
        loaded.predict(calibration.features[:3])["mean_margin"],
        prediction["mean_margin"],
    )


def test_residual_threshold_selection_prioritizes_dynamic_start_coverage():
    module_path = (
        ROOT / "v1" / "scripts" / "calibrate_residual_risk_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location(
        "calibrate_residual_risk_thresholds_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    conservative = {
        "dynamic_starts": 2,
        "margin_q25": 0.0,
        "margin_mean": 0.01,
        "margin_min": -0.001,
        "negative_starts": 1,
        "dynamic_groups": 4,
        "min_risk_lower_bound": -0.14,
        "max_negative_probability": 0.60,
        "min_predicted_mean_margin": 0.0,
    }
    broader = {
        **conservative,
        "dynamic_starts": 3,
        "margin_mean": 0.002,
        "margin_min": -0.01,
        "dynamic_groups": 5,
        "min_risk_lower_bound": -0.17,
    }

    assert module.selection_priority(broader) > module.selection_priority(
        conservative
    )


def test_hist_gbdt_window_risk_model_trains_and_persists(tmp_path):
    records = []
    for start in range(12):
        for controller_idx in range(3):
            margin = (
                0.03 * float(controller_idx)
                + 0.01 * float(start % 3)
                - 0.025
            )
            records.append(
                WindowRiskRecord(
                    seed=41,
                    split_name=(
                        "risk_fit" if start < 8 else "risk_calibration"
                    ),
                    start=start,
                    anchor_action_idx=1,
                    anchor_mask=(True, False),
                    controller_id=f"controller_{controller_idx}",
                    controller_config={"controller_idx": controller_idx},
                    paired_seed_offset=1000 + start,
                    static_objective=1.0,
                    candidate_objective=1.0 - margin,
                    margin=margin,
                    power_mean=0.5,
                    warmup_abort_count=0,
                    constraint_violation_count=0,
                    feature_vector=(
                        float(start % 3),
                        float(controller_idx),
                    ),
                    feature_names=("regime", "controller_idx"),
                )
            )
    fit = build_window_risk_dataset(
        [row for row in records if row.split_name == "risk_fit"]
    )
    calibration = build_window_risk_dataset(
        [row for row in records if row.split_name == "risk_calibration"]
    )
    bundle, metrics = train_window_risk_models(
        fit,
        calibration,
        cfg=WindowRiskTrainingConfig(
            model_family="hist_gbdt",
            n_estimators=20,
            learning_rate=0.08,
            max_leaf_nodes=5,
            min_samples_leaf=2,
            seed=41,
        ),
        out_dir=tmp_path,
    )

    assert metrics["training_config"]["model_family"] == "hist_gbdt"
    assert bundle.predict(calibration.features)["mean_margin"].shape == (
        len(calibration.margins),
    )
    assert WindowRiskModelBundle.load(
        tmp_path / "window_risk_model.joblib"
    ).feature_names == fit.feature_names


def test_residual_anchor_audit_uses_heldout_starts():
    module_path = (
        ROOT / "v1" / "scripts" / "calibrate_residual_risk_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location(
        "calibrate_residual_anchor_audit_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    margins = (0.04, 0.03, 0.02, 0.01, 0.03, 0.02, 0.01, -0.005)
    for start, margin in enumerate(margins):
        rows.append(
            {
                "start": start,
                "anchor_action_idx": 7,
                "controller_id": "residual",
                "margin": margin,
                "mean_margin_pred": margin,
                "risk_lower_bound": margin,
                "negative_probability": float(margin < 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    anchor_audit, heldout = module.audit_anchor_leave_one_start_out(
        frame,
        lower_grid=(-0.01, 0.0),
        negative_grid=(0.0, 1.0),
        mean_grid=(-0.01, 0.0),
        max_negative_starts=1,
    )

    assert len(heldout) == 8
    assert set(heldout["train_starts"]) == {7}
    assert int(anchor_audit.iloc[0]["loso_dynamic_starts"]) > 0
    negative_row = heldout[heldout["heldout_start"] == 7].iloc[0]
    assert float(negative_row["heldout_margin"]) <= 0.0


def test_window_risk_explicit_anchor_bank_is_validated():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location(
        "run_window_risk_explicit_anchor_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    static_table = pd.DataFrame(
        {
            "action_idx": [3, 7, 9],
            "objective_loss_mean": [0.3, 0.1, 0.2],
        }
    )

    assert module.resolve_anchor_bank(
        static_table,
        top_k=2,
        explicit_action_indices=(9, 3, 9),
        eligible_action_indices={3, 7, 9},
    ) == (9, 3)
    with pytest.raises(ValueError, match="not eligible"):
        module.resolve_anchor_bank(
            static_table,
            top_k=2,
            explicit_action_indices=(9, 8),
            eligible_action_indices={3, 7, 9},
        )


def test_one_sided_conformal_correction_never_raises_quantile_bound():
    predictions = np.asarray([-0.02, -0.01, 0.00, 0.01])
    targets = np.asarray([-0.03, 0.00, 0.02, 0.04])
    correction = one_sided_conformal_correction(predictions, targets, alpha=0.25)
    assert correction >= 0.0
    assert np.all(predictions - correction <= predictions)


def test_risk_bin_gate_checks_tail_order_not_bin_means():
    prediction = np.repeat(np.arange(4, dtype=float), 4)
    margins = np.asarray(
        [
            -4.0,
            -3.0,
            -2.0,
            10.0,
            -2.0,
            -1.0,
            0.0,
            20.0,
            -0.5,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.1,
        ]
    )
    diagnostics = risk_bin_diagnostics(prediction, margins)
    assert diagnostics["mean_monotonic_fraction"] < 0.5
    assert diagnostics["q25_monotonic_fraction"] == 1.0
    assert diagnostics["negative_rate_monotonic_fraction"] == 1.0
    assert diagnostics["monotonic_fraction"] == 1.0


def test_one_sided_conformal_correction_uses_independent_start_groups():
    predictions = np.asarray([0.02, 0.02, 0.02, 0.02])
    targets = np.asarray([0.01, -0.08, 0.01, 0.00])
    groups = np.asarray([10, 10, 20, 20])
    grouped = one_sided_conformal_correction(
        predictions,
        targets,
        alpha=0.25,
        groups=groups,
    )
    assert np.isclose(grouped, 0.10)


def test_mean_risk_controller_selects_once_and_has_static_fallback():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location("run_window_risk_policy_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controllers = module.balanced_proxy_controller_specs(2)
    env = make_env()
    env.reset(start_idx=8)
    state_before = (
        int(env.current_idx),
        float(env.current_energy),
        env.previous_action_mask.copy(),
        tuple(
            (int(env.runtimes[sid].mode), int(env.runtimes[sid].warm_remaining))
            for sid in env.sensor_ids
        ),
    )
    candidate_masks = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    target_rates = np.asarray([1.0, 0.5, 0.5])
    _, feature_names = build_window_risk_feature(
        env=env,
        forecast_cfg=ForecastContextConfig(horizon=3),
        candidate_masks=candidate_masks,
        anchor_idx=0,
        controller=controllers[0],
        support=(0, 1),
        target_rates=target_rates,
        preserve_warming=True,
    )
    state_after = (
        int(env.current_idx),
        float(env.current_energy),
        env.previous_action_mask.copy(),
        tuple(
            (int(env.runtimes[sid].mode), int(env.runtimes[sid].warm_remaining))
            for sid in env.sensor_ids
        ),
    )
    assert state_after[0] == state_before[0]
    assert state_after[1] == state_before[1]
    assert np.array_equal(state_after[2], state_before[2])
    assert state_after[3] == state_before[3]

    class StubBundle:
        def __init__(self, lower: float):
            self.feature_names = feature_names
            self.negative_model = None
            self.lower = float(lower)
            self.calls = 0

        def predict(self, features):
            self.calls += 1
            x = np.asarray(features, dtype=float)
            event_idx = self.feature_names.index("controller_event_weight")
            mean = x[:, event_idx]
            return {
                "mean_margin": mean,
                "q25_margin": np.full(x.shape[0], self.lower),
                "risk_lower_bound": np.full(x.shape[0], self.lower),
                "negative_probability": np.full(x.shape[0], np.nan),
            }

    selecting_bundle = StubBundle(0.02)
    policy = ForecastAwareMeanRiskControllerPolicy(
        model_bundle=selecting_bundle,
        controllers=controllers,
        candidate_masks=candidate_masks,
        forecast_cfg=ForecastContextConfig(horizon=3),
        anchor_action_idx=0,
        support=(0, 1),
        target_rates=target_rates,
    )
    first = policy.act_mask(env)
    second = policy.act_mask(env)
    assert first.shape == (3,)
    assert second.shape == (3,)
    assert selecting_bundle.calls == 1
    assert policy.selected_controller_id is not None
    assert policy.static_fallback is False

    fallback_bundle = StubBundle(-0.01)
    fallback = ForecastAwareMeanRiskControllerPolicy(
        model_bundle=fallback_bundle,
        controllers=controllers,
        candidate_masks=candidate_masks,
        forecast_cfg=ForecastContextConfig(horizon=3),
        anchor_action_idx=0,
        support=(0, 1),
        target_rates=target_rates,
    )
    assert fallback.act_mask(env).tolist() == candidate_masks[0].tolist()
    assert fallback.static_fallback is True
    assert fallback.selected_controller_id is None


def test_receding_mean_risk_controller_reselects_at_macro_boundaries():
    module_path = ROOT / "v1" / "scripts" / "run_window_risk_pilot.py"
    spec = importlib.util.spec_from_file_location("run_window_risk_receding_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controllers = module.balanced_proxy_controller_specs(2)
    env = make_env()
    env.reset(start_idx=8)
    candidate_masks = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    target_rates = np.asarray([1.0, 0.5, 0.5])
    _, feature_names = build_window_risk_feature(
        env=env,
        forecast_cfg=ForecastContextConfig(horizon=3),
        candidate_masks=candidate_masks,
        anchor_idx=0,
        controller=controllers[0],
        support=(0, 1),
        target_rates=target_rates,
        preserve_warming=True,
    )

    class StubBundle:
        negative_model = None

        def __init__(self):
            self.feature_names = feature_names
            self.calls = 0

        def predict(self, features):
            self.calls += 1
            rows = np.asarray(features).shape[0]
            return {
                "mean_margin": np.full(rows, 0.02),
                "q25_margin": np.full(rows, 0.01),
                "risk_lower_bound": np.full(rows, 0.01),
                "negative_probability": np.full(rows, np.nan),
            }

    bundle = StubBundle()
    policy = RecedingForecastAwareMeanRiskControllerPolicy(
        model_bundle=bundle,
        controllers=controllers,
        candidate_masks=candidate_masks,
        forecast_cfg=ForecastContextConfig(horizon=3),
        anchor_action_idx=0,
        support=(0, 1),
        target_rates=target_rates,
        decision_interval=2,
    )
    for _ in range(5):
        policy.act_mask(env)
    assert bundle.calls == 3
    assert len(policy.block_history) == 3
    assert all(not bool(row["static_fallback"]) for row in policy.block_history)


def test_residual_risk_controller_selects_safe_action_and_recedes():
    env = make_env()
    env.reset(start_idx=18)
    masks = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    controller = residual_action_controller_specs((1,))[0]
    _, feature_names = build_residual_risk_feature(
        env=env,
        forecast_cfg=ForecastContextConfig(horizon=3),
        candidate_masks=masks,
        anchor_idx=0,
        controller=controller,
    )

    class StubBundle:
        negative_model = None

        def __init__(self, lower: float):
            self.feature_names = feature_names
            self.lower = float(lower)
            self.calls = 0

        def predict(self, features):
            self.calls += 1
            rows = np.asarray(features).shape[0]
            return {
                "mean_margin": np.full(rows, 0.02),
                "q25_margin": np.full(rows, self.lower),
                "risk_lower_bound": np.full(rows, self.lower),
                "negative_probability": np.full(rows, np.nan),
            }

    selecting = StubBundle(0.01)
    policy = ForecastAwareResidualRiskControllerPolicy(
        model_bundle=selecting,
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(horizon=3),
        anchor_action_idx=0,
        support=(0, 1, 2),
        required_sensor_indices=(0,),
        decision_interval=2,
    )
    selected = []
    for _ in range(5):
        mask = policy.act_mask(env)
        selected.append(mask.tolist())
        env.step_mask(mask)
    assert selected == [
        masks[0].tolist(),
        masks[0].tolist(),
        masks[1].tolist(),
        masks[1].tolist(),
        masks[0].tolist(),
    ]
    assert selecting.calls == 1
    assert len(policy.block_history) == 3
    assert [bool(row["conditioning_block"]) for row in policy.block_history] == [
        True,
        False,
        True,
    ]

    fallback = ForecastAwareResidualRiskControllerPolicy(
        model_bundle=StubBundle(-0.01),
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(horizon=3),
        anchor_action_idx=0,
        support=(0, 1, 2),
        required_sensor_indices=(0,),
        decision_interval=2,
    )
    fallback_env = make_env()
    fallback_env.reset(start_idx=18)
    first = fallback.act_mask(fallback_env)
    fallback_env.step_mask(first)
    second = fallback.act_mask(fallback_env)
    fallback_env.step_mask(second)
    third = fallback.act_mask(fallback_env)
    assert first.tolist() == masks[0].tolist()
    assert third.tolist() == masks[0].tolist()
    assert fallback.block_history[0]["static_fallback"] is True
    assert fallback.block_history[1]["static_fallback"] is True


def test_mean_risk_validation_gate_rejects_all_static_fallback():
    module_path = ROOT / "v1" / "scripts" / "evaluate_mean_risk_controller.py"
    spec = importlib.util.spec_from_file_location("evaluate_mean_risk_controller_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.validation_gate_pass(
        {
            "dynamic_windows": 0,
            "margin_mean": 0.0,
            "margin_q25": 0.0,
            "negative_starts": 0,
            "hard_constraint_violations": 0,
        },
        min_mean_margin=0.0,
        min_q25_margin=0.0,
        max_negative_starts=1,
    )
    assert module.validation_gate_pass(
        {
            "dynamic_windows": 3,
            "margin_mean": 0.01,
            "margin_q25": 0.001,
            "negative_starts": 1,
            "hard_constraint_violations": 0,
        },
        min_mean_margin=0.0,
        min_q25_margin=0.0,
        max_negative_starts=1,
    )


def test_concat_recurrent_action_cost_datasets_preserves_sequence_breaks():
    masks = np.asarray([[1, 0], [0, 1]], dtype=bool)
    first = RecurrentActionCostDataset(
        features=np.zeros((2, 3), dtype=np.float32),
        costs=np.zeros((2, 2), dtype=np.float32),
        action_masks=np.ones((2, 2), dtype=bool),
        labels=np.asarray([0, 1], dtype=np.int64),
        candidate_masks=masks,
        step_indices=np.asarray([10, 11], dtype=np.int64),
        feature_dim=3,
        n_sensors=2,
    )
    second = RecurrentActionCostDataset(
        features=np.ones((2, 3), dtype=np.float32),
        costs=np.ones((2, 2), dtype=np.float32),
        action_masks=np.ones((2, 2), dtype=bool),
        labels=np.asarray([1, 0], dtype=np.int64),
        candidate_masks=masks,
        step_indices=np.asarray([10, 11], dtype=np.int64),
        feature_dim=3,
        n_sensors=2,
    )
    merged = concat_recurrent_action_cost_datasets([first, second])
    assert merged.features.shape == (4, 3)
    assert merged.costs.shape == (4, 2)
    assert merged.step_indices.tolist() == [0, 1, 3, 4]


def test_sequence_value_model_and_policy_init():
    masks = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=bool)
    features = np.asarray([[0.0, 0.1, 0.2], [0.5, 0.1, -0.2]], dtype=np.float32)
    bank = np.asarray([[0, 0], [2, 1]], dtype=np.int64)
    rows = [
        np.concatenate([features[0], masks[bank[0]].astype(np.float32).reshape(-1)]),
        np.concatenate([features[1], masks[bank[1]].astype(np.float32).reshape(-1)]),
    ]
    dataset = SequenceValueDataset(
        inputs=np.vstack(rows).astype(np.float32),
        advantages=np.asarray([0.0, 0.25], dtype=np.float32),
        sequence_bank=bank,
        feature_dim=3,
        n_sensors=2,
        sequence_len=2,
    )
    model, history = train_sequence_value_model(
        dataset,
        ActionCostTrainingConfig(hidden_dim=8, epochs=1, batch_size=2, device="cpu"),
    )
    policy = ForecastAwareSequenceValuePolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(horizon=1),
        anchor_mask=masks[0],
        sequence_bank=dataset.sequence_bank,
        device="cpu",
    )
    assert history["loss"]
    assert policy.sequence_bank.shape == (2, 2)


def test_augmented_sequence_value_bank_adds_static_and_cycle_candidates():
    module_path = ROOT / "v1" / "scripts" / "run_protocol_gate.py"
    spec = importlib.util.spec_from_file_location("run_protocol_gate_augmented_sequence_bank_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    masks = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    bank = module.build_augmented_sequence_value_bank(
        labels=np.asarray([3, 3, 1, 2, 3, 1], dtype=np.int64),
        candidate_masks=masks,
        anchor_idx=0,
        train_static_table=pd.DataFrame({"action_idx": [2, 1]}),
        sequence_len=4,
        static_top_k=2,
        support_top_k=2,
        dwell_grid=(1, 2),
        max_sequences=32,
    )
    rows = {tuple(int(x) for x in row) for row in bank}
    assert (0, 0, 0, 0) in rows
    assert (2, 2, 2, 2) in rows
    assert any(len(set(row)) > 1 for row in rows)


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)


class DummyOracle:
    is_fitted = True

    class Cfg:
        horizon = 3

    cfg = Cfg()

    def loss(self, feature, future):
        del feature
        return float(np.mean(np.abs(future)))


class SaturatingOracle(DummyOracle):
    def loss(self, feature, future):
        del feature, future
        return 10.0


def make_truth(n: int = 64) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    event = (t >= 20) & (t < 32)
    return pd.DataFrame(
        {
            "wind_speed_ms": 6.0 + 5.0 * event.astype(float),
            "wind_direction_deg": np.full(n, 180.0),
            "air_temperature_c": -20.0 + 0.01 * t,
            "relative_humidity": np.full(n, 70.0),
            "air_pressure_pa": np.full(n, 70000.0),
            "solar_radiation_wm2": np.zeros(n),
            "snow_surface_temperature_c": -22.0 + 0.01 * t,
            "snow_particle_mean_diameter_mm": 0.2 * event.astype(float),
            "snow_particle_mean_velocity_ms": 5.0 * event.astype(float),
            "snow_mass_flux_kg_m2_s": 1.0e-5 * event.astype(float),
            "event_flag": event,
        }
    )


def make_sensors() -> list[SensorSpecV2]:
    return [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.2, 0.2, warmup_steps=0),
        SensorSpecV2("snow", ("snow_particle_mean_velocity_ms",), 0.4, 0.5, warmup_steps=2),
        SensorSpecV2("flux", ("snow_mass_flux_kg_m2_s",), 0.1, 0.1, warmup_steps=0),
    ]


def make_env() -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        make_truth(),
        make_sensors(),
        PowerConstraintsV2(max_active=2, per_step_budget=0.6, startup_peak_budget=0.7, required_sensor_ids=("met",)),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=16,
            seed=7,
            energy_account_enabled=True,
            energy_capacity=10.0,
            initial_energy=10.0,
            harvest_per_step=0.3,
            reserve_energy=1.0,
        ),
        oracle=DummyOracle(),
    )


def make_saturating_env() -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        make_truth(),
        make_sensors(),
        PowerConstraintsV2(max_active=2, per_step_budget=0.6, startup_peak_budget=0.7),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=16,
            seed=7,
        ),
        oracle=SaturatingOracle(),
    )


def test_event_forecast_truth_future_vector_shape():
    truth = make_truth()
    forecast = build_event_forecast(truth, 18, ForecastContextConfig(horizon=5, truth_future=True))
    assert forecast.probabilities.shape == (5,)
    assert forecast.confidence.shape == (5,)
    assert forecast.as_vector().shape == (11,)
    assert forecast.probabilities[1] == 1.0
    assert 0.0 <= forecast.time_to_event <= 1.0


def test_event_forecast_truth_future_continuous_context():
    truth = make_truth()
    truth["snow_mass_flux_kg_m2_s"] = np.arange(len(truth), dtype=float) * 1.0e-4
    forecast = build_event_forecast(
        truth,
        10,
        ForecastContextConfig(
            horizon=4,
            truth_future=True,
            continuous_truth_future=True,
            continuous_columns=("snow_mass_flux_kg_m2_s",),
            continuous_scales=(1.0e-4,),
        ),
    )
    assert forecast.probabilities.shape == (4,)
    assert forecast.continuous.shape == (7,)
    assert np.isclose(forecast.continuous[0], 10.0)
    assert np.isclose(forecast.continuous[1], 12.5)
    assert np.isclose(forecast.continuous[-1], 4.0)
    names = event_forecast_feature_names(
        horizon=4,
        continuous_columns=("snow_mass_flux_kg_m2_s",),
    )
    assert len(names) == forecast.as_vector().shape[0]
    assert names[-1] == "event_forecast_snow_mass_flux_kg_m2_s_future_delta"


def test_event_forecast_can_read_learned_probability_columns():
    truth = make_truth()
    truth["learned_event_p_h1"] = np.linspace(0.0, 1.0, len(truth))
    truth["learned_event_p_h2"] = 0.25
    forecast = build_event_forecast(
        truth,
        10,
        ForecastContextConfig(
            horizon=3,
            learned_event_probability_columns=("learned_event_p_h1", "learned_event_p_h2"),
        ),
    )
    assert forecast.probabilities.shape == (3,)
    assert np.isclose(forecast.probabilities[0], truth.loc[10, "learned_event_p_h1"])
    assert np.isclose(forecast.probabilities[1], 0.25)
    assert forecast.probabilities[2] == 0.0


def test_event_forecast_can_read_learned_continuous_columns():
    truth = make_truth()
    truth["snow_mass_flux_kg_m2_s"] = 1.0e-4
    truth["learned_cont_snow_mass_flux_kg_m2_s_h1"] = 2.0e-4
    truth["learned_cont_snow_mass_flux_kg_m2_s_h2"] = 4.0e-4
    truth["learned_cont_snow_mass_flux_kg_m2_s_h3"] = 8.0e-4
    forecast = build_event_forecast(
        truth,
        10,
        ForecastContextConfig(
            horizon=3,
            continuous_columns=("snow_mass_flux_kg_m2_s",),
            continuous_scales=(1.0e-4,),
        ),
    )
    assert forecast.continuous.shape == (7,)
    assert np.isclose(forecast.continuous[0], 1.0)
    assert np.isclose(forecast.continuous[1], (2.0 + 4.0 + 8.0) / 3.0)
    assert np.isclose(forecast.continuous[5], 8.0)
    assert np.isclose(forecast.continuous[6], 7.0)


def test_event_forecast_can_use_learned_h1_instead_of_current_task_truth():
    truth = make_truth()
    truth["snow_mass_flux_kg_m2_s"] = 99.0
    truth["learned_cont_snow_mass_flux_kg_m2_s_h1"] = 2.0e-4
    truth["learned_cont_snow_mass_flux_kg_m2_s_h2"] = 4.0e-4
    truth["learned_cont_snow_mass_flux_kg_m2_s_h3"] = 8.0e-4
    forecast = build_event_forecast(
        truth,
        10,
        ForecastContextConfig(
            horizon=3,
            continuous_columns=("snow_mass_flux_kg_m2_s",),
            continuous_scales=(1.0e-4,),
            continuous_current_source="learned_h1",
        ),
    )
    assert np.isclose(forecast.continuous[0], 2.0)
    assert np.isclose(forecast.continuous[1], (2.0 + 4.0 + 8.0) / 3.0)
    assert np.isclose(forecast.continuous[6], 6.0)


def test_event_forecast_learned_h1_requires_causal_prediction_columns():
    truth = make_truth()
    with pytest.raises(ValueError, match="requires learned forecasts"):
        build_event_forecast(
            truth,
            10,
            ForecastContextConfig(
                horizon=3,
                continuous_columns=("snow_mass_flux_kg_m2_s",),
                continuous_current_source="learned_h1",
            ),
        )


def test_window_risk_agent_state_excludes_current_truth_event():
    env = make_env()
    env.reset(start_idx=10)
    before = causal_window_agent_state(env)
    env.event_flags[env.current_idx] = not bool(env.event_flags[env.current_idx])
    after = causal_window_agent_state(env)
    assert np.array_equal(before, after)


def test_window_risk_history_is_strictly_past_and_causal():
    env = make_env()
    env.reset(start_idx=18)
    cfg = ForecastContextConfig(horizon=3)
    before, names = causal_history_summary(env, cfg, windows=(8,))
    env.truth_df.loc[18:, "wind_speed_ms"] = 999.0
    after_future_change, after_names = causal_history_summary(env, cfg, windows=(8,))
    assert names == after_names
    assert np.array_equal(before, after_future_change)
    env.truth_df.loc[17, "wind_speed_ms"] = -999.0
    after_past_change, _ = causal_history_summary(env, cfg, windows=(8,))
    assert not np.array_equal(before, after_past_change)


def test_residual_history_is_compact_and_strictly_past():
    env = make_env()
    env.reset(start_idx=18)
    before, names = causal_residual_history_summary(env, windows=(8,))
    assert len(names) == 24
    assert all("learned_" not in name for name in names)
    env.truth_df.loc[18:, "wind_speed_ms"] = 999.0
    after_future_change, after_names = causal_residual_history_summary(
        env, windows=(8,)
    )
    assert names == after_names
    assert np.array_equal(before, after_future_change)


def test_anchor_neighborhood_support_filters_far_masks_and_keeps_anchor():
    masks = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=bool,
    )
    assert anchor_neighborhood_support(
        masks,
        anchor_mask=masks[0],
        support=(0, 1, 2, 3),
        max_hamming_distance=1,
    ) == (0, 1)
    assert anchor_neighborhood_support(
        masks,
        anchor_mask=masks[0],
        support=(0, 1, 2, 3),
        max_hamming_distance=2,
    ) == (0, 1, 2)
    assert anchor_neighborhood_support(
        masks,
        anchor_mask=masks[0],
        support=(0, 1, 2, 3),
        max_hamming_distance=-1,
    ) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="retain the static anchor"):
        anchor_neighborhood_support(
            masks,
            anchor_mask=masks[0],
            support=(1, 2, 3),
            max_hamming_distance=2,
        )


def test_residual_action_specs_filter_and_feature_schema():
    masks = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    specs = residual_action_controller_specs((3, 1, 0, 2))
    assert [spec.parameters["target_action_idx"] for spec in specs] == [
        0,
        1,
        2,
        3,
    ]
    assert not valid_anchor_residual_action(
        masks,
        anchor_idx=0,
        controller=specs[0],
        allowed_action_indices=(0, 1, 2, 3),
        required_sensor_indices=(0,),
    )
    assert not valid_anchor_residual_action(
        masks,
        anchor_idx=0,
        controller=specs[1],
        allowed_action_indices=(0, 2, 3),
        required_sensor_indices=(0,),
    )
    assert valid_anchor_residual_action(
        masks,
        anchor_idx=0,
        controller=specs[1],
        allowed_action_indices=(0, 1, 2, 3),
        required_sensor_indices=(0,),
    )
    assert not valid_anchor_residual_action(
        masks,
        anchor_idx=0,
        controller=specs[2],
        allowed_action_indices=(0, 1, 2, 3),
        required_sensor_indices=(0,),
    )
    assert not valid_anchor_residual_action(
        masks,
        anchor_idx=0,
        controller=specs[3],
        allowed_action_indices=(0, 1, 2, 3),
        required_sensor_indices=(0,),
    )

    env = make_env()
    env.reset(start_idx=18)
    feature, names = build_residual_risk_feature(
        env=env,
        forecast_cfg=ForecastContextConfig(horizon=3),
        candidate_masks=masks,
        anchor_idx=0,
        controller=specs[1],
    )
    assert feature.shape == (len(names),)
    values = dict(zip(names, feature, strict=True))
    assert values["residual_add"] == 1.0
    assert values["residual_drop"] == 0.0
    assert values["residual_hamming"] == 1.0
    assert not any(name.startswith("residual_target_mask_") for name in names)
    assert not any(name.startswith("causal_history_w1024_") for name in names)


def test_continue_rollout_preserves_prefix_state_and_boundary_features():
    env = make_env()
    env.reset(start_idx=18)
    anchor = StaticMaskPolicy(
        (True, True, False),
        name="anchor_prefix",
    )
    prefix = continue_policy_rollout(env, anchor, steps=4)
    assert prefix.step_indices.tolist() == [18, 19, 20, 21]
    assert env.current_idx == 22

    feature, names = residual_boundary_state(env)
    values = dict(zip(names, feature, strict=True))
    assert values[
        f"residual_boundary_previous_mask_{env.sensor_ids[0]}"
    ] == 1.0
    assert values[
        f"residual_boundary_previous_mask_{env.sensor_ids[1]}"
    ] == 1.0
    assert values[
        f"residual_boundary_previous_mask_{env.sensor_ids[2]}"
    ] == 0.0

    boundary = snapshot_env(env)
    continuation = continue_policy_rollout(env, anchor, steps=3)
    assert continuation.step_indices.tolist() == [22, 23, 24]
    restore_env(env, boundary)
    assert env.current_idx == 22
    replay = continue_policy_rollout(env, anchor, steps=3)
    np.testing.assert_allclose(replay.observations, continuation.observations)
    np.testing.assert_array_equal(replay.selected_masks, continuation.selected_masks)


def test_common_random_numbers_do_not_depend_on_selected_mask():
    base = make_env()
    cfg = replace(base.cfg, common_random_numbers=True)
    left = WarmupSchedulingEnv(
        base.truth_df,
        base.sensor_specs,
        base.projector.constraints,
        cfg,
        oracle=base.oracle,
    )
    right = WarmupSchedulingEnv(
        base.truth_df,
        base.sensor_specs,
        base.projector.constraints,
        cfg,
        oracle=base.oracle,
    )
    left.reset(start_idx=18)
    right.reset(start_idx=18)
    left.step_mask(np.asarray([1, 1, 0], dtype=bool))
    right.step_mask(np.asarray([1, 0, 1], dtype=bool))
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    left.step_mask(np.asarray([1, 1, 0], dtype=bool))
    right.step_mask(np.asarray([1, 1, 0], dtype=bool))
    assert left.rng.bit_generator.state == right.rng.bit_generator.state


def test_utility_planner_uses_learned_continuous_risk_for_sensor_choice():
    env = make_env()
    env.reset(start_idx=8)
    for horizon_idx, value in enumerate((3.0e-4, 5.0e-4, 7.0e-4), start=1):
        env.truth_df[f"learned_cont_snow_mass_flux_kg_m2_s_h{horizon_idx}"] = value
    candidate_masks = np.asarray(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    policy = ForecastAwareUtilityPlannerPolicy(
        candidate_masks=candidate_masks,
        forecast_cfg=ForecastContextConfig(
            horizon=3,
            continuous_columns=("snow_mass_flux_kg_m2_s",),
            continuous_scales=(1.0e-4,),
        ),
        anchor_mask=candidate_masks[0],
        allowed_action_indices=(0, 1, 2),
        event_weight=0.0,
        magnitude_weight=1.0,
        variability_weight=0.5,
        freshness_weight=0.0,
        power_weight=0.0,
        switch_weight=0.0,
        min_dwell=1,
    )
    selected = policy.act_mask(env)
    assert selected.tolist() == [True, False, True]


def test_proxy_mpc_rotates_to_stale_forecast_column():
    env = make_env()
    env.reset(start_idx=8)
    for horizon_idx, value in enumerate((3.0, 3.0, 3.0), start=1):
        env.truth_df[f"learned_cont_snow_particle_mean_velocity_ms_h{horizon_idx}"] = value
    for horizon_idx, value in enumerate((4.0e-4, 4.0e-4, 4.0e-4), start=1):
        env.truth_df[f"learned_cont_snow_mass_flux_kg_m2_s_h{horizon_idx}"] = value
    candidate_masks = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    )
    policy = ForecastAwareProxyMPCPolicy(
        candidate_masks=candidate_masks,
        forecast_cfg=ForecastContextConfig(
            horizon=3,
            continuous_columns=("snow_particle_mean_velocity_ms", "snow_mass_flux_kg_m2_s"),
            continuous_scales=(1.0, 1.0e-4),
        ),
        anchor_mask=candidate_masks[1],
        allowed_action_indices=(0, 1),
        event_weight=0.0,
        magnitude_weight=1.0,
        variability_weight=0.0,
        freshness_weight=0.0,
        target_rate_weight=0.0,
        power_weight=0.0,
        switch_weight=0.0,
        age_weight=2.0,
        planning_depth=2,
        beam_width=2,
        max_branch=2,
        anchor_improvement_threshold=-1.0e9,
        min_dwell=1,
    )
    first = policy.act_mask(env)
    assert first.tolist() == [True, True, False]
    second = policy.act_mask(env)
    assert second.tolist() == [True, False, True]


def test_learned_event_forecaster_augments_truth_without_future_columns():
    truth = make_truth(72)
    cfg = EventForecasterTrainingConfig(
        horizon=3,
        lookback=4,
        hidden_dim=16,
        epochs=2,
        batch_size=16,
        seed=11,
        device="cpu",
        period_steps=8,
    )
    columns = select_event_forecast_columns(
        truth,
        preferred_columns=("wind_speed_ms", "snow_particle_mean_velocity_ms"),
        event_column="event_flag",
    )
    dataset = build_event_forecast_dataset(
        truth,
        bounds=(0, 48),
        feature_columns=columns,
        event_column="event_flag",
        cfg=cfg,
    )
    assert dataset.features.shape[0] == dataset.targets.shape[0]
    assert dataset.targets.shape[1] == 3
    bundle = train_event_forecaster(dataset, cfg)
    augmented, probability_columns = augment_truth_with_event_forecasts(truth, bundle)
    assert len(probability_columns) == 3
    for column in probability_columns:
        assert column in augmented.columns
        assert np.all((augmented[column].to_numpy() >= 0.0) & (augmented[column].to_numpy() <= 1.0))
    forecast = build_event_forecast(
        augmented,
        20,
        ForecastContextConfig(horizon=3, learned_event_probability_columns=probability_columns),
    )
    assert forecast.probabilities.shape == (3,)


def test_learned_continuous_forecaster_augments_truth_without_future_columns():
    truth = make_truth(72)
    cfg = ContinuousForecasterTrainingConfig(
        horizon=3,
        lookback=4,
        target_columns=("snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms"),
        hidden_dim=16,
        epochs=2,
        batch_size=16,
        seed=13,
        device="cpu",
        period_steps=8,
    )
    feature_columns = select_continuous_forecast_columns(
        truth,
        preferred_columns=("wind_speed_ms", "snow_particle_mean_velocity_ms", "snow_mass_flux_kg_m2_s"),
    )
    dataset = build_continuous_forecast_dataset(
        truth,
        bounds=(0, 48),
        feature_columns=feature_columns,
        target_columns=cfg.target_columns,
        cfg=cfg,
    )
    assert dataset.features.shape[0] == dataset.targets.shape[0]
    assert dataset.targets.shape[1] == 6
    bundle = train_continuous_forecaster(dataset, cfg)
    augmented, prediction_columns = augment_truth_with_continuous_forecasts(truth, bundle)
    assert len(prediction_columns) == 6
    for column in prediction_columns:
        assert column in augmented.columns
        assert np.all(np.isfinite(augmented[column].to_numpy()))
    forecast = build_event_forecast(
        augmented,
        20,
        ForecastContextConfig(
            horizon=3,
            continuous_columns=("snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms"),
            continuous_scales=(1.0e-4, 5.0),
        ),
    )
    assert forecast.continuous.shape == (14,)


def test_continuous_forecaster_predicts_from_causal_history():
    truth = make_truth(72)
    cfg = ContinuousForecasterTrainingConfig(
        horizon=3,
        lookback=4,
        target_columns=("wind_speed_ms", "air_temperature_c"),
        hidden_dim=16,
        epochs=1,
        batch_size=16,
        seed=13,
        device="cpu",
        period_steps=8,
    )
    dataset = build_continuous_forecast_dataset(
        truth,
        bounds=(0, 48),
        feature_columns=cfg.target_columns,
        target_columns=cfg.target_columns,
        cfg=cfg,
    )
    bundle = train_continuous_forecaster(dataset, cfg)
    history = truth.loc[16:20, cfg.target_columns].to_numpy(dtype=float)
    prediction = predict_continuous_from_history(
        history,
        history_columns=cfg.target_columns,
        current_idx=20,
        bundle=bundle,
    )
    assert prediction.shape == (3, 2)
    assert np.all(np.isfinite(prediction))


def test_probabilistic_world_model_uses_chronological_residual_calibration(
    tmp_path,
):
    truth = make_truth(128)
    forecaster_cfg = ContinuousForecasterTrainingConfig(
        horizon=4,
        lookback=4,
        target_columns=STATE_COLUMNS,
        hidden_dim=16,
        epochs=1,
        batch_size=32,
        seed=19,
        device="cpu",
        period_steps=8,
    )
    model = train_probabilistic_world_model(
        truth,
        bounds=(0, 120),
        state_columns=STATE_COLUMNS,
        cfg=ProbabilisticWorldModelTrainingConfig(
            member_count=2,
            fit_fraction=0.60,
            calibration_fraction=0.20,
            bootstrap_fraction=0.75,
            seed=19,
            forecaster=forecaster_cfg,
        ),
    )
    env = make_env()
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    batch = model.sample(
        context,
        horizon=5,
        n_scenarios=3,
        rng=np.random.default_rng(5),
    )
    assert batch.values.shape == (3, 5, len(STATE_COLUMNS))
    assert batch.event_flags.shape == (3, 5)
    assert model.residual_bank.shape[1:] == (4, len(STATE_COLUMNS))
    assert model.audit_metrics["bounds"]["fit"][1] <= model.audit_metrics[
        "bounds"
    ]["calibration"][0]
    assert np.isfinite(model.audit_metrics["normalized_rmse"])
    path = tmp_path / "world_model.pt"
    save_probabilistic_world_model(model, path)
    restored = load_probabilistic_world_model(path)
    restored_batch = restored.sample(
        context,
        horizon=5,
        n_scenarios=3,
        rng=np.random.default_rng(5),
    )
    np.testing.assert_allclose(restored_batch.values, batch.values)
    member_predictions = restored.predict_members(context)
    assert member_predictions.shape == (2, 4, len(STATE_COLUMNS))


def test_rollout_world_model_trains_on_masked_histories(tmp_path):
    env = make_env()
    rollout = run_policy_rollout(
        env,
        StaticMaskPolicy((True, True, False), name="static_train"),
        steps=56,
        start_idx=4,
    )
    cfg = RolloutWorldModelTrainingConfig(
        horizon=3,
        lookback=4,
        hidden_dim=16,
        epochs=1,
        batch_size=16,
        member_count=2,
        bootstrap_fraction=0.75,
        seed=23,
        device="cpu",
        period_steps=8,
        event_probability_horizon=3,
    )
    event_values = np.zeros((80, 3), dtype=np.float32)
    fit = build_rollout_world_model_dataset(
        [rollout],
        state_columns=STATE_COLUMNS,
        cfg=cfg,
        event_probability_values=event_values,
    )
    calibration = build_rollout_world_model_dataset(
        [rollout],
        state_columns=STATE_COLUMNS,
        cfg=cfg,
        event_probability_values=event_values,
        normalization=fit,
    )
    audit = build_rollout_world_model_dataset(
        [rollout],
        state_columns=STATE_COLUMNS,
        cfg=cfg,
        event_probability_values=event_values,
        normalization=fit,
    )
    model = train_rollout_world_model(
        fit_dataset=fit,
        calibration_dataset=calibration,
        audit_dataset=audit,
        cfg=cfg,
    )
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    batch = model.sample(
        context,
        horizon=4,
        n_scenarios=3,
        rng=np.random.default_rng(3),
    )
    assert batch.values.shape == (3, 4, len(STATE_COLUMNS))
    assert np.isfinite(model.audit_metrics["normalized_rmse"])
    path = tmp_path / "rollout_world_model.pt"
    save_rollout_world_model(model, path)
    restored = load_rollout_world_model(path)
    restored_batch = restored.sample(
        context,
        horizon=4,
        n_scenarios=3,
        rng=np.random.default_rng(3),
    )
    np.testing.assert_allclose(restored_batch.values, batch.values)


def test_sensor_timing_features_shape():
    features = sensor_timing_features(
        warmup_steps=[0, 2, 3],
        power_costs=[0.2, 0.4, 0.9],
        startup_peaks=[0.2, 0.5, 1.2],
        horizon=8,
    )
    assert features.shape == (3, 3)
    assert np.all(features >= 0.0)


def test_mpc_teacher_returns_feasible_action_and_restores_env():
    env = make_env()
    env.reset(start_idx=18)
    before = snapshot_env(env)
    masks = enumerate_action_masks(3, max_active=2)
    action = beam_search_teacher_action(env, masks, MpcTeacherConfig(planning_horizon=3, beam_width=3, max_branch=4))
    assert action.shape == (3,)
    projection = env.projector.project_mask(action, env.runtimes)
    assert projection.feasible
    after = snapshot_env(env)
    assert after["current_idx"] == before["current_idx"]
    assert np.allclose(after["history"], before["history"])
    assert np.allclose(after["previous_action_mask"], before["previous_action_mask"])


def test_mpc_teacher_bootstraps_coverage_when_oracle_loss_saturates():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    action = beam_search_teacher_action(env, masks, MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4))
    assert np.any(action)
    assert action[0]


def test_mpc_teacher_candidate_prior_breaks_saturated_ties():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    desired_idx = int(np.flatnonzero(np.all(masks == np.asarray([[0, 0, 1]], dtype=bool), axis=1))[0])
    costs = np.ones(masks.shape[0], dtype=float)
    costs[desired_idx] = 0.0
    action = beam_search_teacher_action(
        env,
        masks,
        MpcTeacherConfig(
            planning_horizon=1,
            beam_width=2,
            max_branch=masks.shape[0],
            saturated_coverage_bonus=0.0,
            candidate_prior_weight=5.0,
            candidate_prior_costs=tuple(float(x) for x in costs),
        ),
    )
    assert action.tolist() == [False, False, True]


def test_mpc_teacher_anchor_guard_defaults_to_anchor_without_regret_gain():
    env = make_saturating_env()
    env.reset(start_idx=18)
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (False, False, True)
    action = beam_search_teacher_action(
        env,
        masks,
        MpcTeacherConfig(
            planning_horizon=1,
            beam_width=2,
            max_branch=masks.shape[0],
            saturated_coverage_bonus=0.0,
            anchor_mask=anchor,
            anchor_regret_guard=True,
            anchor_improvement_margin=0.1,
        ),
    )
    assert action.tolist() == [False, False, True]


def test_scenario_environment_contains_no_source_future_truth():
    env = make_env()
    env.reset(start_idx=18)
    current = np.asarray(env.last_observation, dtype=float)
    scenario = np.repeat(current.reshape(1, -1), 8, axis=0)
    scenario[:, env.state_index["wind_speed_ms"]] = np.arange(8, dtype=float)
    shadow = build_scenario_environment(
        env,
        scenario,
        np.zeros(8, dtype=bool),
        planning_horizon=4,
    )
    assert len(shadow.truth_values) == env.current_idx + 8
    np.testing.assert_allclose(
        shadow.truth_values[env.current_idx :],
        scenario,
    )
    assert not np.array_equal(
        shadow.truth_values[env.current_idx :],
        env.truth_values[env.current_idx : env.current_idx + 8],
    )


def test_robust_cost_uses_upper_tail_cvar():
    score, expected, cvar = robust_cost(
        np.asarray([1.0, 1.0, 2.0, 10.0]),
        alpha=0.75,
        cvar_weight=0.5,
    )
    assert expected == 3.5
    assert cvar == 10.0
    assert score == 8.5


def test_robust_planner_is_invariant_to_hidden_future_truth():
    left_truth = make_truth()
    right_truth = left_truth.copy()
    right_truth.loc[19:, "wind_speed_ms"] = 9999.0
    right_truth.loc[19:, "snow_mass_flux_kg_m2_s"] = 123.0
    right_truth.loc[19:, "event_flag"] = ~right_truth.loc[19:, "event_flag"]
    history = left_truth.loc[:18, STATE_COLUMNS].to_numpy(dtype=float)
    base = make_env()
    cfg = replace(
        base.cfg,
        normalization_mean=tuple(np.mean(history, axis=0)),
        normalization_std=tuple(np.maximum(np.std(history, axis=0), 1.0e-6)),
    )
    left = WarmupSchedulingEnv(
        left_truth,
        make_sensors(),
        base.projector.constraints,
        cfg,
        oracle=base.oracle,
    )
    right = WarmupSchedulingEnv(
        right_truth,
        make_sensors(),
        base.projector.constraints,
        cfg,
        oracle=base.oracle,
    )
    left.reset(start_idx=18)
    right.reset(start_idx=18)
    np.testing.assert_allclose(left.history, right.history)
    context = build_causal_world_model_context(left)
    future = np.repeat(context.last_observation.reshape(1, -1), 8, axis=0)
    future[:, left.state_index["snow_particle_mean_velocity_ms"]] = 4.0
    model = FixedScenarioModel(future_values=future)
    masks = enumerate_action_masks(3, max_active=2)
    cfg = RobustPlannerConfig(
        planning_horizon=3,
        beam_width=4,
        max_branch=6,
        n_scenarios=3,
        seed=17,
        step_cost=MpcTeacherConfig(
            planning_horizon=3,
            beam_width=4,
            max_branch=6,
        ),
    )
    left_before = snapshot_env(left)
    right_before = snapshot_env(right)
    left_plan = robust_beam_search_plan(left, model, masks, cfg)
    right_plan = robust_beam_search_plan(right, model, masks, cfg)
    np.testing.assert_array_equal(left_plan.action, right_plan.action)
    np.testing.assert_allclose(left_plan.scenario_costs, right_plan.scenario_costs)
    assert snapshot_env(left)["current_idx"] == left_before["current_idx"]
    assert snapshot_env(right)["current_idx"] == right_before["current_idx"]
    projection = left.projector.project_mask(left_plan.action, left.runtimes)
    assert projection.feasible


def test_robust_planner_anchor_guard_uses_same_scenarios():
    env = make_saturating_env()
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    future = np.repeat(context.last_observation.reshape(1, -1), 6, axis=0)
    model = FixedScenarioModel(future_values=future)
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (False, False, True)
    result = robust_beam_search_plan(
        env,
        model,
        masks,
        RobustPlannerConfig(
            planning_horizon=2,
            beam_width=3,
            max_branch=6,
            n_scenarios=3,
            cvar_weight=0.5,
            step_cost=MpcTeacherConfig(
                anchor_mask=anchor,
                anchor_regret_guard=True,
                anchor_improvement_margin=100.0,
                saturated_coverage_bonus=0.0,
            ),
        ),
    )
    assert result.action.tolist() == list(anchor)
    assert result.anchor_guard_applied
    assert result.raw_robust_cost is not None
    assert result.anchor_robust_cost is not None


def test_robust_planner_component_guard_can_fallback_to_anchor():
    env = make_saturating_env()
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    future = np.repeat(context.last_observation.reshape(1, -1), 6, axis=0)
    model = FixedScenarioModel(future_values=future)
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (False, False, True)
    result = robust_beam_search_plan(
        env,
        model,
        masks,
        RobustPlannerConfig(
            planning_horizon=2,
            beam_width=3,
            max_branch=6,
            n_scenarios=3,
            component_guard_min_task_margin=1.0,
            step_cost=MpcTeacherConfig(
                anchor_mask=anchor,
                anchor_regret_guard=False,
                saturated_coverage_bonus=0.0,
            ),
        ),
    )
    assert result.action.tolist() == list(anchor)
    assert result.component_guard_applied
    assert not result.anchor_guard_applied
    assert result.component_guard_stats["task_error_margin_mean"] == 0.0


def test_robust_planner_hold_component_guard_extends_shadow_depth():
    env = make_saturating_env()
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    future = np.repeat(context.last_observation.reshape(1, -1), 10, axis=0)
    model = FixedScenarioModel(future_values=future)
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (False, False, True)
    result = robust_beam_search_plan(
        env,
        model,
        masks,
        RobustPlannerConfig(
            planning_horizon=2,
            beam_width=3,
            max_branch=6,
            n_scenarios=3,
            replan_interval=4,
            component_guard_mode="hold",
            component_guard_min_task_margin=1.0,
            step_cost=MpcTeacherConfig(
                anchor_mask=anchor,
                anchor_regret_guard=False,
                saturated_coverage_bonus=0.0,
            ),
        ),
    )
    assert result.action.tolist() == list(anchor)
    assert result.component_guard_applied
    assert result.component_guard_stats["task_error_margin_mean"] == 0.0


def test_robust_receding_policy_holds_action_until_replan():
    env = make_env()
    env.reset(start_idx=18)
    context = build_causal_world_model_context(env)
    future = np.repeat(context.last_observation.reshape(1, -1), 6, axis=0)
    policy = RobustRecedingHorizonPolicy(
        scenario_model=FixedScenarioModel(future_values=future),
        candidate_masks=enumerate_action_masks(3, max_active=2),
        cfg=RobustPlannerConfig(
            planning_horizon=2,
            beam_width=2,
            max_branch=4,
            n_scenarios=2,
            replan_interval=3,
        ),
    )
    first = policy.act_mask(env)
    second = policy.act_mask(env)
    third = policy.act_mask(env)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)
    assert policy._remaining_steps == 0


def test_event_transport_rich_selection_prefers_harder_transport_windows():
    truth = make_truth(100)
    truth["event_flag"] = False
    truth.loc[0:9, "event_flag"] = True
    truth.loc[20:29, "event_flag"] = True
    truth.loc[40:47, "event_flag"] = True
    truth.loc[60:67, "event_flag"] = True
    truth["snow_mass_flux_kg_m2_s"] = 0.0
    truth["snow_particle_mean_diameter_mm"] = 0.0
    truth["snow_particle_mean_velocity_ms"] = 0.0
    truth.loc[40:49, "snow_particle_mean_diameter_mm"] = np.arange(10, dtype=float)
    truth.loc[60:69, "snow_particle_mean_diameter_mm"] = np.arange(10, dtype=float)
    truth.loc[40:49, "snow_particle_mean_velocity_ms"] = np.arange(10, dtype=float) * 10.0
    truth.loc[60:69, "snow_particle_mean_velocity_ms"] = np.arange(10, dtype=float) * 10.0
    truth.loc[40:49, "snow_mass_flux_kg_m2_s"] = np.arange(10, dtype=float) * 1.0e-5
    truth.loc[60:69, "snow_mass_flux_kg_m2_s"] = np.arange(10, dtype=float) * 1.0e-5

    event_only = choose_non_overlapping_starts(
        truth,
        bounds=(0, 100),
        window_steps=10,
        horizon=1,
        count=2,
        selection="event_rich",
        stride=10,
        event_column="event_flag",
        seed=0,
    )
    event_transport = choose_non_overlapping_starts(
        truth,
        bounds=(0, 100),
        window_steps=10,
        horizon=1,
        count=2,
        selection="event_transport_rich",
        stride=10,
        event_column="event_flag",
        seed=0,
    )

    assert event_only.starts == (0, 20)
    assert event_transport.starts == (40, 60)
    assert event_transport.diagnostics["selected_snow_particle_mean_velocity_ms_std_mean"] > 0.0


def test_task_focus_metrics_uses_event_filtered_normalized_error():
    result = RolloutResult(
        policy_name="p",
        observations=np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=float),
        masks=np.zeros((2, 2), dtype=float),
        truth=np.asarray([[1.0, 1.0], [4.0, 10.0]], dtype=float),
        rewards=np.zeros(2, dtype=float),
        scores=np.zeros((2, 1), dtype=float),
        powers=np.zeros(2, dtype=float),
        peaks=np.zeros(2, dtype=float),
        selected_masks=np.zeros((2, 1), dtype=bool),
        mode_ids=np.zeros((2, 1), dtype=int),
        event_flags=np.asarray([0.0, 1.0], dtype=float),
        oracle_losses=np.zeros(2, dtype=float),
        step_indices=np.asarray([0, 1], dtype=int),
        warmup_abort_count=0,
        warmup_abort_deltas=np.zeros(2, dtype=int),
        energy_guard_dropped=np.zeros(2, dtype=int),
        soc=np.zeros(2, dtype=float),
    )
    metrics = task_focus_metrics(
        result,
        state_columns=("a", "b"),
        task_error_columns=("a", "b"),
        task_error_scales=(2.0, 3.0),
        event_only=True,
    )
    assert np.isclose(metrics["task_error_event_mean"], 1.5)
    assert np.isclose(metrics["task_error_mean"], 1.5)


def test_static_margin_guard_prefers_robust_validation_policy():
    rows = [
        {
            "policy": "forecast_aware_advantage_residual",
            "objective": 0.91,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.03,
            "objective_margin_min": -0.05,
            "negative_start_count": 2,
        },
        {
            "policy": "forecast_aware_value_residual",
            "objective": 0.93,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.01,
            "objective_margin_min": -0.005,
            "negative_start_count": 1,
        },
    ]
    selected_mean = choose_deployable_validation_row(rows.copy(), criterion="mean_objective")
    assert selected_mean["policy"] == "forecast_aware_advantage_residual"
    selected_guard = choose_deployable_validation_row(
        rows.copy(),
        criterion="static_margin_guard",
        min_mean_margin=0.0,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_guard["policy"] == "forecast_aware_value_residual"
    assert selected_guard["static_margin_guard_pass"]


def test_static_margin_guard_rejects_noop_when_positive_margin_required():
    rows = [
        {
            "policy": "forecast_aware_recurrent_value",
            "objective": 0.90,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.0,
            "objective_margin_min": 0.0,
            "negative_start_count": 0,
        },
        {
            "policy": "forecast_aware_event_threshold",
            "objective": 0.95,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.002,
            "objective_margin_min": -0.005,
            "negative_start_count": 1,
        },
    ]
    selected = choose_deployable_validation_row(
        rows,
        criterion="static_margin_guard",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert rows[0]["static_margin_guard_pass"] is False
    assert selected["policy"] == "forecast_aware_event_threshold"
    assert selected["static_margin_guard_pass"]


def test_static_margin_guard_can_require_a_passing_candidate():
    rows = [
        {
            "policy": "forecast_aware_event_threshold",
            "objective": 0.90,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "objective_margin_mean": 0.0005,
            "objective_margin_min": -0.02,
            "negative_start_count": 2,
        }
    ]
    selected_legacy = choose_deployable_validation_row(
        rows.copy(),
        criterion="static_margin_guard",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_legacy is not None
    assert selected_legacy["static_margin_guard_pass"] is False

    selected_strict = choose_deployable_validation_row(
        rows.copy(),
        criterion="static_margin_guard",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
        require_guard_pass=True,
    )
    assert selected_strict is None


def test_static_margin_risk_prefers_lower_transfer_risk_when_guard_fails():
    rows = [
        {
            "policy": "low_objective_high_risk",
            "objective": 0.90,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "static_start_objectives": [1.0, 1.0, 1.0, 1.0],
            "candidate_start_objectives": [1.03, 1.02, 0.99, 0.98],
            "objective_margin_mean": -0.0025,
            "objective_margin_min": -0.03,
            "negative_start_count": 2,
        },
        {
            "policy": "higher_objective_lower_risk",
            "objective": 0.95,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "static_start_objectives": [1.0, 1.0, 1.0, 1.0],
            "candidate_start_objectives": [1.018, 0.998, 0.996, 0.99],
            "objective_margin_mean": 0.002,
            "objective_margin_min": -0.018,
            "negative_start_count": 2,
        },
    ]

    selected_guard = choose_deployable_validation_row(
        [dict(row) for row in rows],
        criterion="static_margin_guard",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_guard["policy"] == "low_objective_high_risk"
    assert selected_guard["static_margin_guard_pass"] is False

    selected_risk = choose_deployable_validation_row(
        [dict(row) for row in rows],
        criterion="static_margin_risk",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_risk["policy"] == "higher_objective_lower_risk"
    assert selected_risk["static_margin_guard_pass"] is False
    assert selected_risk["objective_margin_median"] > 0.0


def test_static_margin_risk_can_require_positive_center():
    rows = [
        {
            "policy": "negative_center",
            "objective": 0.90,
            "power_mean": 1.0,
            "warmup_abort_count": 0,
            "static_start_objectives": [1.0, 1.0, 1.0, 1.0],
            "candidate_start_objectives": [1.04, 1.03, 1.02, 0.99],
            "objective_margin_mean": -0.0125,
            "objective_margin_min": -0.04,
            "negative_start_count": 3,
        }
    ]

    selected_legacy = choose_deployable_validation_row(
        [dict(row) for row in rows],
        criterion="static_margin_risk",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
    )
    assert selected_legacy["policy"] == "negative_center"
    assert selected_legacy["static_margin_positive_center"] is False

    selected_center = choose_deployable_validation_row(
        [dict(row) for row in rows],
        criterion="static_margin_risk",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
        require_positive_center=True,
    )
    assert selected_center is None


def test_static_margin_risk_band_can_reject_weak_lower_tail():
    row = {
        "policy": "positive_center_weak_tail",
        "objective": 0.90,
        "power_mean": 1.0,
        "warmup_abort_count": 0,
        "static_start_objectives": [1.0, 1.0, 1.0, 1.0],
        "candidate_start_objectives": [1.018, 1.002, 0.990, 0.980],
        "objective_margin_mean": 0.0025,
        "objective_margin_min": -0.018,
        "negative_start_count": 2,
    }

    selected_loose = choose_deployable_validation_row(
        [dict(row)],
        criterion="static_margin_risk",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
        require_positive_center=True,
        require_risk_band=True,
        risk_min_q25_margin=-0.007,
        risk_max_negative_starts=4,
    )
    assert selected_loose is not None
    assert selected_loose["static_margin_positive_center"] is True
    assert selected_loose["objective_margin_q25"] < 0.0

    selected_strict = choose_deployable_validation_row(
        [dict(row)],
        criterion="static_margin_risk",
        min_mean_margin=0.001,
        min_start_margin=-0.01,
        max_negative_starts=1,
        require_positive_center=True,
        require_risk_band=True,
        risk_min_q25_margin=-0.005,
        risk_max_negative_starts=4,
    )
    assert selected_strict is None


def test_teacher_dataset_and_bc_policy_smoke(tmp_path):
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=2, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=True)
    dataset = collect_teacher_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=6,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.features.shape[0] == 6
    assert dataset.action_masks.shape == (6, masks.shape[0])
    assert len(dataset.feature_names) == dataset.features.shape[1]
    assert dataset.feature_names[-7:] == event_forecast_feature_names(horizon=3)
    dataset_path = tmp_path / "teacher_dataset.npz"
    dataset.save_npz(str(dataset_path))
    loaded = TeacherDataset.load_npz(str(dataset_path))
    assert loaded.feature_names == dataset.feature_names
    model, history = train_bc_classifier(
        dataset.features,
        dataset.labels,
        dataset.action_masks,
        cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareBCPolicy(model=model, candidate_masks=masks, forecast_cfg=forecast_cfg, device="cpu")
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible
    fallback = tuple(bool(x) for x in masks[1])
    guarded_policy = ForecastAwareBCPolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        fallback_mask=fallback,
        min_logit_margin=1.0e9,
    )
    env.reset(start_idx=18)
    guarded_action = guarded_policy.act_mask(env)
    assert guarded_action.tolist() == list(fallback)
    env = make_env()
    dagger = collect_dagger_dataset(
        env,
        masks,
        policy=policy,
        start_indices=(18,),
        steps_per_start=2,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    combined = concat_teacher_datasets([dataset, dagger])
    assert combined.features.shape[0] == dataset.features.shape[0] + dagger.features.shape[0]
    checkpoint = tmp_path / "bc_policy.pt"
    save_bc_policy_checkpoint(
        checkpoint,
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        train_cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
        history=history,
    )
    loaded = load_bc_policy_checkpoint(checkpoint, device="cpu")
    env.reset(start_idx=18)
    loaded_action = loaded.act_mask(env)
    assert loaded_action.shape == (3,)
    assert env.projector.project_mask(loaded_action, env.runtimes).feasible
    knn_policy = ForecastAwareKNNPolicy(
        features=combined.features,
        labels=combined.labels,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        k=3,
    )
    env.reset(start_idx=18)
    knn_action = knn_policy.act_mask(env)
    assert knn_action.shape == (3,)
    assert env.projector.project_mask(knn_action, env.runtimes).feasible
    cycle_policy = ForecastAwareCyclePolicy(labels=combined.labels, candidate_masks=masks)
    env.reset(start_idx=18)
    cycle_action = cycle_policy.act_mask(env)
    assert cycle_action.shape == (3,)
    assert env.projector.project_mask(cycle_action, env.runtimes).feasible
    mask_model, mask_history = train_mask_bc(
        dataset.features,
        dataset.labels,
        dataset.candidate_masks,
        cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
    )
    assert len(mask_history["loss"]) == 2
    mask_policy = ForecastAwareMaskBCPolicy(
        model=mask_model,
        forecast_cfg=forecast_cfg,
        device="cpu",
        required_sensor_indices=(0,),
    )
    env.reset(start_idx=18)
    mask_action = mask_policy.act_mask(env)
    assert mask_action.shape == (3,)
    assert mask_action[0]
    assert env.projector.project_mask(mask_action, env.runtimes).feasible
    sequence_model, sequence_history = train_sequence_mask_bc(
        combined.features,
        combined.labels,
        combined.candidate_masks,
        combined.step_indices,
        cfg=BCTrainingConfig(epochs=2, batch_size=3, hidden_dim=16, device="cpu"),
    )
    assert len(sequence_history["loss"]) == 2
    sequence_policy = ForecastAwareSequenceMaskPolicy(
        model=sequence_model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        allowed_action_indices=tuple(np.unique(combined.labels)),
        anchor_mask=tuple(bool(x) for x in masks[0]),
    )
    env.reset(start_idx=18)
    sequence_first = sequence_policy.act_mask(env)
    assert sequence_first.shape == (3,)
    assert env.projector.project_mask(sequence_first, env.runtimes).feasible
    env.step_mask(sequence_first)
    sequence_second = sequence_policy.act_mask(env)
    assert sequence_second.shape == (3,)
    assert env.projector.project_mask(sequence_second, env.runtimes).feasible


def test_bc_policy_can_preserve_warming_sensor():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    met_only_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    met_snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])

    class FixedLogitModel:
        input_dim = env._state().shape[0] + forecast_cfg.horizon * 2 + 1
        hidden_dim = 1
        n_actions = masks.shape[0]

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            logits = torch.zeros((x.shape[0], masks.shape[0]), dtype=torch.float32, device=x.device)
            logits[:, met_only_idx] = 10.0
            logits[:, met_snow_idx] = 1.0
            return logits

    env.reset(start_idx=18)
    env.step_mask(np.asarray([True, True, False], dtype=bool))
    assert str(env.runtimes["snow"].mode.name).lower() == "warming"
    unguarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=False,
    )
    guarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=True,
    )
    assert unguarded.act_mask(env).tolist() == [True, False, False]
    assert guarded.act_mask(env).tolist() == [True, True, False]
    support_guarded = ForecastAwareBCPolicy(
        model=FixedLogitModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        preserve_warming=False,
        allowed_action_indices=(met_snow_idx,),
    )
    assert support_guarded.act_mask(env).tolist() == [True, True, False]


def test_validation_cyclic_dwell_preserves_warming_sensor():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    met_snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])
    met_flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 1]], dtype=bool), axis=1))[0])
    policy = ValidationCyclicDwellPolicy(
        candidate_masks=masks,
        action_indices=(met_snow_idx, met_flux_idx),
        dwell_steps=1,
        fallback_action_idx=met_flux_idx,
        preserve_warming=True,
    )

    env.reset(start_idx=18)
    first = policy.act_mask(env)
    assert first.tolist() == [True, True, False]
    env.step_mask(first)
    assert str(env.runtimes["snow"].mode.name).lower() == "warming"
    second = policy.act_mask(env)
    assert second.tolist() == [True, True, False]


def test_runtime_risk_guard_keeps_static_until_window_risk_opens():
    env = make_env()

    class DynamicFluxPolicy:
        name = "dynamic_flux"

        def reset(self):
            self.calls = 0

        def act_mask(self, env):
            del env
            self.calls += 1
            return np.asarray([True, False, True], dtype=bool)

    low_risk = ForecastAwareRuntimeRiskGuardPolicy(
        dynamic_policy=DynamicFluxPolicy(),
        forecast_cfg=ForecastContextConfig(horizon=3, truth_future=False),
        anchor_mask=(True, False, False),
        threshold=10.0,
        window_steps=2,
    )
    env.reset(start_idx=22)
    assert low_risk.act_mask(env).tolist() == [True, False, False]

    high_risk = ForecastAwareRuntimeRiskGuardPolicy(
        dynamic_policy=DynamicFluxPolicy(),
        forecast_cfg=ForecastContextConfig(horizon=3, truth_future=False),
        anchor_mask=(True, False, False),
        threshold=0.1,
        window_steps=2,
    )
    env.reset(start_idx=22)
    assert high_risk.act_mask(env).tolist() == [True, False, True]


def test_event_threshold_policy_switches_on_forecast_probability():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.0
    truth.loc[20:, "learned_event_p_h1"] = 0.8
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    event_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventThresholdPolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_idx=event_idx,
        threshold=0.5,
        aggregation="max",
    )
    env.reset(start_idx=18)
    assert policy.act_mask(env).tolist() == [True, False, False]
    env.reset(start_idx=22)
    assert policy.act_mask(env).tolist() == [True, False, True]


def test_event_support_cycle_policy_rotates_teacher_supported_actions():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventSupportCyclePolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_indices=(snow_idx, flux_idx),
        threshold=0.5,
        aggregation="max",
        cycle_period=1,
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env).tolist()
    env.reset(start_idx=19)
    second = policy.act_mask(env).tolist()
    assert first != second
    assert first in ([True, True, False], [True, False, True])
    assert second in ([True, True, False], [True, False, True])


def test_event_support_cycle_policy_can_select_by_freshness():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareEventSupportCyclePolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        event_action_indices=(flux_idx, snow_idx),
        threshold=0.5,
        aggregation="max",
        selection_mode="freshness",
    )
    env.reset(start_idx=50)
    env.runtimes["met"].last_observed_step = 50
    env.runtimes["snow"].last_observed_step = None
    env.runtimes["flux"].last_observed_step = 50
    assert policy.act_mask(env).tolist() == [True, True, False]


def test_option_planner_policy_uses_static_fallback_and_dwell():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.0
    truth.loc[20:, "learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareOptionPlannerPolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        option_action_indices=(snow_idx, flux_idx),
        target_rates=(1.0, 1.0, 0.0),
        threshold=0.5,
        aggregation="max",
        min_dwell=2,
        cooldown=0,
        target_rate_weight=1.0,
        freshness_weight=0.0,
        transport_weight=0.1,
        power_weight=0.0,
        switch_weight=0.0,
        preserve_warming=False,
    )
    env.reset(start_idx=18)
    assert policy.act_mask(env).tolist() == [True, False, False]
    policy.reset()
    env.reset(start_idx=22)
    first = policy.act_mask(env)
    assert first.tolist() == [True, True, False]
    env.step_mask(first)
    env.truth_df.loc[:, "learned_event_p_h1"] = 0.0
    second = policy.act_mask(env)
    assert second.tolist() == first.tolist()


def test_option_planner_rate_balance_prefers_target_duty_match():
    env = make_env()
    truth = env.truth_df.copy()
    truth["learned_event_p_h1"] = 0.9
    env.truth_df = truth
    masks = enumerate_action_masks(3, max_active=2)
    anchor = (True, False, False)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareOptionPlannerPolicy(
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(
            horizon=1,
            learned_event_probability_columns=("learned_event_p_h1",),
        ),
        anchor_mask=anchor,
        option_action_indices=(snow_idx, flux_idx),
        target_rates=(1.0, 0.0, 1.0),
        threshold=0.5,
        target_rate_weight=0.0,
        rate_balance_weight=10.0,
        freshness_weight=0.0,
        transport_weight=0.0,
        power_weight=0.0,
        switch_weight=0.0,
        preserve_warming=False,
    )
    env.reset(start_idx=20)
    assert policy.act_mask(env).tolist() == [True, False, True]


def test_teacher_rate_policy_tracks_target_duty_cycle():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])
    policy = ForecastAwareTeacherRatePolicy(
        candidate_masks=masks,
        target_rates=(1.0, 0.5, 0.5),
        allowed_action_indices=(snow_idx, flux_idx),
        freshness_weight=0.0,
        power_weight=0.0,
        preserve_warming=False,
        anchor_mask=(True, False, False),
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env)
    env.step_mask(first)
    second = policy.act_mask(env)
    assert first.tolist() in ([True, True, False], [True, False, True])
    assert second.tolist() in ([True, True, False], [True, False, True])
    assert first.tolist() != second.tolist()


def test_contextual_duty_policy_uses_probability_deficit_feedback():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, True, False]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[True, False, True]], dtype=bool), axis=1))[0])

    class FixedMaskModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            return torch.as_tensor([[8.0, 0.0, 0.0]], dtype=torch.float32, device=x.device)

    policy = ForecastAwareContextualDutyPolicy(
        model=FixedMaskModel(),
        candidate_masks=masks,
        forecast_cfg=ForecastContextConfig(horizon=3, truth_future=False),
        device="cpu",
        allowed_action_indices=(snow_idx, flux_idx),
        anchor_mask=(True, False, False),
        blend=1.0,
        deficit_weight=2.0,
        freshness_weight=0.0,
        power_weight=0.0,
        preserve_warming=False,
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env)
    env.step_mask(first)
    second = policy.act_mask(env)
    assert first.tolist() in ([True, True, False], [True, False, True])
    assert second.tolist() in ([True, True, False], [True, False, True])
    assert first.tolist() != second.tolist()


def test_action_cost_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    dataset = collect_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    model, history = train_action_cost_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareCostPolicy(model=model, candidate_masks=masks, forecast_cfg=forecast_cfg, device="cpu")
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible

    import torch

    met_snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])

    class FixedCostModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            # Prefer actions without the snow sensor unless the support guard
            # removes them from the runtime candidate set.
            return x[:, -2:-1].clone()

    guarded_policy = ForecastAwareCostPolicy(
        model=FixedCostModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        device="cpu",
        allowed_action_indices=(met_snow_idx,),
    )
    env.reset(start_idx=18)
    assert guarded_policy.act_mask(env).tolist() == [True, True, False]


def test_collect_executed_outcome_datasets_uses_projected_actions():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=1, truth_future=False)

    class OverBudgetPolicy:
        name = "over_budget_policy"

        def reset(self):
            pass

        def act_mask(self, env):
            del env
            return np.asarray([True, True, True], dtype=bool)

    cost_dataset, transition_dataset = collect_executed_outcome_datasets(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
        rollout_policy=OverBudgetPolicy(),
    )
    assert cost_dataset.inputs.shape[0] > 0
    assert transition_dataset.inputs.shape == cost_dataset.inputs.shape
    assert transition_dataset.deltas.shape[0] == cost_dataset.inputs.shape[0]
    assert np.all(np.isfinite(cost_dataset.costs))
    executed_action_features = cost_dataset.inputs[:, -3:].astype(bool)
    assert not np.any(np.all(executed_action_features, axis=1))
    assert all(np.any(np.all(masks == row, axis=1)) for row in executed_action_features)


def test_ensemble_value_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    dataset = collect_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    models, histories = train_action_cost_ensemble(
        dataset,
        ActionCostTrainingConfig(
            epochs=2,
            batch_size=8,
            hidden_dim=16,
            device="cpu",
            ensemble_size=2,
            bootstrap_fraction=0.75,
        ),
    )
    assert len(models) == 2
    assert len(histories) == 2
    anchor = tuple(bool(x) for x in masks[0])
    policy = ForecastAwareEnsembleValuePolicy(
        models=models,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        uncertainty_beta=0.5,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_rollout_value_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    cost_dataset = collect_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    cost_model, _ = train_action_cost_model(
        cost_dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    transition_dataset = collect_feature_transition_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
        allowed_action_indices=(anchor_idx,),
        anchor_mask=anchor,
    )
    assert transition_dataset.inputs.shape[0] > 0
    assert transition_dataset.deltas.shape[1] == transition_dataset.feature_dim
    transition_model, history = train_feature_transition_model(
        transition_dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareRolloutValuePolicy(
        cost_model=cost_model,
        transition_model=transition_model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(anchor_idx,),
        advantage_threshold=-1.0,
        planning_depth=2,
        beam_width=2,
        max_branch=2,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert action.tolist() == [True, False, False]


def test_window_candidate_policy_selects_positive_tail_candidate():
    class FixedMaskPolicy:
        def __init__(self, mask: list[bool], name: str) -> None:
            self.mask = np.asarray(mask, dtype=bool)
            self.name = name
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
            del env
            return self.mask.copy()

    env = make_env()
    env.reset(start_idx=18)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    feature = append_event_forecast(
        env._state().astype(np.float32),
        build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg),
    )
    memory_features = np.vstack(
        [
            feature + 0.01,
            feature + 0.02,
            feature,
            feature + 0.005,
        ]
    ).astype(np.float32)
    policy = ForecastAwareWindowCandidatePolicy(
        memory_features=memory_features,
        memory_margins=np.asarray([-0.05, -0.03, 0.04, 0.06], dtype=float),
        memory_candidate_ids=np.asarray([0, 0, 1, 1], dtype=int),
        candidate_policies=(
            FixedMaskPolicy([True, False, True], "bad_candidate"),
            FixedMaskPolicy([True, True, False], "good_candidate"),
        ),
        forecast_cfg=forecast_cfg,
        anchor_mask=(True, False, False),
        k=2,
        margin_threshold=0.0,
        score_quantile=0.25,
        window_steps=3,
        distance_weighting="inverse",
        preserve_warming=False,
    )
    action = policy.act_mask(env)
    assert action.tolist() == [True, True, False]
    assert policy.active_candidate_id == 1
    assert policy.last_predicted_margin > 0.0


def test_recurrent_value_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    dataset = collect_recurrent_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=4,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
        allowed_action_indices=(anchor_idx,),
        anchor_mask=anchor,
    )
    assert dataset.features.shape[0] > 0
    assert dataset.costs.shape == dataset.action_masks.shape
    model, history = train_recurrent_action_cost_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu", rank_weight=0.5),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareRecurrentValuePolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(anchor_idx,),
        advantage_threshold=-1.0,
    )
    env.reset(start_idx=18)
    first = policy.act_mask(env)
    assert first.shape == (3,)
    assert first.tolist() == [True, False, False]
    env.step_mask(first)
    second = policy.act_mask(env)
    assert second.shape == (3,)
    assert second.tolist() == [True, False, False]


def test_cost_knn_policy_uses_teacher_cost_memory_and_anchor_threshold():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 1]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    dataset = collect_recurrent_action_cost_dataset(
        env,
        masks,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
        allowed_action_indices=(anchor_idx, flux_idx),
        anchor_mask=anchor,
    )
    costs = np.full_like(dataset.costs, 10.0, dtype=np.float32)
    action_masks = np.zeros_like(dataset.action_masks, dtype=bool)
    costs[:, anchor_idx] = 1.0
    costs[:, flux_idx] = 0.0
    action_masks[:, anchor_idx] = True
    action_masks[:, flux_idx] = True

    policy = ForecastAwareCostKNNPolicy(
        features=dataset.features,
        costs=costs,
        action_masks=action_masks,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        allowed_action_indices=(anchor_idx, flux_idx),
        k=2,
        advantage_threshold=0.0,
        distance_weighting="uniform",
        preserve_warming=False,
    )
    env.reset(start_idx=18)
    assert policy.act_mask(env).tolist() == [True, False, True]

    conservative = ForecastAwareCostKNNPolicy(
        features=dataset.features,
        costs=costs,
        action_masks=action_masks,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        allowed_action_indices=(anchor_idx, flux_idx),
        k=2,
        advantage_threshold=2.0,
        distance_weighting="uniform",
        preserve_warming=False,
    )
    env.reset(start_idx=18)
    assert conservative.act_mask(env).tolist() == [True, False, False]


def test_macro_option_policy_selects_nearest_teacher_snippet_and_static_fallback():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=True)
    env.reset(start_idx=20)
    event_feature = append_event_forecast(
        env._state().astype(np.float32),
        build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg),
    )
    features = np.vstack(
        [
            event_feature + 50.0,
            event_feature,
            event_feature + 5.0,
        ]
    ).astype(np.float32)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    snow_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 1, 0]], dtype=bool), axis=1))[0])
    flux_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 1]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])

    policy = ForecastAwareMacroOptionPolicy(
        features=features,
        labels=np.asarray([flux_idx, snow_idx, flux_idx], dtype=np.int64),
        candidate_masks=masks,
        step_indices=np.asarray([0, 1, 2], dtype=np.int64),
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        segment_len=2,
        k=1,
        event_threshold=0.0,
        aggregation="max",
        distance_weighting="inverse",
        refresh_interval=0,
        preserve_warming=False,
    )
    env.reset(start_idx=20)
    assert policy.act_mask(env).tolist() == [True, True, False]
    assert policy.act_mask(env).tolist() == [True, False, True]

    fallback = ForecastAwareMacroOptionPolicy(
        features=features,
        labels=np.asarray([flux_idx, snow_idx, flux_idx], dtype=np.int64),
        candidate_masks=masks,
        step_indices=np.asarray([0, 1, 2], dtype=np.int64),
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        segment_len=2,
        k=1,
        event_threshold=1.1,
        aggregation="max",
        distance_weighting="inverse",
        preserve_warming=False,
    )
    env.reset(start_idx=20)
    assert fallback.act_mask(env).tolist() == [True, False, False]


def test_recurrent_anchor_advantage_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    dataset = collect_recurrent_anchor_advantage_dataset(
        env,
        masks,
        anchor_mask=anchor,
        start_indices=(18,),
        steps_per_start=4,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
        allowed_action_indices=(anchor_idx,),
    )
    assert dataset.features.shape[0] > 0
    assert dataset.advantages.shape == dataset.action_masks.shape
    assert dataset.anchor_idx == anchor_idx
    model, history = train_recurrent_anchor_advantage_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu", rank_weight=0.5),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareRecurrentAdvantagePolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(anchor_idx,),
        advantage_threshold=-1.0,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert action.tolist() == [True, False, False]


def test_anchor_advantage_residual_policy_smoke():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=2)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    anchor = tuple(bool(x) for x in masks[anchor_idx])
    dataset = collect_anchor_advantage_dataset(
        env,
        masks,
        anchor_mask=anchor,
        start_indices=(18,),
        steps_per_start=3,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    assert dataset.anchor_idx == anchor_idx
    model, history = train_anchor_advantage_model(
        dataset,
        ActionCostTrainingConfig(epochs=2, batch_size=8, hidden_dim=16, device="cpu"),
    )
    assert len(history["loss"]) == 2
    policy = ForecastAwareAdvantageResidualPolicy(
        model=model,
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.shape == (3,)
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_anchor_advantage_dataset_handles_projected_anchor():
    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    teacher_cfg = MpcTeacherConfig(planning_horizon=1, beam_width=2, max_branch=4)
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)
    anchor = (True, True, True)
    env.reset(start_idx=18)
    assert not np.array_equal(env.projector.project_mask(np.asarray(anchor), env.runtimes).selected_mask, np.asarray(anchor))
    dataset = collect_anchor_advantage_dataset(
        env,
        masks,
        anchor_mask=anchor,
        start_indices=(18,),
        steps_per_start=2,
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )
    assert dataset.inputs.shape[0] > 0
    assert dataset.anchor_idx is not None
    assert np.any(np.isclose(dataset.advantages, 0.0))


def test_advantage_residual_policy_falls_back_to_projected_anchor():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    anchor = (True, True, True)
    met_only_idx = int(np.flatnonzero(np.all(masks == np.asarray([[1, 0, 0]], dtype=bool), axis=1))[0])
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)

    class NegativeAdvantageModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            out = torch.full((x.shape[0], 1), -1.0, dtype=torch.float32, device=x.device)
            # The deployable candidate set still contains a valid action, but a
            # non-positive predicted advantage should choose the static anchor
            # mask and let the environment projector execute it.
            out[:, 0] = -1.0
            return out

    policy = ForecastAwareAdvantageResidualPolicy(
        model=NegativeAdvantageModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(met_only_idx,),
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    action = policy.act_mask(env)
    assert action.tolist() == [True, True, True]
    assert env.projector.project_mask(action, env.runtimes).feasible


def test_advantage_residual_policy_does_not_open_full_space_when_support_invalid():
    import torch

    env = make_env()
    masks = enumerate_action_masks(3, max_active=3)
    anchor = (True, True, True)
    anchor_idx = int(np.flatnonzero(np.all(masks == np.asarray([anchor], dtype=bool), axis=1))[0])
    forecast_cfg = ForecastContextConfig(horizon=3, truth_future=False)

    class PositiveAdvantageModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, x):
            return torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)

    policy = ForecastAwareAdvantageResidualPolicy(
        model=PositiveAdvantageModel(),
        candidate_masks=masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor,
        device="cpu",
        allowed_action_indices=(anchor_idx,),
        advantage_threshold=0.0,
    )
    env.reset(start_idx=18)
    projection = env.projector.project_mask(np.asarray(anchor), env.runtimes)
    assert not np.array_equal(projection.selected_mask, np.asarray(anchor))
    action = policy.act_mask(env)
    assert action.tolist() == [True, True, True]


def test_budget_matrix_parse_budget_ignores_matrix_root_name():
    module_path = ROOT / "v1" / "scripts" / "aggregate_budget_matrix.py"
    spec = importlib.util.spec_from_file_location("aggregate_budget_matrix_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path(
        "v1/artifacts/budget_matrix_learned_hybrid_event_guarded_paired/"
        "budget1p20/learned_hybrid_event_guarded_safe_seed41/gate_summary.json"
    )
    assert module.parse_budget(path) == 1.20


def test_claim_assessment_enforces_win_rate_against_actual_n():
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = pd.DataFrame(
        {
            "preset": ["main"] * 10,
            "deployable_margin": [1.0] * 7 + [-1.0] * 3,
            "teacher_margin": [1.0] * 10,
        }
    )
    assessment = module.assess_claim(runs, main_preset="main", min_seeds=5, min_win_rate=0.8)
    assert assessment["required_wins"] == 8
    assert assessment["deployable_wins"] == 7
    assert not assessment["claim_pass"]


def test_claim_aggregate_collects_multiple_roots(tmp_path):
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for idx, root in enumerate((tmp_path / "suite_a", tmp_path / "suite_b"), start=1):
        run_dir = root / f"main_seed{idx}"
        run_dir.mkdir(parents=True)
        (run_dir / "gate_summary.json").write_text(
            """
{
  "validation_selected_static_objective": 1.0,
  "teacher_reference_objective": 0.8,
  "best_deployable_objective": 0.9,
  "teacher_beats_static": true,
  "gate_pass": true,
  "objective_metric": "task_composite",
  "best_deployable_policy": "forecast_aware_event_threshold"
}
""".strip(),
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text('{"seed": %d}' % idx, encoding="utf-8")
    run_rows, policy_rows = module.collect_suite_roots([tmp_path / "suite_a", tmp_path / "suite_b"])
    assert len(run_rows) == 2
    assert not policy_rows


def test_claim_aggregate_counts_static_fallback_as_zero_margin(tmp_path):
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_dir = tmp_path / "suite" / "main_seed44"
    run_dir.mkdir(parents=True)
    (run_dir / "gate_summary.json").write_text(
        json.dumps(
            {
                "validation_selected_static_objective": 1.0,
                "teacher_reference_objective": 0.8,
                "best_deployable_objective": None,
                "teacher_beats_static": True,
                "gate_pass": False,
                "objective_metric": "task_composite",
                "best_deployable_policy": None,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text('{"seed": 44}', encoding="utf-8")

    run_rows, _ = module.collect_suite_roots([tmp_path / "suite"])
    assert len(run_rows) == 1
    assert run_rows[0]["deployable_selected"] is False
    assert run_rows[0]["deployable_margin"] == 0.0
    assert np.isnan(run_rows[0]["deployable_margin_selected"])


def test_claim_aggregate_collects_validation_selection_rows(tmp_path):
    module_path = ROOT / "v1" / "scripts" / "aggregate_claim_suite.py"
    spec = importlib.util.spec_from_file_location("aggregate_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_dir = tmp_path / "suite" / "main_seed41"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 41,
                "deployable_selection": {
                    "selected_policy": "forecast_aware_rollout_value",
                    "validation_rows": [
                        {
                            "policy": "forecast_aware_rollout_value",
                            "objective": 0.9,
                            "power_mean": 1.1,
                            "warmup_abort_count": 0,
                            "objective_margin_mean": 0.02,
                            "objective_margin_min": -0.001,
                            "negative_start_count": 1,
                            "static_margin_guard_pass": True,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = module.collect_validation_rows([tmp_path / "suite"])
    assert len(rows) == 1
    assert rows[0]["seed"] == 41
    assert rows[0]["policy"] == "forecast_aware_rollout_value"
    assert rows[0]["is_selected"]
    assert rows[0]["static_margin_guard_pass"]


def test_contextual_duty_guard_calibration_prefers_guarded_combo(monkeypatch):
    module_path = ROOT / "v1" / "scripts" / "run_protocol_gate.py"
    spec = importlib.util.spec_from_file_location("run_protocol_gate_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyPolicy:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "forecast_aware_contextual_duty")
            self.blend = float(kwargs.get("blend", 0.0))

    def fake_eval(args, **kwargs):
        del args
        policy = kwargs["policy"]
        start = int(kwargs["starts"][0])
        if str(policy.name) == "contextual_duty_calibration_static":
            objective = 1.0
        elif float(policy.blend) == 0.0:
            objective = 0.8 if start == 10 else 1.05
        else:
            objective = 0.98
        return {"power_mean": 1.0, "warmup_abort_count": 0}, float(objective)

    monkeypatch.setattr(module, "ForecastAwareContextualDutyPolicy", DummyPolicy)
    monkeypatch.setattr(module, "evaluate_validation_policy_metrics", fake_eval)
    monkeypatch.setattr(module, "action_support_from_labels", lambda *args, **kwargs: (0, 1))

    args = SimpleNamespace(
        contextual_duty_support_top_k=2,
        bc_action_support_min_count=0,
        contextual_duty_blend_grid=[0.0, 1.0],
        contextual_duty_deficit_grid=[0.5],
        contextual_duty_freshness_grid=[0.0],
        contextual_duty_power_grid=[0.0],
        contextual_duty_calibration_criterion="static_margin_guard",
        deployable_selection_min_mean_margin=0.0,
        deployable_selection_min_start_margin=-0.01,
        deployable_selection_max_negative_starts=1,
        bc_device="cpu",
        bc_preserve_warming=True,
    )
    result = module.calibrate_contextual_duty_policy(
        args,
        truth=pd.DataFrame(),
        sensors=[],
        constraints=None,
        cfg=None,
        oracle=None,
        model=object(),
        candidate_masks=np.asarray([[True], [False]], dtype=bool),
        forecast_cfg=ForecastContextConfig(horizon=1),
        labels=np.asarray([0, 1]),
        anchor_idx=0,
        anchor_mask=(True,),
        state_columns=(),
        starts=(10, 20),
    )
    _, blend, _, _, _, objective, row = result
    assert blend == 1.0
    assert objective == 0.98
    assert row is not None
    assert row["static_margin_guard_pass"]
    assert row["negative_start_count"] == 0


def test_contextual_duty_risk_calibration_prefers_positive_center_combo(monkeypatch):
    module_path = ROOT / "v1" / "scripts" / "run_protocol_gate.py"
    spec = importlib.util.spec_from_file_location("run_protocol_gate_contextual_risk_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyPolicy:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "forecast_aware_contextual_duty")
            self.blend = float(kwargs.get("blend", 0.0))

    def fake_eval(args, **kwargs):
        del args
        policy = kwargs["policy"]
        start = int(kwargs["starts"][0])
        if str(policy.name) == "contextual_duty_calibration_static":
            objective = 1.0
        elif float(policy.blend) == 0.0:
            objective = 0.70 if start == 10 else 1.02
        else:
            objective = 1.01
        return {"power_mean": 1.0, "warmup_abort_count": 0}, float(objective)

    monkeypatch.setattr(module, "ForecastAwareContextualDutyPolicy", DummyPolicy)
    monkeypatch.setattr(module, "evaluate_validation_policy_metrics", fake_eval)
    monkeypatch.setattr(module, "action_support_from_labels", lambda *args, **kwargs: (0, 1))

    args = SimpleNamespace(
        contextual_duty_support_top_k=2,
        bc_action_support_min_count=0,
        contextual_duty_blend_grid=[0.0, 1.0],
        contextual_duty_deficit_grid=[0.5],
        contextual_duty_freshness_grid=[0.0],
        contextual_duty_power_grid=[0.0],
        contextual_duty_calibration_criterion="static_margin_risk",
        deployable_selection_min_mean_margin=0.001,
        deployable_selection_min_start_margin=-0.01,
        deployable_selection_max_negative_starts=0,
        bc_device="cpu",
        bc_preserve_warming=True,
    )
    result = module.calibrate_contextual_duty_policy(
        args,
        truth=pd.DataFrame(),
        sensors=[],
        constraints=None,
        cfg=None,
        oracle=None,
        model=object(),
        candidate_masks=np.asarray([[True], [False]], dtype=bool),
        forecast_cfg=ForecastContextConfig(horizon=1),
        labels=np.asarray([0, 1]),
        anchor_idx=0,
        anchor_mask=(True,),
        state_columns=(),
        starts=(10, 20),
    )
    _, blend, _, _, _, objective, row = result
    assert blend == 0.0
    assert abs(objective - 0.86) < 1e-12
    assert row is not None
    assert not row["static_margin_guard_pass"]
    assert row["static_margin_positive_center"]


def test_final_deployable_selection_honors_positive_center(monkeypatch):
    module_path = ROOT / "v1" / "scripts" / "run_protocol_gate.py"
    spec = importlib.util.spec_from_file_location("run_protocol_gate_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyPolicy:
        def __init__(self, name):
            self.name = name

    def fake_eval(args, **kwargs):
        del args
        policy = kwargs["policy"]
        starts = tuple(int(x) for x in kwargs["starts"])
        if str(policy.name) == "validation_selected_static":
            objective = 1.0
        elif len(starts) > 1:
            objective = 0.8
        else:
            objective = 1.04 if starts[0] == 10 else 1.03
        return {"power_mean": 1.0, "warmup_abort_count": 0}, float(objective)

    monkeypatch.setattr(module, "evaluate_validation_policy_metrics", fake_eval)
    args = SimpleNamespace(
        deployable_selection_criterion="static_margin_risk",
        deployable_selection_min_mean_margin=0.001,
        deployable_selection_min_start_margin=-0.01,
        deployable_selection_max_negative_starts=1,
        deployable_selection_require_guard_pass=False,
        deployable_selection_require_positive_center=True,
    )
    fixed_static = DummyPolicy("validation_selected_static")
    candidate = DummyPolicy("forecast_aware_event_threshold")
    selected_policies, selected_name, rows = module.select_deployables_for_final(
        args,
        policies=[fixed_static, candidate],
        truth=pd.DataFrame(),
        sensors=[],
        constraints=None,
        cfg=None,
        oracle=None,
        state_columns=(),
        sensor_ids=(),
        starts=(10, 20),
    )
    assert selected_name is None
    assert [policy.name for policy in selected_policies] == ["validation_selected_static"]
    assert rows[0]["static_margin_positive_center"] is False


def test_recurrent_advantage_is_counted_as_deployable_policy():
    module_path = ROOT / "v1" / "scripts" / "run_protocol_gate.py"
    spec = importlib.util.spec_from_file_location("run_protocol_gate_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "forecast_aware_recurrent_advantage" in module.DEPLOYABLE_POLICY_NAMES


def test_event_threshold_only_preset_keeps_policy_set_clean():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_event_threshold_guarded_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--oracle-type") + 1 : command.index("--oracle-device")] == ["tcn"]
    assert "--bc-preserve-warming" in command
    assert "--include-event-threshold-policy" in command
    assert "--no-include-value-residual-policy" in command
    assert "--no-include-advantage-residual-policy" in command
    assert "--include-value-residual-policy" not in command


def test_event_threshold_valguard_preset_uses_guarded_threshold_calibration():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_valguard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_event_threshold_valguard_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_guard"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"
    assert "--include-event-threshold-policy" in command
    assert "--no-include-value-residual-policy" in command


def test_event_threshold_strict_valguard_requires_guard_pass():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_strict_valguard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_event_threshold_strict_valguard_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_guard"
    assert "--deployable-selection-require-guard-pass" in command
    assert "--include-event-threshold-policy" in command
    assert "--no-include-value-residual-policy" in command
    assert "--no-include-advantage-residual-policy" in command
    assert "--include-advantage-residual-policy" not in command


def test_event_threshold_riskcalib_uses_static_margin_risk():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_riskcalib_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_event_threshold_riskcalib_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--deployable-selection-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"
    assert "--include-event-threshold-policy" in command
    assert "--no-include-value-residual-policy" in command


def test_event_threshold_riskcenter_requires_positive_center():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_riskcenter_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_event_threshold_riskcenter_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--deployable-selection-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_risk"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-guard-pass" not in command
    assert "--include-event-threshold-policy" in command
    assert "--no-include-value-residual-policy" in command


def test_option_runtime_risk_guard_preset_is_pure_guarded_option_interface():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_runtime_risk_guard_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_option_runtime_risk_guard_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.8,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--learned-event-forecast" in command
    assert command[command.index("--deployable-selection-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-guard-pass" not in command
    assert "--include-runtime-risk-guard-policy" in command
    assert command[command.index("--runtime-risk-guard-calibration-criterion") + 1] == "static_margin_risk"
    assert "--no-include-option-planner-policy" in command
    assert "--include-option-planner-policy" not in command
    assert command[command.index("--option-planner-calibration-criterion") + 1] == "static_margin_risk"
    assert "--no-include-value-residual-policy" in command
    assert "--include-value-residual-policy" not in command
    assert "--no-include-event-threshold-policy" in command
    assert "--include-event-threshold-policy" not in command
    assert "--no-include-rollout-value-policy" in command
    assert "--include-rollout-value-policy" not in command


def test_option_runtime_risk_denseval_preset_adds_risk_band_and_small_grid():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_runtime_risk_denseval_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=42,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_option_runtime_risk_denseval_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.8,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--include-runtime-risk-guard-policy" in command
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command
    assert "--window-candidate-full-rollout-calibration" not in command

    full_command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_window_candidate_fullrollout_margin_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--include-window-candidate-policy" in full_command
    assert "--window-candidate-full-rollout-calibration" in full_command
    assert command[command.index("--deployable-selection-risk-min-q25-margin") + 1] == "0.0"
    assert command[command.index("--deployable-selection-risk-max-negative-starts") + 1] == "1"
    assert command[command.index("--runtime-risk-aggregation-grid") + 1] == "mean"
    min_soc_idx = command.index("--runtime-risk-min-soc-grid")
    assert command[min_soc_idx + 1 : min_soc_idx + 2] == ["0.0"]
    assert command[command.index("--static-selection-rollouts") + 1] == "12"
    assert "--no-include-option-planner-policy" in command
    assert "--include-option-planner-policy" not in command
    assert "--no-include-value-residual-policy" in command
    assert "--include-value-residual-policy" not in command


def test_cost_knn_riskband_preset_is_pure_cost_memory_interface():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_cost_knn_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=42,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_cost_knn_riskband_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.8,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--include-cost-knn-policy" in command
    assert "--cost-knn-calibration-criterion" in command
    assert command[command.index("--cost-knn-calibration-criterion") + 1] == "static_margin_risk"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command
    assert "--learned-event-forecast" in command
    assert "--no-include-bc-policy" in command
    assert "--no-include-cost-policy" in command
    assert "--no-include-event-threshold-policy" in command
    assert "--no-include-option-planner-policy" in command
    assert "--no-include-runtime-risk-guard-policy" in command
    assert "--no-include-recurrent-value-policy" in command
    assert "--include-value-residual-policy" not in command
    assert "--include-option-planner-policy" not in command


def test_macro_option_riskband_preset_is_pure_trajectory_interface():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_macro_option_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=44,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_macro_option_riskband_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.8,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--include-macro-option-policy" in command
    assert "--macro-option-calibration-criterion" in command
    assert command[command.index("--macro-option-calibration-criterion") + 1] == "static_margin_risk"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command
    assert "--learned-event-forecast" in command
    assert "--no-include-bc-policy" in command
    assert "--no-include-cost-policy" in command
    assert "--no-include-cost-knn-policy" in command
    assert "--no-include-option-planner-policy" in command
    assert "--no-include-runtime-risk-guard-policy" in command
    assert "--include-cost-knn-policy" not in command
    assert "--include-option-planner-policy" not in command
    assert "--include-runtime-risk-guard-policy" not in command


def test_contextual_duty_riskcenter_preset_uses_risk_selection_without_value_residual():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_contextual_riskcenter_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=48,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_hybrid_contextual_duty_riskcenter_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--deployable-selection-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--contextual-duty-calibration-criterion") + 1] == "static_margin_risk"
    assert "--deployable-selection-require-positive-center" in command
    assert "--include-event-threshold-policy" in command
    assert "--include-contextual-duty-policy" in command
    assert "--no-include-value-residual-policy" in command
    assert "--include-value-residual-policy" not in command
    assert "--no-include-teacher-rate-policy" in command


def test_recurrent_rank_costdagger_preset_enables_cost_dagger():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_recurrent_costdagger_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert "--include-recurrent-value-policy" in command
    assert command[command.index("--recurrent-value-rank-weight") + 1] == "0.5"
    assert command[command.index("--recurrent-value-cost-dagger-iters") + 1] == "1"
    assert command[command.index("--recurrent-value-cost-dagger-threshold") + 1] == "0.0"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"


def test_contextual_duty_riskband_preset_adds_transfer_risk_band():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_contextual_riskband_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=52,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_hybrid_contextual_duty_riskband_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cuda",
        bc_device="cuda",
        budget=1.2,
        startup_peak_budget=1.6,
        energy_capacity=180,
        harvest_per_step=0.92,
        train_steps=128,
        train_rollouts=4,
        static_selection_steps=256,
        static_selection_rollouts=12,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )
    assert command[command.index("--deployable-selection-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--deployable-selection-min-mean-margin") + 1] == "0.001"
    assert command[command.index("--event-threshold-calibration-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--contextual-duty-calibration-criterion") + 1] == "static_margin_risk"
    assert command[command.index("--deployable-selection-risk-min-q25-margin") + 1] == "-0.005"
    assert command[command.index("--deployable-selection-risk-max-negative-starts") + 1] == "4"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command
    assert "--include-event-threshold-policy" in command
    assert "--include-contextual-duty-policy" in command
    assert "--no-include-value-residual-policy" in command
    assert "--include-value-residual-policy" not in command


def test_twin_rollout_preset_uses_executed_step_digital_twin():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_twin_rollout_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_twin_rollout_posguard_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )

    def last_value(option: str) -> str:
        positions = [idx for idx, token in enumerate(command) if token == option]
        assert positions, option
        return command[positions[-1] + 1]

    assert "--learned-event-forecast" in command
    assert "--learned-continuous-forecast" in command
    assert "--include-rollout-value-policy" in command
    assert last_value("--rollout-value-cost-target") == "executed_step"
    assert last_value("--rollout-value-random-rollouts") == "1"
    assert last_value("--rollout-value-depth") == "3"
    assert last_value("--rollout-value-support-top-k") == "12"
    assert last_value("--deployable-selection-criterion") == "static_margin_risk"
    assert last_value("--deployable-selection-risk-min-q25-margin") == "0.0"
    assert last_value("--deployable-selection-risk-max-negative-starts") == "1"
    assert last_value("--dagger-iters") == "0"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command
    assert "--no-include-sequence-value-policy" in command
    assert "--no-include-value-residual-policy" in command


def test_window_candidate_margin_preset_uses_direct_window_verifier():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_window_candidate_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_window_candidate_margin_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )

    def last_value(option: str) -> str:
        positions = [idx for idx, token in enumerate(command) if token == option]
        assert positions, option
        return command[positions[-1] + 1]

    assert "--learned-event-forecast" in command
    assert "--learned-continuous-forecast" in command
    assert "--include-window-candidate-policy" in command
    assert "--include-rollout-value-policy" not in command
    assert "--include-sequence-value-policy" not in command
    assert "--no-include-rollout-value-policy" in command
    assert "--no-include-sequence-value-policy" in command
    assert last_value("--window-candidate-calibration-criterion") == "static_margin_risk"
    assert last_value("--window-candidate-support-top-k") == "16"
    assert last_value("--window-candidate-max-candidates") == "12"
    assert last_value("--deployable-selection-criterion") == "static_margin_risk"
    assert last_value("--deployable-selection-risk-min-q25-margin") == "0.0"
    assert last_value("--deployable-selection-risk-max-negative-starts") == "1"
    assert last_value("--dagger-iters") == "0"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command


def test_utility_planner_preset_uses_causal_forecast_rollout_interface():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_utility_planner_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_utility_planner_riskband_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )

    def last_value(option: str) -> str:
        positions = [idx for idx, token in enumerate(command) if token == option]
        assert positions, option
        return command[positions[-1] + 1]

    assert "--learned-event-forecast" in command
    assert "--learned-continuous-forecast" in command
    assert "--include-utility-planner-policy" in command
    assert "--no-include-window-candidate-policy" in command
    assert "--no-include-rollout-value-policy" in command
    assert "--no-include-bc-policy" in command
    assert last_value("--utility-planner-calibration-criterion") == "static_margin_risk"
    assert last_value("--utility-planner-support-top-k") == "16"
    assert last_value("--deployable-selection-criterion") == "static_margin_risk"
    assert last_value("--deployable-selection-risk-min-q25-margin") == "0.0"
    assert last_value("--deployable-selection-risk-max-negative-starts") == "1"
    assert last_value("--dagger-iters") == "0"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command


def test_proxy_mpc_preset_uses_multistep_static_aware_interface():
    module_path = ROOT / "v1" / "scripts" / "run_claim_suite.py"
    spec = importlib.util.spec_from_file_location("run_claim_suite_proxy_mpc_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.base_command(
        seed=41,
        truth_csv=Path("truth.csv"),
        oracle_path=Path("oracle.pt"),
        out_dir=Path("out"),
        preset="learned_proxy_mpc_riskband_safe",
        sensor_cfg="sensors.yaml",
        oracle_device="cpu",
        bc_device="cpu",
        budget=1.36,
        startup_peak_budget=1.75,
        energy_capacity=70,
        harvest_per_step=0.80,
        train_steps=128,
        train_rollouts=12,
        static_selection_steps=256,
        static_selection_rollouts=4,
        eval_steps=256,
        eval_rollouts=4,
        include_rule_baselines=False,
    )

    def last_value(option: str) -> str:
        positions = [idx for idx, token in enumerate(command) if token == option]
        assert positions, option
        return command[positions[-1] + 1]

    assert "--learned-event-forecast" in command
    assert "--learned-continuous-forecast" in command
    assert "--include-proxy-mpc-policy" in command
    assert "--no-include-utility-planner-policy" in command
    assert "--no-include-window-candidate-policy" in command
    assert "--no-include-bc-policy" in command
    assert last_value("--proxy-mpc-calibration-criterion") == "static_margin_risk"
    assert last_value("--proxy-mpc-support-top-k") == "16"
    assert last_value("--proxy-mpc-depth-grid") == "2"
    assert last_value("--deployable-selection-criterion") == "static_margin_risk"
    assert last_value("--deployable-selection-risk-min-q25-margin") == "0.0"
    assert last_value("--deployable-selection-risk-max-negative-starts") == "1"
    assert last_value("--dagger-iters") == "0"
    assert "--deployable-selection-require-positive-center" in command
    assert "--deployable-selection-require-risk-band" in command


def test_budget_matrix_assessment_enforces_win_rate_against_actual_n():
    module_path = ROOT / "v1" / "scripts" / "aggregate_budget_matrix.py"
    spec = importlib.util.spec_from_file_location("aggregate_budget_matrix_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = pd.DataFrame(
        {
            "budget": [1.2],
            "preset": ["main"],
            "n": [10],
            "deployable_wins": [7],
            "deployable_margin_mean": [0.1],
            "teacher_wins": [10],
        }
    )
    assessment = module.assess(summary, min_seeds=5, min_win_rate=0.8)
    assert assessment["required_wins_by_group"]["budget=1.20 preset=main"] == 8
    assert not assessment["matrix_pass"]
