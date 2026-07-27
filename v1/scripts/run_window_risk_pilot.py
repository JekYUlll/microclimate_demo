#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    continue_policy_rollout,
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
)
from forecast_cmdp.continuous_forecaster import (  # noqa: E402
    ContinuousForecasterTrainingConfig,
    augment_truth_with_continuous_forecasts,
    build_continuous_forecast_dataset,
    select_continuous_forecast_columns,
    train_continuous_forecaster,
)
from forecast_cmdp.event_forecaster import (  # noqa: E402
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    select_event_forecast_columns,
    train_event_forecaster,
)
from forecast_cmdp.features import ForecastContextConfig  # noqa: E402
from forecast_cmdp.mean_risk_policy import (  # noqa: E402
    build_residual_risk_feature,
    build_window_risk_feature,
    make_proxy_mpc_controller,
    residual_action_controller_specs,
    valid_anchor_residual_action,
)
from forecast_cmdp.mpc_teacher import restore_env, snapshot_env  # noqa: E402
from forecast_cmdp.protocol import final_objective, task_focus_metrics  # noqa: E402
from forecast_cmdp.window_risk import (  # noqa: E402
    ControllerSpec,
    WindowOutcome,
    assign_balanced_anchors,
    collect_paired_window_risk_dataset,
    refresh_window_risk_features,
    select_train_anchor_bank,
    split_train_risk_starts,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.rollout import RolloutResult, run_policy_rollout  # noqa: E402


CORE_FORECAST_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
)
CORE_FORECAST_SENSOR_IDS = ("met_station_core",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect split-compliant paired full-window outcomes for the Branch H risk controller."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-steps", type=int, default=256)
    parser.add_argument(
        "--prefix-steps",
        type=int,
        default=None,
        help=(
            "Anchor-conditioning steps before each residual label. Defaults "
            "to window-steps for residual_action and zero otherwise."
        ),
    )
    parser.add_argument("--fit-count", type=int, default=32)
    parser.add_argument("--calibration-count", type=int, default=12)
    parser.add_argument("--anchor-top-k", type=int, default=8)
    parser.add_argument("--anchors-per-start", type=int, default=3)
    parser.add_argument(
        "--anchor-action-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit train-only anchor bank. When provided, bypasses static "
            "ranking and assigns these anchors to every selected start."
        ),
    )
    parser.add_argument("--controller-limit", type=int, default=16)
    parser.add_argument(
        "--controller-family",
        choices=[
            "balanced",
            "anchor_guard",
            "anchor_neighborhood",
            "residual_action",
        ],
        default="balanced",
    )
    parser.add_argument("--selection", choices=["source", "event_rich", "event_transport_rich", "uniform"], default="source")
    parser.add_argument("--selection-stride", type=int, default=None)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument(
        "--forecast-training-scope",
        choices=["oracle_pretrain", "source"],
        default="oracle_pretrain",
        help="Formal Branch H uses oracle_pretrain. source is only for compatibility/engineering smoke.",
    )
    parser.add_argument(
        "--forecast-input-mode",
        choices=["core_exogenous", "source"],
        default="core_exogenous",
        help="Formal Branch H restricts forecast inputs to met_station_core variables.",
    )
    parser.add_argument(
        "--reuse-source-static",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the source run's rl_train-only static table. This is formal "
            "only when the residual anchor bank includes every eligible action; "
            "otherwise recompute on risk_fit."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh-features-only",
        action="store_true",
        help="Rebuild causal features for saved outcomes without rerunning any rollout.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def source_truth_path(source_run: Path, manifest: Mapping[str, object]) -> Path:
    for name in (
        "truth_with_learned_continuous_forecast.csv",
        "truth_with_learned_event_forecast.csv",
    ):
        candidate = source_run / name
        if candidate.exists():
            return candidate
    return resolve_project_path(str(manifest["truth_csv"]))


def resolve_anchor_bank(
    static_table: pd.DataFrame,
    *,
    top_k: int,
    explicit_action_indices: Sequence[int] | None,
    eligible_action_indices: set[int],
) -> tuple[int, ...]:
    if explicit_action_indices is None:
        return select_train_anchor_bank(static_table, top_k=int(top_k))
    requested = tuple(
        dict.fromkeys(int(value) for value in explicit_action_indices)
    )
    if not requested:
        raise ValueError("Explicit anchor bank must not be empty")
    available = set(static_table["action_idx"].astype(int).tolist())
    invalid = [
        action_idx
        for action_idx in requested
        if action_idx not in eligible_action_indices or action_idx not in available
    ]
    if invalid:
        raise ValueError(
            f"Explicit anchors are not eligible static actions: {invalid}"
        )
    return requested


def prepare_window_risk_truth(
    *,
    source_run: Path,
    manifest: Mapping[str, object],
    run_args: Mapping[str, object],
    state_columns: tuple[str, ...],
    out_dir: Path,
    training_scope: str,
    input_mode: str,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    if str(training_scope) == "source":
        path = source_truth_path(source_run, manifest)
        return pd.read_csv(path), path, {
            "training_scope": "source",
            "input_mode": str(input_mode),
            "source_truth_csv": str(path),
        }
    cache_path = out_dir / f"truth_with_oracle_pretrain_{input_mode}_forecasts.csv"
    summary_path = out_dir / f"oracle_pretrain_{input_mode}_forecast_summary.json"
    if cache_path.exists() and summary_path.exists():
        return (
            pd.read_csv(cache_path),
            cache_path,
            json.loads(summary_path.read_text(encoding="utf-8")),
        )
    truth_path = resolve_project_path(str(manifest["truth_csv"]))
    truth = pd.read_csv(truth_path)
    pretrain_bounds = tuple(int(x) for x in manifest["bounds"]["oracle_pretrain"])
    horizon = int(run_args["horizon"])
    period_steps = max(1, int(round(86400.0 / max(float(run_args["freq_s"]), 1.0))))
    learned_event_columns: tuple[str, ...] = ()
    event_summary: dict[str, object] | None = None
    if bool(run_args.get("learned_event_forecast", False)):
        if str(input_mode) == "core_exogenous":
            preferred = tuple(column for column in CORE_FORECAST_COLUMNS if column in truth.columns)
        else:
            preferred = tuple(str(x) for x in run_args.get("event_forecast_feature_columns", ()))
        event_features = select_event_forecast_columns(
            truth,
            preferred_columns=preferred or state_columns,
            event_column=str(run_args["event_column"]),
        )
        event_cfg = EventForecasterTrainingConfig(
            horizon=horizon,
            lookback=int(run_args["event_forecast_lookback"]),
            event_column=str(run_args["event_column"]),
            hidden_dim=int(run_args["event_forecast_hidden_dim"]),
            epochs=int(run_args["event_forecast_epochs"]),
            batch_size=int(run_args["event_forecast_batch_size"]),
            learning_rate=float(run_args["event_forecast_learning_rate"]),
            weight_decay=float(run_args["event_forecast_weight_decay"]),
            seed=int(manifest["seed"]),
            device="cpu",
            probability_prefix=str(run_args["event_forecast_probability_prefix"]),
            period_steps=period_steps,
        )
        event_dataset = build_event_forecast_dataset(
            truth,
            bounds=pretrain_bounds,
            feature_columns=event_features,
            event_column=str(run_args["event_column"]),
            cfg=event_cfg,
        )
        event_model = train_event_forecaster(event_dataset, event_cfg)
        truth, learned_event_columns = augment_truth_with_event_forecasts(truth, event_model)
        event_summary = {
            "feature_columns": list(event_features),
            "prediction_columns": list(learned_event_columns),
            "history": event_model.history,
        }
    continuous_summary: dict[str, object] | None = None
    if bool(run_args.get("learned_continuous_forecast", False)):
        if str(input_mode) == "core_exogenous":
            targets = tuple(str(x) for x in run_args.get("forecast_continuous_columns", ()))
        else:
            targets = tuple(str(x) for x in run_args.get("continuous_forecast_target_columns", ()))
        if not targets:
            targets = tuple(str(x) for x in run_args.get("forecast_continuous_columns", ()))
        if str(input_mode) == "core_exogenous":
            preferred = tuple(
                [
                    *(column for column in CORE_FORECAST_COLUMNS if column in truth.columns),
                    *learned_event_columns,
                ]
            )
        else:
            preferred = tuple(str(x) for x in run_args.get("continuous_forecast_feature_columns", ()))
        continuous_features = select_continuous_forecast_columns(
            truth,
            preferred_columns=preferred or tuple([*state_columns, *learned_event_columns]),
        )
        continuous_cfg = ContinuousForecasterTrainingConfig(
            horizon=horizon,
            lookback=int(run_args["continuous_forecast_lookback"]),
            target_columns=targets,
            hidden_dim=int(run_args["continuous_forecast_hidden_dim"]),
            epochs=int(run_args["continuous_forecast_epochs"]),
            batch_size=int(run_args["continuous_forecast_batch_size"]),
            learning_rate=float(run_args["continuous_forecast_learning_rate"]),
            weight_decay=float(run_args["continuous_forecast_weight_decay"]),
            seed=int(manifest["seed"]),
            device="cpu",
            prediction_prefix=str(run_args["continuous_forecast_prediction_prefix"]),
            period_steps=period_steps,
        )
        continuous_dataset = build_continuous_forecast_dataset(
            truth,
            bounds=pretrain_bounds,
            feature_columns=continuous_features,
            target_columns=targets,
            cfg=continuous_cfg,
        )
        continuous_model = train_continuous_forecaster(continuous_dataset, continuous_cfg)
        truth, learned_continuous_columns = augment_truth_with_continuous_forecasts(
            truth,
            continuous_model,
        )
        continuous_summary = {
            "feature_columns": list(continuous_features),
            "target_columns": list(targets),
            "prediction_columns": list(learned_continuous_columns),
            "history": continuous_model.history,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(cache_path, index=False)
    summary = {
        "training_scope": "oracle_pretrain",
        "input_mode": str(input_mode),
        "training_bounds": list(pretrain_bounds),
        "source_truth_csv": str(truth_path),
        "prepared_truth_csv": str(cache_path),
        "event_forecaster": event_summary,
        "continuous_forecaster": continuous_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return truth, cache_path, summary


def balanced_proxy_controller_specs(limit: int = 16) -> tuple[ControllerSpec, ...]:
    specs: list[ControllerSpec] = []
    combo_idx = 0
    for event_idx, event_weight in enumerate((0.5, 1.5)):
        for target_idx, target_rate_weight in enumerate((0.0, 0.5)):
            for dwell_idx, min_dwell in enumerate((1, 2)):
                for depth_idx, planning_depth in enumerate((2, 3)):
                    for age_idx, age_weight in enumerate((0.25, 0.75)):
                        if (event_idx + target_idx + dwell_idx + depth_idx + age_idx) % 2:
                            continue
                        parameters: dict[str, object] = {
                            "event_weight": float(event_weight),
                            "magnitude_weight": 1.0,
                            "variability_weight": 0.5,
                            "freshness_weight": 0.25,
                            "target_rate_weight": float(target_rate_weight),
                            "anchor_bias": 0.0,
                            "power_weight": 0.03,
                            "switch_weight": 0.03,
                            "min_soc": 0.0,
                            "min_dwell": int(min_dwell),
                            "aggregation": "max",
                            "planning_depth": int(planning_depth),
                            "beam_width": 4,
                            "max_branch": 8,
                            "age_weight": float(age_weight),
                            "anchor_improvement_threshold": 0.0,
                            "max_anchor_hamming": -1,
                        }
                        specs.append(ControllerSpec(controller_id=f"proxy_balanced_{combo_idx:03d}", parameters=parameters))
                        combo_idx += 1
    requested = max(1, int(limit))
    if requested >= len(specs):
        return tuple(specs)
    positions = np.linspace(0, len(specs) - 1, num=requested)
    selected = [specs[int(round(position))] for position in positions]
    unique: dict[str, ControllerSpec] = {}
    for spec in selected:
        unique.setdefault(spec.controller_id, spec)
    return tuple(unique.values())


def anchor_guard_controller_specs(limit: int = 16) -> tuple[ControllerSpec, ...]:
    balanced = {
        controller.controller_id: controller
        for controller in balanced_proxy_controller_specs(16)
    }
    base_ids = (
        "proxy_balanced_005",
        "proxy_balanced_004",
        "proxy_balanced_007",
        "proxy_balanced_003",
    )
    specs: list[ControllerSpec] = []
    for base_id in base_ids:
        if base_id not in balanced:
            raise ValueError(f"Missing locked anchor-guard base controller: {base_id}")
        for threshold in (0.0, 0.01, 0.02, 0.04):
            parameters = dict(balanced[base_id].parameters)
            parameters["anchor_improvement_threshold"] = float(threshold)
            specs.append(
                ControllerSpec(
                    controller_id=(
                        f"proxy_guard_{base_id.rsplit('_', 1)[-1]}"
                        f"_t{int(round(threshold * 1000)):03d}"
                    ),
                    parameters=parameters,
                )
            )
    return tuple(specs[: max(1, int(limit))])


def anchor_neighborhood_controller_specs(limit: int = 16) -> tuple[ControllerSpec, ...]:
    balanced = {
        controller.controller_id: controller
        for controller in balanced_proxy_controller_specs(16)
    }
    base_ids = (
        "proxy_balanced_005",
        "proxy_balanced_004",
        "proxy_balanced_007",
        "proxy_balanced_003",
    )
    specs: list[ControllerSpec] = []
    for base_id in base_ids:
        if base_id not in balanced:
            raise ValueError(f"Missing locked anchor-neighborhood base controller: {base_id}")
        for distance in (1, 2, 3, 4):
            parameters = dict(balanced[base_id].parameters)
            parameters["anchor_improvement_threshold"] = 0.0
            parameters["max_anchor_hamming"] = int(distance)
            specs.append(
                ControllerSpec(
                    controller_id=(
                        f"proxy_neighbor_{base_id.rsplit('_', 1)[-1]}"
                        f"_h{int(distance)}"
                    ),
                    parameters=parameters,
                )
            )
    return tuple(specs[: max(1, int(limit))])


def action_support_from_teacher(
    labels: np.ndarray,
    *,
    n_actions: int,
    top_k: int,
    anchor_bank: Sequence[int],
    candidate_masks: np.ndarray,
    required_sensor_indices: Sequence[int] = (),
) -> tuple[int, ...]:
    values = np.asarray(labels, dtype=int).reshape(-1)
    values = values[(values >= 0) & (values < int(n_actions))]
    required = tuple(int(x) for x in required_sensor_indices)
    if required:
        masks = np.asarray(candidate_masks, dtype=bool)
        eligible = np.all(masks[:, required], axis=1)
        values = values[eligible[values]]
    counts = np.bincount(values, minlength=int(n_actions))
    ranked = sorted(np.flatnonzero(counts > 0).tolist(), key=lambda idx: (-int(counts[idx]), int(idx)))
    selected = {int(idx) for idx in ranked[: max(1, int(top_k))]}
    selected.update(int(idx) for idx in anchor_bank)
    if required:
        masks = np.asarray(candidate_masks, dtype=bool)
        selected = {idx for idx in selected if bool(np.all(masks[int(idx), required]))}
    if not selected:
        raise ValueError("No teacher-supported action satisfies required forecast sensors")
    return tuple(sorted(selected))


def teacher_rows_within_windows(
    labels: np.ndarray,
    step_indices: np.ndarray,
    *,
    starts: Sequence[int],
    window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    if values.shape != steps.shape:
        raise ValueError("Teacher labels and step indices must have matching shapes")
    selected = np.zeros(steps.shape, dtype=bool)
    for start in starts:
        begin = int(start)
        selected |= (steps >= begin) & (steps < begin + int(window_steps))
    if not np.any(selected):
        raise ValueError("No teacher rows fall inside risk_fit windows")
    return values[selected], steps[selected]


def make_proxy_policy(
    *,
    controller: ControllerSpec,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    anchor_mask: Sequence[bool],
    support: Sequence[int],
    target_rates: np.ndarray,
    preserve_warming: bool,
) -> object:
    return make_proxy_mpc_controller(
        controller=controller,
        candidate_masks=candidate_masks,
        forecast_cfg=forecast_cfg,
        anchor_mask=anchor_mask,
        support=support,
        target_rates=target_rates,
        preserve_warming=preserve_warming,
    )


def evaluate_window(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    policy: object,
    start: int,
    steps: int,
    seed_offset: int,
    state_columns: tuple[str, ...],
    objective_mode: str,
    task_error_columns: tuple[str, ...],
    task_error_scales: tuple[float, ...],
    task_error_event_only: bool,
    task_error_weight: float,
    budget: float,
    startup_peak_budget: float,
) -> WindowOutcome:
    env_cfg = replace(cfg, episode_len=int(steps), seed=int(cfg.seed) + int(seed_offset))
    env = WarmupSchedulingEnv(truth, sensors, constraints, env_cfg, oracle=oracle)
    result = run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start))
    return window_outcome_from_rollout(
        result,
        state_columns=state_columns,
        objective_mode=objective_mode,
        task_error_columns=task_error_columns,
        task_error_scales=task_error_scales,
        task_error_event_only=task_error_event_only,
        task_error_weight=task_error_weight,
        budget=budget,
        startup_peak_budget=startup_peak_budget,
    )


def window_outcome_from_rollout(
    result: RolloutResult,
    *,
    state_columns: tuple[str, ...],
    objective_mode: str,
    task_error_columns: tuple[str, ...],
    task_error_scales: tuple[float, ...],
    task_error_event_only: bool,
    task_error_weight: float,
    budget: float,
    startup_peak_budget: float,
) -> WindowOutcome:
    finite_oracle = result.oracle_losses[np.isfinite(result.oracle_losses)]
    metrics: dict[str, object] = {
        "oracle_loss_mean": float(np.mean(finite_oracle)) if finite_oracle.size else float("nan"),
        "instant_mae": float(np.mean(np.abs(result.observations - result.truth))),
    }
    metrics.update(
        task_focus_metrics(
            result,
            state_columns=state_columns,
            task_error_columns=task_error_columns,
            task_error_scales=task_error_scales,
            event_only=bool(task_error_event_only),
        )
    )
    violation_count = int(np.sum(result.powers > float(budget) + 1.0e-9))
    violation_count += int(np.sum(result.peaks > float(startup_peak_budget) + 1.0e-9))
    return WindowOutcome(
        objective=float(final_objective(metrics, mode=str(objective_mode), task_error_weight=float(task_error_weight))),
        power_mean=float(np.mean(result.powers)) if result.powers.size else float("nan"),
        warmup_abort_count=int(result.warmup_abort_count),
        constraint_violation_count=int(violation_count),
    )


def collect_fit_static_table(
    *,
    out_dir: Path,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    candidate_masks: np.ndarray,
    starts: Sequence[int],
    window_steps: int,
    state_columns: tuple[str, ...],
    run_args: Mapping[str, object],
    eligible_action_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "risk_fit_static_candidates.jsonl"
    rows: list[dict[str, object]] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
    completed = {int(row["action_idx"]) for row in rows}
    eligible = (
        {int(x) for x in eligible_action_indices}
        if eligible_action_indices is not None
        else None
    )
    for action_idx, mask in enumerate(np.asarray(candidate_masks, dtype=bool)):
        if eligible is not None and int(action_idx) not in eligible:
            continue
        if int(action_idx) in completed:
            continue
        outcomes = []
        for start_pos, start in enumerate(starts):
            outcomes.append(
                evaluate_window(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=cfg,
                    oracle=oracle,
                    policy=StaticMaskPolicy(tuple(bool(x) for x in mask), name=f"risk_fit_static_{action_idx}"),
                    start=int(start),
                    steps=int(window_steps),
                    seed_offset=150_000 + int(action_idx) * 1_000 + int(start_pos),
                    state_columns=state_columns,
                    objective_mode=str(run_args["objective_mode"]),
                    task_error_columns=tuple(str(x) for x in run_args["task_error_columns"]),
                    task_error_scales=tuple(float(x) for x in run_args["task_error_scales"]),
                    task_error_event_only=bool(run_args["task_error_event_only"]),
                    task_error_weight=float(run_args["task_error_weight"]),
                    budget=float(run_args["budget"]),
                    startup_peak_budget=float(run_args["startup_peak_budget"]),
                )
            )
        row = {
            "action_idx": int(action_idx),
            "objective_loss_mean": float(np.mean([outcome.objective for outcome in outcomes])),
            "power_mean": float(np.mean([outcome.power_mean for outcome in outcomes])),
            "warmup_abort_count": int(sum(outcome.warmup_abort_count for outcome in outcomes)),
            "constraint_violation_count": int(sum(outcome.constraint_violation_count for outcome in outcomes)),
            "sensor_ids": "|".join(
                str(sensors[idx].sensor_id) for idx in np.flatnonzero(np.asarray(mask, dtype=bool))
            ),
        }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
        rows.append(row)
        print(
            f"[window_risk] static action={action_idx}/{len(candidate_masks) - 1} "
            f"objective={row['objective_loss_mean']:.6f}",
            flush=True,
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(["objective_loss_mean", "power_mean", "warmup_abort_count", "action_idx"])
        .reset_index(drop=True)
    )
    table.to_csv(out_dir / "risk_fit_static_candidates.csv", index=False)
    return table


def build_feature_vector(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    forecast_cfg: ForecastContextConfig,
    candidate_masks: np.ndarray,
    anchor_idx: int,
    controller: ControllerSpec,
    support: Sequence[int],
    target_rates: np.ndarray,
    preserve_warming: bool,
    start: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    env = WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
    env.reset(start_idx=int(start))
    return build_window_risk_feature(
        env=env,
        forecast_cfg=forecast_cfg,
        candidate_masks=candidate_masks,
        anchor_idx=int(anchor_idx),
        controller=controller,
        support=support,
        target_rates=target_rates,
        preserve_warming=preserve_warming,
    )


def main() -> None:
    args = parse_args()
    prefix_steps = (
        int(args.window_steps)
        if args.prefix_steps is None
        and str(args.controller_family) == "residual_action"
        else int(args.prefix_steps or 0)
    )
    if prefix_steps < 0:
        raise ValueError("--prefix-steps must be non-negative")
    sample_span_steps = int(prefix_steps) + int(args.window_steps)
    source_run = resolve_project_path(args.source_run)
    out_dir = resolve_project_path(args.out_dir)
    manifest_path = source_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_args = dict(manifest["run_args"])
    helpers = load_v2_helpers()
    state_columns = tuple(str(x) for x in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(x) for x in helpers.REWARD_TARGET_COLUMNS)
    truth, truth_path, forecast_preparation = prepare_window_risk_truth(
        source_run=source_run,
        manifest=manifest,
        run_args=run_args,
        state_columns=state_columns,
        out_dir=out_dir,
        training_scope=str(args.forecast_training_scope),
        input_mode=str(args.forecast_input_mode),
    )
    sensors = load_archived_sensor_specs(resolve_project_path(str(manifest["sensor_cfg"])))
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    normalization_bounds = tuple(int(x) for x in manifest["normalization_bounds"])
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=normalization_bounds[0],
        end_idx=normalization_bounds[1],
    )
    cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(run_args["lookback"]),
        episode_len=int(sample_span_steps),
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
        resolve_project_path(str(manifest["oracle_path"])),
        oracle_type=str(manifest["oracle_type"]),
        device=str(args.oracle_device),
    )
    if str(manifest.get("oracle_target_weight_mode", "checkpoint")) != "checkpoint":
        raise ValueError("Window-risk pilot currently requires a checkpoint-weighted source oracle")
    forecast_cfg_data = dict(manifest["forecast_cfg"])
    if str(args.forecast_input_mode) == "core_exogenous":
        forecast_cfg_data["continuous_current_source"] = "learned_h1"
    forecast_cfg = ForecastContextConfig(**forecast_cfg_data)
    teacher_path = resolve_project_path(str(manifest["teacher_dataset"]))
    with np.load(teacher_path, allow_pickle=False) as teacher:
        labels = np.asarray(teacher["labels"], dtype=np.int64)
        teacher_step_indices = np.asarray(teacher["step_indices"], dtype=np.int64)
        candidate_masks = np.asarray(teacher["candidate_masks"], dtype=bool)
    if candidate_masks.shape[1] != len(sensors):
        raise ValueError("Teacher candidate masks do not match source sensors")

    selection = str(run_args["selection"]) if str(args.selection) == "source" else str(args.selection)
    stride = int(run_args["selection_stride"]) if args.selection_stride is None else int(args.selection_stride)
    split = split_train_risk_starts(
        truth,
        bounds=tuple(int(x) for x in manifest["bounds"]["rl_train"]),
        window_steps=int(sample_span_steps),
        horizon=int(run_args["horizon"]),
        fit_count=int(args.fit_count),
        calibration_count=int(args.calibration_count),
        selection=selection,
        stride=stride,
        event_column=str(run_args["event_column"]),
        seed=int(manifest["seed"]) + 701,
    )
    fit_teacher_labels, fit_teacher_steps = teacher_rows_within_windows(
        labels,
        teacher_step_indices,
        starts=split.fit,
        window_steps=int(sample_span_steps),
    )
    if str(args.controller_family) == "anchor_guard":
        controllers = anchor_guard_controller_specs(int(args.controller_limit))
    elif str(args.controller_family) == "anchor_neighborhood":
        controllers = anchor_neighborhood_controller_specs(int(args.controller_limit))
    elif str(args.controller_family) == "residual_action":
        controllers = residual_action_controller_specs(
            range(int(candidate_masks.shape[0]))
        )
    else:
        controllers = balanced_proxy_controller_specs(int(args.controller_limit))
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_summary = {
        "source_run": str(source_run),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "truth_csv": str(truth_path),
        "forecast_preparation": forecast_preparation,
        "forecast_training_scope": str(args.forecast_training_scope),
        "forecast_input_mode": str(args.forecast_input_mode),
        "effective_forecast_cfg": asdict(forecast_cfg),
        "seed": int(manifest["seed"]),
        "window_steps": int(args.window_steps),
        "prefix_steps": int(prefix_steps),
        "sample_span_steps": int(sample_span_steps),
        "fit_starts": list(split.fit),
        "calibration_starts": list(split.calibration),
        "start_diagnostics": split.diagnostics,
        "controller_ids": [controller.controller_id for controller in controllers],
        "controller_configs": [controller.parameters for controller in controllers],
        "controller_family": str(args.controller_family),
        "teacher_support_scope": "risk_fit_windows_only",
        "teacher_support_rows": int(fit_teacher_labels.size),
        "teacher_support_step_min": int(np.min(fit_teacher_steps)),
        "teacher_support_step_max": int(np.max(fit_teacher_steps)),
        "reuse_source_static": bool(args.reuse_source_static),
        "static_table_scope": (
            "source_rl_train"
            if bool(args.reuse_source_static)
            else "risk_fit_only"
        ),
        "common_random_numbers": bool(cfg.common_random_numbers),
        "residual_label_semantics": (
            "shared_anchor_prefix_snapshot_then_counterfactual_continuation"
            if str(args.controller_family) == "residual_action"
            else "full_window_controller_rollout"
        ),
        "anchor_assignment": (
            "uniform_rotation_over_train_anchor_bank"
            if str(args.controller_family) == "residual_action"
            else "best_anchor_plus_rotation"
        ),
    }
    (out_dir / "window_risk_protocol.json").write_text(
        json.dumps(protocol_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(protocol_summary, indent=2, sort_keys=True))
        return

    required_sensor_indices = tuple(
        idx
        for idx, sensor in enumerate(sensors)
        if str(sensor.sensor_id) in set(CORE_FORECAST_SENSOR_IDS)
    ) if str(args.forecast_input_mode) == "core_exogenous" else ()
    if str(args.forecast_input_mode) == "core_exogenous" and not required_sensor_indices:
        raise ValueError("core_exogenous forecast mode requires met_station_core")
    eligible_actions = (
        {
            int(action_idx)
            for action_idx, mask in enumerate(candidate_masks)
            if bool(np.all(np.asarray(mask, dtype=bool)[list(required_sensor_indices)]))
        }
        if required_sensor_indices
        else set(range(int(candidate_masks.shape[0])))
    )
    if bool(args.reuse_source_static):
        static_table = pd.read_csv(source_run / "train_static_candidates.csv")
    else:
        static_table = collect_fit_static_table(
            out_dir=out_dir,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            starts=split.fit,
            window_steps=int(args.window_steps),
            state_columns=state_columns,
            run_args=run_args,
            eligible_action_indices=tuple(sorted(eligible_actions)),
        )
    eligible_static_table = static_table
    if required_sensor_indices:
        eligible_static_table = static_table[
            static_table["action_idx"].astype(int).isin(eligible_actions)
        ].reset_index(drop=True)
    anchor_bank = resolve_anchor_bank(
        eligible_static_table,
        top_k=int(args.anchor_top_k),
        explicit_action_indices=args.anchor_action_indices,
        eligible_action_indices=eligible_actions,
    )
    anchor_masks = {int(idx): tuple(bool(x) for x in candidate_masks[int(idx)]) for idx in anchor_bank}
    if str(args.controller_family) == "residual_action":
        support = tuple(sorted(int(x) for x in eligible_actions))
        action_support_scope = "all_projector_feasible_required_sensor_actions"
    else:
        support = action_support_from_teacher(
            fit_teacher_labels,
            n_actions=int(candidate_masks.shape[0]),
            top_k=int(run_args["proxy_mpc_support_top_k"]),
            anchor_bank=anchor_bank,
            candidate_masks=candidate_masks,
            required_sensor_indices=required_sensor_indices,
        )
        action_support_scope = "risk_fit_teacher_top_k_plus_anchor_bank"
    valid_labels = fit_teacher_labels[
        (fit_teacher_labels >= 0) & (fit_teacher_labels < candidate_masks.shape[0])
    ]
    target_rates = (
        np.mean(candidate_masks[valid_labels].astype(float), axis=0)
        if valid_labels.size
        else np.asarray(candidate_masks[anchor_bank[0]], dtype=float)
    )
    if required_sensor_indices:
        target_rates[np.asarray(required_sensor_indices, dtype=int)] = 1.0
    preserve_warming = bool(run_args["bc_preserve_warming"])
    protocol_summary.update(
        {
            "anchor_bank": [int(x) for x in anchor_bank],
            "anchor_bank_selection": (
                "explicit_train_only"
                if args.anchor_action_indices is not None
                else "static_objective_rank"
            ),
            "support": [int(x) for x in support],
            "action_support_scope": action_support_scope,
            "target_rates": [float(x) for x in target_rates],
            "required_forecast_sensor_indices": [int(x) for x in required_sensor_indices],
            "required_forecast_sensor_ids": [
                str(sensors[idx].sensor_id) for idx in required_sensor_indices
            ],
        }
    )
    protocol_summary["feature_version"] = (
        "prefix_conditioned_residual_v1"
        if str(args.controller_family) == "residual_action"
        else "causal_history_v1"
    )
    boundary_cache: dict[
        tuple[int, int, int],
        tuple[WarmupSchedulingEnv, dict[str, object]],
    ] = {}

    def residual_boundary(
        start: int,
        anchor_idx: int,
        seed_offset: int,
    ) -> tuple[WarmupSchedulingEnv, dict[str, object]]:
        key = (int(start), int(anchor_idx), int(seed_offset))
        if key not in boundary_cache:
            env_cfg = replace(
                cfg,
                episode_len=int(sample_span_steps),
                seed=int(cfg.seed) + int(seed_offset),
            )
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                env_cfg,
                oracle=oracle,
            )
            env.reset(start_idx=int(start))
            if int(prefix_steps) > 0:
                continue_policy_rollout(
                    env,
                    StaticMaskPolicy(
                        anchor_masks[int(anchor_idx)],
                        name=f"risk_prefix_anchor_{anchor_idx}",
                    ),
                    steps=int(prefix_steps),
                )
            boundary_cache[key] = (env, snapshot_env(env))
        return boundary_cache[key]

    def static_evaluator(start: int, anchor_idx: int, seed_offset: int) -> WindowOutcome:
        if str(args.controller_family) == "residual_action":
            env, boundary = residual_boundary(start, anchor_idx, seed_offset)
            restore_env(env, boundary)
            result = continue_policy_rollout(
                env,
                StaticMaskPolicy(
                    anchor_masks[int(anchor_idx)],
                    name=f"risk_static_{anchor_idx}",
                ),
                steps=int(args.window_steps),
            )
            outcome = window_outcome_from_rollout(
                result,
                state_columns=state_columns,
                objective_mode=str(run_args["objective_mode"]),
                task_error_columns=tuple(str(x) for x in run_args["task_error_columns"]),
                task_error_scales=tuple(float(x) for x in run_args["task_error_scales"]),
                task_error_event_only=bool(run_args["task_error_event_only"]),
                task_error_weight=float(run_args["task_error_weight"]),
                budget=float(run_args["budget"]),
                startup_peak_budget=float(run_args["startup_peak_budget"]),
            )
        else:
            outcome = evaluate_window(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=StaticMaskPolicy(anchor_masks[int(anchor_idx)], name=f"risk_static_{anchor_idx}"),
                start=start,
                steps=int(args.window_steps),
                seed_offset=seed_offset,
                state_columns=state_columns,
                objective_mode=str(run_args["objective_mode"]),
                task_error_columns=tuple(str(x) for x in run_args["task_error_columns"]),
                task_error_scales=tuple(float(x) for x in run_args["task_error_scales"]),
                task_error_event_only=bool(run_args["task_error_event_only"]),
                task_error_weight=float(run_args["task_error_weight"]),
                budget=float(run_args["budget"]),
                startup_peak_budget=float(run_args["startup_peak_budget"]),
            )
        print(f"[window_risk] static start={start} anchor={anchor_idx} objective={outcome.objective:.6f}", flush=True)
        return outcome

    def candidate_evaluator(
        start: int,
        anchor_idx: int,
        controller: ControllerSpec,
        seed_offset: int,
    ) -> WindowOutcome:
        if str(args.controller_family) == "residual_action":
            target_idx = int(controller.parameters["target_action_idx"])
            env, boundary = residual_boundary(start, anchor_idx, seed_offset)
            restore_env(env, boundary)
            result = continue_policy_rollout(
                env,
                StaticMaskPolicy(
                    tuple(bool(x) for x in candidate_masks[target_idx]),
                    name=str(controller.controller_id),
                ),
                steps=int(args.window_steps),
            )
            outcome = window_outcome_from_rollout(
                result,
                state_columns=state_columns,
                objective_mode=str(run_args["objective_mode"]),
                task_error_columns=tuple(str(x) for x in run_args["task_error_columns"]),
                task_error_scales=tuple(float(x) for x in run_args["task_error_scales"]),
                task_error_event_only=bool(run_args["task_error_event_only"]),
                task_error_weight=float(run_args["task_error_weight"]),
                budget=float(run_args["budget"]),
                startup_peak_budget=float(run_args["startup_peak_budget"]),
            )
        else:
            policy = make_proxy_policy(
                controller=controller,
                candidate_masks=candidate_masks,
                forecast_cfg=forecast_cfg,
                anchor_mask=anchor_masks[int(anchor_idx)],
                support=support,
                target_rates=target_rates,
                preserve_warming=preserve_warming,
            )
            outcome = evaluate_window(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=policy,
                start=start,
                steps=int(args.window_steps),
                seed_offset=seed_offset,
                state_columns=state_columns,
                objective_mode=str(run_args["objective_mode"]),
                task_error_columns=tuple(str(x) for x in run_args["task_error_columns"]),
                task_error_scales=tuple(float(x) for x in run_args["task_error_scales"]),
                task_error_event_only=bool(run_args["task_error_event_only"]),
                task_error_weight=float(run_args["task_error_weight"]),
                budget=float(run_args["budget"]),
                startup_peak_budget=float(run_args["startup_peak_budget"]),
            )
        print(
            f"[window_risk] candidate start={start} anchor={anchor_idx} "
            f"controller={controller.controller_id} objective={outcome.objective:.6f}",
            flush=True,
        )
        return outcome

    def feature_builder(
        start: int,
        anchor_idx: int,
        controller: ControllerSpec,
        seed_offset: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        if str(args.controller_family) == "residual_action":
            env, boundary = residual_boundary(start, anchor_idx, seed_offset)
            restore_env(env, boundary)
            return build_residual_risk_feature(
                env=env,
                forecast_cfg=forecast_cfg,
                candidate_masks=candidate_masks,
                anchor_idx=int(anchor_idx),
                controller=controller,
            )
        return build_feature_vector(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            forecast_cfg=forecast_cfg,
            candidate_masks=candidate_masks,
            anchor_idx=anchor_idx,
            controller=controller,
            support=support,
            target_rates=target_rates,
            preserve_warming=preserve_warming,
            start=start,
        )

    if bool(args.refresh_features_only):
        for split_name in ("risk_fit", "risk_calibration"):
            dataset = refresh_window_risk_features(
                out_dir / split_name,
                feature_builder=feature_builder,
            )
            print(
                f"[window_risk] refreshed split={split_name} rows={len(dataset.records)} "
                f"features={dataset.features.shape[1]}",
                flush=True,
            )
        protocol_summary["feature_version"] = (
            "prefix_conditioned_residual_v1"
            if str(args.controller_family) == "residual_action"
            else "causal_history_v1"
        )
        (out_dir / "window_risk_protocol.json").write_text(
            json.dumps(protocol_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    for split_name, split_starts, seed_base in (
        ("risk_fit", split.fit, 210_000),
        ("risk_calibration", split.calibration, 410_000),
    ):
        assignments = assign_balanced_anchors(
            split_starts,
            anchor_bank,
            anchors_per_start=int(args.anchors_per_start),
            always_include_best=(
                str(args.controller_family) != "residual_action"
            ),
        )
        dataset = collect_paired_window_risk_dataset(
            out_dir=out_dir / split_name,
            seed=int(manifest["seed"]),
            split_name=split_name,
            starts=split_starts,
            anchor_assignments=assignments,
            anchor_masks=anchor_masks,
            controllers=controllers,
            static_evaluator=static_evaluator,
            candidate_evaluator=candidate_evaluator,
            feature_builder=feature_builder,
            controller_filter=(
                (
                    lambda anchor_idx, controller: valid_anchor_residual_action(
                        candidate_masks,
                        anchor_idx=anchor_idx,
                        controller=controller,
                        allowed_action_indices=support,
                        required_sensor_indices=required_sensor_indices,
                    )
                )
                if str(args.controller_family) == "residual_action"
                else None
            ),
            seed_offset_base=int(seed_base),
        )
        print(
            f"[window_risk] complete split={split_name} rows={len(dataset.records)} "
            f"margin_mean={float(np.mean(dataset.margins)):.6f} "
            f"margin_q25={float(np.quantile(dataset.margins, 0.25)):.6f}",
            flush=True,
        )

    (out_dir / "window_risk_protocol.json").write_text(
        json.dumps(protocol_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
