#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
SCRIPTS_ROOT = V1_ROOT / "scripts"
sys.path.insert(0, str(V1_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
)
from forecast_cmdp.features import ForecastContextConfig  # noqa: E402
from forecast_cmdp.mean_risk_policy import (  # noqa: E402
    ForecastAwareMeanRiskControllerPolicy,
    ForecastAwareResidualRiskControllerPolicy,
    RecedingForecastAwareMeanRiskControllerPolicy,
)
from forecast_cmdp.window_risk import ControllerSpec, WindowOutcome, static_candidate_margin  # noqa: E402
from forecast_cmdp.window_risk_model import WindowRiskModelBundle  # noqa: E402
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.rollout import run_policy_rollout  # noqa: E402

from run_window_risk_pilot import evaluate_window, source_truth_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation/final gate for the Branch H mean-risk controller.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--min-validation-mean-margin", type=float, default=0.0)
    parser.add_argument("--min-validation-q25-margin", type=float, default=0.0)
    parser.add_argument("--max-validation-negative-starts", type=int, default=1)
    parser.add_argument("--min-risk-lower-bound", type=float, default=0.0)
    parser.add_argument("--max-negative-probability", type=float, default=0.25)
    parser.add_argument("--min-predicted-mean-margin", type=float, default=0.0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--final-limit", type=int, default=0)
    parser.add_argument(
        "--policy-anchor-index",
        type=int,
        default=None,
        help="Optional validation-selection candidate; residual anchors must belong to the train-only anchor bank.",
    )
    parser.add_argument("--allow-data-model-gate-fail", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run and gate validation without reading or evaluating final-test starts.",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=0,
        help="Zero selects once per evaluation window; positive values reselect at this interval.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_source_context(
    source_run: Path,
    *,
    oracle_device: str,
    truth_path_override: Path | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    list[object],
    object,
    object | None,
    object,
    tuple[str, ...],
    np.ndarray,
    ForecastContextConfig,
]:
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    run_args = dict(manifest["run_args"])
    truth = pd.read_csv(
        truth_path_override
        if truth_path_override is not None
        else source_truth_path(source_run, manifest)
    )
    helpers = load_v2_helpers()
    state_columns = tuple(str(x) for x in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(x) for x in helpers.REWARD_TARGET_COLUMNS)
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
        episode_len=int(run_args["eval_steps"]),
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
        device=str(oracle_device),
    )
    if str(manifest.get("oracle_target_weight_mode", "checkpoint")) != "checkpoint":
        raise ValueError("Mean-risk evaluator currently requires a checkpoint-weighted source oracle")
    teacher_path = resolve_project_path(str(manifest["teacher_dataset"]))
    with np.load(teacher_path, allow_pickle=False) as teacher:
        candidate_masks = np.asarray(teacher["candidate_masks"], dtype=bool)
    return (
        manifest,
        run_args,
        truth,
        sensors,
        constraints,
        oracle,
        cfg,
        state_columns,
        candidate_masks,
        ForecastContextConfig(**dict(manifest["forecast_cfg"])),
    )


def evaluate_split(
    *,
    split_name: str,
    starts: Sequence[int],
    seed_offset_base: int,
    deploy_dynamic: bool,
    out_path: Path,
    model_bundle: WindowRiskModelBundle,
    controllers: tuple[ControllerSpec, ...],
    candidate_masks: np.ndarray,
    comparator_anchor_idx: int,
    policy_anchor_idx: int,
    support: tuple[int, ...],
    target_rates: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    preserve_warming: bool,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    oracle: object | None,
    cfg: object,
    state_columns: tuple[str, ...],
    run_args: Mapping[str, object],
    min_risk_lower_bound: float,
    max_negative_probability: float,
    min_predicted_mean_margin: float,
    run_signature: str,
    decision_interval: int,
    controller_family: str,
    required_sensor_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    existing: list[dict[str, object]] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = dict(json.loads(line))
                if str(row.get("run_signature", "")) != str(run_signature):
                    raise ValueError(
                        f"Existing {split_name} rows were produced by a different model or deployment mode"
                    )
                existing.append(row)
    completed = {(str(row["split_name"]), int(row["start"])) for row in existing}
    comparator_anchor_mask = tuple(
        bool(x) for x in candidate_masks[int(comparator_anchor_idx)]
    )
    for start_pos, start in enumerate(int(x) for x in starts):
        key = (str(split_name), int(start))
        if key in completed:
            continue
        seed_offset = int(seed_offset_base) + int(start_pos) * 101
        static_outcome = evaluate_window(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=cfg,
            oracle=oracle,
            policy=StaticMaskPolicy(
                comparator_anchor_mask,
                name=f"{split_name}_static_comparator",
            ),
            start=int(start),
            steps=int(run_args["eval_steps"]),
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
        policy = None
        if deploy_dynamic:
            if str(controller_family) == "residual_action":
                policy = ForecastAwareResidualRiskControllerPolicy(
                    model_bundle=model_bundle,
                    candidate_masks=candidate_masks,
                    forecast_cfg=forecast_cfg,
                    anchor_action_idx=int(policy_anchor_idx),
                    support=support,
                    required_sensor_indices=required_sensor_indices,
                    decision_interval=max(1, int(decision_interval)),
                    min_risk_lower_bound=float(min_risk_lower_bound),
                    max_negative_probability=float(max_negative_probability),
                    min_mean_margin=float(min_predicted_mean_margin),
                )
            else:
                policy_class = (
                    RecedingForecastAwareMeanRiskControllerPolicy
                    if int(decision_interval) > 0
                    else ForecastAwareMeanRiskControllerPolicy
                )
                policy_kwargs = {
                    "model_bundle": model_bundle,
                    "controllers": controllers,
                    "candidate_masks": candidate_masks,
                    "forecast_cfg": forecast_cfg,
                    "anchor_action_idx": int(policy_anchor_idx),
                    "support": support,
                    "target_rates": target_rates,
                    "min_risk_lower_bound": float(min_risk_lower_bound),
                    "max_negative_probability": float(max_negative_probability),
                    "min_mean_margin": float(min_predicted_mean_margin),
                    "preserve_warming": bool(preserve_warming),
                }
                if int(decision_interval) > 0:
                    policy_kwargs["decision_interval"] = int(decision_interval)
                policy = policy_class(
                    **policy_kwargs,
                )
            candidate_outcome = evaluate_window(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=policy,
                start=int(start),
                steps=int(run_args["eval_steps"]),
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
        else:
            candidate_outcome = WindowOutcome(
                objective=float(static_outcome.objective),
                power_mean=float(static_outcome.power_mean),
                warmup_abort_count=int(static_outcome.warmup_abort_count),
                constraint_violation_count=int(static_outcome.constraint_violation_count),
            )
        block_history = list(getattr(policy, "block_history", [])) if policy is not None else []
        dynamic_blocks = int(
            sum(not bool(block["static_fallback"]) for block in block_history)
        )
        row = {
            "split_name": str(split_name),
            "run_signature": str(run_signature),
            "start": int(start),
            "paired_seed_offset": int(seed_offset),
            "comparator_anchor_action_idx": int(comparator_anchor_idx),
            "policy_anchor_action_idx": int(policy_anchor_idx),
            "static_objective": float(static_outcome.objective),
            "candidate_objective": float(candidate_outcome.objective),
            "margin": static_candidate_margin(static_outcome.objective, candidate_outcome.objective),
            "candidate_power_mean": float(candidate_outcome.power_mean),
            "warmup_abort_count": int(candidate_outcome.warmup_abort_count),
            "constraint_violation_count": int(candidate_outcome.constraint_violation_count),
            "selected_controller_id": (
                "|".join(
                    (
                        f"residual_action_{int(block['selected_action_idx']):03d}"
                        if "selected_action_idx" in block
                        else str(block["selected_controller_id"])
                    )
                    for block in block_history
                    if not bool(block["static_fallback"])
                )
                if block_history
                else str(policy.selected_controller_id)
                if policy is not None and policy.selected_controller_id is not None
                else "static_fallback"
            ),
            "static_fallback": bool(
                policy is None
                or (bool(block_history) and dynamic_blocks == 0)
                or (
                    not block_history
                    and bool(getattr(policy, "static_fallback", True))
                )
            ),
            "dynamic_blocks": dynamic_blocks,
            "total_blocks": int(len(block_history)),
            "block_history": block_history,
            "selection_rows": (
                policy.all_selection_rows
                if policy is not None and hasattr(policy, "all_selection_rows")
                else policy.selection_rows
                if policy is not None
                else []
            ),
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
        existing.append(row)
        completed.add(key)
        print(
            f"[mean_risk] split={split_name} start={start} "
            f"controller={row['selected_controller_id']} margin={row['margin']:.6f}",
            flush=True,
        )
    return [row for row in existing if str(row["split_name"]) == str(split_name)]


def split_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    margins = np.asarray([float(row["margin"]) for row in rows], dtype=float)
    return {
        "rows": int(margins.size),
        "margin_mean": float(np.mean(margins)),
        "margin_q25": float(np.quantile(margins, 0.25)),
        "margin_min": float(np.min(margins)),
        "negative_starts": int(np.sum(margins < 0.0)),
        "dynamic_windows": int(sum(not bool(row["static_fallback"]) for row in rows)),
        "hard_constraint_violations": int(sum(int(row["constraint_violation_count"]) for row in rows)),
        "warmup_abort_count": int(sum(int(row["warmup_abort_count"]) for row in rows)),
    }


def validation_gate_pass(
    summary: Mapping[str, object],
    *,
    min_mean_margin: float,
    min_q25_margin: float,
    max_negative_starts: int,
) -> bool:
    return bool(
        int(summary["dynamic_windows"]) > 0
        and float(summary["margin_mean"]) > float(min_mean_margin)
        and float(summary["margin_q25"]) >= float(min_q25_margin)
        and int(summary["negative_starts"]) <= int(max_negative_starts)
        and int(summary["hard_constraint_violations"]) == 0
    )


def main() -> None:
    args = parse_args()
    source_run = resolve_project_path(args.source_run)
    data_root = resolve_project_path(args.data_root)
    model_dir = resolve_project_path(args.model_dir) if args.model_dir else data_root / "model"
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = json.loads((model_dir / "window_risk_model_metrics.json").read_text(encoding="utf-8"))
    data_model_gate = bool(model_metrics["pilot_gate_pass"])
    if not data_model_gate and not bool(args.allow_data_model_gate_fail):
        raise RuntimeError("Data/model gate failed; validation evaluation is forbidden")
    bundle = WindowRiskModelBundle.load(model_dir / "window_risk_model.joblib")
    protocol = json.loads((data_root / "window_risk_protocol.json").read_text(encoding="utf-8"))
    controller_family = str(protocol.get("controller_family", "balanced"))
    controllers = tuple(
        ControllerSpec(str(controller_id), dict(parameters))
        for controller_id, parameters in zip(protocol["controller_ids"], protocol["controller_configs"])
    )
    anchor_bank = tuple(int(x) for x in protocol["anchor_bank"])
    support = tuple(int(x) for x in protocol["support"])
    target_rates = np.asarray(protocol["target_rates"], dtype=float)
    required_forecast_sensor_indices = tuple(
        int(x) for x in protocol.get("required_forecast_sensor_indices", ())
    )
    (
        manifest,
        run_args,
        truth,
        sensors,
        constraints,
        oracle,
        cfg,
        state_columns,
        candidate_masks,
        forecast_cfg,
    ) = load_source_context(
        source_run,
        oracle_device=str(args.oracle_device),
        truth_path_override=resolve_project_path(str(protocol["truth_csv"])),
    )
    if "effective_forecast_cfg" in protocol:
        forecast_cfg = ForecastContextConfig(**dict(protocol["effective_forecast_cfg"]))
    comparator_anchor_idx = int(manifest["selected_static"]["action_idx"])
    policy_anchor_idx = (
        int(args.policy_anchor_index)
        if args.policy_anchor_index is not None
        else int(anchor_bank[0])
        if controller_family == "residual_action"
        else int(comparator_anchor_idx)
    )
    if (
        controller_family == "residual_action"
        and policy_anchor_idx
        not in {*anchor_bank, int(comparator_anchor_idx)}
    ):
        raise ValueError(
            "Residual policy anchor must be train-only or the shared static comparator"
        )
    if required_forecast_sensor_indices and not bool(
        np.all(
            candidate_masks[
                comparator_anchor_idx,
                list(required_forecast_sensor_indices),
            ]
        )
    ):
        raise ValueError(
            "Validation static comparator does not provide required forecast sensors"
        )
    if required_forecast_sensor_indices and not bool(
        np.all(
            candidate_masks[
                policy_anchor_idx,
                list(required_forecast_sensor_indices),
            ]
        )
    ):
        raise ValueError(
            "Train policy anchor does not provide required forecast sensors"
        )
    validation_starts = tuple(int(x) for x in manifest["starts"]["validation"]["starts"])
    final_starts = tuple(int(x) for x in manifest["starts"]["final_test"]["starts"])
    if int(args.validation_limit) > 0:
        validation_starts = validation_starts[: int(args.validation_limit)]
    if int(args.final_limit) > 0:
        final_starts = final_starts[: int(args.final_limit)]
    if policy_anchor_idx not in support:
        support = tuple(sorted({*support, int(policy_anchor_idx)}))
    decision_interval = int(args.decision_interval)
    if controller_family == "residual_action" and decision_interval <= 0:
        decision_interval = int(protocol["window_steps"])
    signature_payload = {
        "model_sha256": hashlib.sha256((model_dir / "window_risk_model.joblib").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((data_root / "window_risk_protocol.json").read_bytes()).hexdigest(),
        "comparator_anchor_idx": int(comparator_anchor_idx),
        "policy_anchor_idx": int(policy_anchor_idx),
        "support": list(support),
        "required_forecast_sensor_indices": list(required_forecast_sensor_indices),
        "decision_interval": int(decision_interval),
        "controller_family": controller_family,
        "min_risk_lower_bound": float(args.min_risk_lower_bound),
        "max_negative_probability": float(args.max_negative_probability),
        "min_predicted_mean_margin": float(args.min_predicted_mean_margin),
    }
    base_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    validation_rows = evaluate_split(
        split_name="validation",
        starts=validation_starts,
        seed_offset_base=610_000,
        deploy_dynamic=True,
        out_path=out_dir / "mean_risk_validation_rows.jsonl",
        model_bundle=bundle,
        controllers=controllers,
        candidate_masks=candidate_masks,
        comparator_anchor_idx=comparator_anchor_idx,
        policy_anchor_idx=policy_anchor_idx,
        support=support,
        target_rates=target_rates,
        forecast_cfg=forecast_cfg,
        preserve_warming=bool(run_args["bc_preserve_warming"]),
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        oracle=oracle,
        cfg=replace(cfg, episode_len=int(run_args["eval_steps"])),
        state_columns=state_columns,
        run_args=run_args,
        min_risk_lower_bound=float(args.min_risk_lower_bound),
        max_negative_probability=float(args.max_negative_probability),
        min_predicted_mean_margin=float(args.min_predicted_mean_margin),
        run_signature=f"{base_signature}:validation:dynamic",
        decision_interval=int(decision_interval),
        controller_family=controller_family,
        required_sensor_indices=required_forecast_sensor_indices,
    )
    validation_summary = split_summary(validation_rows)
    validation_gate = validation_gate_pass(
        validation_summary,
        min_mean_margin=float(args.min_validation_mean_margin),
        min_q25_margin=float(args.min_validation_q25_margin),
        max_negative_starts=int(args.max_validation_negative_starts),
    )
    deploy_final = bool(data_model_gate and validation_gate)
    if bool(args.allow_data_model_gate_fail):
        deploy_final = bool(validation_gate)
    if bool(args.validation_only):
        summary = {
            "seed": int(manifest["seed"]),
            "source_run": str(source_run),
            "data_root": str(data_root),
            "model_dir": str(model_dir),
            "validation_anchor_action_idx": int(comparator_anchor_idx),
            "policy_anchor_action_idx": int(policy_anchor_idx),
            "train_anchor_bank": list(anchor_bank),
            "data_model_gate_pass": bool(data_model_gate),
            "validation_gate_pass": bool(validation_gate),
            "deploy_final_dynamic": False,
            "final_test_not_run": True,
            "engineering_gate_override": bool(args.allow_data_model_gate_fail),
            "run_signature": str(base_signature),
            **signature_payload,
            "validation": validation_summary,
            "final_test": None,
            "thresholds": {
                "min_validation_mean_margin": float(
                    args.min_validation_mean_margin
                ),
                "min_validation_q25_margin": float(
                    args.min_validation_q25_margin
                ),
                "max_validation_negative_starts": int(
                    args.max_validation_negative_starts
                ),
                "min_risk_lower_bound": float(args.min_risk_lower_bound),
                "max_negative_probability": float(
                    args.max_negative_probability
                ),
                "min_predicted_mean_margin": float(
                    args.min_predicted_mean_margin
                ),
            },
        }
        pd.DataFrame(validation_rows).drop(
            columns=["selection_rows", "block_history"]
        ).to_csv(
            out_dir / "mean_risk_validation_rows.csv",
            index=False,
        )
        (out_dir / "mean_risk_gate_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    final_rows = evaluate_split(
        split_name="final_test",
        starts=final_starts,
        seed_offset_base=810_000,
        deploy_dynamic=deploy_final,
        out_path=out_dir / "mean_risk_final_rows.jsonl",
        model_bundle=bundle,
        controllers=controllers,
        candidate_masks=candidate_masks,
        comparator_anchor_idx=comparator_anchor_idx,
        policy_anchor_idx=policy_anchor_idx,
        support=support,
        target_rates=target_rates,
        forecast_cfg=forecast_cfg,
        preserve_warming=bool(run_args["bc_preserve_warming"]),
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        oracle=oracle,
        cfg=replace(cfg, episode_len=int(run_args["eval_steps"])),
        state_columns=state_columns,
        run_args=run_args,
        min_risk_lower_bound=float(args.min_risk_lower_bound),
        max_negative_probability=float(args.max_negative_probability),
        min_predicted_mean_margin=float(args.min_predicted_mean_margin),
        run_signature=f"{base_signature}:final:{'dynamic' if deploy_final else 'static'}",
        decision_interval=int(decision_interval),
        controller_family=controller_family,
        required_sensor_indices=required_forecast_sensor_indices,
    )
    summary = {
        "seed": int(manifest["seed"]),
        "source_run": str(source_run),
        "data_root": str(data_root),
        "model_dir": str(model_dir),
        "validation_anchor_action_idx": int(comparator_anchor_idx),
        "policy_anchor_action_idx": int(policy_anchor_idx),
        "train_anchor_bank": list(anchor_bank),
        "data_model_gate_pass": bool(data_model_gate),
        "validation_gate_pass": bool(validation_gate),
        "deploy_final_dynamic": bool(deploy_final),
        "engineering_gate_override": bool(args.allow_data_model_gate_fail),
        "run_signature": str(base_signature),
        **signature_payload,
        "validation": validation_summary,
        "final_test": split_summary(final_rows),
        "thresholds": {
            "min_validation_mean_margin": float(args.min_validation_mean_margin),
            "min_validation_q25_margin": float(args.min_validation_q25_margin),
            "max_validation_negative_starts": int(args.max_validation_negative_starts),
            "min_risk_lower_bound": float(args.min_risk_lower_bound),
            "max_negative_probability": float(args.max_negative_probability),
            "min_predicted_mean_margin": float(args.min_predicted_mean_margin),
        },
    }
    pd.DataFrame(validation_rows).drop(columns=["selection_rows", "block_history"]).to_csv(
        out_dir / "mean_risk_validation_rows.csv",
        index=False,
    )
    pd.DataFrame(final_rows).drop(columns=["selection_rows", "block_history"]).to_csv(
        out_dir / "mean_risk_final_rows.csv",
        index=False,
    )
    (out_dir / "mean_risk_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
