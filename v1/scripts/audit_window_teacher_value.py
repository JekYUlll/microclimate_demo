#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

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
    normalization_stats,
    resolve_archive_path,
)
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, enumerate_action_masks  # noqa: E402
from forecast_cmdp.protocol import evaluate_policy_over_starts, final_objective, rich_metrics, task_focus_metrics  # noqa: E402
from scripts.run_protocol_gate import (  # noqa: E402
    apply_oracle_target_weight_mode,
    make_common_env_config,
    parse_args as parse_protocol_args,
)
from v2.policies import StaticMaskPolicy  # noqa: E402


DEFAULT_ROOTS = [
    "v1/artifacts/claim_suite_v6_transport_teacher_improvement_gate_smoke_20260604",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay static anchor and MPC teacher on train/validation/final "
            "starts to audit sequence-level teacher value without retraining."
        )
    )
    parser.add_argument("suite_roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--out-dir", default="v1/artifacts/window_teacher_value_audit_v6_20260605")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validation", "final_test"],
        choices=["train", "validation", "final_test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    for root_value in args.suite_roots:
        root = Path(root_value)
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            print(f"[audit] start run={manifest_path.parent}", flush=True)
            run_rows, availability = audit_run(
                root=root,
                manifest_path=manifest_path,
                splits=tuple(str(x) for x in args.splits),
            )
            rows.extend(run_rows)
            availability_rows.append(availability)
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / "window_teacher_value_rows.partial.csv", index=False)
            if availability_rows:
                pd.DataFrame(availability_rows).to_csv(
                    out_dir / "window_teacher_value_artifact_availability.partial.csv",
                    index=False,
                )
            print(
                f"[audit] complete run={manifest_path.parent} rows={len(run_rows)} cumulative_rows={len(rows)}",
                flush=True,
            )

    if not rows:
        raise FileNotFoundError(f"No auditable runs found under {args.suite_roots}")

    table = pd.DataFrame(rows).sort_values(["root", "preset", "seed", "split", "start_rank"])
    availability = pd.DataFrame(availability_rows).sort_values(["root", "preset", "seed"])
    summary = summarize(table)

    table.to_csv(out_dir / "window_teacher_value_rows.csv", index=False)
    summary.to_csv(out_dir / "window_teacher_value_summary.csv", index=False)
    availability.to_csv(out_dir / "window_teacher_value_artifact_availability.csv", index=False)
    report = render_report(summary, availability)
    (out_dir / "window_teacher_value_audit.md").write_text(report, encoding="utf-8")
    print(report)


def audit_run(
    *,
    root: Path,
    manifest_path: Path,
    splits: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preset, seed = parse_preset_seed(run_dir.name, manifest)
    run_args = protocol_args_from_manifest(manifest)

    helpers = load_v2_helpers()
    truth_path = resolve_archive_path(str(manifest.get("truth_csv") or run_args.truth_csv))
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(str(manifest.get("sensor_cfg") or run_args.sensor_cfg))
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)

    bounds = manifest.get("bounds", {})
    if not isinstance(bounds, dict):
        bounds = {}
    norm_start = getattr(run_args, "normalization_start_idx", None)
    norm_end = getattr(run_args, "normalization_end_idx", None)
    if norm_start is None and norm_end is None and "oracle_pretrain" in bounds:
        pretrain = bounds["oracle_pretrain"]
        norm_start, norm_end = int(pretrain[0]), int(pretrain[1])
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=norm_start,
        end_idx=norm_end,
    )
    oracle = load_archived_oracle(
        getattr(run_args, "oracle_path", None),
        oracle_type=str(getattr(run_args, "oracle_type", "tcn")),
        device=str(getattr(run_args, "oracle_device", "cpu")),
    )
    apply_oracle_target_weight_mode(
        oracle,
        reward_target_columns=reward_target_columns,
        mode=str(getattr(run_args, "oracle_target_weight_mode", "checkpoint")),
    )
    constraints = make_constraints(
        max_active=int(run_args.max_active),
        budget=float(run_args.budget),
        startup_peak_budget=float(run_args.startup_peak_budget),
    )
    cfg_by_split = {
        "train": make_common_env_config(
            run_args,
            state_columns=state_columns,
            reward_target_columns=reward_target_columns,
            episode_len=int(run_args.train_steps),
            seed=int(seed),
            norm_mean=norm_mean,
            norm_std=norm_std,
        ),
        "validation": make_common_env_config(
            run_args,
            state_columns=state_columns,
            reward_target_columns=reward_target_columns,
            episode_len=int(run_args.static_selection_steps),
            seed=int(seed) + 10_000,
            norm_mean=norm_mean,
            norm_std=norm_std,
        ),
        "final_test": make_common_env_config(
            run_args,
            state_columns=state_columns,
            reward_target_columns=reward_target_columns,
            episode_len=int(run_args.eval_steps),
            seed=int(seed) + 20_000,
            norm_mean=norm_mean,
            norm_std=norm_std,
        ),
    }
    steps_by_split = {
        "train": int(run_args.train_steps),
        "validation": int(run_args.static_selection_steps),
        "final_test": int(run_args.eval_steps),
    }

    candidate_masks = enumerate_action_masks(len(sensors), max_active=int(run_args.max_active))
    static_mask, static_action_idx, static_source = load_validation_static_mask(run_dir, candidate_masks)
    candidate_prior_costs = load_candidate_prior_costs(run_dir, n_actions=int(candidate_masks.shape[0]))
    teacher_cfg = MpcTeacherConfig(
        planning_horizon=int(run_args.planning_horizon),
        beam_width=int(run_args.beam_width),
        max_branch=int(run_args.max_branch),
        event_weight_alpha=float(run_args.event_weight_alpha),
        lambda_warmup_abort=float(run_args.teacher_lambda_warmup_abort),
        lambda_switch=float(run_args.teacher_lambda_switch),
        lambda_energy_deficit=float(run_args.teacher_lambda_energy_deficit),
        saturated_loss_threshold=float(run_args.saturated_loss_threshold),
        saturated_coverage_bonus=float(run_args.saturated_coverage_bonus),
        candidate_prior_weight=float(run_args.candidate_prior_weight),
        candidate_prior_costs=candidate_prior_costs
        if float(run_args.candidate_prior_weight) > 0.0
        else None,
        candidate_prefilter_top_k=int(run_args.candidate_prefilter_top_k),
        anchor_mask=tuple(bool(x) for x in static_mask),
        anchor_regret_guard=bool(run_args.anchor_regret_guard),
        anchor_improvement_margin=float(run_args.anchor_improvement_margin),
        task_error_weight=float(run_args.task_error_weight)
        if str(run_args.objective_mode) == "task_composite"
        else 0.0,
        task_error_columns=tuple(str(x) for x in run_args.task_error_columns),
        task_error_scales=tuple(float(x) for x in run_args.task_error_scales)
        if getattr(run_args, "task_error_scales", None)
        else None,
        task_error_event_only=bool(run_args.task_error_event_only),
    )
    static_policy = StaticMaskPolicy(mask=tuple(bool(x) for x in static_mask), name="validation_selected_static")
    teacher_policy = MpcTeacherPolicy(candidate_masks=candidate_masks, cfg=teacher_cfg, name="mpc_teacher")

    starts_block = manifest.get("starts", {})
    rows: list[dict[str, object]] = []
    for split in splits:
        split_info = starts_block.get(split, {}) if isinstance(starts_block, dict) else {}
        starts = tuple(int(x) for x in split_info.get("starts", [])) if isinstance(split_info, dict) else ()
        if not starts:
            continue
        cfg = cfg_by_split[split]
        steps = int(steps_by_split[split])
        static_objectives: list[float] = []
        teacher_objectives: list[float] = []
        for start_rank, start_idx in enumerate(starts):
            seed_offset = split_seed_offset(split) + int(start_rank) * 101
            print(
                "[audit] "
                f"seed={seed} split={split} start_rank={start_rank} "
                f"start_idx={start_idx} steps={steps} evaluating",
                flush=True,
            )
            static_metrics, static_objective = evaluate_single_start(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=static_policy,
                steps=steps,
                start_idx=int(start_idx),
                seed_offset=seed_offset,
                sensor_ids=sensor_ids,
                state_columns=state_columns,
                run_args=run_args,
            )
            teacher_metrics, teacher_objective = evaluate_single_start(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=cfg,
                oracle=oracle,
                policy=teacher_policy,
                steps=steps,
                start_idx=int(start_idx),
                seed_offset=seed_offset,
                sensor_ids=sensor_ids,
                state_columns=state_columns,
                run_args=run_args,
            )
            static_objectives.append(float(static_objective))
            teacher_objectives.append(float(teacher_objective))
            margin = float(static_objective - teacher_objective)
            print(
                "[audit] "
                f"seed={seed} split={split} start_rank={start_rank} "
                f"static={static_objective:.6f} teacher={teacher_objective:.6f} "
                f"margin={margin:.6f}",
                flush=True,
            )
            rows.append(
                {
                    "root": str(root),
                    "preset": preset,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "split": split,
                    "start_rank": int(start_rank),
                    "start_idx": int(start_idx),
                    "steps": int(steps),
                    "static_action_idx": int(static_action_idx),
                    "static_source": static_source,
                    "static_objective": float(static_objective),
                    "teacher_objective": float(teacher_objective),
                    "objective_margin": margin,
                    "teacher_win": bool(margin > 0.0),
                    "static_oracle_loss": safe_float(static_metrics.get("oracle_loss_mean")),
                    "teacher_oracle_loss": safe_float(teacher_metrics.get("oracle_loss_mean")),
                    "oracle_margin": safe_float(static_metrics.get("oracle_loss_mean"))
                    - safe_float(teacher_metrics.get("oracle_loss_mean")),
                    "static_task_error": safe_float(static_metrics.get("task_error_mean")),
                    "teacher_task_error": safe_float(teacher_metrics.get("task_error_mean")),
                    "task_error_margin": safe_float(static_metrics.get("task_error_mean"))
                    - safe_float(teacher_metrics.get("task_error_mean")),
                    "static_power_mean": safe_float(static_metrics.get("power_mean")),
                    "teacher_power_mean": safe_float(teacher_metrics.get("power_mean")),
                    "teacher_event_rate": safe_float(teacher_metrics.get("event_rate")),
                    "teacher_switch_any_rate": safe_float(teacher_metrics.get("switch_any_rate")),
                    "teacher_switch_ge2_rate": safe_float(teacher_metrics.get("switch_ge2_rate")),
                    "teacher_warmup_abort_count": int(teacher_metrics.get("warmup_abort_count", 0)),
                }
            )

    availability = {
        "root": str(root),
        "preset": preset,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "truth_csv": str(truth_path),
        "sensor_cfg": str(manifest.get("sensor_cfg") or run_args.sensor_cfg),
        "validation_static_candidates": str(run_dir / "validation_static_candidates.csv"),
        "train_static_candidates": str(run_dir / "train_static_candidates.csv"),
        "static_action_idx": int(static_action_idx),
        "static_source": static_source,
        "n_rows": int(len(rows)),
    }
    return rows, availability


def protocol_args_from_manifest(manifest: dict[str, object]) -> argparse.Namespace:
    argv = [
        "run_protocol_gate.py",
        "--truth-csv",
        str(manifest.get("truth_csv", "__missing_truth__")),
        "--out-dir",
        "__audit_defaults__",
    ]
    original_argv = sys.argv
    try:
        sys.argv = argv
        args = parse_protocol_args()
    finally:
        sys.argv = original_argv
    data = vars(args)
    run_args = manifest.get("run_args", {})
    if isinstance(run_args, dict):
        data.update(run_args)
    for key in ("truth_csv", "sensor_cfg", "objective_mode", "task_error_weight"):
        if key in manifest and manifest[key] is not None:
            data[key] = manifest[key]
    return SimpleNamespace(**data)


def load_validation_static_mask(run_dir: Path, candidate_masks: np.ndarray) -> tuple[np.ndarray, int, str]:
    path = run_dir / "validation_static_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation static table: {path}")
    table = pd.read_csv(path)
    if table.empty or "action_idx" not in table.columns:
        raise ValueError(f"Invalid validation static table: {path}")
    action_idx = int(table.iloc[0]["action_idx"])
    if action_idx < 0 or action_idx >= int(candidate_masks.shape[0]):
        raise ValueError(f"Static action_idx out of range in {path}: {action_idx}")
    return np.asarray(candidate_masks[action_idx], dtype=bool), action_idx, "validation_static_candidates"


def load_candidate_prior_costs(run_dir: Path, *, n_actions: int) -> tuple[float, ...] | None:
    path = run_dir / "train_static_candidates.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if "action_idx" not in table.columns or "objective_loss_mean" not in table.columns:
        return None
    values = np.full(int(n_actions), np.nan, dtype=float)
    for _, row in table.iterrows():
        idx = int(row["action_idx"])
        if 0 <= idx < int(n_actions):
            values[idx] = safe_float(row["objective_loss_mean"])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 1.0e-12:
        out = np.zeros_like(values, dtype=float)
    else:
        out = (values - mean) / std
    out[~np.isfinite(out)] = float(np.nanmax(out[np.isfinite(out)])) if np.any(np.isfinite(out)) else 0.0
    return tuple(float(x) for x in out)


def evaluate_single_start(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    oracle: object | None,
    policy: object,
    steps: int,
    start_idx: int,
    seed_offset: int,
    sensor_ids: tuple[str, ...],
    state_columns: tuple[str, ...],
    run_args: object,
) -> tuple[dict[str, object], float]:
    result, _ = evaluate_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=cfg,
        oracle=oracle,
        policy=policy,
        steps=int(steps),
        start_indices=(int(start_idx),),
        seed_offset=int(seed_offset),
    )
    metrics = rich_metrics(
        result,
        sensor_ids=sensor_ids,
        state_columns=state_columns,
        per_step_budget=float(run_args.budget),
        startup_peak_budget=float(run_args.startup_peak_budget),
    )
    metrics.update(
        task_focus_metrics(
            result,
            state_columns=state_columns,
            task_error_columns=tuple(str(x) for x in run_args.task_error_columns),
            task_error_scales=tuple(float(x) for x in run_args.task_error_scales)
            if getattr(run_args, "task_error_scales", None)
            else None,
            event_only=bool(run_args.task_error_event_only),
        )
    )
    objective = final_objective(
        metrics,
        mode=str(run_args.objective_mode),
        task_error_weight=float(run_args.task_error_weight),
    )
    return metrics, float(objective)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        table.groupby(["root", "preset", "seed", "split"], sort=True)
        .agg(
            n_windows=("objective_margin", "size"),
            teacher_wins=("teacher_win", "sum"),
            teacher_win_rate=("teacher_win", "mean"),
            margin_mean=("objective_margin", "mean"),
            margin_median=("objective_margin", "median"),
            margin_q25=("objective_margin", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.25))),
            margin_min=("objective_margin", "min"),
            oracle_margin_mean=("oracle_margin", "mean"),
            task_error_margin_mean=("task_error_margin", "mean"),
            teacher_power_mean=("teacher_power_mean", "mean"),
            static_power_mean=("static_power_mean", "mean"),
            teacher_switch_any_rate=("teacher_switch_any_rate", "mean"),
        )
        .reset_index()
    )
    return grouped


def render_report(summary: pd.DataFrame, availability: pd.DataFrame) -> str:
    lines = ["# Window-Level Teacher Value Audit", ""]
    lines.append("## Summary")
    lines.append("")
    display = summary[
        [
            "preset",
            "seed",
            "split",
            "n_windows",
            "teacher_wins",
            "teacher_win_rate",
            "margin_mean",
            "margin_q25",
            "margin_min",
            "oracle_margin_mean",
            "task_error_margin_mean",
        ]
    ]
    lines.append(markdown_table(display))
    lines.append("")
    lines.append("## Decision Signal")
    lines.append("")
    final = summary[summary["split"].eq("final_test")]
    validation = summary[summary["split"].eq("validation")]
    train = summary[summary["split"].eq("train")]
    if not final.empty:
        final_positive = int((final["margin_mean"] > 0.0).sum())
        lines.append(
            f"- Final teacher value: {final_positive}/{len(final)} seeds have positive mean teacher margin."
        )
    if not validation.empty:
        val_positive = int((validation["margin_mean"] > 0.0).sum())
        lines.append(
            f"- Validation teacher value: {val_positive}/{len(validation)} seeds have positive mean teacher margin."
        )
    if not train.empty:
        train_positive = int((train["margin_mean"] > 0.0).sum())
        lines.append(
            f"- Train teacher value: {train_positive}/{len(train)} seeds have positive mean teacher margin."
        )
    if not final.empty and not validation.empty:
        merged = validation[["seed", "margin_mean"]].merge(
            final[["seed", "margin_mean"]],
            on="seed",
            suffixes=("_validation", "_final"),
        )
        if not merged.empty:
            same_sign = int(
                np.sum(
                    np.sign(merged["margin_mean_validation"].to_numpy(dtype=float))
                    == np.sign(merged["margin_mean_final"].to_numpy(dtype=float))
                )
            )
            lines.append(
                f"- Validation/final sign agreement: {same_sign}/{len(merged)} seeds at mean-margin level."
            )
    lines.append("")
    lines.append("## Artifact Availability")
    lines.append("")
    lines.append(markdown_table(availability))
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    body = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in body)) for idx in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not np.isfinite(out):
            return "nan"
        return f"{out:.6g}"
    return str(value)


def parse_preset_seed(run_name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", run_name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    return run_name, int(manifest.get("seed", -1))


def split_seed_offset(split: str) -> int:
    return {
        "train": 210_000,
        "validation": 220_000,
        "final_test": 230_000,
    }[str(split)]


def safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


if __name__ == "__main__":
    main()
