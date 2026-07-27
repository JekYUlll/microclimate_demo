#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.protocol import final_objective, rich_metrics, task_focus_metrics  # noqa: E402
from v2.rollout import RolloutResult  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final-test margins by start window.")
    parser.add_argument("suite_roots", nargs="+")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.suite_roots]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif len(roots) == 1:
        out_dir = roots[0] / "aggregate"
    else:
        raise SystemExit("--out-dir is required when auditing multiple suite roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            rows.extend(audit_run(manifest_path, suite_root=root))
    if not rows:
        raise FileNotFoundError(f"No auditable completed runs under {roots}")
    table = pd.DataFrame(rows).sort_values(["root", "preset", "seed", "start_rank"])
    table.to_csv(out_dir / "start_transfer_rows.csv", index=False)

    summary = (
        table.groupby(["root", "preset", "policy"], sort=True)
        .agg(
            n_starts=("final_margin", "size"),
            start_wins=("final_win", "sum"),
            final_margin_mean=("final_margin", "mean"),
            final_margin_median=("final_margin", "median"),
            final_margin_min=("final_margin", "min"),
            event_rate_mean=("event_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "start_transfer_summary.csv", index=False)
    markdown = render_markdown(table, summary)
    (out_dir / "start_transfer_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def audit_run(manifest_path: Path, *, suite_root: Path) -> list[dict[str, object]]:
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate_summary = manifest.get("gate_summary", {})
    if not isinstance(gate_summary, dict):
        summary_path = run_dir / "gate_summary.json"
        if summary_path.exists():
            gate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            return []
    policy_name = str(gate_summary.get("best_deployable_policy") or "")
    if not policy_name:
        return []
    static_path = run_dir / "rollout_validation_selected_static.npz"
    policy_path = run_dir / f"rollout_{policy_name}.npz"
    if not static_path.exists() or not policy_path.exists():
        return []

    static = load_rollout(static_path, policy_name="validation_selected_static")
    deployable = load_rollout(policy_path, policy_name=policy_name)
    sensor_ids = tuple(str(x) for x in np.load(static_path, allow_pickle=True)["sensor_ids"])
    state_columns = tuple(str(x) for x in np.load(static_path, allow_pickle=True)["state_columns"])
    run_args = manifest.get("run_args", {})
    if not isinstance(run_args, dict):
        run_args = {}
    starts_info = manifest.get("starts", {}).get("final_test", {}) if isinstance(manifest.get("starts"), dict) else {}
    starts = list(starts_info.get("starts", [])) if isinstance(starts_info, dict) else []
    eval_steps = int(run_args.get("eval_steps") or infer_eval_steps(static, starts))
    if eval_steps <= 0:
        return []

    objective_mode = str(manifest.get("objective_mode", "oracle"))
    task_error_weight = float(manifest.get("task_error_weight", 0.0))
    task_error_columns = tuple(str(x) for x in manifest.get("task_error_columns", []))
    task_error_scales_raw = manifest.get("task_error_scales", None)
    task_error_scales = (
        tuple(float(x) for x in task_error_scales_raw)
        if isinstance(task_error_scales_raw, list)
        else None
    )
    task_error_event_only = bool(manifest.get("task_error_event_only", True))
    budget = float(run_args.get("budget", np.nan))
    startup_peak_budget = float(run_args.get("startup_peak_budget", np.nan))

    rows: list[dict[str, object]] = []
    n_segments = min(len(starts) if starts else static.observations.shape[0] // eval_steps, deployable.observations.shape[0] // eval_steps)
    preset, seed = parse_preset_seed(run_dir.name, manifest)
    for rank in range(int(n_segments)):
        start = int(starts[rank]) if rank < len(starts) else int(static.step_indices[rank * eval_steps])
        static_seg = slice_rollout(static, rank * eval_steps, (rank + 1) * eval_steps)
        deploy_seg = slice_rollout(deployable, rank * eval_steps, (rank + 1) * eval_steps)
        static_obj = objective_for_segment(
            static_seg,
            sensor_ids=sensor_ids,
            state_columns=state_columns,
            budget=budget,
            startup_peak_budget=startup_peak_budget,
            objective_mode=objective_mode,
            task_error_weight=task_error_weight,
            task_error_columns=task_error_columns,
            task_error_scales=task_error_scales,
            task_error_event_only=task_error_event_only,
        )
        deploy_obj = objective_for_segment(
            deploy_seg,
            sensor_ids=sensor_ids,
            state_columns=state_columns,
            budget=budget,
            startup_peak_budget=startup_peak_budget,
            objective_mode=objective_mode,
            task_error_weight=task_error_weight,
            task_error_columns=task_error_columns,
            task_error_scales=task_error_scales,
            task_error_event_only=task_error_event_only,
        )
        margin = float(static_obj - deploy_obj)
        rows.append(
            {
                "root": str(suite_root),
                "preset": preset,
                "seed": seed,
                "run_dir": str(run_dir),
                "policy": policy_name,
                "start_rank": rank,
                "start_idx": start,
                "static_objective": float(static_obj),
                "deployable_objective": float(deploy_obj),
                "final_margin": margin,
                "final_win": bool(margin > 0.0),
                "event_rate": float(np.mean(deploy_seg.event_flags.astype(bool))),
                "deployable_power_mean": float(np.mean(deploy_seg.powers)),
                "static_power_mean": float(np.mean(static_seg.powers)),
            }
        )
    return rows


def objective_for_segment(
    result: RolloutResult,
    *,
    sensor_ids: tuple[str, ...],
    state_columns: tuple[str, ...],
    budget: float,
    startup_peak_budget: float,
    objective_mode: str,
    task_error_weight: float,
    task_error_columns: tuple[str, ...],
    task_error_scales: tuple[float, ...] | None,
    task_error_event_only: bool,
) -> float:
    metrics = rich_metrics(
        result,
        sensor_ids=sensor_ids,
        state_columns=state_columns,
        per_step_budget=budget if np.isfinite(budget) else None,
        startup_peak_budget=startup_peak_budget if np.isfinite(startup_peak_budget) else None,
    )
    metrics.update(
        task_focus_metrics(
            result,
            state_columns=state_columns,
            task_error_columns=task_error_columns,
            task_error_scales=task_error_scales,
            event_only=bool(task_error_event_only),
        )
    )
    return final_objective(metrics, mode=objective_mode, task_error_weight=task_error_weight)


def load_rollout(path: Path, *, policy_name: str) -> RolloutResult:
    data = np.load(path, allow_pickle=True)
    return RolloutResult(
        policy_name=policy_name,
        observations=np.asarray(data["observations"], dtype=float),
        masks=np.asarray(data["masks"], dtype=float),
        truth=np.asarray(data["truth"], dtype=float),
        rewards=np.asarray(data["rewards"], dtype=float),
        scores=np.asarray(data["scores"], dtype=float),
        powers=np.asarray(data["powers"], dtype=float),
        peaks=np.asarray(data["peaks"], dtype=float),
        selected_masks=np.asarray(data["selected_masks"], dtype=int),
        mode_ids=np.asarray(data["mode_ids"], dtype=int),
        event_flags=np.asarray(data["event_flags"], dtype=float),
        oracle_losses=np.asarray(data["oracle_losses"], dtype=float),
        step_indices=np.asarray(data["step_indices"], dtype=int),
        warmup_abort_count=int(np.asarray(data["warmup_abort_count"]).reshape(-1)[0]),
        warmup_abort_deltas=np.asarray(data["warmup_abort_deltas"], dtype=int),
        energy_guard_dropped=np.asarray(data["energy_guard_dropped"], dtype=int),
        soc=np.asarray(data["soc"], dtype=float),
    )


def slice_rollout(result: RolloutResult, start: int, end: int) -> RolloutResult:
    end = min(int(end), int(result.observations.shape[0]))
    start = min(int(start), end)
    return RolloutResult(
        policy_name=result.policy_name,
        observations=result.observations[start:end],
        masks=result.masks[start:end],
        truth=result.truth[start:end],
        rewards=result.rewards[start:end],
        scores=result.scores[start:end],
        powers=result.powers[start:end],
        peaks=result.peaks[start:end],
        selected_masks=result.selected_masks[start:end],
        mode_ids=result.mode_ids[start:end],
        event_flags=result.event_flags[start:end],
        oracle_losses=result.oracle_losses[start:end],
        step_indices=result.step_indices[start:end],
        warmup_abort_count=int(np.sum(result.warmup_abort_deltas[start:end])),
        warmup_abort_deltas=result.warmup_abort_deltas[start:end],
        energy_guard_dropped=result.energy_guard_dropped[start:end],
        soc=result.soc[start:end],
    )


def infer_eval_steps(result: RolloutResult, starts: list[int]) -> int:
    if len(starts) > 0:
        return int(result.observations.shape[0] // len(starts))
    return int(result.observations.shape[0])


def parse_preset_seed(run_name: str, manifest: dict[str, object]) -> tuple[str, int]:
    seed = int(manifest.get("seed", -1))
    marker = "_seed"
    if marker in run_name:
        preset, seed_text = run_name.rsplit(marker, 1)
        try:
            seed = int(seed_text)
        except ValueError:
            pass
        return preset, seed
    return run_name, seed


def render_markdown(table: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = ["# Start-Level Transfer Audit", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    display_cols = [
        "root",
        "preset",
        "seed",
        "policy",
        "start_rank",
        "start_idx",
        "final_margin",
        "final_win",
        "event_rate",
    ]
    lines.append(markdown_table(table[display_cols]))
    lines.append("")
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = frame.to_dict(orient="records")
    widths = [len(column) for column in columns]
    formatted_rows: list[list[str]] = []
    for row in rows:
        formatted: list[str] = []
        for column in columns:
            value = row.get(column, "")
            text = format_cell(value)
            formatted.append(text)
        formatted_rows.append(formatted)
        widths = [max(width, len(text)) for width, text in zip(widths, formatted)]
    header = "| " + " | ".join(column.ljust(width) for column, width in zip(columns, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(text.ljust(width) for text, width in zip(row, widths)) + " |"
        for row in formatted_rows
    ]
    return "\n".join([header, sep, *body])


def format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6g}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


if __name__ == "__main__":
    main()
