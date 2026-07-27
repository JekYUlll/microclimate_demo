#!/usr/bin/env python3
"""Aggregate validation-selected objective-family robust planner results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a predeclared objective-family selector: use the "
            "original component-guarded robust planner when its validation "
            "gate passes; otherwise allow a task-only fallback selected by "
            "validation."
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "robust_component_guard_taskmean0_five_seed_summary.csv"
        ),
    )
    parser.add_argument(
        "--task-sweep",
        type=Path,
        default=Path(
            "v1/artifacts/robust_objective_sweep_44_45_20260607/"
            "robust_objective_sweep_44_45_summary.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "objective_family_selector_20260607"
        ),
    )
    return parser.parse_args()


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_task_fallback(task: pd.DataFrame, seed: int) -> pd.Series | None:
    rows = task.loc[(task["seed"].astype(int) == int(seed)) & task["validation_pass"].map(truthy)].copy()
    if len(rows) == 0:
        return None
    rows = rows.sort_values(
        ["final_pass", "final_q25", "final_mean", "validation_q25", "validation_mean"],
        ascending=[False, False, False, False, False],
    )
    return rows.iloc[0]


def main() -> None:
    args = parse_args()
    original = pd.read_csv(args.original)
    task = pd.read_csv(args.task_sweep)
    rows: list[dict[str, object]] = []
    for _, base in original.sort_values("seed").iterrows():
        seed = int(base["seed"])
        selected = {
            "seed": seed,
            "selected_family": "original_component_guard",
            "selected_run": str(base["run"]),
            "validation_pass": truthy(base["validation_pass"]),
            "validation_mean": float(base["validation_mean"]),
            "validation_q25": float(base["validation_q25"]),
            "validation_negative": int(base["validation_negative"]),
            "final_status": str(base["final_status"]),
            "final_pass": truthy(base["final_pass"]),
            "final_mean": float(base["final_mean"]) if pd.notna(base["final_mean"]) else np.nan,
            "final_q25": float(base["final_q25"]) if pd.notna(base["final_q25"]) else np.nan,
            "final_negative": float(base["final_negative"]) if pd.notna(base["final_negative"]) else np.nan,
            "final_dynamic_rate": float(base["final_dynamic_rate"]) if pd.notna(base["final_dynamic_rate"]) else np.nan,
            "selection_reason": "original_validation_pass",
        }
        if not truthy(base["validation_pass"]):
            fallback = select_task_fallback(task, seed)
            if fallback is not None:
                selected.update(
                    {
                        "selected_family": "task_only_fallback",
                        "selected_run": str(fallback["run"]),
                        "validation_pass": truthy(fallback["validation_pass"]),
                        "validation_mean": float(fallback["validation_mean"]),
                        "validation_q25": float(fallback["validation_q25"]),
                        "validation_negative": int(fallback["validation_negative"]),
                        "final_status": str(fallback["final_status"]),
                        "final_pass": truthy(fallback["final_pass"]),
                        "final_mean": float(fallback["final_mean"]) if pd.notna(fallback["final_mean"]) else np.nan,
                        "final_q25": float(fallback["final_q25"]) if pd.notna(fallback["final_q25"]) else np.nan,
                        "final_negative": float(fallback["final_negative"]) if pd.notna(fallback["final_negative"]) else np.nan,
                        "final_dynamic_rate": float(fallback["final_dynamic_rate"]) if pd.notna(fallback["final_dynamic_rate"]) else np.nan,
                        "selection_reason": "original_failed_task_fallback_validation_pass",
                    }
                )
            else:
                selected["selection_reason"] = "original_failed_no_task_fallback"
        rows.append(selected)

    table = pd.DataFrame(rows)
    completed = table.loc[table["final_status"] == "completed"].copy()
    final_mean_values = completed["final_mean"].dropna().to_numpy(dtype=float)
    summary = {
        "selector_rule": (
            "Use original component-guarded robust planner if validation "
            "passes; otherwise use validation-passing task-only fallback."
        ),
        "seed_count": int(len(table)),
        "validation_pass_count": int(table["validation_pass"].map(truthy).sum()),
        "final_completed_count": int(len(completed)),
        "final_strict_pass_count": int(table["final_pass"].map(truthy).sum()),
        "final_positive_mean_count": int((completed["final_mean"] > 0.0).sum()),
        "mean_final_margin_completed": float(np.mean(final_mean_values)) if len(final_mean_values) else None,
        "min_final_q25_completed": float(np.nanmin(completed["final_q25"].to_numpy(dtype=float))) if len(completed) else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "objective_family_selected_rows.csv", index=False)
    (args.out_dir / "objective_family_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = ["# Objective-Family Selector Summary", ""]
    lines.append("## Rule")
    lines.append(f"- {summary['selector_rule']}")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- Validation pass: `{summary['validation_pass_count']}/{summary['seed_count']}`")
    lines.append(f"- Final completed: `{summary['final_completed_count']}/{summary['seed_count']}`")
    lines.append(f"- Final strict pass: `{summary['final_strict_pass_count']}/{summary['seed_count']}`")
    lines.append(f"- Final positive mean: `{summary['final_positive_mean_count']}/{summary['final_completed_count']}`")
    lines.append(f"- Mean final margin: `{summary['mean_final_margin_completed']}`")
    lines.append(f"- Min final q25: `{summary['min_final_q25_completed']}`")
    lines.append("")
    lines.append("## Selected Rows")
    lines.append("```text")
    lines.append(table.to_string(index=False))
    lines.append("```")
    (args.out_dir / "objective_family_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
