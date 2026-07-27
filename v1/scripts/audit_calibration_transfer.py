#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit event-threshold calibration rows against final-test margins."
    )
    parser.add_argument("suite_roots", nargs="+", help="One or more claim-suite roots.")
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
        raise SystemExit("--out-dir is required when auditing multiple roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(roots)
    if not rows:
        raise FileNotFoundError(f"No calibration-transfer rows found under {roots}")
    table = pd.DataFrame(rows).sort_values(["root", "preset", "seed"])
    table.to_csv(out_dir / "calibration_transfer_rows.csv", index=False)

    summary = summarize(table)
    summary.to_csv(out_dir / "calibration_transfer_summary.csv", index=False)

    report = render_report(table, summary)
    (out_dir / "calibration_transfer_audit.md").write_text(report, encoding="utf-8")
    print(report)


def collect_rows(roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            metrics_path = run_dir / "metrics_final.csv"
            if not metrics_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            event_policy = manifest.get("event_threshold_policy", {})
            if not isinstance(event_policy, dict):
                continue
            calibration_row = event_policy.get("calibration_row")
            if not isinstance(calibration_row, dict):
                continue
            metrics = pd.read_csv(metrics_path)
            if "policy" not in metrics.columns or "objective_loss_mean" not in metrics.columns:
                continue
            final_by_policy = {
                str(row["policy"]): row
                for row in metrics.to_dict(orient="records")
                if isinstance(row, dict)
            }
            static_final = final_by_policy.get("validation_selected_static")
            event_final = final_by_policy.get("forecast_aware_event_threshold")
            if static_final is None or event_final is None:
                continue
            static_objective = safe_float(static_final.get("objective_loss_mean"))
            event_objective = safe_float(event_final.get("objective_loss_mean"))
            margins = margin_values(calibration_row)
            preset, seed = parse_preset_seed(run_dir.name, manifest)
            deployable_selection = manifest.get("deployable_selection", {})
            rows.append(
                {
                    "root": str(root),
                    "preset": preset,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "event_calibration_criterion": str(event_policy.get("calibration_criterion", "")),
                    "deployable_selection_criterion": str(deployable_selection.get("criterion", ""))
                    if isinstance(deployable_selection, dict)
                    else "",
                    "action_idx": safe_int(calibration_row.get("action_idx")),
                    "threshold": safe_float(calibration_row.get("threshold")),
                    "aggregation": str(calibration_row.get("aggregation", "")),
                    "validation_objective": safe_float(calibration_row.get("objective")),
                    "validation_margin_mean": safe_float(calibration_row.get("objective_margin_mean")),
                    "validation_margin_median": quantile(margins, 0.5),
                    "validation_margin_q25": quantile(margins, 0.25),
                    "validation_margin_min": safe_float(calibration_row.get("objective_margin_min")),
                    "validation_negative_starts": safe_int(calibration_row.get("negative_start_count")),
                    "validation_guard_pass": bool(calibration_row.get("static_margin_guard_pass", False)),
                    "static_final_objective": static_objective,
                    "event_final_objective": event_objective,
                    "final_margin": static_objective - event_objective,
                    "final_win": bool(static_objective > event_objective),
                    "transfer_gap": (static_objective - event_objective)
                    - safe_float(calibration_row.get("objective_margin_mean")),
                }
            )
    return rows


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for preset, group in table.groupby("preset", sort=True):
        final_wins = int(group["final_win"].astype(bool).sum())
        rows.append(
            {
                "preset": preset,
                "n": int(len(group)),
                "final_wins": final_wins,
                "final_win_rate": final_wins / len(group) if len(group) else np.nan,
                "validation_guard_pass": int(group["validation_guard_pass"].astype(bool).sum()),
                "validation_margin_mean": float(np.nanmean(group["validation_margin_mean"].to_numpy(dtype=float))),
                "validation_margin_median_mean": float(
                    np.nanmean(group["validation_margin_median"].to_numpy(dtype=float))
                ),
                "validation_negative_starts_mean": float(
                    np.nanmean(group["validation_negative_starts"].to_numpy(dtype=float))
                ),
                "final_margin_mean": float(np.nanmean(group["final_margin"].to_numpy(dtype=float))),
                "transfer_gap_mean": float(np.nanmean(group["transfer_gap"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values("preset")


def render_report(table: pd.DataFrame, summary: pd.DataFrame) -> str:
    display = table[
        [
            "preset",
            "seed",
            "event_calibration_criterion",
            "action_idx",
            "threshold",
            "aggregation",
            "validation_margin_mean",
            "validation_margin_median",
            "validation_margin_q25",
            "validation_negative_starts",
            "validation_guard_pass",
            "final_margin",
            "final_win",
        ]
    ]
    lines = [
        "# Calibration Transfer Audit",
        "",
        "## Selected Calibration Rows",
        "",
        markdown_table(display),
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
    ]
    return "\n".join(lines)


def margin_values(row: dict[str, object]) -> list[float]:
    static_values = row.get("static_start_objectives")
    candidate_values = row.get("candidate_start_objectives")
    if not isinstance(static_values, list) or not isinstance(candidate_values, list):
        return []
    out: list[float] = []
    for static, candidate in zip(static_values, candidate_values):
        margin = safe_float(static) - safe_float(candidate)
        if np.isfinite(margin):
            out.append(float(margin))
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), float(q)))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    body = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in body))
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, np.floating):
        if np.isnan(float(value)):
            return "nan"
        return f"{float(value):.6g}"
    return str(value)


def parse_preset_seed(name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    return name, safe_int(manifest.get("seed"))


def safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


if __name__ == "__main__":
    main()
