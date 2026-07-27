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
        description="Audit validation-to-final regime transfer for selected v1 deployables."
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
        raise SystemExit("--out-dir is required when auditing multiple suite roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(roots)
    if not rows:
        raise FileNotFoundError(f"No completed regime-transfer rows found under {roots}")
    table = pd.DataFrame(rows).sort_values(["root", "preset", "seed"])
    table.to_csv(out_dir / "regime_transfer_rows.csv", index=False)

    selected = table.loc[table["selected_policy"].ne("STATIC_FALLBACK")].copy()
    selected.to_csv(out_dir / "regime_transfer_selected.csv", index=False)

    summary = summarize(table, by=("selected_family",))
    summary.to_csv(out_dir / "regime_transfer_summary.csv", index=False)

    report = render_report(table, selected, summary)
    (out_dir / "regime_transfer_audit.md").write_text(report, encoding="utf-8")
    print(report)


def collect_rows(roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            gate_path = run_dir / "gate_summary.json"
            if not gate_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            preset, seed = parse_preset_seed(run_dir.name, manifest)
            selected_policy = str(gate.get("best_deployable_policy") or "STATIC_FALLBACK")
            static_obj = safe_float(gate.get("validation_selected_static_objective"))
            deploy_obj = safe_float(gate.get("best_deployable_objective"))
            teacher_obj = safe_float(gate.get("teacher_reference_objective"))
            final_margin = 0.0 if selected_policy == "STATIC_FALLBACK" else static_obj - deploy_obj
            validation_row = selected_validation_row(manifest, selected_policy)
            event_row = calibration_row(manifest, "event_threshold_policy")
            contextual_row = calibration_row(manifest, "contextual_duty_policy")
            train_diag = start_diag(manifest, "train")
            val_diag = start_diag(manifest, "validation")
            final_diag = start_diag(manifest, "final_test")
            selected_static = manifest.get("selected_static", {})
            if not isinstance(selected_static, dict):
                selected_static = {}
            rows.append(
                {
                    "root": str(root),
                    "preset": preset,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "selected_policy": selected_policy,
                    "selected_family": policy_family(selected_policy),
                    "static_action_idx": safe_int(selected_static.get("action_idx")),
                    "static_sensor_ids": str(selected_static.get("sensor_ids", "")),
                    "final_margin": float(final_margin),
                    "final_win": bool(final_margin > 0.0),
                    "teacher_margin": float(static_obj - teacher_obj),
                    "teacher_win": bool(gate.get("teacher_beats_static", False)),
                    "train_event_mean": safe_float(train_diag.get("mean")),
                    "validation_event_mean": safe_float(val_diag.get("mean")),
                    "validation_event_min": safe_float(val_diag.get("min")),
                    "validation_event_max": safe_float(val_diag.get("max")),
                    "validation_event_std": safe_float(val_diag.get("std")),
                    "final_event_mean": safe_float(final_diag.get("mean")),
                    "final_event_min": safe_float(final_diag.get("min")),
                    "final_event_max": safe_float(final_diag.get("max")),
                    "final_event_std": safe_float(final_diag.get("std")),
                    "final_minus_validation_event": safe_float(final_diag.get("mean")) - safe_float(val_diag.get("mean")),
                    "selected_validation_margin_mean": safe_float(validation_row.get("objective_margin_mean")),
                    "selected_validation_margin_median": safe_float(validation_row.get("objective_margin_median")),
                    "selected_validation_margin_q25": safe_float(validation_row.get("objective_margin_q25")),
                    "selected_validation_margin_min": safe_float(validation_row.get("objective_margin_min")),
                    "selected_validation_negative_starts": safe_int(validation_row.get("negative_start_count")),
                    "selected_validation_guard_pass": bool(validation_row.get("static_margin_guard_pass", False)),
                    "selected_validation_positive_center": bool(
                        validation_row.get("static_margin_positive_center", False)
                    ),
                    "event_action_idx": safe_int(event_row.get("action_idx")),
                    "event_threshold": safe_float(event_row.get("threshold")),
                    "event_aggregation": str(event_row.get("aggregation", "")),
                    "event_validation_margin_mean": safe_float(event_row.get("objective_margin_mean")),
                    "event_validation_margin_q25": safe_float(event_row.get("objective_margin_q25")),
                    "event_validation_negative_starts": safe_int(event_row.get("negative_start_count")),
                    "contextual_blend": safe_float(contextual_row.get("blend")),
                    "contextual_deficit_weight": safe_float(contextual_row.get("deficit_weight")),
                    "contextual_freshness_weight": safe_float(contextual_row.get("freshness_weight")),
                    "contextual_power_weight": safe_float(contextual_row.get("power_weight")),
                    "contextual_validation_margin_mean": safe_float(contextual_row.get("objective_margin_mean")),
                    "contextual_validation_margin_q25": safe_float(contextual_row.get("objective_margin_q25")),
                    "contextual_validation_negative_starts": safe_int(contextual_row.get("negative_start_count")),
                }
            )
    return rows


def summarize(table: pd.DataFrame, *, by: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in table.groupby(list(by), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(by, key, strict=True))
        wins = int(group["final_win"].astype(bool).sum())
        rows.append(
            {
                **values,
                "n": int(len(group)),
                "wins": wins,
                "win_rate": wins / len(group) if len(group) else np.nan,
                "final_margin_mean": nanmean(group["final_margin"]),
                "event_shift_mean": nanmean(group["final_minus_validation_event"]),
                "validation_margin_mean": nanmean(group["selected_validation_margin_mean"]),
                "validation_q25_mean": nanmean(group["selected_validation_margin_q25"]),
                "teacher_win_rate": float(group["teacher_win"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(list(by))


def render_report(table: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = ["# Regime Transfer Audit", ""]
    lines.extend(["## Selected/Fallback Rows", ""])
    display_cols = [
        "root",
        "preset",
        "seed",
        "selected_policy",
        "static_action_idx",
        "validation_event_mean",
        "final_event_mean",
        "final_minus_validation_event",
        "selected_validation_margin_mean",
        "selected_validation_margin_q25",
        "selected_validation_negative_starts",
        "final_margin",
        "final_win",
    ]
    lines.append(markdown_table(table[display_cols]))
    lines.extend(["", "## Selected Deployables Only", ""])
    if selected.empty:
        lines.append("No selected deployables with final metrics.")
    else:
        lines.append(markdown_table(selected[display_cols]))
    lines.extend(["", "## Summary By Selected Family", ""])
    lines.append(markdown_table(summary))
    lines.append("")
    return "\n".join(lines)


def selected_validation_row(manifest: dict[str, object], selected_policy: str) -> dict[str, object]:
    if selected_policy == "STATIC_FALLBACK":
        return {}
    selection = manifest.get("deployable_selection", {})
    if not isinstance(selection, dict):
        return {}
    rows = selection.get("validation_rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("policy", "")) == selected_policy:
            return row
    return {}


def calibration_row(manifest: dict[str, object], key: str) -> dict[str, object]:
    block = manifest.get(key, {})
    if not isinstance(block, dict):
        return {}
    row = block.get("calibration_row", {})
    return row if isinstance(row, dict) else {}


def start_diag(manifest: dict[str, object], split: str) -> dict[str, float]:
    starts = manifest.get("starts", {})
    if not isinstance(starts, dict):
        return {}
    block = starts.get(split, {})
    if not isinstance(block, dict):
        return {}
    diag = block.get("diagnostics", {})
    if not isinstance(diag, dict):
        return {}
    rates = diag.get("selected_event_rates", [])
    if not isinstance(rates, list) or not rates:
        mean = safe_float(diag.get("selected_event_rate_mean"))
        return {"mean": mean, "min": mean, "max": mean, "std": 0.0}
    arr = np.asarray([safe_float(x) for x in rates], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def policy_family(policy: str) -> str:
    if policy == "STATIC_FALLBACK":
        return "static_fallback"
    if "contextual_duty" in policy:
        return "contextual_duty"
    if "event_threshold" in policy:
        return "event_threshold"
    if "value_residual" in policy:
        return "value_residual"
    return policy


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


def nanmean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


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


if __name__ == "__main__":
    main()
