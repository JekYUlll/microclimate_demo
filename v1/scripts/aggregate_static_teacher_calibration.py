#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECT_FULL = {"met_station_core", "laser_disdrometer", "fc4_flux"}
DIRECT_SNOW = {"laser_disdrometer", "fc4_flux"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v1 static/teacher scenario-calibration gates.")
    parser.add_argument("--root", required=True, help="Root containing per-seed calibration_summary.json files.")
    parser.add_argument("--out-dir", default=None, help="Defaults to <root>/aggregate.")
    parser.add_argument("--require-all", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_run(path) for path in sorted(root.glob("*/calibration_summary.json"))]
    if not rows:
        raise FileNotFoundError(f"No calibration_summary.json files found under {root}")
    df = pd.DataFrame(rows).sort_values(["selection", "seed"]).reset_index(drop=True)
    summary = build_summary(df, require_all=bool(args.require_all))
    df.to_csv(out_dir / "static_teacher_calibration_rows.csv", index=False)
    (out_dir / "static_teacher_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = render_report(df, summary)
    (out_dir / "static_teacher_calibration_summary.md").write_text(report, encoding="utf-8")
    print(report)


def load_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    static_sensors = split_sensors(str(data.get("selected_static_sensor_ids", "")))
    teacher_npz = load_rollout_stats(run_dir / "rollout_mpc_teacher.npz")
    static_npz = load_rollout_stats(run_dir / "rollout_validation_selected_static.npz")
    raw_static_direct_full = DIRECT_FULL.issubset(static_sensors)
    raw_static_direct_snow = DIRECT_SNOW.issubset(static_sensors)
    static_direct_full_duty = as_float(static_npz.get("direct_full_duty"))
    static_direct_snow_duty = as_float(static_npz.get("direct_snow_duty"))
    teacher_beats = bool(data.get("teacher_beats_static", False))
    teacher_switches = int(teacher_npz.get("unique_masks", 0)) > 1
    seed_gate_pass = (
        teacher_beats
        and static_direct_full_duty <= 1.0e-9
        and static_direct_snow_duty <= 1.0e-9
        and teacher_switches
    )
    return {
        "run_dir": str(run_dir),
        "seed": int(data["seed"]),
        "selection": str(data.get("selection", "")),
        "selected_static_action_idx": int(data.get("selected_static_action_idx", -1)),
        "selected_static_sensor_ids": "|".join(sorted(static_sensors)),
        "static_objective": as_float(data.get("static_objective")),
        "teacher_objective": as_float(data.get("teacher_objective")),
        "teacher_margin": as_float(data.get("teacher_margin")),
        "teacher_beats_static": teacher_beats,
        "raw_static_direct_full": bool(raw_static_direct_full),
        "raw_static_direct_snow": bool(raw_static_direct_snow),
        "static_direct_full_duty": static_direct_full_duty,
        "static_direct_snow_duty": static_direct_snow_duty,
        "static_laser_duty": as_float(static_npz.get("laser_duty")),
        "static_fc4_duty": as_float(static_npz.get("fc4_duty")),
        "static_spc_duty": as_float(static_npz.get("spc_duty")),
        "teacher_laser_duty": as_float(teacher_npz.get("laser_duty")),
        "teacher_fc4_duty": as_float(teacher_npz.get("fc4_duty")),
        "teacher_spc_duty": as_float(teacher_npz.get("spc_duty")),
        "teacher_unique_masks": int(teacher_npz.get("unique_masks", 0)),
        "teacher_nontrivial_switch": bool(teacher_switches),
        "validation_event_rate_mean": as_float(data.get("validation_event_rate_mean")),
        "final_event_rate_mean": as_float(data.get("final_event_rate_mean")),
        "static_power_mean": as_float(data.get("static_power_mean")),
        "teacher_power_mean": as_float(data.get("teacher_power_mean")),
        "static_soc_mean": nullable_float(data.get("static_soc_mean")),
        "teacher_soc_mean": nullable_float(data.get("teacher_soc_mean")),
        "seed_gate_pass": bool(seed_gate_pass),
    }


def split_sensors(value: str) -> set[str]:
    return {item for item in value.split("|") if item}


def load_rollout_stats(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    masks = np.asarray(data["selected_masks"], dtype=int)
    sensor_ids = [str(x) for x in np.asarray(data["sensor_ids"]).tolist()]
    stats: dict[str, float | int] = {"unique_masks": int(np.unique(masks, axis=0).shape[0])}
    index = {sensor_id: pos for pos, sensor_id in enumerate(sensor_ids)}
    for key, sensor_id in (
        ("laser_duty", "laser_disdrometer"),
        ("fc4_duty", "fc4_flux"),
        ("spc_duty", "snow_particle_counter"),
    ):
        if sensor_id in index:
            idx = index[sensor_id]
            stats[key] = float(np.mean(masks[:, idx])) if masks.size else 0.0
        else:
            stats[key] = 0.0
    stats["direct_full_duty"] = stack_duty(masks, index, DIRECT_FULL)
    stats["direct_snow_duty"] = stack_duty(masks, index, DIRECT_SNOW)
    return stats


def build_summary(df: pd.DataFrame, *, require_all: bool) -> dict[str, Any]:
    n = int(len(df))
    pass_count = int(df["seed_gate_pass"].sum())
    event_rich = bool((df["selection"] == "event_rich").any())
    event_rich_mechanism_pass = (not event_rich) or int((df["teacher_laser_duty"] > 0.0).sum()) > 0
    summary = {
        "n": n,
        "seed_gate_passes": pass_count,
        "teacher_wins": int(df["teacher_beats_static"].sum()),
        "raw_static_direct_full_count": int(df["raw_static_direct_full"].sum()),
        "raw_static_direct_snow_count": int(df["raw_static_direct_snow"].sum()),
        "static_direct_full_execution_count": int((df["static_direct_full_duty"] > 0.0).sum()),
        "static_direct_snow_execution_count": int((df["static_direct_snow_duty"] > 0.0).sum()),
        "teacher_laser_use_count": int((df["teacher_laser_duty"] > 0.0).sum()),
        "teacher_nontrivial_switch_count": int(df["teacher_nontrivial_switch"].sum()),
        "teacher_margin_mean": finite_mean(df["teacher_margin"]),
        "teacher_margin_min": finite_min(df["teacher_margin"]),
        "teacher_laser_duty_mean": finite_mean(df["teacher_laser_duty"]),
        "static_laser_duty_mean": finite_mean(df["static_laser_duty"]),
        "final_event_rate_mean": finite_mean(df["final_event_rate_mean"]),
        "require_all": bool(require_all),
        "event_rich_mechanism_pass": bool(event_rich_mechanism_pass),
    }
    per_seed_pass = pass_count == n if require_all else pass_count >= max(1, n - 1)
    summary["calibration_gate_pass"] = bool(per_seed_pass and event_rich_mechanism_pass)
    return summary


def render_report(df: pd.DataFrame, summary: dict[str, Any]) -> str:
    columns = [
        "seed",
        "selection",
        "seed_gate_pass",
        "teacher_margin",
        "selected_static_sensor_ids",
        "raw_static_direct_full",
        "static_direct_full_duty",
        "static_direct_snow_duty",
        "teacher_laser_duty",
        "teacher_fc4_duty",
        "teacher_spc_duty",
        "teacher_unique_masks",
        "final_event_rate_mean",
    ]
    return "\n".join(
        [
            "# Static/Teacher Scenario Calibration",
            "",
            "## Summary",
            "",
            markdown_table(pd.DataFrame([summary])),
            "",
            "## Per-Seed Rows",
            "",
            markdown_table(df[columns]),
            "",
            "## Gate Definition",
            "",
            "- Per-seed pass: teacher beats validation-selected static; the executed static rollout never contains the old direct laser+fc4 stack; teacher uses more than one mask.",
            "- Aggregate pass: all per-seed rows pass by default; event-rich suites must show selective teacher laser use in at least one seed.",
            "",
        ]
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def as_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def nullable_float(value: object) -> float | None:
    val = as_float(value)
    return val if np.isfinite(val) else None


def stack_duty(masks: np.ndarray, index: dict[str, int], sensors: set[str]) -> float:
    if not sensors.issubset(index) or masks.size == 0:
        return 0.0
    cols = [index[sensor_id] for sensor_id in sensors]
    return float(np.mean(np.all(masks[:, cols] > 0, axis=1)))


def finite_mean(values: pd.Series) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def finite_min(values: pd.Series) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else None


if __name__ == "__main__":
    main()
