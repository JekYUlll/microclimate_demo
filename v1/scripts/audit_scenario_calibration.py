#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SCENARIOS = (
    {
        "name": "current_v4_b1p20",
        "sensor_cfg": "rl_sensor_scheduling_framework/configs/sensors/windblown_sensors_physical_event_v4.yaml",
        "budget": 1.20,
        "startup_peak_budget": 1.60,
        "max_active": 4,
        "energy_capacity": 180.0,
        "initial_energy": 180.0,
        "harvest_per_step": 0.92,
        "reserve_energy": 20.0,
        "eval_steps": 256,
    },
    {
        "name": "v5_constraint_active_b1p20_e70",
        "sensor_cfg": "v1/configs/sensors/windblown_sensors_physical_event_v5_constraint_active.yaml",
        "budget": 1.20,
        "startup_peak_budget": 1.60,
        "max_active": 4,
        "energy_capacity": 70.0,
        "initial_energy": 70.0,
        "harvest_per_step": 0.92,
        "reserve_energy": 20.0,
        "eval_steps": 256,
    },
    {
        "name": "v5_constraint_active_b1p20_e90",
        "sensor_cfg": "v1/configs/sensors/windblown_sensors_physical_event_v5_constraint_active.yaml",
        "budget": 1.20,
        "startup_peak_budget": 1.60,
        "max_active": 4,
        "energy_capacity": 90.0,
        "initial_energy": 90.0,
        "harvest_per_step": 0.92,
        "reserve_energy": 20.0,
        "eval_steps": 256,
    },
    {
        "name": "v6_complex_static_break_b1p36_e70_h0p80",
        "sensor_cfg": "v1/configs/sensors/windblown_sensors_physical_event_v6_complex_static_break.yaml",
        "budget": 1.36,
        "startup_peak_budget": 1.75,
        "max_active": 4,
        "energy_capacity": 70.0,
        "initial_energy": 70.0,
        "harvest_per_step": 0.80,
        "reserve_energy": 20.0,
        "eval_steps": 256,
    },
)


STACKS = {
    "core_only": ("met_station_core",),
    "direct_full_core_laser_fc4": ("met_station_core", "laser_disdrometer", "fc4_flux"),
    "direct_snow_laser_fc4": ("laser_disdrometer", "fc4_flux"),
    "event_core_laser": ("met_station_core", "laser_disdrometer"),
    "proxy_core_spc_fc4": ("met_station_core", "snow_particle_counter", "fc4_flux"),
    "proxy_core_spc_fc4_surface": (
        "met_station_core",
        "snow_particle_counter",
        "fc4_flux",
        "surface_temp_ir",
    ),
    "context_core_fc4_surface": ("met_station_core", "fc4_flux", "surface_temp_ir"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structurally audit candidate v1 sensor/power scenarios before expensive reruns."
    )
    parser.add_argument("--out-dir", default="v1/artifacts/scenario_calibration_structural_20260603")
    parser.add_argument("--scenario-yaml", default=None, help="Optional YAML list of scenario dictionaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.scenario_yaml)
    rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        summary, masks = audit_scenario(scenario)
        rows.append(summary)
        mask_rows.extend(masks)
    summary_df = pd.DataFrame(rows)
    masks_df = pd.DataFrame(mask_rows)
    summary_df.to_csv(out_dir / "scenario_calibration_summary.csv", index=False)
    masks_df.to_csv(out_dir / "scenario_feasible_masks.csv", index=False)
    report = render_report(summary_df, masks_df)
    (out_dir / "scenario_calibration_audit.md").write_text(report, encoding="utf-8")
    print(report)


def load_scenarios(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return [dict(item) for item in DEFAULT_SCENARIOS]
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--scenario-yaml must contain a YAML list")
    return [dict(item) for item in data]


def audit_scenario(scenario: dict[str, Any]) -> tuple[dict[str, object], list[dict[str, object]]]:
    sensor_cfg = resolve_path(str(scenario["sensor_cfg"]))
    sensors = load_sensors(sensor_cfg)
    sensor_ids = [sensor["sensor_id"] for sensor in sensors]
    powers = np.asarray([float(sensor["power_cost"]) for sensor in sensors], dtype=float)
    peaks = np.asarray([float(sensor["startup_peak_power"]) for sensor in sensors], dtype=float)
    budget = float(scenario["budget"])
    peak_budget = float(scenario["startup_peak_budget"])
    max_active = int(scenario["max_active"])
    harvest = float(scenario["harvest_per_step"])
    initial = float(scenario["initial_energy"])
    reserve = float(scenario["reserve_energy"])
    eval_steps = int(scenario["eval_steps"])
    masks = enumerate_masks(len(sensors), max_active=max_active)
    feasible = [
        mask
        for mask in masks
        if float(np.dot(mask, powers)) <= budget + 1.0e-12
        and float(np.dot(mask, peaks)) <= peak_budget + 1.0e-12
    ]
    feasible_arr = np.asarray(feasible, dtype=int) if feasible else np.zeros((0, len(sensors)), dtype=int)

    stack_metrics: dict[str, object] = {}
    for stack_name, names in STACKS.items():
        mask = mask_for(sensor_ids, names)
        stack_power = float(np.dot(mask, powers))
        stack_peak = float(np.dot(mask, peaks))
        stack_metrics[f"{stack_name}_power"] = stack_power
        stack_metrics[f"{stack_name}_peak"] = stack_peak
        stack_metrics[f"{stack_name}_feasible"] = is_feasible(mask, powers, peaks, budget, peak_budget, max_active)
        stack_metrics[f"{stack_name}_continuous_steps"] = continuous_steps(
            power=stack_power,
            harvest=harvest,
            initial=initial,
            reserve=reserve,
        )

    event_mask = mask_for(sensor_ids, STACKS["event_core_laser"])
    proxy_mask = mask_for(sensor_ids, STACKS["proxy_core_spc_fc4"])
    event_power = float(np.dot(event_mask, powers))
    proxy_power = float(np.dot(proxy_mask, powers))
    allowed_average = harvest + max(0.0, initial - reserve) / max(eval_steps, 1)
    laser_duty_over_proxy = duty_fraction(
        event_power=event_power,
        base_power=proxy_power,
        allowed_average=allowed_average,
    )
    core_mask = mask_for(sensor_ids, STACKS["core_only"])
    core_power = float(np.dot(core_mask, powers))
    laser_duty_over_core = duty_fraction(
        event_power=event_power,
        base_power=core_power,
        allowed_average=allowed_average,
    )

    summary = {
        "scenario": str(scenario["name"]),
        "sensor_cfg": str(sensor_cfg),
        "budget": budget,
        "startup_peak_budget": peak_budget,
        "max_active": max_active,
        "energy_capacity": float(scenario["energy_capacity"]),
        "initial_energy": initial,
        "harvest_per_step": harvest,
        "reserve_energy": reserve,
        "eval_steps": eval_steps,
        "feasible_masks": int(len(feasible)),
        "feasible_with_laser": int(sum(1 for mask in feasible if mask[index_of(sensor_ids, "laser_disdrometer")])),
        "feasible_with_fc4": int(sum(1 for mask in feasible if mask[index_of(sensor_ids, "fc4_flux")])),
        "feasible_with_spc": int(sum(1 for mask in feasible if mask[index_of(sensor_ids, "snow_particle_counter")])),
        "laser_inclusion_rate": inclusion_rate(feasible_arr, sensor_ids, "laser_disdrometer"),
        "fc4_inclusion_rate": inclusion_rate(feasible_arr, sensor_ids, "fc4_flux"),
        "spc_inclusion_rate": inclusion_rate(feasible_arr, sensor_ids, "snow_particle_counter"),
        "allowed_average_power": allowed_average,
        "laser_duty_over_proxy": laser_duty_over_proxy,
        "laser_duty_over_core": laser_duty_over_core,
        **stack_metrics,
    }
    summary["structural_gate_pass"] = structural_gate(summary)
    summary["energy_gate_pass"] = energy_gate(summary, eval_steps=eval_steps)
    summary["static_anchor_gate_pass"] = static_anchor_gate(summary)
    summary["calibration_gate_pass"] = bool(summary["structural_gate_pass"] and summary["energy_gate_pass"])

    mask_rows = []
    for action_idx, mask in enumerate(feasible_arr):
        ids = [sensor_ids[idx] for idx, active in enumerate(mask) if int(active)]
        mask_rows.append(
            {
                "scenario": str(scenario["name"]),
                "action_idx": int(action_idx),
                "sensor_ids": "|".join(ids),
                "power": float(np.dot(mask, powers)),
                "peak": float(np.dot(mask, peaks)),
                "active_count": int(np.sum(mask)),
                "contains_laser": bool(mask[index_of(sensor_ids, "laser_disdrometer")]),
                "contains_fc4": bool(mask[index_of(sensor_ids, "fc4_flux")]),
                "contains_spc": bool(mask[index_of(sensor_ids, "snow_particle_counter")]),
                "contains_core": bool(mask[index_of(sensor_ids, "met_station_core")]),
            }
        )
    return summary, mask_rows


def structural_gate(row: dict[str, object]) -> bool:
    return (
        not bool(row["direct_full_core_laser_fc4_feasible"])
        and not bool(row["direct_snow_laser_fc4_feasible"])
        and bool(row["event_core_laser_feasible"])
        and bool(row["proxy_core_spc_fc4_feasible"])
    )


def energy_gate(row: dict[str, object], *, eval_steps: int) -> bool:
    return (
        finite_float(row["event_core_laser_continuous_steps"]) < float(eval_steps)
        and finite_float(row["proxy_core_spc_fc4_continuous_steps"]) >= float(eval_steps)
        and 0.35 <= finite_float(row["laser_duty_over_proxy"]) <= 0.95
    )


def static_anchor_gate(row: dict[str, object]) -> bool:
    return (
        bool(row["event_core_laser_feasible"])
        and finite_float(row["laser_duty_over_core"]) <= 0.75
        and finite_float(row["event_core_laser_continuous_steps"]) < finite_float(row["proxy_core_spc_fc4_continuous_steps"])
    )


def load_sensors(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sensors = data.get("sensors") if isinstance(data, dict) else None
    if not isinstance(sensors, list):
        raise ValueError(f"Sensor config does not contain a sensors list: {path}")
    return [dict(sensor) for sensor in sensors]


def enumerate_masks(n_sensors: int, *, max_active: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for count in range(1, min(int(max_active), int(n_sensors)) + 1):
        for combo in combinations(range(int(n_sensors)), count):
            mask = np.zeros(int(n_sensors), dtype=int)
            mask[list(combo)] = 1
            masks.append(mask)
    return masks


def mask_for(sensor_ids: list[str], names: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(len(sensor_ids), dtype=int)
    for name in names:
        mask[index_of(sensor_ids, name)] = 1
    return mask


def index_of(sensor_ids: list[str], name: str) -> int:
    try:
        return sensor_ids.index(str(name))
    except ValueError as exc:
        raise ValueError(f"Sensor {name!r} not found in {sensor_ids}") from exc


def is_feasible(
    mask: np.ndarray,
    powers: np.ndarray,
    peaks: np.ndarray,
    budget: float,
    peak_budget: float,
    max_active: int,
) -> bool:
    return (
        int(np.sum(mask)) <= int(max_active)
        and float(np.dot(mask, powers)) <= float(budget) + 1.0e-12
        and float(np.dot(mask, peaks)) <= float(peak_budget) + 1.0e-12
    )


def continuous_steps(*, power: float, harvest: float, initial: float, reserve: float) -> float:
    drain = float(power) - float(harvest)
    if drain <= 0.0:
        return float("inf")
    return max(0.0, float(initial) - float(reserve)) / drain


def duty_fraction(*, event_power: float, base_power: float, allowed_average: float) -> float:
    if event_power <= base_power:
        return float("nan")
    return (float(allowed_average) - float(base_power)) / (float(event_power) - float(base_power))


def inclusion_rate(masks: np.ndarray, sensor_ids: list[str], name: str) -> float:
    if masks.size == 0:
        return float("nan")
    return float(np.mean(masks[:, index_of(sensor_ids, name)]))


def finite_float(value: object) -> float:
    out = float(value)
    if np.isinf(out):
        return 1.0e12
    return out


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ROOT / path


def render_report(summary: pd.DataFrame, masks: pd.DataFrame) -> str:
    lines = ["# Scenario Calibration Structural Audit", ""]
    lines.append("## Summary")
    lines.append("")
    display_cols = [
        "scenario",
        "calibration_gate_pass",
        "structural_gate_pass",
        "energy_gate_pass",
        "static_anchor_gate_pass",
        "feasible_masks",
        "direct_full_core_laser_fc4_feasible",
        "direct_snow_laser_fc4_feasible",
        "event_core_laser_feasible",
        "proxy_core_spc_fc4_feasible",
        "event_core_laser_continuous_steps",
        "proxy_core_spc_fc4_continuous_steps",
        "laser_duty_over_proxy",
        "laser_duty_over_core",
    ]
    lines.append(markdown_table(summary[display_cols]))
    lines.append("")
    lines.append("## Sensor Inclusion In Feasible Masks")
    lines.append("")
    inclusion_cols = [
        "scenario",
        "laser_inclusion_rate",
        "fc4_inclusion_rate",
        "spc_inclusion_rate",
        "feasible_with_laser",
        "feasible_with_fc4",
        "feasible_with_spc",
    ]
    lines.append(markdown_table(summary[inclusion_cols]))
    lines.append("")
    lines.append("## Top Feasible Masks By Power")
    lines.append("")
    if masks.empty:
        lines.append("No feasible masks.")
    else:
        top = masks.sort_values(["scenario", "power", "peak"], ascending=[True, False, False]).groupby("scenario").head(12)
        lines.append(markdown_table(top[["scenario", "sensor_ids", "power", "peak", "active_count"]]))
    lines.append("")
    lines.append("## Gate Definition")
    lines.append("")
    lines.append("- Structural pass: `core+laser+fc4` and `laser+fc4` are infeasible; `core+laser` and `core+SPC+fc4` remain feasible.")
    lines.append("- Energy pass: constant `core+laser` cannot last a full eval window; proxy stack can; max laser duty over proxy is between 0.35 and 0.95.")
    lines.append("- Static-anchor pass: `core+laser` remains feasible but max laser duty over `core` is at most 0.75.")
    return "\n".join(lines)


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
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if np.isnan(out):
            return "nan"
        if np.isinf(out):
            return "inf"
        return f"{out:.6g}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


if __name__ == "__main__":
    main()
