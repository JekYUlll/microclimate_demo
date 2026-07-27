#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


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

PHASE_NAMES = ("calm", "onset", "active", "decay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a scenario has real regime-specific sensor value. "
            "This is a lightweight masked-predictor audit, not a scheduler run."
        )
    )
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--sensor-cfg", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=1.36)
    parser.add_argument("--startup-peak-budget", type=float, default=1.75)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.30, 0.45, 0.125, 0.125])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--lags", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=[
            "snow_mass_flux_kg_m2_s",
            "snow_particle_mean_diameter_mm",
            "snow_particle_mean_velocity_ms",
        ],
    )
    parser.add_argument("--target-scales", nargs="+", type=float, default=[1.0e-4, 0.2, 5.0])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-phase-samples", type=int, default=32)
    parser.add_argument("--gate-margin", type=float, default=0.015)
    parser.add_argument("--include-calm", action="store_true")
    parser.add_argument(
        "--average-power-budget",
        type=float,
        default=None,
        help=(
            "Optional long-run average power budget for static and "
            "phase-conditioned selectors. This makes the audit match "
            "energy-limited scenes where high-value sensors are instantaneously "
            "feasible but cannot be held continuously."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth_csv)
    sensors = load_sensors(Path(args.sensor_cfg))
    masks = enumerate_feasible_masks(
        sensors,
        budget=float(args.budget),
        peak_budget=float(args.startup_peak_budget),
        max_active=int(args.max_active),
    )
    audit = audit_static_dominance(
        truth,
        sensors=sensors,
        masks=masks,
        seed=int(args.seed),
        split_ratios=tuple(float(x) for x in args.split_ratios),
        horizon=int(args.horizon),
        lags=tuple(int(x) for x in args.lags),
        target_columns=tuple(str(x) for x in args.target_columns),
        target_scales=tuple(float(x) for x in args.target_scales),
        ridge_alpha=float(args.ridge_alpha),
        min_phase_samples=int(args.min_phase_samples),
        gate_margin=float(args.gate_margin),
        include_calm=bool(args.include_calm),
        average_power_budget=None if args.average_power_budget is None else float(args.average_power_budget),
    )
    audit["mask_metrics"].to_csv(out_dir / "mask_predictor_metrics.csv", index=False)
    audit["phase_selection"].to_csv(out_dir / "regime_phase_selection.csv", index=False)
    audit["summary"].to_csv(out_dir / "regime_static_dominance_summary.csv", index=False)
    report = render_report(audit["summary"], audit["phase_selection"], audit["mask_metrics"])
    (out_dir / "regime_static_dominance_audit.md").write_text(report, encoding="utf-8")
    print(report)


def audit_static_dominance(
    truth: pd.DataFrame,
    *,
    sensors: list[dict[str, Any]],
    masks: list[np.ndarray],
    seed: int,
    split_ratios: tuple[float, ...],
    horizon: int,
    lags: tuple[int, ...],
    target_columns: tuple[str, ...],
    target_scales: tuple[float, ...],
    ridge_alpha: float,
    min_phase_samples: int,
    gate_margin: float,
    include_calm: bool,
    average_power_budget: float | None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    missing = [col for col in STATE_COLUMNS if col not in truth.columns]
    if missing:
        raise ValueError(f"truth CSV is missing state columns: {missing}")
    target_missing = [col for col in target_columns if col not in truth.columns]
    if target_missing:
        raise ValueError(f"truth CSV is missing target columns: {target_missing}")

    phase_id = phase_ids(truth)
    common = common_features(truth)
    target = future_target_mean(truth, target_columns=target_columns, horizon=int(horizon))
    max_lag = max(max(lags), 0)
    valid_idx = np.arange(max_lag, len(truth) - int(horizon), dtype=int)
    bounds = split_bounds(len(truth), split_ratios)
    split_masks = {
        name: (valid_idx >= start) & (valid_idx < end)
        for name, (start, end) in bounds.items()
    }
    fit_mask = split_masks["oracle_pretrain"] | split_masks["rl_train"]
    validation_mask = split_masks["validation"]
    final_mask = split_masks["final_test"]
    phase_valid = phase_id[valid_idx]

    scales = np.asarray(target_scales, dtype=float).reshape(-1)
    if scales.shape[0] != len(target_columns):
        raise ValueError("--target-scales must match --target-columns")
    scales = np.maximum(np.abs(scales), 1.0e-12)

    sensor_ids = [str(sensor["sensor_id"]) for sensor in sensors]
    metric_rows: list[dict[str, object]] = []
    predictions: dict[int, np.ndarray] = {}

    for mask_idx, mask in enumerate(masks):
        observed = observed_series_for_mask(truth, sensors, mask, rng=np.random.default_rng(rng.integers(0, 2**31)))
        x = build_feature_matrix(
            observed,
            common,
            valid_idx=valid_idx,
            lags=lags,
        )
        y = target[valid_idx]
        pred = fit_predict_ridge(
            x_train=x[fit_mask],
            y_train=y[fit_mask],
            x_all=x,
            alpha=float(ridge_alpha),
        )
        predictions[int(mask_idx)] = pred
        row_base = {
            "mask_idx": int(mask_idx),
            "mask_bits": "".join("1" if active else "0" for active in mask.astype(bool)),
            "sensor_ids": "|".join(sensor_ids[idx] for idx, active in enumerate(mask) if bool(active)),
            "power": float(sum(float(sensors[idx].get("power_cost", 0.0)) for idx, active in enumerate(mask) if bool(active))),
            "peak": float(
                sum(float(sensors[idx].get("startup_peak_power", sensors[idx].get("power_cost", 0.0))) for idx, active in enumerate(mask) if bool(active))
            ),
            "active_count": int(np.sum(mask)),
        }
        for split_name, split_mask in (("validation", validation_mask), ("final", final_mask)):
            losses = normalized_mae(pred[split_mask], y[split_mask], scales=scales)
            metric_rows.append({**row_base, "split": split_name, "phase": "all", "loss": float(np.mean(losses))})
            for phase_value, phase_name in enumerate(PHASE_NAMES):
                phase_split = split_mask & (phase_valid == int(phase_value))
                if int(np.sum(phase_split)) < int(min_phase_samples):
                    continue
                phase_losses = normalized_mae(pred[phase_split], y[phase_split], scales=scales)
                metric_rows.append(
                    {
                        **row_base,
                        "split": split_name,
                        "phase": phase_name,
                        "loss": float(np.mean(phase_losses)),
                        "n": int(np.sum(phase_split)),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    objective_phases = list(PHASE_NAMES if bool(include_calm) else PHASE_NAMES[1:])
    validation_all = aggregate_phase_objective(metrics, split="validation", phases=objective_phases)
    final_all = aggregate_phase_objective(metrics, split="final", phases=objective_phases)
    if average_power_budget is not None:
        validation_all_for_static = validation_all.loc[
            validation_all["power"].astype(float) <= float(average_power_budget) + 1.0e-12
        ].copy()
        if validation_all_for_static.empty:
            raise ValueError(
                f"No static mask has power <= average_power_budget={average_power_budget}"
            )
    else:
        validation_all_for_static = validation_all
    best_static = validation_all_for_static.sort_values(["loss", "power", "active_count"]).iloc[0]
    best_static_idx = int(best_static["mask_idx"])
    static_final_loss = float(final_all.loc[final_all["mask_idx"].astype(int) == best_static_idx, "loss"].iloc[0])

    phase_selection = select_phase_conditioned_masks(
        metrics,
        phases=objective_phases,
        best_static_idx=best_static_idx,
        average_power_budget=average_power_budget,
    )
    phase_rows = phase_selection["rows"]
    regime_final_loss = float(phase_selection["regime_final_loss"])
    static_phase_weighted_final = float(phase_selection["static_phase_weighted_final"])
    phase_weighted_power = float(phase_selection["phase_weighted_power"])
    selected_phase_mask_ids = [int(row["selected_mask_idx"]) for row in phase_rows]

    core_stack_idx = find_mask_idx(sensor_ids, masks, ("met_station_core", "snow_particle_counter", "fc4_flux"))
    core_stack_final_loss = float("nan")
    core_stack_validation_rank = -1
    if core_stack_idx is not None:
        core_row = final_all.loc[final_all["mask_idx"].astype(int) == int(core_stack_idx)]
        if len(core_row):
            core_stack_final_loss = float(core_row["loss"].iloc[0])
        ranked = validation_all.sort_values(["loss", "power", "active_count"]).reset_index(drop=True)
        matches = ranked.index[ranked["mask_idx"].astype(int) == int(core_stack_idx)].tolist()
        if matches:
            core_stack_validation_rank = int(matches[0]) + 1

    unique_phase_masks = int(len(set(selected_phase_mask_ids)))
    margin = float(static_phase_weighted_final) - float(regime_final_loss)
    summary = pd.DataFrame(
        [
            {
                "n_feasible_masks": int(len(masks)),
                "best_static_mask_idx": best_static_idx,
                "best_static_sensor_ids": str(best_static["sensor_ids"]),
                "best_static_validation_loss": float(best_static["loss"]),
                "best_static_final_loss": static_final_loss,
                "phase_weighted_static_final_loss": static_phase_weighted_final,
                "phase_weighted_regime_final_loss": regime_final_loss,
                "regime_margin_vs_static": margin,
                "regime_gain_pct": margin / max(static_phase_weighted_final, 1.0e-12),
                "phase_weighted_power": phase_weighted_power,
                "average_power_budget": np.nan if average_power_budget is None else float(average_power_budget),
                "unique_phase_selected_masks": unique_phase_masks,
                "core_spc_fc4_mask_idx": -1 if core_stack_idx is None else int(core_stack_idx),
                "core_spc_fc4_validation_rank": core_stack_validation_rank,
                "core_spc_fc4_final_loss": core_stack_final_loss,
                "gate_margin": float(gate_margin),
                "scenario_gate_pass": bool(
                    np.isfinite(margin)
                    and margin >= float(gate_margin)
                    and unique_phase_masks >= 3
                    and (average_power_budget is None or phase_weighted_power <= float(average_power_budget) + 1.0e-12)
                    and (core_stack_validation_rank < 0 or core_stack_validation_rank > 1)
                ),
            }
        ]
    )
    return {
        "mask_metrics": metrics.sort_values(["split", "phase", "loss", "power"]).reset_index(drop=True),
        "phase_selection": pd.DataFrame(phase_rows),
        "summary": summary,
    }


def select_phase_conditioned_masks(
    metrics: pd.DataFrame,
    *,
    phases: list[str],
    best_static_idx: int,
    average_power_budget: float | None,
) -> dict[str, object]:
    phase_tables: dict[str, pd.DataFrame] = {}
    for phase_name in phases:
        val_phase = metrics.loc[(metrics["split"] == "validation") & (metrics["phase"] == phase_name)].copy()
        if val_phase.empty:
            continue
        phase_tables[phase_name] = val_phase.sort_values(["loss", "power", "active_count"]).reset_index(drop=True)
    if not phase_tables:
        return {
            "rows": [],
            "regime_final_loss": float("nan"),
            "static_phase_weighted_final": float("nan"),
            "phase_weighted_power": float("nan"),
        }

    if average_power_budget is None or len(phase_tables) == 1:
        selected = {phase: table.iloc[0] for phase, table in phase_tables.items()}
    else:
        # Full product over phase-specific masks is small for the current
        # 3-phase audit (typically < 70^3). Keep all candidates so the result is
        # not another top-k heuristic.
        phase_names = list(phase_tables)
        best_combo: dict[str, pd.Series] | None = None
        best_loss = float("inf")
        candidate_lists = [list(table.itertuples(index=False)) for table in (phase_tables[name] for name in phase_names)]
        for combo in product(*candidate_lists):
            weights = np.asarray([float(getattr(row, "n", 0.0)) for row in combo], dtype=float)
            if float(np.sum(weights)) <= 0.0:
                continue
            powers = np.asarray([float(getattr(row, "power")) for row in combo], dtype=float)
            avg_power = float(np.sum(weights * powers) / np.sum(weights))
            if avg_power > float(average_power_budget) + 1.0e-12:
                continue
            losses = np.asarray([float(getattr(row, "loss")) for row in combo], dtype=float)
            weighted_loss = float(np.sum(weights * losses) / np.sum(weights))
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_combo = {
                    phase: pd.Series(row._asdict())
                    for phase, row in zip(phase_names, combo, strict=True)
                }
        if best_combo is None:
            # Fall back to the lowest-power feasible phase rows so the report is
            # still diagnostic. The gate will fail via phase_weighted_power.
            selected = {phase: table.sort_values(["power", "loss"]).iloc[0] for phase, table in phase_tables.items()}
        else:
            selected = best_combo

    rows: list[dict[str, object]] = []
    weighted_regime_final = 0.0
    weighted_static_final = 0.0
    weighted_power = 0.0
    total_weight = 0.0
    for phase_name, chosen in selected.items():
        final_phase = metrics.loc[(metrics["split"] == "final") & (metrics["phase"] == phase_name)].copy()
        if final_phase.empty:
            continue
        chosen_idx = int(chosen["mask_idx"])
        chosen_final = final_phase.loc[final_phase["mask_idx"].astype(int) == chosen_idx].iloc[0]
        static_phase_final = final_phase.loc[final_phase["mask_idx"].astype(int) == int(best_static_idx)].iloc[0]
        weight = int(chosen_final.get("n", 0))
        weighted_regime_final += float(chosen_final["loss"]) * float(weight)
        weighted_static_final += float(static_phase_final["loss"]) * float(weight)
        weighted_power += float(chosen["power"]) * float(weight)
        total_weight += float(weight)
        rows.append(
            {
                "phase": phase_name,
                "selected_mask_idx": chosen_idx,
                "selected_sensor_ids": str(chosen["sensor_ids"]),
                "validation_loss": float(chosen["loss"]),
                "final_loss_selected": float(chosen_final["loss"]),
                "final_loss_static": float(static_phase_final["loss"]),
                "final_margin_vs_static": float(static_phase_final["loss"]) - float(chosen_final["loss"]),
                "final_n": int(weight),
                "selected_power": float(chosen["power"]),
            }
        )
    if total_weight <= 0.0:
        return {
            "rows": rows,
            "regime_final_loss": float("nan"),
            "static_phase_weighted_final": float("nan"),
            "phase_weighted_power": float("nan"),
        }
    return {
        "rows": rows,
        "regime_final_loss": weighted_regime_final / total_weight,
        "static_phase_weighted_final": weighted_static_final / total_weight,
        "phase_weighted_power": weighted_power / total_weight,
    }


def aggregate_phase_objective(metrics: pd.DataFrame, *, split: str, phases: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = metrics.loc[(metrics["split"] == split) & (metrics["phase"].isin(phases))].copy()
    if selected.empty:
        raise ValueError(f"No phase metrics found for split={split}, phases={phases}")
    for mask_idx, group in selected.groupby("mask_idx"):
        total_n = float(group["n"].fillna(0).sum())
        if total_n <= 0:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "mask_idx": int(mask_idx),
                "mask_bits": str(first["mask_bits"]),
                "sensor_ids": str(first["sensor_ids"]),
                "power": float(first["power"]),
                "peak": float(first["peak"]),
                "active_count": int(first["active_count"]),
                "split": split,
                "phase": "event_phases",
                "loss": float(np.sum(group["loss"].to_numpy(dtype=float) * group["n"].fillna(0).to_numpy(dtype=float)) / total_n),
                "n": int(total_n),
            }
        )
    return pd.DataFrame(rows)


def load_sensors(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sensors = data.get("sensors") if isinstance(data, dict) else None
    if not isinstance(sensors, list):
        raise ValueError(f"Sensor config does not contain a sensors list: {path}")
    return [dict(sensor) for sensor in sensors]


def enumerate_feasible_masks(
    sensors: list[dict[str, Any]],
    *,
    budget: float,
    peak_budget: float,
    max_active: int,
) -> list[np.ndarray]:
    n = len(sensors)
    masks: list[np.ndarray] = []
    powers = np.asarray([float(sensor.get("power_cost", 0.0)) for sensor in sensors], dtype=float)
    peaks = np.asarray(
        [float(sensor.get("startup_peak_power", sensor.get("power_cost", 0.0))) for sensor in sensors],
        dtype=float,
    )
    for count in range(1, min(int(max_active), n) + 1):
        for combo in combinations(range(n), count):
            mask = np.zeros(n, dtype=bool)
            mask[list(combo)] = True
            if float(np.dot(mask, powers)) <= float(budget) + 1.0e-12 and float(np.dot(mask, peaks)) <= float(peak_budget) + 1.0e-12:
                masks.append(mask)
    if not masks:
        raise ValueError("No feasible masks under the supplied constraints")
    return masks


def phase_ids(truth: pd.DataFrame) -> np.ndarray:
    if "v7_transport_phase_id" in truth.columns:
        return truth["v7_transport_phase_id"].to_numpy(dtype=int)
    events = truth["event_flag"].astype(bool).to_numpy() if "event_flag" in truth.columns else np.zeros(len(truth), dtype=bool)
    phase = np.zeros(len(truth), dtype=int)
    for start, end in bool_runs(events):
        length = max(1, int(end) - int(start))
        onset_end = min(int(end), int(start) + max(1, int(round(0.30 * length))))
        decay_end = min(len(truth), int(end) + max(2, int(round(0.20 * length))))
        phase[int(start) : onset_end] = 1
        phase[onset_end:int(end)] = 2
        segment = phase[int(end) : decay_end]
        segment[segment == 0] = 3
        phase[int(end) : decay_end] = segment
    return phase


def bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr):
        if bool(value) and start is None:
            start = int(idx)
        elif not bool(value) and start is not None:
            runs.append((start, int(idx)))
            start = None
    if start is not None:
        runs.append((start, int(arr.size)))
    return runs


def common_features(truth: pd.DataFrame) -> np.ndarray:
    n = len(truth)
    if "time_idx" in truth.columns:
        time_idx = truth["time_idx"].to_numpy(dtype=float)
    else:
        time_idx = np.arange(n, dtype=float)
    period = 8.0 if n < 512 else 24.0
    common = [
        np.sin(2.0 * np.pi * time_idx / period),
        np.cos(2.0 * np.pi * time_idx / period),
        truth["event_flag"].astype(float).to_numpy() if "event_flag" in truth.columns else np.zeros(n, dtype=float),
    ]
    return np.vstack(common).T.astype(float)


def observed_series_for_mask(
    truth: pd.DataFrame,
    sensors: list[dict[str, Any]],
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    event = truth["event_flag"].astype(bool).to_numpy() if "event_flag" in truth.columns else np.zeros(len(truth), dtype=bool)
    out: dict[str, np.ndarray] = {}
    for variable in STATE_COLUMNS:
        candidates = []
        for idx, active in enumerate(mask):
            if not bool(active):
                continue
            sensor = sensors[idx]
            variables = sensor.get("observed_variables", sensor.get("variables", []))
            if variable not in variables:
                continue
            candidates.append(sensor)
        if not candidates:
            continue
        base = truth[variable].to_numpy(dtype=float)
        obs_candidates = []
        for sensor in candidates:
            base_std = float(dict(sensor.get("noise_std", {}) or {}).get(variable, 0.0))
            event_std = dict(sensor.get("event_noise_std", {}) or {}).get(variable)
            std = np.full(len(base), base_std, dtype=float)
            if event_std is not None:
                std[event] = float(event_std)
            prob = np.ones(len(base), dtype=float)
            event_prob = dict(sensor.get("event_observation_probability", {}) or {}).get(variable)
            if event_prob is not None:
                prob[event] = float(event_prob)
            observed = base + rng.normal(0.0, std, size=len(base))
            available = rng.random(len(base)) <= prob
            interval = max(1, int(sensor.get("sampling_interval", sensor.get("refresh_interval", 1))))
            if interval > 1:
                available = available & ((np.arange(len(base), dtype=int) % interval) == 0)
            observed = zero_order_hold(observed, available=available)
            expected_std = float(np.mean(std / np.maximum(prob, 1.0e-3)))
            obs_candidates.append((expected_std, observed))
        obs_candidates.sort(key=lambda item: item[0])
        out[variable] = obs_candidates[0][1]
    return pd.DataFrame(out)


def zero_order_hold(values: np.ndarray, *, available: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    avail = np.asarray(available, dtype=bool).reshape(-1)
    out = np.empty_like(arr)
    last = float(arr[0]) if arr.size else 0.0
    for idx, value in enumerate(arr):
        if bool(avail[idx]) and np.isfinite(value):
            last = float(value)
        out[idx] = last
    return out


def build_feature_matrix(
    observed: pd.DataFrame,
    common: np.ndarray,
    *,
    valid_idx: np.ndarray,
    lags: tuple[int, ...],
) -> np.ndarray:
    columns = [col for col in STATE_COLUMNS if col in observed.columns]
    parts = [common[valid_idx]]
    for lag in lags:
        idx = valid_idx - int(lag)
        if columns:
            parts.append(observed[columns].to_numpy(dtype=float)[idx])
    return np.hstack(parts).astype(float)


def future_target_mean(
    truth: pd.DataFrame,
    *,
    target_columns: tuple[str, ...],
    horizon: int,
) -> np.ndarray:
    values = truth[list(target_columns)].to_numpy(dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    for idx in range(0, len(values) - int(horizon)):
        out[idx] = np.mean(values[idx + 1 : idx + int(horizon) + 1], axis=0)
    return out


def fit_predict_ridge(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    alpha: float,
) -> np.ndarray:
    good = np.all(np.isfinite(x_train), axis=1) & np.all(np.isfinite(y_train), axis=1)
    x = np.asarray(x_train[good], dtype=float)
    y = np.asarray(y_train[good], dtype=float)
    if x.shape[0] < max(8, x.shape[1] + 2):
        return np.repeat(np.nanmean(y_train, axis=0, keepdims=True), x_all.shape[0], axis=0)
    mean = np.mean(x, axis=0)
    std = np.maximum(np.std(x, axis=0), 1.0e-6)
    x_norm = (x - mean) / std
    x_all_norm = (np.asarray(x_all, dtype=float) - mean) / std
    x_aug = np.hstack([np.ones((x_norm.shape[0], 1), dtype=float), x_norm])
    all_aug = np.hstack([np.ones((x_all_norm.shape[0], 1), dtype=float), x_all_norm])
    reg = np.eye(x_aug.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return all_aug @ beta


def normalized_mae(pred: np.ndarray, target: np.ndarray, *, scales: np.ndarray) -> np.ndarray:
    err = np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))
    return np.mean(err / scales.reshape(1, -1), axis=1)


def split_bounds(n: int, ratios: tuple[float, ...]) -> dict[str, tuple[int, int]]:
    if len(ratios) != 4:
        raise ValueError("split ratios must have four entries")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    norm = [float(x) / total for x in ratios]
    cut1 = int(round(n * norm[0]))
    cut2 = int(round(n * (norm[0] + norm[1])))
    cut3 = int(round(n * (norm[0] + norm[1] + norm[2])))
    return {
        "oracle_pretrain": (0, cut1),
        "rl_train": (cut1, cut2),
        "validation": (cut2, cut3),
        "final_test": (cut3, n),
    }


def find_mask_idx(
    sensor_ids: list[str],
    masks: list[np.ndarray],
    names: tuple[str, ...],
) -> int | None:
    desired = np.zeros(len(sensor_ids), dtype=bool)
    for name in names:
        if name not in sensor_ids:
            return None
        desired[sensor_ids.index(name)] = True
    for idx, mask in enumerate(masks):
        if np.array_equal(np.asarray(mask, dtype=bool), desired):
            return int(idx)
    return None


def render_report(summary: pd.DataFrame, phase_selection: pd.DataFrame, metrics: pd.DataFrame) -> str:
    lines = ["# Regime Static-Dominance Audit", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Regime-Conditioned Selection")
    lines.append("")
    lines.append(markdown_table(phase_selection))
    lines.append("")
    lines.append("## Top Validation Static Masks")
    lines.append("")
    top = aggregate_phase_objective(metrics, split="validation", phases=list(PHASE_NAMES[1:]))
    if "average_power_budget" in summary.columns:
        budget = float(summary["average_power_budget"].iloc[0])
        if np.isfinite(budget):
            top = top.loc[top["power"].astype(float) <= budget + 1.0e-12]
    top = top.sort_values(["loss", "power"]).head(12)
    lines.append(markdown_table(top[["mask_idx", "sensor_ids", "loss", "power", "peak", "active_count"]]))
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append("- Pass requires positive event-phase-conditioned final margin over the validation-selected static mask.")
    lines.append("- Pass also requires at least three distinct phase-selected masks.")
    lines.append("- `core+SPC+FC4` must not be the top validation static mask when present.")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    rows = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
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
