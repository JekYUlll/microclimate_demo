#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RLSF_ROOT = ROOT / "rl_sensor_scheduling_framework"
RLSF_SRC = RLSF_ROOT / "src"
if str(RLSF_SRC) not in sys.path:
    sys.path.insert(0, str(RLSF_SRC))

from data_sources.public_weather_synthesis import (  # noqa: E402
    PublicWeatherSynthesisConfig,
    build_antaws_anchor,
    generate_public_weather_truth,
    validate_synthetic_against_anchor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v1-local regime-causal windblown truth CSV. The base "
            "weather synthesis is archived-compatible; the v7 postprocess "
            "creates onset/active/decay task phases with different sensor value."
        )
    )
    parser.add_argument("--antaws-root", default="data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--steps", type=int, default=90000)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-keep-fraction", type=float, default=0.15)
    parser.add_argument("--event-coverage", type=float, default=0.30)
    parser.add_argument("--event-model", default="semi_markov")
    parser.add_argument("--min-duration", type=int, default=10)
    parser.add_argument("--max-duration", type=int, default=30)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--lead-steps", type=int, default=6)
    parser.add_argument("--wind-margin-ms", type=float, default=1.6)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.60)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.30)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.4)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.65)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.16)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.035)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=0.90)
    parser.add_argument("--particle-correlation", type=float, default=0.55)

    parser.add_argument("--v7-onset-fraction", type=float, default=0.30)
    parser.add_argument("--v7-decay-steps", type=int, default=10)
    parser.add_argument("--v7-flux-scale", type=float, default=1.45e-4)
    parser.add_argument("--v7-background-flux", type=float, default=1.0e-7)
    parser.add_argument("--v7-noise-scale", type=float, default=0.05)
    parser.add_argument("--out", default="v1/artifacts/regime_causal_v7_seed42/truth_regime_causal_v7.csv")
    parser.add_argument("--report-dir", default="v1/artifacts/regime_causal_v7_seed42/dataset_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PublicWeatherSynthesisConfig(
        antaws_root=resolve_antaws_root(str(args.antaws_root)),
        stations=tuple(str(s) for s in args.stations),
        steps=int(args.steps),
        freq_s=int(args.freq_s),
        seed=int(args.seed),
        phase_keep_fraction=float(args.phase_keep_fraction),
        blowing_snow_event_coverage=float(args.event_coverage),
        blowing_snow_event_model=str(args.event_model),
        blowing_snow_min_duration_steps=int(args.min_duration),
        blowing_snow_max_duration_steps=int(args.max_duration),
        blowing_snow_min_gap_steps=int(args.min_gap),
        blowing_snow_lead_steps=int(args.lead_steps),
        blowing_snow_wind_margin_ms=float(args.wind_margin_ms),
        cred_hysteresis_on=float(args.cred_hysteresis_on),
        cred_hysteresis_off=float(args.cred_hysteresis_off),
        flux_wind_exponent=float(args.flux_wind_exponent),
        event_microstructure_sigma=float(args.event_microstructure_sigma),
        event_microstructure_alpha=float(args.event_microstructure_alpha),
        event_microstructure_diameter_scale=float(args.event_microstructure_diameter_scale),
        event_microstructure_velocity_scale=float(args.event_microstructure_velocity_scale),
        event_particle_microstructure_correlation=float(args.particle_correlation),
    )
    base, meta = generate_public_weather_truth(cfg)
    truth, v7_meta = apply_regime_causal_postprocess(
        base,
        seed=int(args.seed) + 70_000,
        lead_steps=int(args.lead_steps),
        onset_fraction=float(args.v7_onset_fraction),
        decay_steps=int(args.v7_decay_steps),
        flux_scale=float(args.v7_flux_scale),
        background_flux=float(args.v7_background_flux),
        noise_scale=float(args.v7_noise_scale),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(out_path, index=False)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    anchor = build_antaws_anchor(cfg.antaws_root, cfg.stations, freq_s=cfg.freq_s)
    validation = validate_synthetic_against_anchor(anchor, truth)
    validation.to_csv(report_dir / "synthetic_validation.csv", index=False)
    metadata = {
        "base_synthesis": meta,
        "regime_causal_v7": v7_meta,
        "builder_args": vars(args),
    }
    (report_dir / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(out_path)
    print(report_dir / "synthetic_validation.csv")


def apply_regime_causal_postprocess(
    df: pd.DataFrame,
    *,
    seed: int,
    lead_steps: int,
    onset_fraction: float,
    decay_steps: int,
    flux_scale: float,
    background_flux: float,
    noise_scale: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    rng = np.random.default_rng(int(seed))
    n = len(out)
    base_event = out["event_flag"].astype(bool).to_numpy() if "event_flag" in out.columns else np.zeros(n, dtype=bool)
    phase_id = np.zeros(n, dtype=int)

    for start, end in bool_runs(base_event):
        run_len = max(1, int(end) - int(start))
        lead_start = max(0, int(start) - max(0, int(lead_steps)))
        onset_end = min(int(end), int(start) + max(1, int(round(run_len * float(onset_fraction)))))
        decay_end = min(n, int(end) + max(0, int(decay_steps)))
        if lead_start < start:
            phase_id[lead_start:start] = np.maximum(phase_id[lead_start:start], 1)
        phase_id[int(start) : onset_end] = 1
        phase_id[onset_end:int(end)] = 2
        decay_segment = phase_id[int(end) : decay_end]
        decay_segment[decay_segment == 0] = 3
        phase_id[int(end) : decay_end] = decay_segment

    wind = out["wind_speed_ms"].to_numpy(dtype=float)
    temp = out["air_temperature_c"].to_numpy(dtype=float)
    rh = out["relative_humidity"].to_numpy(dtype=float)
    pressure = out["air_pressure_pa"].to_numpy(dtype=float)
    radiation = out["solar_radiation_wm2"].to_numpy(dtype=float)
    surface = out["snow_surface_temperature_c"].to_numpy(dtype=float)

    wind_grad = np.r_[0.0, np.diff(wind)]
    pressure_drop = -np.r_[0.0, np.diff(pressure)] / 80.0
    surface_air_gap = temp - surface
    radiation_grad = np.r_[0.0, np.diff(_lowpass(radiation, alpha=0.22))] / 120.0
    thermal_raw = (
        0.006 * radiation
        + 0.85 * surface_air_gap
        + 0.75 * radiation_grad
        + 0.015 * (rh - np.nanmean(rh))
    )
    thermal_driver = sigmoid(zscore(thermal_raw))
    onset_raw = (
        1.20 * zscore(wind - np.nanpercentile(wind, 72.0))
        + 0.55 * zscore(wind_grad)
        + 0.35 * zscore(rh)
        + 0.25 * zscore(pressure_drop)
    )
    onset_driver = sigmoid(onset_raw)

    micro = _lowpass(rng.normal(0.0, 1.0, size=n), alpha=0.20)
    micro = zscore(micro)
    event_micro = np.where(phase_id > 0, micro, 0.0)
    particle_micro = zscore(
        0.55 * event_micro + 0.45 * _lowpass(rng.normal(0.0, 1.0, size=n), alpha=0.12)
    )
    particle_micro = np.where(phase_id > 0, particle_micro, 0.0)

    intensity = np.zeros(n, dtype=float)
    onset = phase_id == 1
    active = phase_id == 2
    decay = phase_id == 3
    intensity[onset] = 0.20 + 0.60 * onset_driver[onset] + 0.20 * thermal_driver[onset]
    intensity[active] = (
        0.30
        + 0.10 * z01(wind[active])
        + 0.85 * z01(np.abs(particle_micro[active]))
        + 0.05 * thermal_driver[active]
        if np.any(active)
        else np.zeros(0, dtype=float)
    )
    intensity[decay] = 0.12 + 0.68 * thermal_driver[decay] + 0.20 * z01(wind[decay])
    intensity = np.clip(intensity, 0.0, 1.8)

    phase_flux_scale = np.zeros(n, dtype=float)
    phase_flux_scale[onset] = 0.18
    phase_flux_scale[active] = 1.00
    phase_flux_scale[decay] = 0.55
    flux_target = background_flux + float(flux_scale) * phase_flux_scale * np.square(intensity)
    flux_target *= np.exp(0.35 * event_micro)
    flux_target = np.where(phase_id > 0, flux_target, background_flux * (0.4 + 0.2 * z01(wind)))
    flux_target *= np.exp(float(noise_scale) * rng.normal(0.0, 1.0, size=n))
    flux = _asymmetric_filter(flux_target, rise_alpha=0.72, fall_alpha=0.32)
    flux = np.clip(flux, 0.0, None)

    diameter = np.zeros(n, dtype=float)
    velocity = np.zeros(n, dtype=float)
    diameter[onset] = (
        0.15
        + 0.070 * onset_driver[onset]
        + 0.035 * z01(rh[onset])
    )
    velocity[onset] = (
        0.58 * wind[onset]
        + 2.10 * onset_driver[onset]
        + 0.35 * wind_grad[onset]
    )
    diameter[active] = 0.20 + 0.015 * z01(wind[active]) + 0.350 * particle_micro[active]
    velocity[active] = 0.52 * wind[active] + 1.15 * np.sqrt(np.maximum(flux[active], 0.0) / max(float(flux_scale), 1e-12))
    velocity[active] += 6.00 * particle_micro[active]
    diameter[decay] = (
        0.13
        + 0.095 * thermal_driver[decay]
        + 0.045 * z01(surface_air_gap[decay])
        + 0.020 * z01(radiation[decay])
    )
    velocity[decay] = (
        0.18 * wind[decay]
        + 1.75 * thermal_driver[decay]
        + 0.55 * z01(surface_air_gap[decay])
        + 0.30 * z01(radiation[decay])
    )
    diameter += rng.normal(0.0, 0.012, size=n)
    velocity += rng.normal(0.0, 0.14, size=n)
    diameter = np.where(phase_id > 0, np.clip(diameter, 0.04, 0.55), 0.0)
    velocity = np.where(phase_id > 0, np.clip(velocity, 0.0, 20.0), 0.0)

    # Broad event semantics: task evaluation should include onset and decay,
    # not only fully active transport pulses.
    out["snow_mass_flux_kg_m2_s"] = flux
    out["snow_particle_mean_diameter_mm"] = diameter
    out["snow_particle_mean_velocity_ms"] = velocity
    out["event_flag"] = phase_id > 0
    out["storm_flag"] = phase_id > 0
    out["v7_transport_phase_id"] = phase_id
    out["v7_transport_phase"] = pd.Categorical.from_codes(
        phase_id,
        categories=["calm", "onset", "active", "decay"],
    ).astype(str)
    out["v7_onset_driver"] = onset_driver
    out["v7_thermal_driver"] = thermal_driver
    out["v7_transport_intensity"] = intensity
    out["event_microstructure"] = event_micro
    out["event_particle_microstructure"] = particle_micro
    out["parsivel_available"] = phase_id > 0

    phase_counts = pd.Series(out["v7_transport_phase"]).value_counts(normalize=True).to_dict()
    meta = {
        "phase_fraction": {str(key): float(value) for key, value in phase_counts.items()},
        "base_active_fraction": float(np.mean(base_event)),
        "broad_event_fraction": float(np.mean(phase_id > 0)),
        "lead_steps": int(lead_steps),
        "onset_fraction": float(onset_fraction),
        "decay_steps": int(decay_steps),
        "flux_scale": float(flux_scale),
        "background_flux": float(background_flux),
        "noise_scale": float(noise_scale),
    }
    return out, meta


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


def _lowpass(values: np.ndarray, *, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for idx in range(1, arr.size):
        out[idx] = out[idx - 1] + float(alpha) * (arr[idx] - out[idx - 1])
    return out


def _asymmetric_filter(values: np.ndarray, *, rise_alpha: float, fall_alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for idx in range(1, arr.size):
        alpha = float(rise_alpha) if arr[idx] >= out[idx - 1] else float(fall_alpha)
        out[idx] = out[idx - 1] + alpha * (arr[idx] - out[idx - 1])
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    std = float(np.nanstd(arr)) if arr.size else 1.0
    return (arr - mean) / max(std, 1.0e-6)


def z01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.nanpercentile(arr, 5.0))
    hi = float(np.nanpercentile(arr, 95.0))
    return np.clip((arr - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))


def resolve_antaws_root(value: str) -> Path:
    path = Path(value)
    candidates = [
        path,
        ROOT / value,
        RLSF_ROOT / value,
        RLSF_ROOT.parent / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve AntAWS root: {value}")


if __name__ == "__main__":
    main()
