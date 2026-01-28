"""
Utility to synthesize wind-blown snow observations as a CSV for sandbox models.

This version introduces temporal structure and physically-inspired couplings so
that models can learn non-trivial patterns.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SIZE_BINS_UM: List[Tuple[int, int]] = [(50, 100), (100, 200), (200, 400), (400, 800)]
VELOCITY_BINS_MS: List[Tuple[int, int]] = [(0, 2), (2, 4), (4, 6), (6, 8)]
STABILITY_FLAGS = ["stable", "neutral", "unstable"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic wind-blown snow CSV.")
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of observations to generate.",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="Sampling interval in minutes between rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv"),
        help="Output CSV path.",
    )
    return parser


def _ar1(prev: float, mean: float, phi: float, noise: float, rng: np.random.Generator) -> float:
    return phi * prev + (1 - phi) * mean + rng.normal(0.0, noise)


def _solar_flux(minute_of_day: float, rng: np.random.Generator) -> float:
    phase = 2 * np.pi * (minute_of_day / (24 * 60))
    base = max(0.0, np.sin(phase))
    return max(0.0, 600.0 * base + rng.normal(0.0, 40.0))


def _distribute(total: float, bins: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        return np.zeros(bins)
    weights = rng.dirichlet(np.ones(bins))
    return total * weights


def generate_rows(rows: int, freq_minutes: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    start = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    data: List[Dict[str, Any]] = []

    wind_speed = max(0.5, rng.normal(8.0, 1.5))
    wind_dir = rng.uniform(0, 360)
    air_temp = rng.normal(-12.0, 3.0)
    snow_temp = air_temp - rng.normal(1.5, 0.5)
    pressure = rng.normal(70000, 400)
    rh = np.clip(rng.normal(70, 8), 20, 100)
    snow_supply = float(rng.uniform(0.2, 0.8))
    storm_state = rng.random() < 0.1
    storm_intensity = 1.0

    for i in range(rows):
        ts = start + timedelta(minutes=i * freq_minutes)
        minute_of_day = (ts.hour * 60) + ts.minute

        if storm_state:
            if rng.random() < 0.03:
                storm_state = False
        else:
            if rng.random() < 0.01:
                storm_state = True

        storm_intensity = float(np.clip(_ar1(storm_intensity, mean=1.4 if storm_state else 1.0, phi=0.7, noise=0.05, rng=rng), 0.9, 1.8))

        solar = _solar_flux(minute_of_day, rng)
        if storm_state:
            solar *= 0.4 + 0.2 * rng.random()

        temp_mean = -12.0 + 3.0 * np.sin(2 * np.pi * minute_of_day / (24 * 60)) + (0.8 if storm_state else 0.0)
        air_temp = _ar1(air_temp, mean=temp_mean, phi=0.92, noise=0.6, rng=rng)
        snow_temp = _ar1(snow_temp, mean=air_temp - 2.0 + 0.002 * solar, phi=0.9, noise=0.4, rng=rng)

        wind_mean = 8.0 + 1.5 * np.sin(2 * np.pi * minute_of_day / (24 * 60)) + (3.5 if storm_state else 0.0)
        wind_speed = max(0.2, _ar1(wind_speed, mean=wind_mean * storm_intensity, phi=0.85, noise=0.9, rng=rng))
        wind_dir = (wind_dir + rng.normal(0.0, 12.0)) % 360
        pressure = _ar1(pressure, mean=70000 - (300 if storm_state else 0), phi=0.95, noise=80, rng=rng)
        rh_mean = 75.0 - 0.3 * (air_temp + 10.0) + (8.0 if storm_state else 0.0)
        rh = float(np.clip(_ar1(rh, mean=rh_mean, phi=0.9, noise=4.0, rng=rng), 20, 100))

        turbulence = float(np.clip(0.15 + 0.05 * wind_speed + rng.normal(0.0, 0.05), 0.0, 1.0))
        friction_velocity = max(0.05, 0.08 * wind_speed + rng.normal(0.0, 0.03))
        ri_bulk = float(np.clip(0.15 * (air_temp - snow_temp) - 0.08 * wind_speed + rng.normal(0.0, 0.2), -1.5, 1.5))

        # Threshold for snow transport
        Ut = 6.975 + 0.0033 * (air_temp + 27.27) ** 2
        ratio = max(0.0, (wind_speed - Ut) / max(Ut, 0.1))
        precip_rate = max(0.0, rng.gamma(shape=1.1, scale=0.3))
        if storm_state:
            precip_rate += max(0.0, rng.gamma(shape=1.4, scale=0.8))
        if rh > 85:
            precip_rate *= 1.0 + (rh - 85.0) / 40.0

        snow_supply = float(np.clip(snow_supply + 0.02 * precip_rate - 0.25 * ratio, 0.0, 1.0))
        supply_factor = 0.2 + 0.8 * snow_supply
        if storm_state:
            supply_factor *= 1.1

        snow_mass_flux = max(0.0, 0.0015 * (ratio ** 3) * supply_factor * (1 + rng.normal(0.0, 0.25)))
        snow_mass_flux += max(0.0, rng.lognormal(mean=-10.5, sigma=0.8) - 2e-5)

        diameter = 200e-6
        rho_ice = 917.0
        vol = (4 / 3) * np.pi * (diameter / 2) ** 3
        snow_number_flux = max(0.0, snow_mass_flux / (rho_ice * vol) * rng.uniform(0.8, 1.2))

        stability_flag = "neutral"
        if ri_bulk > 0.2:
            stability_flag = "stable"
        elif ri_bulk < -0.2:
            stability_flag = "unstable"

        row: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "air_temperature_c": air_temp,
            "relative_humidity": rh,
            "air_pressure_pa": pressure,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_dir,
            "solar_radiation_wm2": solar,
            "snow_surface_temperature_c": snow_temp,
            "snow_mass_flux_kg_m2_s": snow_mass_flux,
            "snow_number_flux_m2_s": snow_number_flux,
            "turbulence_intensity": turbulence,
            "friction_velocity_ms": friction_velocity,
            "ri_bulk": ri_bulk,
            "stability_flag": stability_flag,
            "snow_density_kg_m3": max(100.0, 150.0 + 20.0 * wind_speed + rng.normal(0.0, 10.0)),
            "snow_grain_temp_c": snow_temp + rng.normal(0.0, 0.5),
            "visibility_m": max(10.0, 2000.0 / (1.0 + snow_mass_flux * 3e5) + rng.normal(0.0, 50.0)),
            "precip_rate_mm_h": precip_rate,
            "quality_flag": rng.choice(
                ["good", "suspect", "missing"],
                p=[0.85, 0.12, 0.03] if wind_speed > 12 or precip_rate > 2 else [0.92, 0.07, 0.01],
            ),
            "data_source": rng.choice(["sim", "field_logger_a", "field_logger_b"]),
            "missing_reason": "",
        }

        # Derived scalars
        row["wind_snow_coupling"] = snow_mass_flux / max(wind_speed, 0.1)
        row["threshold_exceedance"] = int(wind_speed > Ut)
        row["net_radiation_est"] = 0.7 * solar - 20.0
        row["sensible_heat_flux_est"] = 0.1 * (air_temp - snow_temp) * wind_speed
        row["latent_heat_flux_est"] = 0.02 * rh * wind_speed

        # Flatten spectra bins with mass/number conservation per spectrum
        size_mass = _distribute(snow_mass_flux, len(SIZE_BINS_UM), rng)
        size_num = _distribute(snow_number_flux, len(SIZE_BINS_UM), rng)
        for idx, (low, high) in enumerate(SIZE_BINS_UM, start=1):
            prefix = f"size_bin_{idx}_{low}_{high}_um"
            row[f"{prefix}_number_flux_m2_s"] = size_num[idx - 1]
            row[f"{prefix}_mass_flux_kg_m2_s"] = size_mass[idx - 1]

        vel_mass = _distribute(snow_mass_flux, len(VELOCITY_BINS_MS), rng)
        vel_num = _distribute(snow_number_flux, len(VELOCITY_BINS_MS), rng)
        for idx, (low, high) in enumerate(VELOCITY_BINS_MS, start=1):
            prefix = f"velocity_bin_{idx}_{low}_{high}_ms"
            row[f"{prefix}_number_flux_m2_s"] = vel_num[idx - 1]
            row[f"{prefix}_mass_flux_kg_m2_s"] = vel_mass[idx - 1]

        data.append(row)

    return data


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    rows = generate_rows(rows=args.rows, freq_minutes=args.freq, rng=rng)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
