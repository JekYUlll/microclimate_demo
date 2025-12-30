"""
Utility to synthesize wind-blown snow observations as a CSV for EMD + sequence models.

Fields mirror the proto layout in schemas/windblown_snow.proto, with fixed spectrum bins
expanded into flat columns to keep CSV parsing simple.
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
        default=500,
        help="Number of observations to generate.",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
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
        default=Path("data/synthetic/windblown_snow_sample.csv"),
        help="Output CSV path.",
    )
    return parser


def generate_rows(rows: int, freq_minutes: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    start = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    data: List[Dict[str, Any]] = []

    for i in range(rows):
        ts = start + timedelta(minutes=i * freq_minutes)
        wind_speed = rng.normal(8.0, 2.0)
        snow_mass_flux = max(0.0, rng.lognormal(mean=-5, sigma=1.0))
        snow_number_flux = snow_mass_flux * rng.uniform(1e3, 1e4)
        turbulence = abs(rng.normal(0.2, 0.05))
        friction_velocity = max(0.05, rng.normal(0.4, 0.1))

        row: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "air_temperature_c": rng.normal(-5.0, 7.0),
            "relative_humidity": np.clip(rng.normal(70, 15), 20, 100),
            "air_pressure_pa": rng.normal(70000, 500),
            "wind_speed_ms": max(0.1, wind_speed),
            "wind_direction_deg": rng.uniform(0, 360),
            "solar_radiation_wm2": max(0.0, rng.normal(200, 300)),
            "snow_surface_temperature_c": rng.normal(-8.0, 5.0),
            "snow_mass_flux_kg_m2_s": snow_mass_flux,
            "snow_number_flux_m2_s": snow_number_flux,
            "turbulence_intensity": turbulence,
            "friction_velocity_ms": friction_velocity,
            "ri_bulk": rng.normal(0.0, 0.5),
            "stability_flag": rng.choice(STABILITY_FLAGS),
            "snow_density_kg_m3": rng.normal(200, 30),
            "snow_grain_temp_c": rng.normal(-8.0, 3.0),
            "visibility_m": max(10.0, rng.normal(500, 150)),
            "precip_rate_mm_h": max(0.0, rng.gamma(shape=1.2, scale=0.5)),
            "quality_flag": rng.choice(["good", "suspect", "missing"], p=[0.85, 0.1, 0.05]),
            "data_source": rng.choice(["sim", "field_logger_a", "field_logger_b"]),
            "missing_reason": "",
        }

        # Derived scalars
        row["wind_snow_coupling"] = snow_mass_flux / row["wind_speed_ms"]
        row["threshold_exceedance"] = int(row["wind_speed_ms"] > 7.5)
        row["net_radiation_est"] = row["solar_radiation_wm2"] * rng.uniform(0.5, 0.9)
        row["sensible_heat_flux_est"] = 0.1 * (row["air_temperature_c"] - row["snow_surface_temperature_c"]) * row["wind_speed_ms"]
        row["latent_heat_flux_est"] = 0.05 * row["relative_humidity"] * row["wind_speed_ms"]

        # Flatten spectra bins
        for idx, (low, high) in enumerate(SIZE_BINS_UM, start=1):
            prefix = f"size_bin_{idx}_{low}_{high}_um"
            row[f"{prefix}_number_flux_m2_s"] = max(0.0, rng.lognormal(mean=-6, sigma=1.2))
            row[f"{prefix}_mass_flux_kg_m2_s"] = max(0.0, rng.lognormal(mean=-7, sigma=1.2))

        for idx, (low, high) in enumerate(VELOCITY_BINS_MS, start=1):
            prefix = f"velocity_bin_{idx}_{low}_{high}_ms"
            row[f"{prefix}_number_flux_m2_s"] = max(0.0, rng.lognormal(mean=-6, sigma=1.0))
            row[f"{prefix}_mass_flux_kg_m2_s"] = max(0.0, rng.lognormal(mean=-7, sigma=1.0))

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
