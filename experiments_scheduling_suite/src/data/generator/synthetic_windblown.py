from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# 粒径与速度分箱，用于生成谱分箱列
SIZE_BINS_UM: List[Tuple[int, int]] = [(50, 100), (100, 200), (200, 400), (400, 800)]
VELOCITY_BINS_MS: List[Tuple[int, int]] = [(0, 2), (2, 4), (4, 6), (6, 8)]


def _ar1_scaled(prev: float, mean: float, phi_base: float, noise_base: float, dt_scale: float, rng: np.random.Generator) -> float:
    """带时间步长缩放的 AR(1)。"""
    phi_dt = float(phi_base ** dt_scale)
    noise_dt = float(noise_base * np.sqrt(dt_scale))
    return phi_dt * prev + (1 - phi_dt) * mean + rng.normal(0.0, noise_dt)


def _solar_flux(second_of_day: float, rng: np.random.Generator) -> float:
    """日变化太阳辐射（W/m^2）。"""
    phase = 2 * np.pi * (second_of_day / (24 * 60 * 60))
    base = max(0.0, np.sin(phase))
    return max(0.0, 600.0 * base + rng.normal(0.0, 40.0))


def _distribute(total: float, bins: int, rng: np.random.Generator) -> np.ndarray:
    """总量分配到分箱。"""
    if total <= 0:
        return np.zeros(bins)
    weights = rng.dirichlet(np.ones(bins))
    return total * weights


def generate(rows: int, freq_seconds: int, seed: int = 42) -> pd.DataFrame:
    """生成合成风吹雪数据。"""
    rng = np.random.default_rng(seed)
    start = datetime.now(tz=timezone.utc).replace(microsecond=0)
    base_step_seconds = 10 * 60
    dt_scale = max(1e-6, float(freq_seconds) / base_step_seconds)

    wind_speed = max(0.5, rng.normal(8.0, 1.5))
    wind_dir = rng.uniform(0, 360)
    air_temp = rng.normal(-12.0, 3.0)
    snow_temp = air_temp - rng.normal(1.5, 0.5)
    pressure = rng.normal(70000, 400)
    rh = np.clip(rng.normal(70, 8), 20, 100)
    snow_supply = float(rng.uniform(0.2, 0.8))
    storm_state = rng.random() < 0.1
    storm_intensity = 1.0

    rows_list: List[Dict[str, Any]] = []
    for i in range(rows):
        ts = start + timedelta(seconds=i * freq_seconds)
        second_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second

        # 风暴状态转移（按步长缩放）
        if storm_state:
            if rng.random() < 1 - (1 - 0.03) ** dt_scale:
                storm_state = False
        else:
            if rng.random() < 1 - (1 - 0.01) ** dt_scale:
                storm_state = True

        storm_intensity = float(
            np.clip(
                _ar1_scaled(storm_intensity, 1.4 if storm_state else 1.0, 0.7, 0.05, dt_scale, rng),
                0.9,
                1.8,
            )
        )

        solar = _solar_flux(second_of_day, rng)
        if storm_state:
            solar *= 0.4 + 0.2 * rng.random()

        temp_mean = -12.0 + 3.0 * np.sin(2 * np.pi * second_of_day / (24 * 60 * 60)) + (0.8 if storm_state else 0.0)
        air_temp = _ar1_scaled(air_temp, temp_mean, 0.92, 0.6, dt_scale, rng)
        snow_temp = _ar1_scaled(snow_temp, air_temp - 2.0 + 0.002 * solar, 0.9, 0.4, dt_scale, rng)

        wind_mean = 8.0 + 1.5 * np.sin(2 * np.pi * second_of_day / (24 * 60 * 60)) + (3.5 if storm_state else 0.0)
        wind_speed = max(0.2, _ar1_scaled(wind_speed, wind_mean * storm_intensity, 0.85, 0.9, dt_scale, rng))
        wind_dir = (wind_dir + rng.normal(0.0, 12.0)) % 360
        pressure = _ar1_scaled(pressure, 70000 - (300 if storm_state else 0), 0.95, 80, dt_scale, rng)
        rh_mean = 75.0 - 0.3 * (air_temp + 10.0) + (8.0 if storm_state else 0.0)
        rh = float(np.clip(_ar1_scaled(rh, rh_mean, 0.9, 4.0, dt_scale, rng), 20, 100))

        turbulence = float(np.clip(0.15 + 0.05 * wind_speed + rng.normal(0.0, 0.05), 0.0, 1.0))
        friction_velocity = max(0.05, 0.08 * wind_speed + rng.normal(0.0, 0.03))
        ri_bulk = float(np.clip(0.15 * (air_temp - snow_temp) - 0.08 * wind_speed + rng.normal(0.0, 0.2), -1.5, 1.5))

        Ut = 6.975 + 0.0033 * (air_temp + 27.27) ** 2
        ratio = max(0.0, (wind_speed - Ut) / max(Ut, 0.1))
        precip_rate = max(0.0, rng.gamma(shape=1.1, scale=0.3))
        if storm_state:
            precip_rate += max(0.0, rng.gamma(shape=1.4, scale=0.8))
        if rh > 85:
            precip_rate *= 1.0 + (rh - 85.0) / 40.0

        snow_supply = float(np.clip(snow_supply + (0.02 * precip_rate - 0.25 * ratio) * dt_scale, 0.0, 1.0))
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
            "quality_flag": rng.choice(["good", "suspect", "missing"], p=[0.85, 0.12, 0.03]),
            "data_source": rng.choice(["sim", "field_logger_a", "field_logger_b"]),
            "missing_reason": "",
        }

        row["wind_snow_coupling"] = snow_mass_flux / max(wind_speed, 0.1)
        row["threshold_exceedance"] = int(wind_speed > Ut)
        row["net_radiation_est"] = 0.7 * solar - 20.0
        row["sensible_heat_flux_est"] = 0.1 * (air_temp - snow_temp) * wind_speed
        row["latent_heat_flux_est"] = 0.02 * rh * wind_speed

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

        rows_list.append(row)

    return pd.DataFrame(rows_list)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
