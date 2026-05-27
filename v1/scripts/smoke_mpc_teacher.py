#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.features import ForecastContextConfig, build_event_forecast
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, enumerate_action_masks
from forecast_cmdp.reuse import ensure_archive_src

ensure_archive_src()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import rollout_metrics, run_policy_rollout  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


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


class DummyOracle:
    is_fitted = True

    class Cfg:
        horizon = 4

    cfg = Cfg()

    def loss(self, feature, future):
        del feature
        weights = np.asarray([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 2.0, 2.0, 4.0])
        return float(np.mean(np.abs(future) * weights.reshape(1, -1)))


def make_truth(n: int = 96) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    event = ((t >= 24) & (t < 40)) | ((t >= 68) & (t < 82))
    return pd.DataFrame(
        {
            "wind_speed_ms": 6.0 + 6.0 * event.astype(float),
            "wind_direction_deg": np.full(n, 210.0),
            "air_temperature_c": -20.0 + 0.02 * np.sin(t / 8.0),
            "relative_humidity": np.full(n, 72.0),
            "air_pressure_pa": np.full(n, 70000.0),
            "solar_radiation_wm2": np.zeros(n),
            "snow_surface_temperature_c": -22.0 + 0.02 * np.cos(t / 8.0),
            "snow_particle_mean_diameter_mm": 0.2 * event.astype(float),
            "snow_particle_mean_velocity_ms": 4.0 * event.astype(float),
            "snow_mass_flux_kg_m2_s": 1.0e-5 * event.astype(float),
            "event_flag": event,
        }
    )


def main() -> None:
    truth = make_truth()
    sensors = [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.2, 0.2, warmup_steps=0),
        SensorSpecV2("snow", ("snow_particle_mean_velocity_ms",), 0.4, 0.5, warmup_steps=2),
        SensorSpecV2("flux", ("snow_mass_flux_kg_m2_s",), 0.1, 0.1, warmup_steps=0),
    ]
    constraints = PowerConstraintsV2(
        max_active=2,
        per_step_budget=0.6,
        startup_peak_budget=0.7,
        required_sensor_ids=("met",),
    )
    env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=32,
            seed=11,
            event_reward_multiplier=1.0,
            energy_account_enabled=True,
            energy_capacity=12.0,
            initial_energy=12.0,
            harvest_per_step=0.35,
            reserve_energy=1.0,
        ),
        oracle=DummyOracle(),
    )
    candidate_masks = enumerate_action_masks(len(sensors), max_active=2)
    policy = MpcTeacherPolicy(
        candidate_masks=candidate_masks,
        cfg=MpcTeacherConfig(planning_horizon=4, beam_width=4, max_branch=6),
    )
    result = run_policy_rollout(env, policy, steps=32, start_idx=18)
    print(rollout_metrics(result))
    forecast = build_event_forecast(truth, 22, ForecastContextConfig(horizon=4, truth_future=True))
    print({"event_probabilities": forecast.probabilities.tolist(), "time_to_event": forecast.time_to_event})


if __name__ == "__main__":
    main()
