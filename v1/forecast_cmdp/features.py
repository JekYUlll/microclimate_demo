from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .reuse import ensure_archive_src

ensure_archive_src()


@dataclass(frozen=True)
class ForecastContextConfig:
    horizon: int = 8
    event_column: str = "event_flag"
    wind_column: str = "wind_speed_ms"
    wind_threshold_ms: float = 8.0
    wind_scale_ms: float = 1.5
    truth_future: bool = False


@dataclass(frozen=True)
class EventForecast:
    probabilities: np.ndarray
    time_to_event: float
    confidence: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.probabilities.astype(np.float32),
                np.asarray([self.time_to_event], dtype=np.float32),
                self.confidence.astype(np.float32),
            ],
            axis=0,
        )


def build_event_forecast(
    truth: pd.DataFrame,
    current_idx: int,
    cfg: ForecastContextConfig,
) -> EventForecast:
    """Build explicit event forecast context for policy or teacher inputs.

    ``truth_future=True`` is intended for teacher generation only. The default
    uses a causal wind/event heuristic so deployed policies do not receive future
    truth labels.
    """

    horizon = max(1, int(cfg.horizon))
    idx = int(np.clip(int(current_idx), 0, max(len(truth) - 1, 0)))
    if bool(cfg.truth_future):
        probabilities = _future_truth_event_probabilities(truth, idx, horizon, cfg.event_column)
        confidence = np.ones(horizon, dtype=np.float32)
    else:
        probabilities = _causal_event_probabilities(truth, idx, horizon, cfg)
        confidence = np.clip(np.abs(probabilities - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    time_to_event = _normalized_time_to_event(probabilities)
    return EventForecast(
        probabilities=probabilities.astype(np.float32),
        time_to_event=float(time_to_event),
        confidence=confidence.astype(np.float32),
    )


def append_event_forecast(state: np.ndarray, forecast: EventForecast) -> np.ndarray:
    base = np.asarray(state, dtype=np.float32).reshape(-1)
    return np.concatenate([base, forecast.as_vector()], axis=0).astype(np.float32)


def sensor_timing_features(
    *,
    warmup_steps: Sequence[int],
    power_costs: Sequence[float],
    startup_peaks: Sequence[float],
    horizon: int,
) -> np.ndarray:
    horizon_f = max(float(horizon), 1.0)
    warm = np.asarray(warmup_steps, dtype=np.float32).reshape(-1)
    power = np.asarray(power_costs, dtype=np.float32).reshape(-1)
    peak = np.asarray(startup_peaks, dtype=np.float32).reshape(-1)
    if not (warm.shape == power.shape == peak.shape):
        raise ValueError("warmup_steps, power_costs and startup_peaks must have matching lengths")
    power_norm = power / max(float(np.max(power)), 1.0e-6)
    peak_norm = peak / max(float(np.max(peak)), 1.0e-6)
    return np.stack([warm / horizon_f, power_norm, peak_norm], axis=1).astype(np.float32)


def _future_truth_event_probabilities(
    truth: pd.DataFrame,
    current_idx: int,
    horizon: int,
    event_column: str,
) -> np.ndarray:
    if event_column not in truth.columns:
        return np.zeros(horizon, dtype=np.float32)
    events = truth[event_column].astype(bool).to_numpy()
    start = int(current_idx) + 1
    end = start + int(horizon)
    window = events[start:end].astype(np.float32)
    if window.size < horizon:
        window = np.pad(window, (0, horizon - window.size), constant_values=0.0)
    return window.astype(np.float32)


def _causal_event_probabilities(
    truth: pd.DataFrame,
    current_idx: int,
    horizon: int,
    cfg: ForecastContextConfig,
) -> np.ndarray:
    if len(truth) == 0:
        return np.zeros(horizon, dtype=np.float32)
    event_now = 0.0
    if cfg.event_column in truth.columns:
        event_now = float(bool(truth.iloc[current_idx][cfg.event_column]))
    wind = 0.0
    if cfg.wind_column in truth.columns:
        wind = float(truth.iloc[current_idx][cfg.wind_column])
    risk = 1.0 / (1.0 + np.exp(-(wind - float(cfg.wind_threshold_ms)) / max(float(cfg.wind_scale_ms), 1.0e-6)))
    base = max(0.85 * event_now, 0.10 + 0.70 * risk)
    decay = np.power(0.96, np.arange(int(horizon), dtype=np.float32))
    return np.clip(base * decay, 0.0, 1.0).astype(np.float32)


def _normalized_time_to_event(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size == 0:
        return 1.0
    hits = np.flatnonzero(probs >= 0.5)
    if hits.size == 0:
        return 1.0
    return float((int(hits[0]) + 1) / max(int(probs.size), 1))
