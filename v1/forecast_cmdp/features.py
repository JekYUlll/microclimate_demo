from __future__ import annotations

from dataclasses import dataclass, field
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
    learned_event_probability_columns: tuple[str, ...] = ()
    continuous_columns: tuple[str, ...] = ()
    continuous_scales: tuple[float, ...] = ()
    continuous_truth_future: bool = False
    learned_continuous_prefix: str = "learned_cont"
    continuous_current_source: str = "truth"


@dataclass(frozen=True)
class EventForecast:
    probabilities: np.ndarray
    time_to_event: float
    confidence: np.ndarray
    continuous: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.probabilities.astype(np.float32),
                np.asarray([self.time_to_event], dtype=np.float32),
                self.confidence.astype(np.float32),
                self.continuous.astype(np.float32).reshape(-1),
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
    elif cfg.learned_event_probability_columns:
        probabilities = _learned_event_probabilities(truth, idx, horizon, cfg.learned_event_probability_columns)
        confidence = np.clip(np.abs(probabilities - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    else:
        probabilities = _causal_event_probabilities(truth, idx, horizon, cfg)
        confidence = np.clip(np.abs(probabilities - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    time_to_event = _normalized_time_to_event(probabilities)
    continuous = _continuous_context_features(truth, idx, horizon, cfg)
    return EventForecast(
        probabilities=probabilities.astype(np.float32),
        time_to_event=float(time_to_event),
        confidence=confidence.astype(np.float32),
        continuous=continuous.astype(np.float32),
    )


def append_event_forecast(state: np.ndarray, forecast: EventForecast) -> np.ndarray:
    base = np.asarray(state, dtype=np.float32).reshape(-1)
    return np.concatenate([base, forecast.as_vector()], axis=0).astype(np.float32)


def event_forecast_feature_names(
    *,
    horizon: int,
    prefix: str = "event_forecast",
    continuous_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    horizon = max(1, int(horizon))
    names = (
        [f"{prefix}_p_h{idx + 1}" for idx in range(horizon)]
        + [f"{prefix}_time_to_event"]
        + [f"{prefix}_confidence_h{idx + 1}" for idx in range(horizon)]
    )
    for column in tuple(str(x) for x in continuous_columns):
        safe = column.replace(" ", "_")
        names.extend(
            [
                f"{prefix}_{safe}_current",
                f"{prefix}_{safe}_future_mean",
                f"{prefix}_{safe}_future_max",
                f"{prefix}_{safe}_future_min",
                f"{prefix}_{safe}_future_std",
                f"{prefix}_{safe}_future_last",
                f"{prefix}_{safe}_future_delta",
            ]
        )
    return tuple(names)


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


def _learned_event_probabilities(
    truth: pd.DataFrame,
    current_idx: int,
    horizon: int,
    probability_columns: tuple[str, ...],
) -> np.ndarray:
    values: list[float] = []
    row = truth.iloc[int(current_idx)] if len(truth) else None
    for column in tuple(str(x) for x in probability_columns)[: int(horizon)]:
        if row is None or column not in truth.columns:
            values.append(0.0)
            continue
        value = float(row[column])
        values.append(value if np.isfinite(value) else 0.0)
    if len(values) < int(horizon):
        values.extend([0.0] * (int(horizon) - len(values)))
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def _continuous_context_features(
    truth: pd.DataFrame,
    current_idx: int,
    horizon: int,
    cfg: ForecastContextConfig,
) -> np.ndarray:
    columns = tuple(str(x) for x in cfg.continuous_columns)
    if not columns:
        return np.zeros(0, dtype=np.float32)
    scales = tuple(float(x) for x in cfg.continuous_scales)
    values: list[float] = []
    for column_idx, column in enumerate(columns):
        scale = scales[column_idx] if column_idx < len(scales) else 1.0
        scale = max(abs(float(scale)), 1.0e-6)
        if column not in truth.columns or len(truth) == 0:
            values.extend([0.0] * 7)
            continue
        series = truth[column].astype(float).to_numpy()
        idx = int(np.clip(int(current_idx), 0, max(len(series) - 1, 0)))
        learned_future = _learned_continuous_values(
            truth,
            current_idx=idx,
            horizon=int(horizon),
            source_column=column,
            prefix=str(cfg.learned_continuous_prefix),
        )
        current_source = str(cfg.continuous_current_source)
        if current_source == "learned_h1":
            if learned_future is None or learned_future.size == 0:
                raise ValueError(
                    f"continuous_current_source=learned_h1 requires learned forecasts for {column}"
                )
            current = float(learned_future[0])
        elif current_source == "truth":
            current = float(series[idx])
        else:
            raise ValueError(f"Unsupported continuous_current_source: {current_source}")
        if not np.isfinite(current):
            current = 0.0
        if bool(cfg.continuous_truth_future):
            start = idx + 1
            end = start + int(horizon)
            future = series[start:end].astype(float)
            if future.size < int(horizon):
                pad_value = float(future[-1]) if future.size else current
                future = np.pad(future, (0, int(horizon) - future.size), constant_values=pad_value)
            future = np.where(np.isfinite(future), future, current)
        elif learned_future is not None:
            future = np.where(np.isfinite(learned_future), learned_future, current)
        else:
            # Causal fallback for deployable paths until a learned continuous
            # forecaster is added: persistence from the current measured state.
            future = np.full(int(horizon), current, dtype=float)
        stats = [
            current,
            float(np.mean(future)),
            float(np.max(future)),
            float(np.min(future)),
            float(np.std(future)),
            float(future[-1]) if future.size else current,
            (float(future[-1]) if future.size else current) - current,
        ]
        values.extend(float(x) / scale for x in stats)
    return np.asarray(values, dtype=np.float32)


def learned_continuous_column_name(prefix: str, source_column: str, horizon_idx: int) -> str:
    safe = str(source_column).replace(" ", "_")
    return f"{str(prefix)}_{safe}_h{int(horizon_idx)}"


def _learned_continuous_values(
    truth: pd.DataFrame,
    *,
    current_idx: int,
    horizon: int,
    source_column: str,
    prefix: str,
) -> np.ndarray | None:
    columns = [
        learned_continuous_column_name(prefix, str(source_column), horizon_idx + 1)
        for horizon_idx in range(int(horizon))
    ]
    if not all(column in truth.columns for column in columns):
        return None
    if len(truth) == 0:
        return np.zeros(int(horizon), dtype=float)
    row = truth.iloc[int(current_idx)]
    values = [float(row[column]) for column in columns]
    return np.asarray(values, dtype=float)


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
