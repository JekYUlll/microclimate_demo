from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from darts import TimeSeries
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    NaiveDrift,
    NaiveMean,
    NaiveSeasonal,
    Theta,
)

try:  # optional deep models
    from darts.models import NBEATSModel
    from darts.models import TFTModel
except Exception:  # noqa: BLE001
    NBEATSModel = None  # type: ignore[assignment]
    TFTModel = None  # type: ignore[assignment]


@dataclass
class BaselineConfig:
    models: Sequence[str]
    horizon: int = 6
    input_chunk_length: int = 24
    train_ratio: float = 0.8
    stride: int = 1
    max_windows: Optional[int] = None
    season_length: int = 8
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    epochs: int = 20
    seed: int = 42
    tft_hidden_size: int = 32
    tft_num_heads: int = 4
    tft_dropout: float = 0.1
    pl_trainer_kwargs: Optional[Dict[str, object]] = None


def build_model(name: str, cfg: BaselineConfig):
    name = name.lower()
    if name == "naive_mean":
        return NaiveMean()
    if name == "naive_drift":
        return NaiveDrift()
    if name == "naive_seasonal":
        return NaiveSeasonal(K=cfg.season_length)
    if name == "theta":
        return Theta()
    if name == "ets":
        return ExponentialSmoothing()
    if name == "arima":
        p, d, q = cfg.arima_order
        return ARIMA(p=p, d=d, q=q)
    if name == "nbeats":
        if NBEATSModel is None:
            raise RuntimeError("NBEATSModel not available. Ensure darts[torch] is installed.")
        return NBEATSModel(
            input_chunk_length=cfg.input_chunk_length,
            output_chunk_length=cfg.horizon,
            n_epochs=cfg.epochs,
            random_state=cfg.seed,
            force_reset=True,
            pl_trainer_kwargs=cfg.pl_trainer_kwargs,
        )
    if name == "tft":
        if TFTModel is None:
            raise RuntimeError("TFTModel not available. Ensure darts[torch] is installed.")
        return TFTModel(
            input_chunk_length=cfg.input_chunk_length,
            output_chunk_length=cfg.horizon,
            n_epochs=cfg.epochs,
            hidden_size=cfg.tft_hidden_size,
            num_attention_heads=cfg.tft_num_heads,
            dropout=cfg.tft_dropout,
            random_state=cfg.seed,
            force_reset=True,
            pl_trainer_kwargs=cfg.pl_trainer_kwargs,
        )
    raise ValueError(f"Unknown model name: {name}")


Forecasts = Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]


def _normalize_forecasts(forecasts: Forecasts) -> List[TimeSeries]:
    if isinstance(forecasts, TimeSeries):
        return [forecasts]
    if not forecasts:
        return []
    if isinstance(forecasts[0], TimeSeries):  # type: ignore[index]
        return list(forecasts)  # type: ignore[return-value]
    flattened: List[TimeSeries] = []
    for group in forecasts:  # type: ignore[assignment]
        if isinstance(group, TimeSeries):
            flattened.append(group)
        else:
            flattened.extend(list(group))
    return flattened


def _stack_forecasts(forecasts: Forecasts) -> np.ndarray:
    normalized = _normalize_forecasts(forecasts)
    if not normalized:
        return np.empty((0, 0), dtype=np.float32)
    arrays = [ts.values().squeeze(-1) for ts in normalized]
    return np.stack(arrays).astype(np.float32)


def evaluate_model(
    series: TimeSeries,
    values: np.ndarray,
    cfg: BaselineConfig,
    model_name: str,
) -> Dict[str, float | str]:
    split_idx = int(len(values) * cfg.train_ratio)
    split_idx = max(cfg.input_chunk_length, min(split_idx, len(values) - cfg.horizon))

    model = build_model(model_name, cfg)
    train_series = series[:split_idx]
    start = time.time()
    model.fit(train_series)

    forecasts = model.historical_forecasts(
        series,
        start=split_idx,
        forecast_horizon=cfg.horizon,
        stride=cfg.stride,
        retrain=False,
        last_points_only=False,
    )
    preds = _stack_forecasts(forecasts)

    actuals = []
    for idx in range(split_idx, len(values) - cfg.horizon + 1, cfg.stride):
        actuals.append(values[idx : idx + cfg.horizon])
    if not actuals:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "rmse_t1": float("nan"),
            "mae_t1": float("nan"),
            "windows": 0.0,
        }
    actuals_arr = np.stack(actuals).astype(np.float32)

    if cfg.max_windows is not None and len(actuals_arr) > cfg.max_windows:
        actuals_arr = actuals_arr[-cfg.max_windows :]
        preds = preds[-cfg.max_windows :]

    diff = preds - actuals_arr
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))

    step_diff = diff[:, 0]
    rmse_t1 = float(np.sqrt(np.mean(step_diff**2)))
    mae_t1 = float(np.mean(np.abs(step_diff)))

    elapsed = time.time() - start
    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_t1": rmse_t1,
        "mae_t1": mae_t1,
        "windows": float(len(actuals_arr)),
        "seconds": float(elapsed),
    }


def evaluate_all(series: TimeSeries, values: np.ndarray, cfg: BaselineConfig) -> List[Dict[str, float | str]]:
    results: List[Dict[str, float | str]] = []
    for name in cfg.models:
        metrics = evaluate_model(series, values, cfg, name)
        metrics["model"] = name
        results.append(metrics)
    return results
