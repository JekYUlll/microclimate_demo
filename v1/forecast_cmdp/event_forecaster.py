from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventForecasterTrainingConfig:
    horizon: int = 8
    lookback: int = 8
    event_column: str = "event_flag"
    hidden_dim: int = 128
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    device: str = "auto"
    probability_prefix: str = "learned_event_p"
    period_steps: int = 8


@dataclass
class EventForecastDataset:
    features: np.ndarray
    targets: np.ndarray
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


@dataclass
class EventForecasterBundle:
    model: Any
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    cfg: EventForecasterTrainingConfig
    history: dict[str, list[float]]

    @property
    def probability_columns(self) -> tuple[str, ...]:
        return tuple(f"{self.cfg.probability_prefix}_h{idx + 1}" for idx in range(int(self.cfg.horizon)))


def select_event_forecast_columns(
    truth: pd.DataFrame,
    *,
    preferred_columns: tuple[str, ...] | list[str] | None = None,
    event_column: str = "event_flag",
) -> tuple[str, ...]:
    if preferred_columns:
        columns = [str(col) for col in preferred_columns if str(col) in truth.columns]
    else:
        columns = [
            col
            for col in truth.columns
            if col != str(event_column) and np.issubdtype(np.asarray(truth[col]).dtype, np.number)
        ]
    if not columns:
        raise ValueError("No numeric columns available for event forecasting")
    return tuple(columns)


def build_event_forecast_dataset(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    feature_columns: tuple[str, ...],
    event_column: str,
    cfg: EventForecasterTrainingConfig,
) -> EventForecastDataset:
    start, end = int(bounds[0]), int(bounds[1])
    horizon = max(1, int(cfg.horizon))
    max_idx = min(int(end), len(truth) - horizon - 1)
    if max_idx <= start:
        raise ValueError(f"Training bounds {bounds} are too short for horizon={horizon}")
    columns = tuple(str(x) for x in feature_columns)
    raw = truth.loc[:, columns].astype(float).to_numpy(dtype=np.float32)
    train_slice = raw[start:max_idx]
    mean = np.nanmean(train_slice, axis=0).astype(np.float32)
    std = np.nanstd(train_slice, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
    events = _event_array(truth, event_column)
    features = [
        _causal_feature_at(raw, events, idx, mean, std, cfg)
        for idx in range(start, max_idx)
    ]
    targets = [
        events[idx + 1 : idx + 1 + horizon]
        for idx in range(start, max_idx)
    ]
    return EventForecastDataset(
        features=np.vstack(features).astype(np.float32),
        targets=np.vstack(targets).astype(np.float32),
        feature_columns=columns,
        mean=mean,
        std=std,
    )


def train_event_forecaster(
    dataset: EventForecastDataset,
    cfg: EventForecasterTrainingConfig,
) -> EventForecasterBundle:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.features, dtype=np.float32)
    y = np.asarray(dataset.targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("event forecast features and targets must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("event forecast features and targets must have matching rows")
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = EventForecastNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim), horizon=y.shape[1]).to(device)
    pos = np.maximum(np.sum(y, axis=0), 1.0)
    neg = np.maximum(float(y.shape[0]) - pos, 1.0)
    pos_weight = np.clip(neg / pos, 0.5, 10.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "brier": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        briers: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                briers.append(float(torch.mean((prob - yb) ** 2).detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["brier"].append(float(np.mean(briers)) if briers else float("nan"))
    return EventForecasterBundle(
        model=model.eval(),
        feature_columns=dataset.feature_columns,
        mean=dataset.mean.astype(np.float32),
        std=dataset.std.astype(np.float32),
        cfg=cfg,
        history=history,
    )


def predict_event_probabilities(
    truth: pd.DataFrame,
    bundle: EventForecasterBundle,
) -> np.ndarray:
    torch, _, _, _ = _torch_modules()
    raw = truth.loc[:, bundle.feature_columns].astype(float).to_numpy(dtype=np.float32)
    events = _event_array(truth, bundle.cfg.event_column)
    rows = [
        _causal_feature_at(raw, events, idx, bundle.mean, bundle.std, bundle.cfg)
        for idx in range(len(truth))
    ]
    if not rows:
        return np.zeros((0, int(bundle.cfg.horizon)), dtype=np.float32)
    device = next(bundle.model.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=device)
        probs = torch.sigmoid(bundle.model(x)).detach().cpu().numpy().astype(np.float32)
    return np.clip(probs, 0.0, 1.0)


def augment_truth_with_event_forecasts(
    truth: pd.DataFrame,
    bundle: EventForecasterBundle,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = truth.copy()
    probs = predict_event_probabilities(out, bundle)
    columns = bundle.probability_columns
    for idx, column in enumerate(columns):
        out[column] = probs[:, idx] if probs.size else np.zeros(len(out), dtype=np.float32)
    return out, columns


class EventForecastNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int, horizon: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _EventForecastNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.horizon = int(horizon)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(horizon)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _EventForecastNet()


def _causal_feature_at(
    raw: np.ndarray,
    events: np.ndarray,
    idx: int,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: EventForecasterTrainingConfig,
) -> np.ndarray:
    idx = int(np.clip(idx, 0, max(raw.shape[0] - 1, 0)))
    lookback = max(1, int(cfg.lookback))
    start = max(0, idx - lookback + 1)
    current = raw[idx]
    previous = raw[max(0, idx - 1)]
    window = raw[start : idx + 1]
    current_z = _normalize(current, mean, std)
    delta_z = _normalize(current - previous, np.zeros_like(mean), std)
    mean_z = _normalize(np.nanmean(window, axis=0), mean, std)
    event_window = events[start : idx + 1]
    event_rate = float(np.mean(event_window)) if event_window.size else 0.0
    recent_event = float(events[idx]) if events.size else 0.0
    phase = _phase_features(idx, max(1, int(cfg.period_steps)))
    return np.concatenate(
        [
            current_z,
            delta_z,
            mean_z,
            np.asarray([event_rate, recent_event, *phase], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, mean)
    return ((arr - mean) / std).astype(np.float32)


def _phase_features(idx: int, period_steps: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * float(int(idx) % max(1, int(period_steps))) / float(max(1, int(period_steps)))
    return float(np.sin(angle)), float(np.cos(angle))


def _event_array(truth: pd.DataFrame, event_column: str) -> np.ndarray:
    if str(event_column) not in truth.columns:
        return np.zeros(len(truth), dtype=np.float32)
    return truth[str(event_column)].astype(bool).to_numpy(dtype=np.float32)


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
