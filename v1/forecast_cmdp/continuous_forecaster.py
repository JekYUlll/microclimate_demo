from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import learned_continuous_column_name


@dataclass(frozen=True)
class ContinuousForecasterTrainingConfig:
    horizon: int = 8
    lookback: int = 8
    target_columns: tuple[str, ...] = ()
    hidden_dim: int = 128
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    device: str = "auto"
    prediction_prefix: str = "learned_cont"
    period_steps: int = 8


@dataclass
class ContinuousForecastDataset:
    features: np.ndarray
    targets: np.ndarray
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


@dataclass
class ContinuousForecasterBundle:
    model: Any
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    cfg: ContinuousForecasterTrainingConfig
    history: dict[str, list[float]]

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for target in self.target_columns:
            for horizon_idx in range(int(self.cfg.horizon)):
                columns.append(
                    learned_continuous_column_name(
                        str(self.cfg.prediction_prefix),
                        str(target),
                        horizon_idx + 1,
                    )
                )
        return tuple(columns)


def select_continuous_forecast_columns(
    truth: pd.DataFrame,
    *,
    preferred_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    if preferred_columns:
        columns = [str(col) for col in preferred_columns if str(col) in truth.columns]
    else:
        columns = [
            str(col)
            for col in truth.columns
            if np.issubdtype(np.asarray(truth[col]).dtype, np.number)
        ]
    if not columns:
        raise ValueError("No numeric columns available for continuous forecasting")
    return tuple(columns)


def build_continuous_forecast_dataset(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    cfg: ContinuousForecasterTrainingConfig,
    normalization: ContinuousForecastDataset | None = None,
) -> ContinuousForecastDataset:
    start, end = int(bounds[0]), int(bounds[1])
    horizon = max(1, int(cfg.horizon))
    max_idx = min(int(end), len(truth) - horizon - 1)
    if max_idx <= start:
        raise ValueError(f"Training bounds {bounds} are too short for horizon={horizon}")
    feature_columns = tuple(str(x) for x in feature_columns)
    target_columns = tuple(str(x) for x in target_columns)
    missing_features = [col for col in feature_columns if col not in truth.columns]
    missing_targets = [col for col in target_columns if col not in truth.columns]
    if missing_features:
        raise ValueError(f"Continuous forecast feature columns not found: {missing_features}")
    if missing_targets:
        raise ValueError(f"Continuous forecast target columns not found: {missing_targets}")
    raw_features = truth.loc[:, feature_columns].astype(float).to_numpy(dtype=np.float32)
    raw_targets = truth.loc[:, target_columns].astype(float).to_numpy(dtype=np.float32)
    train_feature_slice = raw_features[start:max_idx]
    if normalization is None:
        feature_mean = np.nanmean(train_feature_slice, axis=0).astype(np.float32)
        feature_std = np.nanstd(train_feature_slice, axis=0).astype(np.float32)
        feature_mean = np.where(
            np.isfinite(feature_mean), feature_mean, 0.0
        ).astype(np.float32)
        feature_std = np.where(
            feature_std > 1.0e-6, feature_std, 1.0
        ).astype(np.float32)
    else:
        if tuple(normalization.feature_columns) != feature_columns:
            raise ValueError("normalization feature columns do not match")
        if tuple(normalization.target_columns) != target_columns:
            raise ValueError("normalization target columns do not match")
        feature_mean = np.asarray(normalization.feature_mean, dtype=np.float32)
        feature_std = np.asarray(normalization.feature_std, dtype=np.float32)
    features = [
        _causal_feature_at(raw_features, idx, feature_mean, feature_std, cfg)
        for idx in range(start, max_idx)
    ]
    targets = [
        raw_targets[idx + 1 : idx + 1 + horizon].reshape(-1)
        for idx in range(start, max_idx)
    ]
    targets_arr = np.vstack(targets).astype(np.float32)
    if normalization is None:
        target_mean = np.nanmean(targets_arr, axis=0).astype(np.float32)
        target_std = np.nanstd(targets_arr, axis=0).astype(np.float32)
        target_mean = np.where(
            np.isfinite(target_mean), target_mean, 0.0
        ).astype(np.float32)
        target_std = np.where(
            target_std > 1.0e-6, target_std, 1.0
        ).astype(np.float32)
    else:
        target_mean = np.asarray(normalization.target_mean, dtype=np.float32)
        target_std = np.asarray(normalization.target_std, dtype=np.float32)
    normalized_targets = ((targets_arr - target_mean.reshape(1, -1)) / target_std.reshape(1, -1)).astype(np.float32)
    normalized_targets = np.where(np.isfinite(normalized_targets), normalized_targets, 0.0).astype(np.float32)
    return ContinuousForecastDataset(
        features=np.vstack(features).astype(np.float32),
        targets=normalized_targets,
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def train_continuous_forecaster(
    dataset: ContinuousForecastDataset,
    cfg: ContinuousForecasterTrainingConfig,
) -> ContinuousForecasterBundle:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.features, dtype=np.float32)
    y = np.asarray(dataset.targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("continuous forecast features and targets must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("continuous forecast features and targets must have matching rows")
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ContinuousForecastNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim), output_dim=y.shape[1]).to(device)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "rmse": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        rmses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            with torch.no_grad():
                rmses.append(float(torch.sqrt(torch.mean((pred - yb) ** 2)).detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["rmse"].append(float(np.mean(rmses)) if rmses else float("nan"))
    return ContinuousForecasterBundle(
        model=model.eval(),
        feature_columns=dataset.feature_columns,
        target_columns=dataset.target_columns,
        feature_mean=dataset.feature_mean.astype(np.float32),
        feature_std=dataset.feature_std.astype(np.float32),
        target_mean=dataset.target_mean.astype(np.float32),
        target_std=dataset.target_std.astype(np.float32),
        cfg=cfg,
        history=history,
    )


def predict_continuous_values(
    truth: pd.DataFrame,
    bundle: ContinuousForecasterBundle,
) -> np.ndarray:
    torch, _, _, _ = _torch_modules()
    raw = truth.loc[:, bundle.feature_columns].astype(float).to_numpy(dtype=np.float32)
    rows = [
        _causal_feature_at(raw, idx, bundle.feature_mean, bundle.feature_std, bundle.cfg)
        for idx in range(len(truth))
    ]
    horizon = max(1, int(bundle.cfg.horizon))
    n_targets = max(1, len(bundle.target_columns))
    if not rows:
        return np.zeros((0, horizon, n_targets), dtype=np.float32)
    device = next(bundle.model.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=device)
        normalized = bundle.model(x).detach().cpu().numpy().astype(np.float32)
    values = normalized * bundle.target_std.reshape(1, -1) + bundle.target_mean.reshape(1, -1)
    values = np.where(np.isfinite(values), values, bundle.target_mean.reshape(1, -1)).astype(np.float32)
    return values.reshape(len(rows), horizon, n_targets).astype(np.float32)


def predict_continuous_features(
    features: np.ndarray,
    bundle: ContinuousForecasterBundle,
) -> np.ndarray:
    torch, _, _, _ = _torch_modules()
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("continuous forecast features must be 1D or 2D")
    device = next(bundle.model.parameters()).device
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        normalized = bundle.model(tensor).detach().cpu().numpy().astype(np.float32)
    values = (
        normalized * bundle.target_std.reshape(1, -1)
        + bundle.target_mean.reshape(1, -1)
    )
    horizon = max(1, int(bundle.cfg.horizon))
    return values.reshape(x.shape[0], horizon, len(bundle.target_columns)).astype(
        np.float32
    )


def predict_continuous_from_history(
    history: np.ndarray,
    *,
    history_columns: tuple[str, ...],
    current_idx: int,
    bundle: ContinuousForecasterBundle,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("history must be 2D")
    index = {str(name): idx for idx, name in enumerate(history_columns)}
    missing = [name for name in bundle.feature_columns if name not in index]
    if missing:
        raise ValueError(f"history is missing forecast features: {missing}")
    raw = values[:, [index[name] for name in bundle.feature_columns]]
    feature = _causal_feature_at(
        raw,
        raw.shape[0] - 1,
        bundle.feature_mean,
        bundle.feature_std,
        bundle.cfg,
        phase_idx=int(current_idx),
    )
    return predict_continuous_features(feature, bundle)[0]


def augment_truth_with_continuous_forecasts(
    truth: pd.DataFrame,
    bundle: ContinuousForecasterBundle,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = truth.copy()
    values = predict_continuous_values(out, bundle)
    columns = bundle.prediction_columns
    col_idx = 0
    for target_idx, _target in enumerate(bundle.target_columns):
        for horizon_idx in range(int(bundle.cfg.horizon)):
            column = columns[col_idx]
            out[column] = values[:, horizon_idx, target_idx] if values.size else np.zeros(len(out), dtype=np.float32)
            col_idx += 1
    return out, columns


class ContinuousForecastNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int, output_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _ContinuousForecastNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.output_dim = int(output_dim)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(output_dim)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _ContinuousForecastNet()


def _causal_feature_at(
    raw: np.ndarray,
    idx: int,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: ContinuousForecasterTrainingConfig,
    phase_idx: int | None = None,
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
    std_z = np.nanstd(window, axis=0).astype(np.float32) / std
    std_z = np.where(np.isfinite(std_z), std_z, 0.0).astype(np.float32)
    phase = _phase_features(
        idx if phase_idx is None else int(phase_idx),
        max(1, int(cfg.period_steps)),
    )
    return np.concatenate(
        [
            current_z,
            delta_z,
            mean_z,
            std_z,
            np.asarray(phase, dtype=np.float32),
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
