from __future__ import annotations

import math
import random
import time
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..RAW_LSTM.model import LSTMForecaster


@dataclass
class ExperimentConfig:
    data_path: Path
    target_col: str = "Temperature(Ąć)"
    feature_cols: Optional[List[str]] = None
    encoding: Optional[str] = None
    input_window: int = 24
    horizon: int = 6
    train_ratio: float = 0.8
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    device: Optional[str] = None
    max_imfs: Optional[int] = 8
    max_points: Optional[int] = 2000
    plot_last_days: Optional[int] = None
    plot_path: Optional[Path] = None
    log_path: Optional[Path] = None
    log_every: int = 1
    verbose: bool = True
    cudnn_benchmark: bool = True
    matmul_precision: Optional[str] = "high"
    emd_log_interval: int = 300
    max_samples: Optional[int] = None
    emd_max_sift: Optional[int] = None
    emd_spline_kind: Optional[str] = "slinear"


def _log_message(cfg: ExperimentConfig, message: str) -> None:
    if cfg.verbose:
        print(message)
    if cfg.log_path:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def _log_header(cfg: ExperimentConfig) -> None:
    if not cfg.log_path:
        return
    header = [
        "",
        "=" * 80,
        f"EMD-LSTM run started at {datetime.now().isoformat(timespec='seconds')}",
        f"data_path={cfg.data_path}",
        f"target_col={cfg.target_col}",
        f"feature_cols={cfg.feature_cols}",
        f"input_window={cfg.input_window}",
        f"horizon={cfg.horizon}",
        f"train_ratio={cfg.train_ratio}",
        f"epochs={cfg.epochs}",
        f"batch_size={cfg.batch_size}",
        f"learning_rate={cfg.learning_rate}",
        f"hidden_size={cfg.hidden_size}",
        f"num_layers={cfg.num_layers}",
        f"dropout={cfg.dropout}",
        f"seed={cfg.seed}",
        f"device={cfg.device}",
        f"max_imfs={cfg.max_imfs}",
        f"matmul_precision={cfg.matmul_precision}",
        f"cudnn_benchmark={cfg.cudnn_benchmark}",
        f"emd_log_interval={cfg.emd_log_interval}",
        f"max_samples={cfg.max_samples}",
        f"emd_max_sift={cfg.emd_max_sift}",
        f"emd_spline_kind={cfg.emd_spline_kind}",
        "=" * 80,
    ]
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")


class ComponentDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        window_size: int,
        horizon: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.features = features
        self.target = target
        self.window_size = window_size
        self.horizon = horizon
        self.start = start
        self.end = len(features) if end is None else end

        if self.end - self.start < window_size + horizon:
            raise ValueError("Not enough samples to construct a single window.")

    def __len__(self) -> int:
        return self.end - self.start - self.window_size - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = idx + self.start
        window = self.features[idx : idx + self.window_size]
        target_slice = self.target[idx + self.window_size : idx + self.window_size + self.horizon]

        features = torch.from_numpy(window).float()
        target = torch.from_numpy(target_slice).float()
        return features, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_with_fallback(path: Path, encoding: Optional[str]) -> pd.DataFrame:
    encodings: Sequence[str] = [encoding] if encoding else ("utf-8", "latin1")
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, errors="replace") as handle:
                return pd.read_csv(handle)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error:
        raise last_error
    raise ValueError("Unable to read CSV with provided encodings.")


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    lower_map = {str(col).lower(): col for col in df.columns}
    if "timestamp" in lower_map:
        ts = pd.to_datetime(df[lower_map["timestamp"]], errors="coerce")
        if ts.notna().sum() > 0:
            return ts

    ymd = [lower_map.get("year"), lower_map.get("month"), lower_map.get("day")]
    hour_candidates = [
        "three-hourly observation time(utc)",
        "hour",
        "hours",
        "time(utc)",
        "observation time",
        "hour(utc)",
    ]
    hour_col = next((lower_map[c] for c in hour_candidates if c in lower_map), None)
    if all(col is not None for col in ymd) and hour_col is not None:
        ts = pd.to_datetime(
            {
                "year": df[ymd[0]],
                "month": df[ymd[1]],
                "day": df[ymd[2]],
                "hour": df[hour_col],
            },
            errors="coerce",
        )
        if ts.notna().sum() > 0:
            return ts

    for col in df.columns:
        col_str = str(col).lower()
        if "time" in col_str or "时间" in col_str:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.notna().sum() > 0:
                return ts

    raise ValueError("Unable to infer timestamp column.")


def resolve_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name

    lower_map = {str(col).lower(): col for col in df.columns}
    name_lower = name.lower()
    if name_lower in lower_map:
        return lower_map[name_lower]

    return ""


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    resolved = resolve_column(df, target_col)
    if resolved:
        return resolved

    for col in df.columns:
        col_str = str(col).lower()
        if "temperature" in col_str or "气温" in col_str:
            return col

    raise KeyError(f"Target column not found: {target_col}. Available columns: {list(df.columns)}")


def prepare_series(cfg: ExperimentConfig) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    df = read_csv_with_fallback(cfg.data_path, cfg.encoding)
    df = df.replace(["NA", "NaN", "nan", ""], pd.NA)
    df["timestamp"] = build_timestamp(df)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("No valid timestamps after parsing.")

    df = df.sort_values("timestamp")
    target_col = resolve_target_column(df, cfg.target_col)

    target = pd.to_numeric(df[target_col], errors="coerce")
    target = target.interpolate(limit_direction="both").ffill().bfill()
    if target.isna().all():
        raise ValueError("Target column is entirely missing after cleaning.")

    candidate_cols = [col for col in df.columns if col != "timestamp"]
    if cfg.feature_cols:
        resolved_features: List[str] = []
        for name in cfg.feature_cols:
            resolved = resolve_column(df, name)
            if resolved:
                resolved_features.append(resolved)
        feature_cols = [col for col in resolved_features if col in candidate_cols]
    else:
        feature_cols = candidate_cols
    if target_col not in feature_cols:
        feature_cols.append(target_col)
    if not feature_cols:
        raise ValueError(f"No valid feature columns found. Available columns: {list(df.columns)}")

    features_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features_df = features_df.interpolate(limit_direction="both").ffill().bfill()

    timestamps = df["timestamp"].to_numpy()
    features = features_df.to_numpy(dtype=np.float32)
    return target, features, timestamps


def decompose_signal(signal: np.ndarray, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from PyEMD import EMD  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyEMD is required for EMD decomposition. Install with `pip install EMD-signal`."
        ) from exc

    emd_kwargs: dict[str, Any] = {}
    if cfg.emd_max_sift is not None:
        emd_kwargs["max_sift"] = cfg.emd_max_sift
    if cfg.max_imfs is not None:
        emd_kwargs["max_imf"] = cfg.max_imfs
    if cfg.emd_spline_kind:
        emd_kwargs["spline_kind"] = str(cfg.emd_spline_kind)

    emd = EMD(**emd_kwargs) if emd_kwargs else EMD()
    imfs = emd.emd(signal)
    if imfs.size == 0:
        residue = signal
    else:
        residue = signal - np.sum(imfs, axis=0)
    return imfs, residue
def build_datasets(
    features: np.ndarray,
    target: np.ndarray,
    split_idx: int,
    window_size: int,
    horizon: int,
) -> Tuple[ComponentDataset, ComponentDataset]:
    train_dataset = ComponentDataset(features, target, window_size, horizon, start=0, end=split_idx)
    val_start = max(split_idx - window_size - horizon + 1, 0)
    val_dataset = ComponentDataset(features, target, window_size, horizon, start=val_start)
    return train_dataset, val_dataset


def train_component(
    features: np.ndarray,
    target_series: np.ndarray,
    split_idx: int,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    train_features = features[:split_idx]
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    scaled_features = (features - feat_mean) / feat_std

    train_target = target_series[:split_idx]
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if target_std == 0:
        target_std = 1.0
    scaled_target = (target_series - target_mean) / target_std

    train_dataset, val_dataset = build_datasets(
        scaled_features,
        scaled_target,
        split_idx=split_idx,
        window_size=cfg.input_window,
        horizon=cfg.horizon,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = LSTMForecaster(
        input_size=features.shape[1],
        hidden_size=cfg.hidden_size,
        horizon=cfg.horizon,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()
        for features, targets in train_loader:
            features_tensor = torch.as_tensor(features).to(device)
            targets_tensor = torch.as_tensor(targets).to(device)
            optimizer.zero_grad()
            preds = model(features_tensor)
            loss = criterion(preds, targets_tensor)
            loss.backward()
            optimizer.step()
            batch_size = features_tensor.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
        if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
            avg_loss = running_loss / max(seen, 1)
            elapsed = time.time() - epoch_start
            _log_message(cfg, f"  epoch {epoch:03d} | train_loss={avg_loss:.4f} | {elapsed:.1f}s")

    model.eval()
    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    with torch.no_grad():
        for features, targets in val_loader:
            features_tensor = torch.as_tensor(features).to(device)
            preds = model(features_tensor)
            preds_list.append(preds.cpu())
            targets_list.append(targets)

    preds = torch.cat(preds_list).numpy()
    targets = torch.cat(targets_list).numpy()

    preds = preds * target_std + target_mean
    targets = targets * target_std + target_mean
    return preds, targets


def aggregate_metrics(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    diff = preds - targets
    mse_all = float(np.mean(diff**2))
    rmse_all = math.sqrt(mse_all)
    mae_all = float(np.mean(np.abs(diff)))

    step_diff = diff[:, 0]
    mse_step1 = float(np.mean(step_diff**2))
    rmse_step1 = math.sqrt(mse_step1)
    mae_step1 = float(np.mean(np.abs(step_diff)))

    return rmse_all, mae_all, rmse_step1, mae_step1


def run_experiment(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(cfg.seed)
    if torch.cuda.is_available() and (cfg.device is None or "cuda" in cfg.device):
        if cfg.matmul_precision:
            torch.set_float32_matmul_precision(cfg.matmul_precision)
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    _log_header(cfg)

    target, features, timestamps = prepare_series(cfg)
    signal = target.to_numpy(dtype=np.float32)

    if cfg.max_samples is not None and cfg.max_samples > 0 and len(signal) > cfg.max_samples:
        start_idx = len(signal) - cfg.max_samples
        _log_message(cfg, f"Truncating to last {cfg.max_samples} samples (from {len(signal)})")
        signal = signal[start_idx:]
        features = features[start_idx:]
        timestamps = timestamps[start_idx:]

    min_required = cfg.input_window + cfg.horizon
    if len(signal) < min_required * 2:
        raise ValueError("Not enough samples for the requested window/horizon.")

    split_idx = int(len(signal) * cfg.train_ratio)
    split_idx = max(split_idx, min_required)
    split_idx = min(split_idx, len(signal) - min_required)

    extrema = (
        int(np.sum(np.diff(np.sign(np.diff(signal))) != 0)) if len(signal) > 2 else 0
    )
    _log_message(cfg, f"Starting EMD decomposition (samples={len(signal)}, extrema={extrema})...")
    emd_start = time.time()
    stop_event = threading.Event()

    def _emd_heartbeat() -> None:
        while not stop_event.wait(cfg.emd_log_interval):
            elapsed = time.time() - emd_start
            _log_message(cfg, f"EMD still running... elapsed {elapsed:.1f}s")

    heartbeat_thread: Optional[threading.Thread] = None
    if cfg.emd_log_interval > 0:
        heartbeat_thread = threading.Thread(target=_emd_heartbeat, daemon=True)
        heartbeat_thread.start()
    imfs, residue = decompose_signal(signal, cfg)
    stop_event.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)
    _log_message(cfg, f"EMD decomposition done in {time.time() - emd_start:.1f}s")
    if cfg.max_imfs is not None and imfs.shape[0] > cfg.max_imfs:
        imfs = imfs[: cfg.max_imfs]
    components = [imf for imf in imfs] + [residue]

    preds_components: List[np.ndarray] = []
    targets_components: List[np.ndarray] = []

    for idx, comp in enumerate(components, start=1):
        _log_message(cfg, f"Training component {idx}/{len(components)}")
        comp_start = time.time()
        preds, targets = train_component(features, comp, split_idx, cfg)
        _log_message(cfg, f"Component {idx} done in {time.time() - comp_start:.1f}s")
        preds_components.append(preds)
        targets_components.append(targets)

    preds_sum = np.sum(preds_components, axis=0)
    targets_sum = np.sum(targets_components, axis=0)

    rmse_all, mae_all, rmse_step1, mae_step1 = aggregate_metrics(preds_sum, targets_sum)
    _log_message(cfg, f"RMSE (all steps): {rmse_all:.4f}")
    _log_message(cfg, f"MAE  (all steps): {mae_all:.4f}")
    _log_message(cfg, f"RMSE (t+1 only): {rmse_step1:.4f}")
    _log_message(cfg, f"MAE  (t+1 only): {mae_step1:.4f}")

    start_time_idx = split_idx + cfg.input_window
    indices = start_time_idx + np.arange(len(preds_sum))
    indices = np.clip(indices, 0, len(timestamps) - 1)
    time_axis = timestamps[indices]

    if cfg.plot_path:
        import matplotlib.pyplot as plt

        plot_path = cfg.plot_path
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        values_to_plot = slice(None)
        time_to_plot = time_axis
        preds_to_plot = preds_sum[:, 0]
        targets_to_plot = targets_sum[:, 0]

        if cfg.plot_last_days is not None and cfg.plot_last_days > 0:
            cutoff = time_axis[-1] - np.timedelta64(cfg.plot_last_days, "D")
            mask = time_axis >= cutoff
            if mask.any():
                time_to_plot = time_axis[mask]
                preds_to_plot = preds_sum[:, 0][mask]
                targets_to_plot = targets_sum[:, 0][mask]
        elif cfg.max_points is not None and cfg.max_points > 0:
            values_to_plot = slice(-cfg.max_points, None)
            time_to_plot = time_axis[values_to_plot]
            preds_to_plot = preds_sum[:, 0][values_to_plot]
            targets_to_plot = targets_sum[:, 0][values_to_plot]

        plt.figure(figsize=(12, 5))
        plt.plot(time_to_plot, targets_to_plot, label="Actual", linewidth=2)
        plt.plot(time_to_plot, preds_to_plot, label="Predicted", linewidth=2)
        plt.title(f"EMD-LSTM forecast (horizon {cfg.horizon}, first step)")
        plt.xlabel("Timestamp")
        plt.ylabel(cfg.target_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    return preds_sum, targets_sum, time_axis
