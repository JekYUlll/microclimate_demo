from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig


def _standardize_column_names(columns: Iterable[str]) -> List[str]:
    return [str(col).strip() for col in columns]


def load_station_dataframe(path: Path, cfg: DataConfig) -> pd.DataFrame:
    """Load a single Excel file and return a cleaned dataframe."""

    df = pd.read_excel(path)
    df.columns = _standardize_column_names(df.columns)

    timestamp_col = cfg.timestamp_col
    if timestamp_col not in df.columns:
        raise KeyError(f"{timestamp_col} not found in {path.name}")

    available_features = [col for col in cfg.feature_cols if col in df.columns]
    if cfg.target_col not in available_features:
        available_features.append(cfg.target_col)

    use_cols = [timestamp_col] + available_features
    df = df[use_cols]
    df = df.replace("NaN", pd.NA)

    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.set_index("timestamp")

    df = df.resample(cfg.freq).mean()
    df["station"] = path.stem
    df = df.reset_index()
    return df


def load_all_stations(raw_dir: Path, cfg: DataConfig) -> pd.DataFrame:
    """Iterate over every Excel file in the raw data directory."""

    files: Sequence[Path] = sorted(raw_dir.glob("*.xls*"))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {raw_dir}")

    frames = [load_station_dataframe(path, cfg) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["station", "timestamp"])

    numeric_cols = [col for col in combined.columns if col not in {"timestamp", "station"}]
    for col in numeric_cols:
        combined[col] = (
            combined.groupby("station")[col]
            .transform(lambda series: series.interpolate(limit_direction="both").ffill().bfill())
        )

    return combined


def select_features(df: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Return a dataframe containing only the requested feature columns."""

    available = [col for col in cfg.feature_cols if col in df.columns]
    if cfg.target_col not in available:
        available.append(cfg.target_col)
    available = list(dict.fromkeys(available))
    if not available:
        raise ValueError("No usable feature columns were found in the dataframe.")

    return df[["timestamp", "station"] + available].copy(), available


def feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Convert the time ordered dataframe to a numpy feature matrix."""

    numeric_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.ffill().bfill()
    return numeric_df.to_numpy(dtype=np.float32)


def normalization_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute column-wise mean and std with numerical safeguards."""

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class WindowDataset(Dataset):
    """Turn a long multivariate sequence into overlapping windows."""

    def __init__(
        self,
        data: np.ndarray,
        target_idx: int,
        window_size: int,
        horizon: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.data = data
        self.target_idx = target_idx
        self.window_size = window_size
        self.horizon = horizon
        self.start = start
        self.end = len(data) if end is None else end

        if self.end - self.start < window_size + horizon:
            raise ValueError("Not enough samples to construct a single window.")

    def __len__(self) -> int:
        return self.end - self.start - self.window_size - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = idx + self.start
        window = self.data[idx : idx + self.window_size]
        target_slice = self.data[idx + self.window_size : idx + self.window_size + self.horizon, self.target_idx]

        features = torch.from_numpy(window).float()
        target = torch.from_numpy(target_slice).float()
        return features, target
