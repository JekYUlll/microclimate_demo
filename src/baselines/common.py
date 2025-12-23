from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class SeriesConfig:
    data_path: Path
    target_col: str
    encoding: Optional[str] = None
    freq: Optional[str] = None
    log_path: Optional[Path] = None
    verbose: bool = True


def log_message(cfg: SeriesConfig, message: str) -> None:
    if cfg.verbose:
        print(message)
    if cfg.log_path:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def log_header(cfg: SeriesConfig) -> None:
    if not cfg.log_path:
        return
    header = [
        "",
        "=" * 80,
        f"Baseline run started at {datetime.now().isoformat(timespec='seconds')}",
        f"data_path={cfg.data_path}",
        f"target_col={cfg.target_col}",
        f"freq={cfg.freq}",
        "=" * 80,
    ]
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")


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


def prepare_series(cfg: SeriesConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    df = read_csv_with_fallback(cfg.data_path, cfg.encoding)
    df = df.replace(["NA", "NaN", "nan", ""], pd.NA)
    df["timestamp"] = build_timestamp(df)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("No valid timestamps after parsing.")

    df = df.sort_values("timestamp")
    target_col = resolve_target_column(df, cfg.target_col)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(limit_direction="both").ffill().bfill()
    if df[target_col].isna().all():
        raise ValueError("Target column is entirely missing after cleaning.")

    if cfg.freq:
        df = df.set_index("timestamp")
        df = df.resample(cfg.freq).mean(numeric_only=True)
        df = df.reset_index()

    values = df[target_col].to_numpy(dtype=np.float32)
    return df[["timestamp", target_col]], values


def train_val_indices(total: int, train_ratio: float) -> Tuple[int, int]:
    split_idx = int(total * train_ratio)
    split_idx = max(1, min(split_idx, total - 1))
    return split_idx, total


def windowed_actuals(values: np.ndarray, start: int, horizon: int, stride: int) -> np.ndarray:
    windows = []
    for idx in range(start, len(values) - horizon + 1, stride):
        windows.append(values[idx : idx + horizon])
    if not windows:
        return np.empty((0, horizon), dtype=np.float32)
    return np.stack(windows)
