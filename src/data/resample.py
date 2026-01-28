from __future__ import annotations

from typing import Iterable

import pandas as pd


def resample_df(df: pd.DataFrame, freq: str, method: str = "linear") -> pd.DataFrame:
    """
    Resample a time-indexed dataframe to a new frequency.

    Uses time-based interpolation after resampling to align to the target grid.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column to resample.")
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.set_index("timestamp").sort_index()
    numeric_cols = work.select_dtypes(include=["number"]).columns
    resampled = work[numeric_cols].resample(freq).mean()
    if method:
        resampled = resampled.interpolate(method=method, limit_direction="both")
    resampled = resampled.reset_index()
    return resampled
