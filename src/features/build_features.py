from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def build(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])

    features = cfg.get("features", {})
    targets = cfg.get("columns", {}).get("targets", [])
    diff_steps = features.get("diff_steps", [1, 2])
    rolling_windows = features.get("rolling_windows", [2, 8])

    if features.get("include_time_features", True):
        hour = work["timestamp"].dt.hour
        doy = work["timestamp"].dt.dayofyear
        work["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        work["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        work["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        work["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        work["month"] = work["timestamp"].dt.month.astype(float)

    for col in targets:
        if col not in work.columns:
            continue
        for step in diff_steps:
            work[f"d_{col}_{step}"] = work[col].diff(step)
        for window in rolling_windows:
            shifted = work[col].shift(1)
            work[f"roll_{col}_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
            work[f"roll_{col}_std_{window}"] = shifted.rolling(window=window, min_periods=1).std()

    feature_cols = [c for c in work.columns if c != "timestamp"]
    return work, feature_cols
