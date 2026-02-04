from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizons: List[int],
    stride: int = 1,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建监督学习窗口。"""
    X_list = []
    y_list = []
    t_list = []
    values = df[feature_cols].to_numpy()
    target = df[target_col].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    max_h = max(horizons)
    for start in range(0, len(df) - lookback - max_h + 1, stride):
        end = start + lookback
        X_list.append(values[start:end])
        y_list.append([target[end + h - 1] for h in horizons])
        t_list.append(timestamps[end - 1])
        if max_windows and len(X_list) >= max_windows:
            break
    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    t_ref = np.asarray(t_list)
    return X, y, t_ref
