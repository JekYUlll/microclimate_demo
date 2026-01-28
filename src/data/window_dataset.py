from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    window_size: int,
    horizons: List[int],
    stride: int = 1,
    max_windows: int | None = None,
    allow_feature_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    max_h = max(horizons)
    X_list = []
    Y_list = []
    t_refs = []

    for i in range(window_size, len(df) - max_h + 1, stride):
        window = df.iloc[i - window_size : i]
        if window[feature_cols].isna().any().any() and not allow_feature_nan:
            continue
        y_vals = []
        invalid = False
        for h in horizons:
            idx = i + h - 1
            row = df.iloc[idx]
            if row[target_cols].isna().any():
                invalid = True
                break
            y_vals.append(row[target_cols].to_numpy(dtype=float))
        if invalid:
            continue
        X_list.append(window[feature_cols].to_numpy(dtype=float))
        Y_list.append(np.stack(y_vals, axis=0))
        t_refs.append(df.loc[i, "timestamp"])

    if max_windows is not None and len(X_list) > max_windows:
        X_list = X_list[-max_windows:]
        Y_list = Y_list[-max_windows:]
        t_refs = t_refs[-max_windows:]

    if not X_list:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, len(horizons), len(target_cols))), np.empty((0,))

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    t_ref = np.array(t_refs)
    return X, Y, t_ref
