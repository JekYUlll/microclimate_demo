from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def metrics_overall(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, target_cols: List[str], horizons: List[int]) -> pd.DataFrame:
    rows = []
    for h_idx, h in enumerate(horizons):
        for t_idx, target in enumerate(target_cols):
            yt = y_true[:, h_idx, t_idx]
            yp = y_pred[:, h_idx, t_idx]
            rows.append({
                "horizon": int(h),
                "target": target,
                "mae": mae(yt, yp),
                "rmse": rmse(yt, yp),
                "r2": r2(yt, yp),
            })
    return pd.DataFrame(rows)
