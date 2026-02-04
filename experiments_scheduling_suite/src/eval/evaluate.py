from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.utils.metrics import mae, mape, rmse


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, horizons: List[int]) -> Dict[str, float]:
    """计算每个 horizon 的指标并汇总。"""
    metrics: Dict[str, float] = {}
    for i, h in enumerate(horizons):
        metrics[f"rmse_h{h}"] = rmse(y_true[:, i], y_pred[:, i])
        metrics[f"mae_h{h}"] = mae(y_true[:, i], y_pred[:, i])
        metrics[f"mape_h{h}"] = mape(y_true[:, i], y_pred[:, i])
    metrics["model"] = model_name
    return metrics


def build_long_table(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, horizons: List[int]) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(horizons):
        rows.append({
            "model": model_name,
            "horizon": h,
            "rmse": rmse(y_true[:, i], y_pred[:, i]),
            "mae": mae(y_true[:, i], y_pred[:, i]),
            "mape": mape(y_true[:, i], y_pred[:, i]),
        })
    return pd.DataFrame(rows)
