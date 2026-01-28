from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.eval.metrics import mae, rmse


def extreme_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    horizons: List[int],
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for t_idx, target in enumerate(target_cols):
        true_vals = y_true[:, :, t_idx].reshape(-1)
        pred_vals = y_pred[:, :, t_idx].reshape(-1)
        n = len(true_vals)
        if n == 0:
            continue
        low_thresh = np.quantile(true_vals, bottom_pct)
        high_thresh = np.quantile(true_vals, 1 - top_pct)

        low_mask = true_vals <= low_thresh
        high_mask = true_vals >= high_thresh

        for label, mask in [("bottom", low_mask), ("top", high_mask)]:
            if mask.sum() == 0:
                continue
            rows.append({
                "target": target,
                "slice": label,
                "mae": mae(true_vals[mask], pred_vals[mask]),
                "rmse": rmse(true_vals[mask], pred_vals[mask]),
                "count": int(mask.sum()),
            })
    return pd.DataFrame(rows)
