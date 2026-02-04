from __future__ import annotations

from typing import List

import pandas as pd


def leaderboard(metrics_overall: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """按平均 RMSE 排序的 leaderboard。"""
    cols = [f"rmse_h{h}" for h in horizons]
    metrics_overall = metrics_overall.copy()
    metrics_overall["rmse_mean"] = metrics_overall[cols].mean(axis=1)
    return metrics_overall.sort_values("rmse_mean")
