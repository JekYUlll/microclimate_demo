from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.imputation.base import BaseImputer


def _kalman_smooth(series: pd.Series, process_var: float, obs_var: float) -> np.ndarray:
    """简单一维局部水平 Kalman 平滑。"""
    y = series.to_numpy(dtype=float)
    n = len(y)
    x = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)

    # 初始化
    x[0] = y[0] if not np.isnan(y[0]) else 0.0
    p[0] = 1.0

    for t in range(1, n):
        # 预测
        x_pred = x[t - 1]
        p_pred = p[t - 1] + process_var
        if np.isnan(y[t]):
            x[t] = x_pred
            p[t] = p_pred
        else:
            k = p_pred / (p_pred + obs_var)
            x[t] = x_pred + k * (y[t] - x_pred)
            p[t] = (1 - k) * p_pred

    # 平滑（简单 RTS）
    for t in range(n - 2, -1, -1):
        if p[t] + process_var == 0:
            continue
        c = p[t] / (p[t] + process_var)
        x[t] = x[t] + c * (x[t + 1] - x[t])
    return x


class KalmanImputer(BaseImputer):
    """按列进行一维 Kalman 平滑插补。"""

    def __init__(self, process_var: float = 1e-4, obs_var: float = 1e-2) -> None:
        self.process_var = process_var
        self.obs_var = obs_var
        self._report: Dict[str, object] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        value_cols = [c for c in out.columns if c != "timestamp"]
        out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")
        for col in value_cols:
            out[col] = _kalman_smooth(out[col], self.process_var, self.obs_var)
        self._report = {"method": "kalman", "process_var": self.process_var, "obs_var": self.obs_var}
        return out

    def report(self) -> Dict[str, object]:
        return self._report
