from __future__ import annotations

from typing import Dict

import pandas as pd

from experiments_scheduling_suite.src.imputation.base import BaseImputer


class SplineImputer(BaseImputer):
    """样条插值（不足点数则退化为线性）。"""

    def __init__(self, order: int = 3, limit_direction: str = "both") -> None:
        self.order = order
        self.limit_direction = limit_direction
        self._report: Dict[str, object] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        value_cols = [c for c in out.columns if c != "timestamp"]
        out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")
        for col in value_cols:
            series = out[col]
            if series.dropna().shape[0] < self.order + 1:
                out[col] = series.interpolate(method="linear", limit_direction=self.limit_direction).ffill().bfill()
            else:
                out[col] = series.interpolate(method="spline", order=self.order, limit_direction=self.limit_direction)
                out[col] = out[col].ffill().bfill()
        self._report = {"method": "spline", "order": self.order}
        return out

    def report(self) -> Dict[str, object]:
        return self._report
