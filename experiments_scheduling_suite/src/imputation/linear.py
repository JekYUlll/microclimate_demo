from __future__ import annotations

from typing import Dict

import pandas as pd

from experiments_scheduling_suite.src.imputation.base import BaseImputer


class LinearImputer(BaseImputer):
    """线性插值（按时间序列方向）。"""

    def __init__(self, limit_direction: str = "both") -> None:
        self.limit_direction = limit_direction
        self._report: Dict[str, object] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        value_cols = [c for c in out.columns if c != "timestamp"]
        out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")
        out[value_cols] = out[value_cols].interpolate(method="linear", limit_direction=self.limit_direction)
        out[value_cols] = out[value_cols].ffill().bfill()
        self._report = {"method": "linear", "limit_direction": self.limit_direction}
        return out

    def report(self) -> Dict[str, object]:
        return self._report
