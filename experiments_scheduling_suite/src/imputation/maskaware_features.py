from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.imputation.base import BaseImputer


def _time_since_last_seen(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask), dtype=float)
    last = None
    for i, missing in enumerate(mask):
        if not missing:
            last = i
            out[i] = 0.0
        else:
            if last is None:
                out[i] = float(i + 1)
            else:
                out[i] = float(i - last)
    return out


class MaskAwareImputer(BaseImputer):
    """不插补，仅增加缺失指示与 time-since-last-seen 特征。"""

    def __init__(self, fill_value: float = 0.0) -> None:
        self.fill_value = float(fill_value)
        self._report: Dict[str, object] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        value_cols = [c for c in out.columns if c != "timestamp"]
        for col in value_cols:
            mask = out[col].isna().to_numpy()
            out[f"is_missing_{col}"] = mask.astype(float)
            out[f"tsls_{col}"] = _time_since_last_seen(mask)
            out[col] = out[col].fillna(self.fill_value)
        self._report = {"method": "maskaware", "fill_value": self.fill_value}
        return out

    def report(self) -> Dict[str, object]:
        return self._report
