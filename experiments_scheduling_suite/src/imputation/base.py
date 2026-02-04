from __future__ import annotations

from typing import Dict

import pandas as pd


class BaseImputer:
    """插补器基类。"""

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def report(self) -> Dict[str, object]:
        return {}
