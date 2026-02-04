from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def time_split(df: pd.DataFrame, train: float, val: float, test: float) -> Dict[str, slice]:
    """按时间顺序切分训练/验证/测试。"""
    n = len(df)
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
