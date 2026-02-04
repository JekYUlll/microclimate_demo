from __future__ import annotations

from typing import Optional

import pandas as pd


def regularize_time_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把数据重采样到规则频率，保留 timestamp 列。"""
    work = df.copy()
    work = work.set_index("timestamp").sort_index()
    work = work.resample(freq).asfreq()
    work = work.reset_index()
    return work


def infer_step_seconds(df: pd.DataFrame) -> Optional[float]:
    """估计时间步长（秒）。"""
    if len(df) < 2:
        return None
    diffs = df["timestamp"].diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.median().total_seconds())
