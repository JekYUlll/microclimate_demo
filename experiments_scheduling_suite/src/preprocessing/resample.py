from __future__ import annotations

import pandas as pd

from experiments_scheduling_suite.src.utils.time import regularize_time_index


def resample_to_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """重采样到指定频率并保持 timestamp 列。"""
    return regularize_time_index(df, freq)
