from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def apply_mask(df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    """将 mask 应用于数据（mask=0 的位置置为 NaN）。"""
    out = df.copy()
    value_cols = [c for c in df.columns if c != "timestamp"]
    missing_mask = mask_df[value_cols].to_numpy() == 0
    values = out[value_cols].to_numpy().copy()
    values[missing_mask] = np.nan
    out[value_cols] = values
    return out


def init_mask(df: pd.DataFrame) -> pd.DataFrame:
    """初始化全观测 mask。"""
    cols = [c for c in df.columns if c != "timestamp"]
    return pd.DataFrame(1, index=df.index, columns=cols, dtype=int)


def enforce_target_observed(mask_df: pd.DataFrame, target_col: str, enable: bool) -> pd.DataFrame:
    """确保目标列始终观测。"""
    if enable and target_col in mask_df.columns:
        mask_df[target_col] = 1
    return mask_df
