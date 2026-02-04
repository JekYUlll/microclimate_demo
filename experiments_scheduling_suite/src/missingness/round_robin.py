from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.missingness.base import enforce_target_observed, init_mask


def generate(df: pd.DataFrame, cfg: Dict, target_col: str, seed: int) -> pd.DataFrame:
    """轮询调度：按固定顺序轮换每步选择 budget_k 个传感器。"""
    mask = init_mask(df)
    rng = np.random.default_rng(seed)
    budget_k = int(cfg.get("budget_k", 1))
    min_on_steps = int(cfg.get("min_on_steps", 1))
    sensor_order = cfg.get("sensor_order")
    cols = list(mask.columns)
    if sensor_order:
        cols = [c for c in sensor_order if c in cols]
    if not cols:
        return mask

    n = len(mask)
    block_len = max(1, min_on_steps)
    pointer = 0
    for start in range(0, n, block_len):
        end = min(n, start + block_len)
        selected = [cols[(pointer + i) % len(cols)] for i in range(min(budget_k, len(cols)))]
        mask.iloc[start:end, :] = 0
        mask.loc[start:end, selected] = 1
        pointer = (pointer + budget_k) % len(cols)

    mask = enforce_target_observed(mask, target_col, bool(cfg.get("target_always_observed", True)))
    return mask
