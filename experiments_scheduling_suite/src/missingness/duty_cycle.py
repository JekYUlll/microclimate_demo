from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.missingness.base import enforce_target_observed, init_mask


def generate(df: pd.DataFrame, cfg: Dict, target_col: str, seed: int) -> pd.DataFrame:
    """占空比缺失：每个传感器周期性开/关。"""
    rng = np.random.default_rng(seed)
    mask = init_mask(df)
    period_steps = int(cfg.get("period_steps", 20))
    on_steps = int(cfg.get("on_steps", 5))
    random_phase = bool(cfg.get("random_phase", True))
    budget_k = cfg.get("budget_k")

    t = np.arange(len(mask))
    for col in mask.columns:
        phase = int(rng.integers(0, period_steps)) if random_phase else 0
        mask[col] = ((t + phase) % period_steps < on_steps).astype(int)

    # 可选预算裁剪
    if budget_k is not None:
        k = int(budget_k)
        for i in range(len(mask)):
            on_idx = np.flatnonzero(mask.iloc[i].to_numpy())
            if len(on_idx) > k:
                drop = rng.choice(on_idx, size=len(on_idx) - k, replace=False)
                mask.iloc[i, drop] = 0

    mask = enforce_target_observed(mask, target_col, bool(cfg.get("target_always_observed", True)))
    return mask
