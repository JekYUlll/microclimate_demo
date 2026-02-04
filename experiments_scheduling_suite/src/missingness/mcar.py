from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.missingness.base import enforce_target_observed, init_mask


def generate(df: pd.DataFrame, cfg: Dict, target_col: str, seed: int) -> pd.DataFrame:
    """MCAR 随机缺失。"""
    rng = np.random.default_rng(seed)
    mask = init_mask(df)
    p_missing = float(cfg.get("p_missing", 0.1))
    per_var = cfg.get("per_variable", {})
    for col in mask.columns:
        p = float(per_var.get(col, p_missing))
        mask[col] = (rng.random(len(mask)) > p).astype(int)
    mask = enforce_target_observed(mask, target_col, bool(cfg.get("target_always_observed", True)))
    return mask
