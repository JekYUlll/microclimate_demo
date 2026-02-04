from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.missingness.base import enforce_target_observed, init_mask


def _apply_blocks(mask: pd.DataFrame, col: str, n_blocks: int, min_len: int, max_len: int, rng: np.random.Generator) -> None:
    n = len(mask)
    for _ in range(n_blocks):
        start = int(rng.integers(0, n))
        length = int(rng.integers(min_len, max_len + 1))
        end = min(n, start + length)
        mask.loc[start:end, col] = 0


def generate(df: pd.DataFrame, cfg: Dict, target_col: str, seed: int) -> pd.DataFrame:
    """块状缺失（连续故障）。"""
    rng = np.random.default_rng(seed)
    mask = init_mask(df)
    n_blocks = int(cfg.get("n_blocks", 5))
    min_len = int(cfg.get("min_len_steps", 5))
    max_len = int(cfg.get("max_len_steps", 20))
    per_variable = bool(cfg.get("per_variable", True))

    if per_variable:
        for col in mask.columns:
            _apply_blocks(mask, col, n_blocks, min_len, max_len, rng)
    else:
        # 共享块：对所有变量同时缺失
        for _ in range(n_blocks):
            start = int(rng.integers(0, len(mask)))
            length = int(rng.integers(min_len, max_len + 1))
            end = min(len(mask), start + length)
            mask.iloc[start:end, :] = 0

    mask = enforce_target_observed(mask, target_col, bool(cfg.get("target_always_observed", True)))
    return mask
