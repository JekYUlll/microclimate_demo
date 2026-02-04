from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.missingness.base import enforce_target_observed, init_mask


def _corr_weight(x: pd.Series, y: pd.Series, lag: int) -> float:
    if lag > 0:
        x = x.shift(lag)
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 2:
        return 0.0
    corr = np.corrcoef(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def _compute_weights(df: pd.DataFrame, target_col: str, candidates: List[str], lag_steps: List[int]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for col in candidates:
        max_corr = 0.0
        for lag in lag_steps:
            max_corr = max(max_corr, _corr_weight(df[col], df[target_col], lag))
        weights[col] = max_corr
    return weights


def generate(
    df: pd.DataFrame,
    cfg: Dict,
    target_col: str,
    seed: int,
    train_slice: Optional[slice] = None,
) -> pd.DataFrame:
    """信息优先级：按与目标相关性排序选择 top-k。"""
    rng = np.random.default_rng(seed)
    mask = init_mask(df)
    budget_k = int(cfg.get("budget_k", 1))
    min_on_steps = int(cfg.get("min_on_steps", 1))
    lag_steps = [int(x) for x in cfg.get("lag_steps", [0, 1, 2, 4])]
    refresh_steps = cfg.get("refresh_steps")

    candidates = [c for c in mask.columns if c != target_col]
    base_df = df.iloc[train_slice] if train_slice is not None else df
    base_weights = _compute_weights(base_df, target_col, candidates, lag_steps)

    n = len(df)
    block_len = max(1, min_on_steps)
    for start in range(0, n, block_len):
        end = min(n, start + block_len)
        weights = base_weights
        if refresh_steps and start > 0 and start % int(refresh_steps) == 0:
            weights = _compute_weights(df.iloc[:start], target_col, candidates, lag_steps)
        sorted_cols = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        selected = [c for c, _ in sorted_cols[: min(budget_k, len(sorted_cols))]]
        mask.iloc[start:end, :] = 0
        mask.loc[start:end, selected] = 1

    mask = enforce_target_observed(mask, target_col, bool(cfg.get("target_always_observed", True)))
    return mask
