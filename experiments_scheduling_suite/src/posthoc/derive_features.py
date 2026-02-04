from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _gap_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for missing in mask:
        if missing:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def compute_missingness_features(mask_df: pd.DataFrame, target_col: str | None = None) -> Dict[str, float]:
    value_cols = [c for c in mask_df.columns if c != "timestamp"]
    if target_col and target_col in value_cols:
        non_target_cols = [c for c in value_cols if c != target_col]
    else:
        non_target_cols = value_cols

    mask = mask_df[value_cols].to_numpy()
    missing = (mask == 0).astype(int)

    overall_missing_rate = float(missing.mean()) if mask.size else 0.0
    co_missingness_mean = float(missing.sum(axis=1).mean()) if mask.size else 0.0

    # 非目标列统计
    if non_target_cols:
        non_mask = mask_df[non_target_cols].to_numpy()
        non_missing = (non_mask == 0).astype(int)
        co_missingness_non_target = float(non_missing.sum(axis=1).mean())
        effective_k = float(non_mask.sum(axis=1).mean())
    else:
        co_missingness_non_target = 0.0
        effective_k = 0.0

    # gap 统计
    all_gaps: List[int] = []
    num_gaps = 0
    obs_events = 0
    total_steps = len(mask_df)

    for col in value_cols:
        series = mask_df[col].to_numpy()
        gaps = _gap_lengths(series == 0)
        all_gaps.extend(gaps)
        num_gaps += len(gaps)
        # 观测事件：0->1 转换计数
        transitions = np.where((series[1:] == 1) & (series[:-1] == 0))[0]
        obs_events += len(transitions)

    if all_gaps:
        mean_gap = float(np.mean(all_gaps))
        p90_gap = float(np.percentile(all_gaps, 90))
        p95_gap = float(np.percentile(all_gaps, 95))
        max_gap = float(np.max(all_gaps))
    else:
        mean_gap = p90_gap = p95_gap = max_gap = 0.0

    obs_event_rate = float(obs_events / max(total_steps, 1))

    return {
        "overall_missing_rate": overall_missing_rate,
        "co_missingness_mean": co_missingness_mean,
        "co_missingness_mean_non_target": co_missingness_non_target,
        "mean_gap_len": mean_gap,
        "p90_gap_len": p90_gap,
        "p95_gap_len": p95_gap,
        "max_gap_len": max_gap,
        "num_gaps": float(num_gaps),
        "obs_event_rate": obs_event_rate,
        "effective_k": effective_k,
    }
