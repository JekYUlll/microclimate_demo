from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _gap_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for i, missing in enumerate(mask):
        if missing and start is None:
            start = i
        if not missing and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _gap_features(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_obs = (~mask).astype(float)
    tslo = np.zeros(len(mask), dtype=float)
    gap_len = np.zeros(len(mask), dtype=float)
    last_obs = None
    current_gap = 0
    for i, missing in enumerate(mask):
        if not missing:
            last_obs = i
            current_gap = 0
            tslo[i] = 0.0
            gap_len[i] = 0.0
        else:
            current_gap += 1
            gap_len[i] = float(current_gap)
            if last_obs is None:
                tslo[i] = float(current_gap)
            else:
                tslo[i] = float(i - last_obs)
    return is_obs, tslo, gap_len


def _apply_gap_limit(series: pd.Series, max_gap_steps: int) -> pd.Series:
    interp = series.interpolate(method="linear", limit_direction="both")
    mask = series.isna().to_numpy()
    for start, end in _gap_segments(mask):
        gap_len = end - start
        if gap_len > max_gap_steps:
            interp.iloc[start:end] = np.nan
    return interp


def impute_A(df: pd.DataFrame, value_cols: Iterable[str], max_gap_steps: int) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        mask = out[col].isna().to_numpy()
        out[col] = _apply_gap_limit(out[col], max_gap_steps=max_gap_steps)
        is_obs, tslo, gap_len = _gap_features(mask)
        out[f"{col}_is_obs"] = is_obs
        out[f"{col}_tslo"] = tslo
        out[f"{col}_gap_len"] = gap_len
    return out


def impute_B_stl(df: pd.DataFrame, value_cols: Iterable[str], period: int = 8) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        series = out[col]
        mask = series.isna().to_numpy()
        interp = series.interpolate(method="linear", limit_direction="both")
        try:
            from statsmodels.tsa.seasonal import STL

            stl = STL(interp, period=period, robust=True)
            res = stl.fit()
            smoothed = res.trend + res.seasonal
            out[col] = smoothed
        except Exception:
            out[col] = interp.rolling(window=period, min_periods=1, center=True).mean()
        is_obs, tslo, gap_len = _gap_features(mask)
        out[f"{col}_is_obs"] = is_obs
        out[f"{col}_tslo"] = tslo
        out[f"{col}_gap_len"] = gap_len
    return out
