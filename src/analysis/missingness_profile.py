from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, cast

import numpy as np
import pandas as pd


def _gap_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for is_missing in mask:
        if is_missing:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def profile(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows_all_missing = int(df[list(cols)].isna().all(axis=1).sum())
    summary_rows = []
    for col in cols:
        missing_ratio = float(df[col].isna().mean())
        gaps = _gap_lengths(df[col].isna().to_numpy())
        if gaps:
            stats = {
                "p50": float(np.percentile(gaps, 50)),
                "p90": float(np.percentile(gaps, 90)),
                "p95": float(np.percentile(gaps, 95)),
                "max_gap": float(np.max(gaps)),
                "longest_gap": float(np.max(gaps)),
            }
        else:
            stats = {"p50": 0.0, "p90": 0.0, "p95": 0.0, "max_gap": 0.0, "longest_gap": 0.0}
        summary_rows.append({
            "variable": col,
            "missing_ratio": missing_ratio,
            **stats,
        })
    summary = pd.DataFrame(summary_rows)
    meta = {"rows_all_missing": float(rows_all_missing)}
    return summary, meta


def missing_by_year(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    work = df.copy()
    work["year"] = pd.to_datetime(work["timestamp"]).dt.year
    rows = []
    for col in cols:
        grp = work.groupby("year")[col].apply(lambda s: float(s.isna().mean()))
        for year, ratio in grp.items():
            year_value = int(cast(int, year))
            rows.append({"year": year_value, "variable": col, "missing_ratio": ratio})
    return pd.DataFrame(rows)


def gap_length_distribution(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    all_gaps: List[int] = []
    for col in cols:
        all_gaps.extend(_gap_lengths(df[col].isna().to_numpy()))
    return np.array(all_gaps, dtype=float)
