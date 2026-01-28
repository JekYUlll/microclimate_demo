from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def time_split(df: pd.DataFrame, cfg: Dict) -> Dict[str, slice]:
    split_cfg = cfg.get("split", {})
    mode = split_cfg.get("mode", "ratio")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if mode == "year":
        years = df["timestamp"].dt.year
        train_years = split_cfg.get("train_years")
        val_years = split_cfg.get("val_years")
        test_years = split_cfg.get("test_years")
        if not (train_years and val_years and test_years):
            raise ValueError("year split requires train_years/val_years/test_years in config")
        train_idx = df[years.isin(train_years)].index
        val_idx = df[years.isin(val_years)].index
        test_idx = df[years.isin(test_years)].index
        return {
            "train": slice(train_idx.min(), train_idx.max() + 1),
            "val": slice(val_idx.min(), val_idx.max() + 1),
            "test": slice(test_idx.min(), test_idx.max() + 1),
        }

    n = len(df)
    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.1))
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def split_summary(df: pd.DataFrame, splits: Dict[str, slice]) -> pd.DataFrame:
    rows = []
    for name, sl in splits.items():
        sub = df.iloc[sl]
        rows.append({
            "split": name,
            "rows": len(sub),
            "start": str(sub["timestamp"].iloc[0]) if len(sub) else "",
            "end": str(sub["timestamp"].iloc[-1]) if len(sub) else "",
        })
    return pd.DataFrame(rows)
