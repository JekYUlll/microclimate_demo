from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.utils.metrics import mae, mape, rmse


def _load_original(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "original.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _event_mask(df: pd.DataFrame, event_def: str) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    if event_def == "threshold_exceedance" and "threshold_exceedance" in df.columns:
        return df["threshold_exceedance"].astype(int) == 1
    if event_def == "q90_flux" and "snow_mass_flux_kg_m2_s" in df.columns:
        thr = df["snow_mass_flux_kg_m2_s"].quantile(0.9)
        return df["snow_mass_flux_kg_m2_s"] > thr
    if event_def == "q90_wind" and "wind_speed_ms" in df.columns:
        thr = df["wind_speed_ms"].quantile(0.9)
        return df["wind_speed_ms"] > thr
    return pd.Series([False] * len(df))


def _auto_event_mask(df: pd.DataFrame, event_def: str) -> tuple[pd.Series, str]:
    """若事件比例过高/过低，自动回退到更稀疏的定义。"""
    mask = _event_mask(df, event_def)
    if df.empty:
        return mask, event_def
    rate = float(mask.mean()) if len(mask) else 0.0
    if 0.05 <= rate <= 0.95:
        return mask, event_def

    # 自动回退顺序：q90_flux -> q90_wind
    for fallback in ["q90_flux", "q90_wind"]:
        fallback_mask = _event_mask(df, fallback)
        rate = float(fallback_mask.mean()) if len(fallback_mask) else 0.0
        if 0.05 <= rate <= 0.95:
            return fallback_mask, fallback
    return mask, event_def


def compute_event_metrics(run_dir: Path, event_def: str) -> pd.DataFrame:
    """基于预测文件计算事件/非事件误差。"""
    preds_dir = run_dir / "preds"
    processed_dir = run_dir.parent.parent / "data" / "processed" / run_dir.name

    original = _load_original(processed_dir)
    if original.empty:
        return pd.DataFrame()

    rows = []
    for path in preds_dir.glob("*_h*.csv"):
        name = path.stem
        model, h_str = name.rsplit("_h", 1)
        h = int(h_str)
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        merged = pd.merge(df, original, on="timestamp", how="left")
        if merged.empty:
            continue
        event_mask, event_used = _auto_event_mask(merged, event_def)
        n_event = int(event_mask.sum())
        n_nonevent = int((~event_mask).sum())
        event_rate = float(n_event / max(len(event_mask), 1))
        for label, mask in [("event", event_mask), ("non_event", ~event_mask)]:
            if mask.sum() == 0:
                continue
            y_true = merged.loc[mask, "y_true"].to_numpy()
            y_pred = merged.loc[mask, "y_pred"].to_numpy()
            rows.append({
                "run_id": run_dir.name,
                "model": model,
                "horizon": h,
                "segment": label,
                "event_def": event_used,
                "n_event": n_event,
                "n_nonevent": n_nonevent,
                "event_rate": event_rate,
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "mape": mape(y_true, y_pred),
            })
    return pd.DataFrame(rows)


def compute_imputation_error(run_dir: Path, target_col: str | None = None) -> pd.DataFrame:
    """计算插补误差：仅在被 mask 掉的位置上比较原始 vs 插补。"""
    processed_dir = run_dir.parent.parent / "data" / "processed" / run_dir.name
    original_path = processed_dir / "original.csv"
    imputed_path = processed_dir / "imputed.csv"
    mask_path = processed_dir / "mask.csv"
    if not (original_path.exists() and imputed_path.exists() and mask_path.exists()):
        return pd.DataFrame()

    original = pd.read_csv(original_path)
    imputed = pd.read_csv(imputed_path)
    mask_df = pd.read_csv(mask_path)

    original["timestamp"] = pd.to_datetime(original["timestamp"])
    imputed["timestamp"] = pd.to_datetime(imputed["timestamp"])

    cols = [c for c in original.columns if c != "timestamp"]
    if target_col and target_col in cols:
        cols = [target_col]

    rows = []
    for col in cols:
        if col not in mask_df.columns:
            continue
        masked_pos = mask_df[col] == 0
        if masked_pos.sum() == 0:
            continue
        y_true = pd.to_numeric(original.loc[masked_pos, col], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(imputed.loc[masked_pos, col], errors="coerce").to_numpy()
        # 去除 NaN
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid.sum() == 0:
            continue
        rows.append({
            "run_id": run_dir.name,
            "variable": col,
            "rmse": rmse(y_true[valid], y_pred[valid]),
            "mae": mae(y_true[valid], y_pred[valid]),
            "mape": mape(y_true[valid], y_pred[valid]),
            "count": int(valid.sum()),
        })
    return pd.DataFrame(rows)
