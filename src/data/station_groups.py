from __future__ import annotations

from typing import Optional

import pandas as pd


def assign_group(meta: pd.DataFrame, elevation_threshold: float = 1000.0) -> pd.DataFrame:
    if "station_id" not in meta.columns:
        raise ValueError("metadata requires 'station_id' column")
    meta = meta.copy()
    if "group_label" in meta.columns:
        meta["group"] = meta["group_label"].where(meta["group_label"].notna() & (meta["group_label"] != ""), None)
    else:
        meta["group"] = None

    if "elevation_m" in meta.columns:
        elev = pd.to_numeric(meta["elevation_m"], errors="coerce")
    else:
        elev = pd.Series([None] * len(meta))

    def _label(row):
        if row["group"]:
            return row["group"]
        val = row["elevation_m"]
        if pd.isna(val):
            return "unknown"
        return "inland" if float(val) >= elevation_threshold else "coastal"

    meta["elevation_m"] = elev
    meta["group"] = meta.apply(_label, axis=1)
    return meta
