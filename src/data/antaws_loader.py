from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import re

import pandas as pd


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(cols, keyword: str) -> str | None:
    for c in cols:
        if keyword in _normalize_col(c):
            return c
    return None


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    mapping: Dict[str, str] = {}
    mapping["year"] = _find_column(cols, "year")
    mapping["month"] = _find_column(cols, "month")
    mapping["day"] = _find_column(cols, "day")
    mapping["obs_time"] = _find_column(cols, "threehourlyobservationtimeutc") or _find_column(cols, "observationtimeutc") or _find_column(cols, "timeutc")
    mapping["temperature"] = _find_column(cols, "temperature")
    mapping["pressure"] = _find_column(cols, "pressure")
    mapping["wind_speed"] = _find_column(cols, "windspeed")
    mapping["wind_dir"] = _find_column(cols, "winddirection")
    mapping["rh"] = _find_column(cols, "relativehumidity") or _find_column(cols, "humidity")

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns found: {cols}")
    return mapping


def load_antaws(path: Path, encoding: str | None = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        encodings = [encoding] if encoding else ["utf-8", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception as exc:  # pragma: no cover - fallback
                last_err = exc
                df = None
        if df is None:
            raise last_err

    mapping = _resolve_columns(df)
    out = pd.DataFrame()
    out["year"] = pd.to_numeric(df[mapping["year"]], errors="coerce")
    out["month"] = pd.to_numeric(df[mapping["month"]], errors="coerce")
    out["day"] = pd.to_numeric(df[mapping["day"]], errors="coerce")
    out["obs_time"] = pd.to_numeric(df[mapping["obs_time"]], errors="coerce")
    out = out.dropna(subset=["year", "month", "day"]).reset_index(drop=True)
    out["obs_time"] = out["obs_time"].fillna(0).astype(int)
    out["timestamp"] = pd.to_datetime(
        dict(year=out["year"].astype(int), month=out["month"].astype(int), day=out["day"].astype(int), hour=out["obs_time"]),
        errors="coerce",
    )
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    out["temperature_c"] = pd.to_numeric(df[mapping["temperature"]], errors="coerce")
    out["pressure_hpa"] = pd.to_numeric(df[mapping["pressure"]], errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(df[mapping["wind_speed"]], errors="coerce")
    out["wind_dir_deg"] = pd.to_numeric(df[mapping["wind_dir"]], errors="coerce")
    out["relative_humidity_pct"] = pd.to_numeric(df[mapping["rh"]], errors="coerce")

    out = out[[
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "wind_speed_ms",
        "wind_dir_deg",
        "relative_humidity_pct",
    ]]

    out = out.sort_values("timestamp").reset_index(drop=True)
    before = len(out)
    out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    dropped = before - len(out)

    log = {
        "rows_read": str(len(df)),
        "rows_out": str(len(out)),
        "duplicates_removed": str(dropped),
        "first_timestamp": str(out["timestamp"].iloc[0]) if len(out) else "",
        "last_timestamp": str(out["timestamp"].iloc[-1]) if len(out) else "",
    }
    return out, log
