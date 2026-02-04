from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.station_groups import assign_group
from src.utils.config import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Layer B station segments for sampling experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Override processed dir.")
    parser.add_argument("--min-temp-missing", type=float, default=0.10)
    parser.add_argument("--max-aux-missing", type=float, default=0.30)
    parser.add_argument("--min-aux-count", type=int, default=3)
    parser.add_argument("--max-gap-days", type=int, default=7)
    parser.add_argument("--min-months", type=int, default=24)
    parser.add_argument("--search-step-days", type=int, default=30)
    parser.add_argument("--coastal-count", type=int, default=20)
    parser.add_argument("--inland-count", type=int, default=20)
    return parser.parse_args()


def _infer_temp_col(columns: List[str], targets: List[str]) -> Optional[str]:
    for col in targets:
        if col in columns and "temp" in col.lower():
            return col
    for col in columns:
        if "temp" in col.lower():
            return col
    return targets[0] if targets else None


def _step_hours(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 3.0
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return 3.0
    median = diffs.median()
    return float(median.total_seconds() / 3600.0)


def _longest_gap(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for missing in mask:
        if missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _segment_metrics(
    segment: pd.DataFrame,
    temp_col: str,
    aux_cols: List[str],
    max_aux_missing: float,
) -> Dict[str, float]:
    temp_missing = float(segment[temp_col].isna().mean())
    if aux_cols:
        aux_missing_series = segment[aux_cols].isna().mean()
        aux_missing_avg = float(aux_missing_series.mean())
        aux_available = int((aux_missing_series < max_aux_missing).sum())
    else:
        aux_missing_avg = 1.0
        aux_available = 0
    longest_gap = _longest_gap(segment[temp_col].isna().to_numpy())
    return {
        "temp_missing": temp_missing,
        "aux_missing_avg": aux_missing_avg,
        "aux_available": aux_available,
        "longest_gap": longest_gap,
    }


def _find_recent_segment(
    df: pd.DataFrame,
    temp_col: str,
    aux_cols: List[str],
    min_months: int,
    max_gap_steps: int,
    min_temp_missing: float,
    max_aux_missing: float,
    min_aux_count: int,
    search_step_days: int,
    step_hours: float,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, Dict[str, float]]]:
    ts = df["timestamp"]
    min_ts = ts.min()
    max_ts = ts.max()

    offset = pd.DateOffset(months=min_months)
    step = pd.Timedelta(days=search_step_days)
    end_time = max_ts
    expected_steps = int((min_months * 30 * 24) / step_hours) if step_hours > 0 else 0
    min_rows = int(expected_steps * 0.9) if expected_steps > 0 else 0

    while end_time - offset >= min_ts:
        start_time = end_time - offset
        segment = df[(ts >= start_time) & (ts <= end_time)]
        if len(segment) < min_rows:
            end_time -= step
            continue
        metrics = _segment_metrics(segment, temp_col, aux_cols, max_aux_missing)
        if metrics["temp_missing"] > min_temp_missing:
            end_time -= step
            continue
        if metrics["longest_gap"] > max_gap_steps:
            end_time -= step
            continue
        if not (
            metrics["aux_missing_avg"] <= max_aux_missing or metrics["aux_available"] >= min_aux_count
        ):
            end_time -= step
            continue
        return start_time, end_time, metrics
    return None


def _station_id_from_path(path: Path) -> Optional[str]:
    name = path.stem
    if name.endswith("_features"):
        return None
    if "_impute" in name or name.endswith("_masked"):
        return None
    return name


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed_dir = args.processed_dir or cfg["processed_dir"]
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise SystemExit(f"Processed dir not found: {processed_dir}")

    targets = cfg.get("columns", {}).get("targets", [])
    stations = []
    for path in sorted(processed_dir.glob("*.csv")):
        station_id = _station_id_from_path(path)
        if not station_id:
            continue
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        columns = [c for c in df.columns if c != "timestamp"]
        temp_col = _infer_temp_col(columns, targets)
        if temp_col is None or temp_col not in df.columns:
            continue
        aux_cols = [c for c in columns if c != temp_col]
        aux_cols = [c for c in aux_cols if not df[c].isna().all()]
        step_hours = _step_hours(df["timestamp"])
        max_gap_steps = int(args.max_gap_days * (24 / step_hours))

        segment = _find_recent_segment(
            df,
            temp_col,
            aux_cols,
            args.min_months,
            max_gap_steps,
            args.min_temp_missing,
            args.max_aux_missing,
            args.min_aux_count,
            args.search_step_days,
            step_hours,
        )
        if not segment:
            continue
        start_time, end_time, metrics = segment
        stations.append({
            "station_id": station_id,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "temp_missing": metrics["temp_missing"],
            "aux_missing_avg": metrics["aux_missing_avg"],
            "aux_available": metrics["aux_available"],
            "longest_gap": metrics["longest_gap"],
            "step_hours": step_hours,
        })

    if not stations:
        raise SystemExit("No stations met the selection criteria.")

    segments_df = pd.DataFrame(stations)

    # Attach group labels
    meta_path = PROJECT_ROOT / "data" / "metadata" / "stations_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta = assign_group(meta)
        group_map = dict(zip(meta["station_id"], meta["group"]))
        segments_df["group"] = segments_df["station_id"].map(group_map).fillna("unknown")
    else:
        segments_df["group"] = "unknown"

    # Relaxation tiers for aux missing
    thresholds = [args.max_aux_missing, 0.4, 0.5, 0.6]
    tiers = []
    for _, row in segments_df.iterrows():
        tier = len(thresholds)
        for i, th in enumerate(thresholds):
            if row["aux_missing_avg"] <= th or row["aux_available"] >= args.min_aux_count:
                tier = i
                break
        tiers.append(tier)
    segments_df["tier"] = tiers

    selected_ids: List[str] = []
    selected_rows: List[int] = []
    for group, target_n in [("coastal", args.coastal_count), ("inland", args.inland_count)]:
        subset = segments_df[segments_df["group"] == group]
        subset = subset.sort_values(["tier", "temp_missing", "longest_gap", "aux_missing_avg"])
        take = subset.head(target_n)
        selected_ids.extend(take["station_id"].tolist())
        selected_rows.extend(take.index.tolist())

    # Fill with remaining stations if under target
    total_target = args.coastal_count + args.inland_count
    if len(selected_ids) < total_target:
        remaining = segments_df.drop(index=selected_rows)
        remaining = remaining.sort_values(["tier", "temp_missing", "longest_gap", "aux_missing_avg"])
        fill = remaining.head(total_target - len(selected_ids))
        selected_ids.extend(fill["station_id"].tolist())
        selected_rows.extend(fill.index.tolist())

    selected_df = segments_df.loc[selected_rows].copy()

    out_table = cfg["reports_dir"] / "tables" / "layerB_station_segments.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(out_table, index=False)

    out_list = cfg["processed_dir"] / "layerB_station_list.txt"
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(selected_ids))

    print(f"Wrote {len(selected_ids)} stations to {out_list}")
    print(f"Segment table saved to {out_table}")


if __name__ == "__main__":
    main()
