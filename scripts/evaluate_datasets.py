from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetReport:
    name: str
    rows: int
    start: str
    end: str
    mode_freq_hours: Optional[float]
    est_missing_steps: Optional[int]
    longest_gap_hours: Optional[float]
    dup_timestamps: int
    overall_missing_pct: float
    target_missing_pct: Optional[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.name,
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "mode_freq_h": self.mode_freq_hours,
            "est_missing": self.est_missing_steps,
            "longest_gap_h": self.longest_gap_hours,
            "dup_ts": self.dup_timestamps,
            "missing_pct": self.overall_missing_pct,
            "target_missing_pct": self.target_missing_pct,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "扫描目录下的时间序列数据集，评估是否适合做预测，输出采样频率、缺失率等指标。"
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "AntAWS" / "3_hourly",
        help="包含多个数据集的目录（默认扫描 data/AntAWS/3_hourly）。",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="匹配数据文件的通配符，默认 *.csv。",
    )
    parser.add_argument(
        "--expected-freq",
        type=str,
        default="3H",
        help="期望的采样频率（例如 3H/6H/1D），用于估计缺测步数。",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Temperature(Ąć)",
        help="预测目标列名，用于单独计算缺失率。留空则跳过。",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="文件编码（默认自动尝试 utf-8 和 latin-1）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评估前 N 个文件，默认全部。",
    )
    return parser.parse_args()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 将常见的缺失标记替换为真正的 NaN
    df = df.replace(["NA", "NaN", "nan", ""], pd.NA)
    return df


def read_csv_with_fallback(path: Path, encoding: Optional[str]) -> pd.DataFrame:
    encodings: Sequence[str] = [encoding] if encoding else ("utf-8", "latin1")
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, errors="replace") as handle:
                return pd.read_csv(handle)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error:
        raise last_error
    raise ValueError("无法读取文件，编码未知。")


def infer_timestamp(df: pd.DataFrame) -> pd.Series:
    columns = list(df.columns)
    lower_map = {str(col).lower(): col for col in columns}

    # 直接存在 timestamp 列
    if "timestamp" in lower_map:
        ts = pd.to_datetime(df[lower_map["timestamp"]], errors="coerce")
        if ts.notna().sum() > 0:
            return ts

    # Year/Month/Day + 小时列组合
    ymd = [lower_map.get("year"), lower_map.get("month"), lower_map.get("day")]
    hour_candidates = [
        "three-hourly observation time(utc)",
        "hour",
        "hours",
        "time(utc)",
        "observation time",
        "hour(utc)",
    ]
    hour_col = next((lower_map[c] for c in hour_candidates if c in lower_map), None)
    if all(col is not None for col in ymd) and hour_col is not None:
        ts = pd.to_datetime(
            {
                "year": df[ymd[0]],
                "month": df[ymd[1]],
                "day": df[ymd[2]],
                "hour": df[hour_col],
            },
            errors="coerce",
        )
        if ts.notna().sum() > 0:
            return ts

    # 兜底：尝试任何包含 time/时间 的列
    for col in columns:
        col_str = str(col).lower()
        if "time" in col_str or "时间" in col_str:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.notna().sum() > 0:
                return ts

    raise ValueError("无法自动识别时间列，请检查文件格式。")


def est_missing_steps(timestamps: pd.Series, expected_freq: str) -> tuple[Optional[int], Optional[float], Optional[float]]:
    if timestamps.size < 2:
        return None, None, None

    diffs = timestamps.diff().dropna()
    try:
        expected_delta = pd.to_timedelta(str(expected_freq).lower())
    except ValueError:
        return None, None, None

    expected_seconds = expected_delta.total_seconds()
    if expected_seconds <= 0 or np.isnan(expected_seconds):
        return None, None, None

    longest_gap_hours = float(diffs.max().total_seconds() / 3600.0)
    mode_delta = diffs.mode().iloc[0]
    mode_freq_hours = float(mode_delta.total_seconds() / 3600.0)

    # 估计缺测步数：每个间隔超出 1 个 expected_delta，就认为有缺测
    multiples = diffs.dt.total_seconds() / expected_seconds
    missing = int(np.clip(np.round(multiples) - 1, a_min=0, a_max=None).sum())

    return missing, longest_gap_hours, mode_freq_hours


def summarise_file(
    path: Path,
    expected_freq: str,
    target_col: Optional[str],
    encoding: Optional[str],
) -> DatasetReport:
    df = clean_dataframe(read_csv_with_fallback(path, encoding))
    df["timestamp"] = infer_timestamp(df)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("没有有效的时间戳数据。")

    df = df.sort_values("timestamp")
    dup_ts = int(df["timestamp"].duplicated().sum())
    numeric_cols = [c for c in df.columns if c not in {"timestamp"}]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    overall_missing_pct = float(df[numeric_cols].isna().mean().mean() * 100) if numeric_cols else 0.0

    target_missing = None
    if target_col and target_col in df.columns:
        target_missing = float(df[target_col].isna().mean() * 100)

    missing_steps, longest_gap_hours, mode_freq_hours = est_missing_steps(
        df["timestamp"], expected_freq
    )

    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    return DatasetReport(
        name=path.name,
        rows=len(df),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        mode_freq_hours=mode_freq_hours,
        est_missing_steps=missing_steps,
        longest_gap_hours=longest_gap_hours,
        dup_timestamps=dup_ts,
        overall_missing_pct=overall_missing_pct,
        target_missing_pct=target_missing,
    )


def find_files(root: Path, pattern: str, limit: Optional[int]) -> List[Path]:
    files = sorted(root.glob(pattern))
    if limit is not None:
        files = files[:limit]
    return files


def main() -> None:
    args = parse_args()
    files = find_files(args.root, args.pattern, args.limit)
    if not files:
        raise FileNotFoundError(f"在 {args.root} 下找不到匹配 {args.pattern} 的数据文件。")

    reports: List[DatasetReport] = []
    for path in files:
        try:
            report = summarise_file(path, args.expected_freq, args.target_col or None, args.encoding)
            reports.append(report)
        except Exception as exc:
            print(f"[WARN] {path.name}: {exc}")

    if not reports:
        print("没有可用的数据集评估结果。")
        return

    table = pd.DataFrame([r.as_dict() for r in reports])
    table = table.sort_values("missing_pct")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", lambda x: f"{x:.2f}",
    ):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
