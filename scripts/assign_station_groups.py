from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.station_groups import assign_group
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign coastal/inland groups for stations.")
    parser.add_argument("--meta", type=Path, default=Path("data/metadata/stations_meta.csv"), help="Station metadata CSV.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/station_groups.csv"), help="Output CSV path.")
    parser.add_argument("--config", type=Path, default=None, help="Config for report paths.")
    parser.add_argument("--elevation-threshold", type=float, default=1000.0, help="Elevation threshold for inland.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else None
    if cfg:
        ensure_dirs(cfg)

    meta = pd.read_csv(args.meta)
    grouped = assign_group(meta, elevation_threshold=args.elevation_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.out, index=False)

    if cfg:
        report_path = cfg["reports_dir"] / "tables" / "station_groups_used.csv"
        grouped.to_csv(report_path, index=False)
    print(f"Wrote station groups to {args.out}")


if __name__ == "__main__":
    main()
