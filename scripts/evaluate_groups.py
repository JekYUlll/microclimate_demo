from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics by station groups.")
    parser.add_argument("--metrics", type=Path, required=True, help="Metrics CSV with station_id column.")
    parser.add_argument("--groups", type=Path, required=True, help="Station groups CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    groups = pd.read_csv(args.groups)

    if "station_id" not in metrics.columns:
        raise SystemExit("metrics CSV must include station_id")
    if "station_id" not in groups.columns:
        raise SystemExit("groups CSV must include station_id")

    merged = metrics.merge(groups[["station_id", "group"]], on="station_id", how="left")
    agg = merged.groupby(["group", "model"]).agg({
        "mae": ["mean", "std"],
        "rmse": ["mean", "std"],
        "r2": ["mean", "std"],
    })
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"Wrote grouped metrics to {args.out}")


if __name__ == "__main__":
    main()
