from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.plots.pretrain_viz import run_pretrain_viz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预训练数据可视化。")
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = PROJECT_ROOT / "data" / "processed" / args.run_id
    reports_dir = PROJECT_ROOT / "reports" / args.run_id

    original = pd.read_csv(processed_dir / "original.csv")
    masked = pd.read_csv(processed_dir / "masked.csv")
    imputed = pd.read_csv(processed_dir / "imputed.csv")
    mask_df = pd.read_csv(processed_dir / "mask.csv")

    for df in (original, masked, imputed):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    target = "wind_speed_ms" if "wind_speed_ms" in imputed.columns else imputed.columns[1]
    feature_cols = [c for c in imputed.columns if c not in {"timestamp", target}]

    out_dir = reports_dir / "figures" / "pretrain"
    run_pretrain_viz(original, masked, imputed, mask_df, target, out_dir, feature_cols)
    print(f"Saved pretrain figures -> {out_dir}")


if __name__ == "__main__":
    main()
