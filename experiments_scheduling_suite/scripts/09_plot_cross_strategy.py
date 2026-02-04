from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.posthoc.plotting.heatmaps import plot_heatmaps
from experiments_scheduling_suite.src.posthoc.plotting.boxplots import plot_boxplot
from experiments_scheduling_suite.src.posthoc.plotting.curves import plot_sensitivity
from experiments_scheduling_suite.src.posthoc.plotting.scatter import plot_scatter
from experiments_scheduling_suite.src.posthoc.plotting.rankings import rank_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后期横向对比绘图。")
    parser.add_argument("--in", dest="input_csv", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["heatmap", "boxplot", "sensitivity", "scatter", "ranking"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--features", type=Path, default=None, help="missingness_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    out_dir = args.out_dir

    if args.models:
        df = df[df["model"].isin(args.models)]

    if args.mode == "heatmap":
        models = sorted(df["model"].unique())
        for model in models:
            for h in args.horizons:
                plot_heatmaps(df, out_dir, model=model, horizon=h, metric=args.metric, baseline_label=args.baseline, use_delta=True)

    elif args.mode == "boxplot":
        for h in args.horizons:
            plot_boxplot(df, out_dir, horizon=h, metric=args.metric)

    elif args.mode == "sensitivity":
        families = sorted(df["missingness"].unique())
        for fam in families:
            for h in args.horizons:
                plot_sensitivity(df, out_dir, missingness=fam, horizon=h, metric=args.metric, top_models=3)

    elif args.mode == "scatter":
        if args.features is None:
            raise SystemExit("scatter 模式需要 --features missingness_features.csv")
        feats = pd.read_csv(args.features)
        for h in args.horizons:
            for x in ["max_gap_len", "p95_gap_len", "co_missingness_mean"]:
                if x in feats.columns:
                    plot_scatter(df, feats, out_dir, horizon=h, x_col=x, metric=args.metric)

    elif args.mode == "ranking":
        for h in args.horizons:
            rank_correlation(df, out_dir, horizon=h, metric=args.metric)


if __name__ == "__main__":
    main()
