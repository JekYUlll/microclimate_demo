from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.missingness_profile import gap_length_distribution, missing_by_year, profile
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile missingness for AntAWS station data.")
    parser.add_argument("--data", type=Path, required=True, help="Processed station CSV.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path for summary table.")
    parser.add_argument("--config", type=Path, default=None, help="Config to infer report paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else None
    if cfg:
        ensure_dirs(cfg)

    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    value_cols = [c for c in df.columns if c != "timestamp"]

    summary, meta = profile(df, value_cols)
    out_path = args.out or (cfg["reports_dir"] / "tables" / "missingness.csv" if cfg else Path("missingness.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    by_year = missing_by_year(df, value_cols)
    fig_path = cfg["reports_dir"] / "figures" / "missingness_by_year.png" if cfg else Path("missingness_by_year.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for col in value_cols:
        sub = by_year[by_year["variable"] == col]
        plt.plot(sub["year"], sub["missing_ratio"], label=col)
    plt.xlabel("Year")
    plt.ylabel("Missing ratio")
    plt.title("Missingness by year")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    gaps = gap_length_distribution(df, value_cols)
    ccdf_path = cfg["reports_dir"] / "figures" / "gap_length_ccdf.png" if cfg else Path("gap_length_ccdf.png")
    if len(gaps) > 0:
        sorted_gaps = pd.Series(gaps).sort_values().to_numpy()
        ccdf = 1.0 - (pd.Series(range(1, len(sorted_gaps) + 1)) / len(sorted_gaps))
        plt.figure(figsize=(6, 4))
        plt.step(sorted_gaps, ccdf, where="post")
        plt.xlabel("Gap length (steps)")
        plt.ylabel("CCDF")
        plt.title("Missing gap length CCDF")
        plt.tight_layout()
        plt.savefig(ccdf_path)
        plt.close()

    meta_path = out_path.with_name("missingness_meta.csv")
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Wrote missingness summary to {out_path}")


if __name__ == "__main__":
    main()
