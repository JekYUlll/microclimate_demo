from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.plots.summary_figure import plot_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制总结大图。")
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    preds_dir = PROJECT_ROOT / "reports" / run_id / "preds"
    tables_dir = PROJECT_ROOT / "reports" / run_id / "tables"
    figs_dir = PROJECT_ROOT / "reports" / run_id / "figures"

    preds_files = sorted(preds_dir.glob("*_h*.csv"))
    data = {}
    for path in preds_files:
        name = path.stem
        model, h_str = name.rsplit("_h", 1)
        h = int(h_str)
        df = pd.read_csv(path)
        data.setdefault(model, {})[h] = df

    if not data:
        raise SystemExit("No prediction files found.")

    # 取第一个模型作为时间轴与真实值
    first_model = next(iter(data))
    horizons = sorted(data[first_model].keys())
    timestamps = pd.to_datetime(data[first_model][horizons[0]]["timestamp"]).to_numpy()
    y_true = np.stack([data[first_model][h]["y_true"].to_numpy() for h in horizons], axis=1)

    model_preds = {}
    for model, h_map in data.items():
        preds = np.stack([h_map[h]["y_pred"].to_numpy() for h in horizons], axis=1)
        model_preds[model] = preds

    metrics_overall = pd.read_csv(tables_dir / "metrics_overall.csv")
    metrics = {row["model"]: row.to_dict() for _, row in metrics_overall.iterrows()}

    out_path = figs_dir / f"summary_{run_id}.png"
    plot_summary(timestamps, y_true, model_preds, metrics, horizons, out_path)
    print(f"Saved summary figure -> {out_path}")


if __name__ == "__main__":
    main()
