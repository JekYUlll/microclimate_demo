from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.plots.per_model_forecast_plot import plot_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制每个模型的预测图。")
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    preds_dir = PROJECT_ROOT / "reports" / run_id / "preds"
    figs_dir = PROJECT_ROOT / "reports" / run_id / "figures" / "preds"
    figs_dir.mkdir(parents=True, exist_ok=True)

    preds_files = sorted(preds_dir.glob("*_h*.csv"))
    data = {}
    for path in preds_files:
        name = path.stem
        model, h_str = name.rsplit("_h", 1)
        h = int(h_str)
        df = pd.read_csv(path)
        data.setdefault(model, {})[h] = df

    for model, h_map in data.items():
        horizons = sorted(h_map.keys())
        timestamps = pd.to_datetime(h_map[horizons[0]]["timestamp"]).to_numpy()
        y_true = np.stack([h_map[h]["y_true"].to_numpy() for h in horizons], axis=1)
        y_pred = np.stack([h_map[h]["y_pred"].to_numpy() for h in horizons], axis=1)
        plot_predictions(timestamps, y_true, y_pred, horizons, figs_dir / f"{model}_pred_h123.png", title=model)

    print(f"Saved prediction figures -> {figs_dir}")


if __name__ == "__main__":
    main()
