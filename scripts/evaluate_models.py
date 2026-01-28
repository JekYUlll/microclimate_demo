from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.extremes import extreme_slice_metrics
from src.eval.metrics import metrics_by_horizon, metrics_overall
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model predictions from reports folder.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    return parser.parse_args()


def load_pred_csv(path: Path):
    df = pd.read_csv(path)
    target_cols = sorted({c.replace("y_true_", "") for c in df.columns if c.startswith("y_true_")})
    horizons = list(pd.unique(df["horizon"]))
    timestamps = list(pd.unique(df["timestamp"]))

    t_index = {ts: i for i, ts in enumerate(timestamps)}
    h_index = {h: i for i, h in enumerate(horizons)}

    import numpy as np

    y_true = np.zeros((len(timestamps), len(horizons), len(target_cols)))
    y_pred = np.zeros_like(y_true)

    for _, row in df.iterrows():
        i = t_index[row["timestamp"]]
        h = h_index[row["horizon"]]
        for t_idx, col in enumerate(target_cols):
            y_true[i, h, t_idx] = row[f"y_true_{col}"]
            y_pred[i, h, t_idx] = row[f"y_pred_{col}"]
    return y_true, y_pred, target_cols, horizons


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    preds_dir = cfg["reports_dir"] / "preds"
    station_id = cfg.get("station_id_main", "")
    overall_rows = []
    horizon_rows = []
    extremes_rows = []

    for path in preds_dir.glob("*.csv"):
        model_name = path.stem
        y_true, y_pred, target_cols, horizons = load_pred_csv(path)
        overall = metrics_overall(y_true, y_pred)
        overall_rows.append({"station_id": station_id, "model": model_name, **overall})

        per_h = metrics_by_horizon(y_true, y_pred, target_cols, horizons)
        per_h["model"] = model_name
        per_h["station_id"] = station_id
        horizon_rows.append(per_h)

        extremes = extreme_slice_metrics(
            y_true,
            y_pred,
            target_cols,
            horizons,
            top_pct=float(cfg.get("extremes", {}).get("top_pct", 0.1)),
            bottom_pct=float(cfg.get("extremes", {}).get("bottom_pct", 0.1)),
        )
        extremes["model"] = model_name
        extremes["station_id"] = station_id
        extremes_rows.append(extremes)

    overall_df = pd.DataFrame(overall_rows).sort_values("rmse")
    overall_df.to_csv(cfg["reports_dir"] / "tables" / "metrics_overall.csv", index=False)

    if horizon_rows:
        horizon_df = pd.concat(horizon_rows, ignore_index=True)
        horizon_df.to_csv(cfg["reports_dir"] / "tables" / "metrics_by_horizon.csv", index=False)

    if extremes_rows:
        extremes_df = pd.concat(extremes_rows, ignore_index=True)
        extremes_df.to_csv(cfg["reports_dir"] / "tables" / "metrics_extremes.csv", index=False)

    print(f"Wrote metrics to {cfg['reports_dir'] / 'tables'}")


if __name__ == "__main__":
    main()
