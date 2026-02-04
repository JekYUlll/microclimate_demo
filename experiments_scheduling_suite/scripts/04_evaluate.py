from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.eval.evaluate import build_long_table, evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估模型预测。")
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    preds_dir = PROJECT_ROOT / "reports" / run_id / "preds"
    tables_dir = PROJECT_ROOT / "reports" / run_id / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    preds_files = sorted(preds_dir.glob("*_h*.csv"))
    if not preds_files:
        raise SystemExit("No prediction files found.")

    # 收集 y_true/y_pred
    data: Dict[str, Dict[int, pd.DataFrame]] = {}
    for path in preds_files:
        name = path.stem
        if "_h" not in name:
            continue
        model, h_str = name.rsplit("_h", 1)
        h = int(h_str)
        df = pd.read_csv(path)
        data.setdefault(model, {})[h] = df

    overall_rows = []
    long_rows = []
    for model, h_map in data.items():
        horizons = sorted(h_map.keys())
        y_true = np.stack([h_map[h]["y_true"].to_numpy() for h in horizons], axis=1)
        y_pred = np.stack([h_map[h]["y_pred"].to_numpy() for h in horizons], axis=1)
        overall_rows.append(evaluate_predictions(y_true, y_pred, model, horizons))
        long_rows.append(build_long_table(y_true, y_pred, model, horizons))

    metrics_overall = pd.DataFrame(overall_rows)
    metrics_long = pd.concat(long_rows, ignore_index=True)

    metrics_overall.to_csv(tables_dir / "metrics_overall.csv", index=False)
    metrics_long.to_csv(tables_dir / "metrics_long.csv", index=False)
    print(f"Saved metrics -> {tables_dir}")


if __name__ == "__main__":
    main()
