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

from experiments_scheduling_suite.src.posthoc.collect import parse_run_metadata
from experiments_scheduling_suite.src.posthoc.stats_test import pairwise_wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="显著性检验（Wilcoxon）。")
    parser.add_argument("--reports_dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "reports" / "_aggregate")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--imputer", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    error_by_strategy: Dict[str, np.ndarray] = {}

    for run_dir in args.reports_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "_aggregate":
            continue
        meta = parse_run_metadata(run_dir)
        if meta.get("imputer") != args.imputer:
            continue
        missingness = meta.get("missingness")
        params = meta.get("missingness_params")
        label = missingness if not params else f"{missingness}_{params}"
        preds_path = run_dir / "preds" / f"{args.model}_h{args.horizon}.csv"
        if not preds_path.exists():
            continue
        df = pd.read_csv(preds_path)
        err = (df["y_true"] - df["y_pred"]).abs().to_numpy()
        if len(err) == 0:
            continue
        error_by_strategy[label] = err

    if not error_by_strategy:
        print("No matching runs found for significance test.")
        return

    result = pairwise_wilcoxon(error_by_strategy)
    out_path = out_tables / f"significance_{args.model}_H{args.horizon}_{args.imputer}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved significance matrix -> {out_path}")


if __name__ == "__main__":
    main()
