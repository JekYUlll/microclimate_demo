#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.window_risk import (  # noqa: E402
    build_window_risk_dataset,
    filter_exact_anchor_boundaries,
    load_window_risk_records,
)
from forecast_cmdp.window_risk_model import (  # noqa: E402
    WindowRiskTrainingConfig,
    train_window_risk_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and calibrate Branch H full-window risk models.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--model-family",
        choices=["gbdt", "hist_gbdt", "xgboost"],
        default="gbdt",
    )
    parser.add_argument("--quantile-alpha", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-leaf-nodes", type=int, default=7)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--controller-id", default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    data_root = resolve_project_path(args.data_root)
    output = (
        resolve_project_path(args.out_dir)
        if args.out_dir is not None
        else data_root / "model"
    )
    fit_records = load_window_risk_records(
        data_root / "risk_fit" / "window_risk_rows.jsonl"
    )
    calibration_records = load_window_risk_records(
        data_root / "risk_calibration" / "window_risk_rows.jsonl"
    )
    if args.controller_id is not None:
        fit_records = [
            record
            for record in fit_records
            if record.controller_id == str(args.controller_id)
        ]
        calibration_records = [
            record
            for record in calibration_records
            if record.controller_id == str(args.controller_id)
        ]
        if not fit_records or not calibration_records:
            raise ValueError(
                f"Missing fit/calibration rows for {args.controller_id}"
            )
    fit, fit_filter = filter_exact_anchor_boundaries(
        build_window_risk_dataset(fit_records)
    )
    calibration, calibration_filter = filter_exact_anchor_boundaries(
        build_window_risk_dataset(calibration_records)
    )
    _, metrics = train_window_risk_models(
        fit,
        calibration,
        cfg=WindowRiskTrainingConfig(
            model_family=str(args.model_family),
            quantile_alpha=float(args.quantile_alpha),
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            max_leaf_nodes=int(args.max_leaf_nodes),
            min_samples_leaf=int(args.min_samples_leaf),
            l2_regularization=float(args.l2_regularization),
            seed=int(args.seed),
        ),
        out_dir=output,
    )
    exact_anchor_filter = {
        "fit": fit_filter,
        "calibration": calibration_filter,
    }
    metrics["exact_anchor_filter"] = exact_anchor_filter
    metrics["controller_id"] = (
        str(args.controller_id)
        if args.controller_id is not None
        else None
    )
    (output / "window_risk_model_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(output),
                "exact_anchor_filter": exact_anchor_filter,
                **metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
