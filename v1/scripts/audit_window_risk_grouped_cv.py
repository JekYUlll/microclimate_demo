#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.window_risk import (  # noqa: E402
    build_window_risk_dataset,
    filter_exact_anchor_boundaries,
    load_window_risk_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit-only grouped CV for Branch H risk models.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--quantile-alpha", type=float, default=0.25)
    parser.add_argument("--controller-id", default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_family(name: str, *, alpha: float, seed: int) -> tuple[Any, Any, Any]:
    if name == "gbdt":
        common = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 32,
            "random_state": int(seed),
        }
        return (
            GradientBoostingRegressor(loss="squared_error", **common),
            GradientBoostingRegressor(loss="quantile", alpha=float(alpha), **common),
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=200,
                max_leaf_nodes=7,
                min_samples_leaf=32,
                l2_regularization=1.0,
                random_state=int(seed),
            ),
        )
    if name == "hist_gbdt":
        common = {
            "learning_rate": 0.04,
            "max_iter": 250,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 32,
            "l2_regularization": 1.0,
            "random_state": int(seed),
        }
        return (
            HistGradientBoostingRegressor(loss="squared_error", **common),
            HistGradientBoostingRegressor(loss="quantile", quantile=float(alpha), **common),
            HistGradientBoostingClassifier(**common),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        common = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 24.0,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
            "n_jobs": 8,
            "random_state": int(seed),
        }
        return (
            XGBRegressor(objective="reg:squarederror", **common),
            XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=float(alpha),
                **common,
            ),
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common,
            ),
        )
    raise ValueError(f"Unknown model family: {name}")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def evaluate_family(
    name: str,
    *,
    features: np.ndarray,
    margins: np.ndarray,
    groups: np.ndarray,
    folds: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=int(folds))
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, margins, groups)):
        mean_model, quantile_model, negative_model = model_family(
            name,
            alpha=float(alpha),
            seed=int(seed) + int(fold_idx) * 17,
        )
        x_train = features[train_idx]
        y_train = margins[train_idx]
        x_test = features[test_idx]
        y_test = margins[test_idx]
        mean_model.fit(x_train, y_train)
        quantile_model.fit(x_train, y_train)
        negative_model.fit(x_train, (y_train < 0.0).astype(np.int8))
        mean_pred = np.asarray(mean_model.predict(x_test), dtype=float)
        quantile_pred = np.asarray(quantile_model.predict(x_test), dtype=float)
        negative_pred = np.asarray(negative_model.predict_proba(x_test), dtype=float)[:, 1]
        constant_mean = np.full(y_test.shape, float(np.mean(y_train)), dtype=float)
        constant_q25 = np.full(y_test.shape, float(np.quantile(y_train, alpha)), dtype=float)
        prevalence = float(np.mean(y_train < 0.0))
        constant_negative = np.full(y_test.shape, prevalence, dtype=float)
        rows.append(
            {
                "fold": int(fold_idx),
                "train_starts": int(np.unique(groups[train_idx]).size),
                "test_starts": int(np.unique(groups[test_idx]).size),
                "mean_mae": float(mean_absolute_error(y_test, mean_pred)),
                "constant_mean_mae": float(mean_absolute_error(y_test, constant_mean)),
                "mean_spearman": safe_spearman(mean_pred, y_test),
                "q25_pinball": float(mean_pinball_loss(y_test, quantile_pred, alpha=alpha)),
                "constant_q25_pinball": float(
                    mean_pinball_loss(y_test, constant_q25, alpha=alpha)
                ),
                "q25_coverage": float(np.mean(y_test <= quantile_pred)),
                "negative_brier": float(
                    brier_score_loss((y_test < 0.0).astype(np.int8), negative_pred)
                ),
                "constant_negative_brier": float(
                    brier_score_loss((y_test < 0.0).astype(np.int8), constant_negative)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    mean_mae = float(frame["mean_mae"].mean())
    constant_mean_mae = float(frame["constant_mean_mae"].mean())
    q25_pinball = float(frame["q25_pinball"].mean())
    constant_q25_pinball = float(frame["constant_q25_pinball"].mean())
    negative_brier = float(frame["negative_brier"].mean())
    constant_negative_brier = float(frame["constant_negative_brier"].mean())
    return {
        "family": str(name),
        "folds": rows,
        "mean_mae": mean_mae,
        "constant_mean_mae": constant_mean_mae,
        "mean_mae_improvement": (
            (constant_mean_mae - mean_mae) / constant_mean_mae
            if constant_mean_mae > 1.0e-12
            else float("nan")
        ),
        "mean_spearman": float(frame["mean_spearman"].mean()),
        "q25_pinball": q25_pinball,
        "constant_q25_pinball": constant_q25_pinball,
        "q25_pinball_improvement": (
            (constant_q25_pinball - q25_pinball) / constant_q25_pinball
            if constant_q25_pinball > 1.0e-12
            else float("nan")
        ),
        "q25_coverage": float(frame["q25_coverage"].mean()),
        "negative_brier": negative_brier,
        "constant_negative_brier": constant_negative_brier,
        "negative_brier_improvement": (
            (constant_negative_brier - negative_brier) / constant_negative_brier
            if constant_negative_brier > 1.0e-12
            else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.data_root)
    records = load_window_risk_records(
        root / "risk_fit" / "window_risk_rows.jsonl"
    )
    if args.controller_id is not None:
        records = [
            record
            for record in records
            if record.controller_id == str(args.controller_id)
        ]
        if not records:
            raise ValueError(
                f"No fit rows for controller {args.controller_id}"
            )
    dataset, exact_filter = filter_exact_anchor_boundaries(
        build_window_risk_dataset(records)
    )
    features = np.asarray(dataset.features, dtype=np.float32)
    varying = np.var(features, axis=0) > 1.0e-12
    features = features[:, varying]
    results = {
        "data_root": str(root),
        "rows": int(features.shape[0]),
        "independent_starts": int(np.unique(dataset.starts).size),
        "input_features": int(dataset.features.shape[1]),
        "varying_features": int(features.shape[1]),
        "folds": int(args.folds),
        "quantile_alpha": float(args.quantile_alpha),
        "controller_id": (
            str(args.controller_id)
            if args.controller_id is not None
            else None
        ),
        "exact_anchor_filter": exact_filter,
        "families": [
            evaluate_family(
                family,
                features=features,
                margins=np.asarray(dataset.margins, dtype=float),
                groups=np.asarray(dataset.starts, dtype=np.int64),
                folds=int(args.folds),
                alpha=float(args.quantile_alpha),
                seed=int(args.seed),
            )
            for family in ("gbdt", "hist_gbdt", "xgboost")
        ],
    }
    output = (
        resolve_project_path(args.out)
        if args.out is not None
        else root / "fit_grouped_cv_causal_history_v1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
