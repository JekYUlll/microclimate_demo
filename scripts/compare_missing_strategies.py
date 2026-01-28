from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.simple import build_tabular_features, fit_tabular_regressor, predict_tabular
from src.data.impute import impute_A, impute_B_stl
from src.data.normalize import StandardScaler
from src.data.split import time_split
from src.data.window_dataset import build_windows
from src.eval.metrics import mae, rmse
from src.features.build_features import build
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare imputation strategies with a tabular baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    return parser.parse_args()


def evaluate_strategy(df: pd.DataFrame, cfg: dict, strategy: str) -> dict:
    value_cols = [c for c in df.columns if c != "timestamp"]
    if strategy == "A":
        work = impute_A(df, value_cols, max_gap_steps=int(cfg.get("impute", {}).get("max_gap_steps", 4)))
    else:
        work = impute_B_stl(df, value_cols, period=int(cfg.get("baseline", {}).get("season_length", 8)))

    feat_df, feature_cols = build(work, cfg)
    feature_cols = [c for c in feature_cols if c != "timestamp"]
    feat_df[feature_cols] = feat_df[feature_cols].apply(lambda s: s.fillna(s.mean()))

    splits = time_split(feat_df, cfg)
    target_cols = cfg.get("columns", {}).get("targets", [])
    window_size = int(cfg.get("window_size", 24))
    horizons = [int(h) for h in cfg.get("horizons", [1])]

    data = {}
    for split_name, sl in splits.items():
        sub = feat_df.iloc[sl].reset_index(drop=True)
        X, Y, _ = build_windows(
            sub,
            feature_cols=feature_cols,
            target_cols=target_cols,
            window_size=window_size,
            horizons=horizons,
            stride=int(cfg.get("stride", 1)),
            allow_feature_nan=True,
        )
        data[split_name] = (X, Y)

    scaler = StandardScaler.fit(data["train"][0], data["train"][1])
    X_train, Y_train = scaler.transform(data["train"][0], data["train"][1])
    X_test, Y_test = scaler.transform(data["test"][0], data["test"][1])

    X_tab_train = build_tabular_features(X_train)
    X_tab_test = build_tabular_features(X_test)

    tab_cfg = cfg.get("models", {}).get("tabular", {})

    y_pred = np.zeros_like(Y_test)
    for h_idx in range(Y_train.shape[1]):
        for t_idx in range(Y_train.shape[2]):
            model = fit_tabular_regressor(X_tab_train, Y_train[:, h_idx, t_idx], tab_cfg)
            y_pred[:, h_idx, t_idx] = predict_tabular(model, X_tab_test)

    y_true_denorm = scaler.inverse_transform_y(Y_test)
    y_pred_denorm = scaler.inverse_transform_y(y_pred)

    return {
        "mae": mae(y_true_denorm, y_pred_denorm),
        "rmse": rmse(y_true_denorm, y_pred_denorm),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    station_id = cfg.get("station_id_main")
    station_path = cfg["processed_dir"] / f"{station_id}.csv"
    df = pd.read_csv(station_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    result_a = evaluate_strategy(df, cfg, "A")
    result_b = evaluate_strategy(df, cfg, "B")

    out_path = cfg["reports_dir"] / "tables" / "missing_strategy_compare.csv"
    pd.DataFrame([
        {"strategy": "A", **result_a},
        {"strategy": "B", **result_b},
    ]).to_csv(out_path, index=False)
    print(f"Wrote missing strategy comparison to {out_path}")


if __name__ == "__main__":
    main()
