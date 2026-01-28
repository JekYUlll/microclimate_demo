from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.impute import impute_A, impute_B_stl
from src.data.normalize import StandardScaler, save_scaler
from src.data.split import split_summary, time_split
from src.data.window_dataset import build_windows
from src.features.build_features import build
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build windowed datasets for AntAWS experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--station-id", type=str, default=None, help="Override station id.")
    return parser.parse_args()


def _save_npz(path: Path, X: np.ndarray, Y: np.ndarray, t_ref: np.ndarray, feature_cols, target_cols, horizons):
    np.savez_compressed(
        path,
        X=X,
        Y=Y,
        t_ref=t_ref.astype(str),
        feature_cols=np.array(feature_cols),
        target_cols=np.array(target_cols),
        horizons=np.array(horizons),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    station_id = args.station_id or cfg.get("station_id_main")
    if station_id is None:
        raise SystemExit("station_id_main missing in config")

    station_path = cfg["processed_dir"] / f"{station_id}.csv"
    if not station_path.exists():
        raise SystemExit(f"Missing processed station file: {station_path}")

    df = pd.read_csv(station_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    value_cols = [c for c in df.columns if c != "timestamp"]

    impute_cfg = cfg.get("impute", {})
    strategy = impute_cfg.get("strategy", "A")
    if strategy.upper() == "A":
        work = impute_A(df, value_cols, max_gap_steps=int(impute_cfg.get("max_gap_steps", 4)))
        impute_tag = "imputeA"
    else:
        work = impute_B_stl(df, value_cols, period=int(cfg.get("baseline", {}).get("season_length", 8)))
        impute_tag = "imputeB"

    impute_path = cfg["processed_dir"] / f"{station_id}_{impute_tag}.csv"
    work.to_csv(impute_path, index=False)

    feat_df, feature_cols = build(work, cfg)
    feature_path = cfg["processed_dir"] / f"{station_id}_features.csv"
    feat_df.to_csv(feature_path, index=False)

    feat_list_path = cfg["reports_dir"] / "tables" / "feature_list.csv"
    pd.DataFrame({"feature": feature_cols}).to_csv(feat_list_path, index=False)

    splits = time_split(feat_df, cfg)
    split_summary(feat_df, splits).to_csv(cfg["reports_dir"] / "tables" / "split_summary.csv", index=False)

    target_cols = cfg.get("columns", {}).get("targets", [])
    missing_targets = [c for c in target_cols if c not in feat_df.columns]
    if missing_targets:
        raise SystemExit(f"Missing target columns in features: {missing_targets}")
    window_size = int(cfg.get("window_size", 24))
    horizons = [int(h) for h in cfg.get("horizons", [1])]
    stride = int(cfg.get("stride", 1))

    # Fill NaNs in features with column means to keep windows; missingness features carry the signal.
    feat_df[feature_cols] = feat_df[feature_cols].apply(lambda s: s.fillna(s.mean()))

    datasets = {}
    for split_name, sl in splits.items():
        sub = feat_df.iloc[sl].reset_index(drop=True)
        X, Y, t_ref = build_windows(
            sub,
            feature_cols=feature_cols,
            target_cols=target_cols,
            window_size=window_size,
            horizons=horizons,
            stride=stride,
            max_windows=cfg.get("max_windows"),
            allow_feature_nan=True,
        )
        datasets[split_name] = (X, Y, t_ref)

    scaler = StandardScaler.fit(datasets["train"][0], datasets["train"][1])
    save_scaler(scaler, cfg["processed_dir"] / "scaler.json")

    for split_name, (X, Y, t_ref) in datasets.items():
        Xn, Yn = scaler.transform(X, Y)
        out_path = cfg["processed_dir"] / f"{split_name}.npz"
        _save_npz(out_path, Xn, Yn, t_ref, feature_cols, target_cols, horizons)

    print(f"Saved datasets to {cfg['processed_dir']}")


if __name__ == "__main__":
    main()
