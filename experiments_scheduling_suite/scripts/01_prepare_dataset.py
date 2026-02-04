from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.imputation.kalman import KalmanImputer
from experiments_scheduling_suite.src.imputation.linear import LinearImputer
from experiments_scheduling_suite.src.imputation.maskaware_features import MaskAwareImputer
from experiments_scheduling_suite.src.imputation.spline import SplineImputer
from experiments_scheduling_suite.src.imputation.gp import GPImputer
from experiments_scheduling_suite.src.missingness import block, duty_cycle, info_priority, mcar, round_robin
from experiments_scheduling_suite.src.missingness.base import apply_mask
from experiments_scheduling_suite.src.preprocessing.normalize import fit_scaler, apply_scaler
from experiments_scheduling_suite.src.preprocessing.resample import resample_to_freq
from experiments_scheduling_suite.src.preprocessing.split import time_split
from experiments_scheduling_suite.src.preprocessing.windowing import build_windows
from experiments_scheduling_suite.src.utils.io import build_run_id, deep_merge, ensure_dirs, load_yaml, read_csv_with_timestamp, save_yaml
from experiments_scheduling_suite.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备数据集（缺失+插补+窗口化）。")
    parser.add_argument("--config", type=Path, required=True, help="base.yaml")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset config")
    parser.add_argument("--missingness", type=Path, required=True, help="missingness config")
    parser.add_argument("--imputation", type=Path, required=True, help="imputation config")
    parser.add_argument("--run-id", type=str, default=None, help="自定义 RUN_ID")
    return parser.parse_args()


def _feature_engineering(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """基础特征工程：风向 sin/cos、稳定度 one-hot。"""
    out = df.copy()
    if cfg.get("features", {}).get("use_wind_dir_sincos", True) and "wind_direction_deg" in out.columns:
        rad = np.deg2rad(out["wind_direction_deg"].astype(float))
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
    if cfg.get("features", {}).get("onehot_stability_flag", True) and "stability_flag" in out.columns:
        onehot = pd.get_dummies(out["stability_flag"], prefix="stability")
        out = pd.concat([out.drop(columns=["stability_flag"]), onehot], axis=1)
    return out


def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留数值列（保留 timestamp）。"""
    work = df.copy()
    for col in work.columns:
        if col == "timestamp":
            continue
        # 强制转为数值，无法转换的置为 NaN
        work[col] = pd.to_numeric(work[col], errors="coerce")
    # 丢弃全为 NaN 的列（避免插补后仍为 NaN）
    drop_cols = [c for c in work.columns if c != "timestamp" and work[c].isna().all()]
    if drop_cols:
        work = work.drop(columns=drop_cols)
    cols = ["timestamp"]
    for col in work.columns:
        if col == "timestamp":
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            cols.append(col)
    return work[cols]


def _build_mask(df: pd.DataFrame, missing_cfg: Dict, target_col: str, seed: int, train_slice: slice) -> pd.DataFrame:
    name = missing_cfg.get("name", "mcar")
    if name == "mcar":
        return mcar.generate(df, missing_cfg, target_col, seed)
    if name == "block":
        return block.generate(df, missing_cfg, target_col, seed)
    if name == "duty_cycle":
        return duty_cycle.generate(df, missing_cfg, target_col, seed)
    if name == "round_robin":
        return round_robin.generate(df, missing_cfg, target_col, seed)
    if name == "info_priority":
        return info_priority.generate(df, missing_cfg, target_col, seed, train_slice=train_slice)
    raise ValueError(f"Unknown missingness: {name}")


def _build_imputer(impute_cfg: Dict):
    name = impute_cfg.get("name", "maskaware")
    if name == "maskaware":
        return MaskAwareImputer(fill_value=float(impute_cfg.get("fill_value", 0.0)))
    if name == "linear":
        return LinearImputer(limit_direction=impute_cfg.get("limit_direction", "both"))
    if name == "spline":
        return SplineImputer(order=int(impute_cfg.get("order", 3)), limit_direction=impute_cfg.get("limit_direction", "both"))
    if name == "kalman":
        return KalmanImputer(process_var=float(impute_cfg.get("process_var", 1e-4)), obs_var=float(impute_cfg.get("obs_var", 1e-2)))
    if name == "gp":
        return GPImputer(length_scale=float(impute_cfg.get("length_scale", 5.0)), noise=float(impute_cfg.get("noise", 1e-3)))
    raise ValueError(f"Unknown imputation: {name}")


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(args.dataset).get("dataset", {})
    missing_cfg = load_yaml(args.missingness).get("missingness", {})
    impute_cfg = load_yaml(args.imputation).get("imputation", {})

    cfg = deep_merge(base_cfg, {"dataset": dataset_cfg, "missingness": missing_cfg, "imputation": impute_cfg})
    set_seed(int(cfg.get("run", {}).get("seed", 42)))

    run_id = args.run_id or build_run_id(dataset_cfg, cfg, missing_cfg, impute_cfg)
    paths = cfg.get("paths", {})
    processed_dir = PROJECT_ROOT / paths.get("processed_dir", "data/processed") / run_id
    reports_dir = PROJECT_ROOT / paths.get("reports_dir", "reports") / run_id
    ensure_dirs({
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "tables_dir": reports_dir / "tables",
        "figures_dir": reports_dir / "figures",
    })

    # 1) 读取数据
    if dataset_cfg.get("mode", "synthetic") == "synthetic":
        input_path = PROJECT_ROOT / paths.get("generated_dir", "data/generated") / dataset_cfg.get("output_csv", "synthetic.csv")
    else:
        input_path = Path(dataset_cfg.get("input_csv", ""))
    df = read_csv_with_timestamp(input_path)

    # 2) 重采样
    base_freq = cfg.get("run", {}).get("base_freq", "1S")
    df = resample_to_freq(df, base_freq)

    # 3) 特征工程 + 仅保留数值列
    df = _feature_engineering(df, cfg)
    df = _select_numeric(df)

    target_col = cfg.get("run", {}).get("target", "wind_speed_ms")
    if target_col not in df.columns:
        raise SystemExit(f"Target column not found: {target_col}")

    # 4) 缺失模拟
    splits = time_split(df, **cfg.get("split", {}))
    mask_df = _build_mask(df, missing_cfg, target_col, seed=int(cfg.get("run", {}).get("seed", 42)), train_slice=splits["train"])
    masked_df = apply_mask(df, mask_df)

    # 5) 插补
    imputer = _build_imputer(impute_cfg)
    imputer.fit(masked_df.iloc[splits["train"]])
    imputed_df = imputer.transform(masked_df)

    # 6) 保存中间数据
    df.to_csv(processed_dir / "original.csv", index=False)
    masked_df.to_csv(processed_dir / "masked.csv", index=False)
    imputed_df.to_csv(processed_dir / "imputed.csv", index=False)
    mask_df.to_csv(processed_dir / "mask.csv", index=False)

    # 7) 统计缺失与插补报告
    missing_stats = mask_df.mean(axis=0).reset_index()
    missing_stats.columns = ["variable", "observed_ratio"]
    missing_stats.to_csv(reports_dir / "tables" / "missingness_stats.csv", index=False)

    impute_report = imputer.report()
    (reports_dir / "tables" / "imputation_report.csv").write_text(
        pd.DataFrame([impute_report]).to_csv(index=False)
    )

    # 8) 归一化（对输入特征；包含目标历史作为输入特征）
    feature_cols = [c for c in imputed_df.columns if c != "timestamp"]
    train_df = imputed_df.iloc[splits["train"]]
    scaler = fit_scaler(train_df, feature_cols)
    imputed_df = apply_scaler(imputed_df, feature_cols, scaler)

    # 9) 窗口化
    lookback = int(cfg.get("run", {}).get("lookback", 60))
    horizons = [int(h) for h in cfg.get("run", {}).get("horizons", [1, 2, 3])]
    max_windows = cfg.get("run", {}).get("max_windows")

    datasets = {}
    for split_name, sl in splits.items():
        sub = imputed_df.iloc[sl].reset_index(drop=True)
        X, y, t_ref = build_windows(
            sub,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            horizons=horizons,
            stride=1,
            max_windows=max_windows,
        )
        datasets[split_name] = (X, y, t_ref)

    np.savez_compressed(
        processed_dir / "dataset.npz",
        X_train=datasets["train"][0],
        y_train=datasets["train"][1],
        t_train=datasets["train"][2].astype(str),
        X_val=datasets["val"][0],
        y_val=datasets["val"][1],
        t_val=datasets["val"][2].astype(str),
        X_test=datasets["test"][0],
        y_test=datasets["test"][1],
        t_test=datasets["test"][2].astype(str),
    )

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "lookback": lookback,
        "horizons": horizons,
        "base_freq": base_freq,
        "run_id": run_id,
    }
    (processed_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 10) 保存 config 快照
    save_yaml(cfg, reports_dir / "config_used.yaml")
    print(f"Prepared dataset -> {processed_dir}")


if __name__ == "__main__":
    main()
