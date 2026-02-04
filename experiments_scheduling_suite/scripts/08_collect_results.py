from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.posthoc.collect import load_metrics_long, parse_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="收集所有 RUN 的指标与元信息。")
    parser.add_argument("--reports_dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "reports" / "_aggregate")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--missingness", type=str, default=None)
    parser.add_argument("--imputer", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--with_missingness_features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _param_value(missingness: str, params: str) -> float | None:
    if not params:
        return None
    if missingness == "mcar":
        # p0.2
        try:
            return float(params.replace("p", ""))
        except Exception:
            return None
    if missingness == "block":
        # b10
        try:
            return float(params.replace("b", ""))
        except Exception:
            return None
    if missingness == "duty_cycle":
        # p20_on5
        try:
            parts = params.split("_")
            p = float(parts[0].replace("p", ""))
            on = float(parts[1].replace("on", "")) if len(parts) > 1 else None
            if on is not None and p != 0:
                return on / p
            return p
        except Exception:
            return None
    if missingness == "round_robin":
        # k2_minon3
        try:
            return float(params.split("_")[0].replace("k", ""))
        except Exception:
            return None
    if missingness == "info_priority":
        try:
            return float(params.split("_")[0].replace("k", ""))
        except Exception:
            return None
    return None


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_logs = args.out_dir / "logs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    manifest_rows = []
    missingness_rows = []
    warnings = []

    for run_dir in args.reports_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "_aggregate":
            continue
        metrics_df = load_metrics_long(run_dir)
        if metrics_df.empty:
            warnings.append(f"Skip {run_dir.name}: missing metrics")
            continue

        meta = parse_run_metadata(run_dir)
        missingness = str(meta.get("missingness", "unknown"))
        params = str(meta.get("missingness_params", ""))
        missingness_label = missingness if not params else f"{missingness}_{params}"
        imputer = str(meta.get("imputer", "unknown"))

        # 过滤
        if args.dataset and meta.get("dataset") != args.dataset:
            continue
        if args.missingness and missingness != args.missingness:
            continue
        if args.imputer and imputer != args.imputer:
            continue

        metrics_df = metrics_df.copy()
        if args.model:
            metrics_df = metrics_df[metrics_df["model"] == args.model]
        if metrics_df.empty:
            continue

        metrics_df["run_id"] = meta.get("run_id")
        metrics_df["dataset"] = meta.get("dataset")
        metrics_df["freq"] = meta.get("freq")
        metrics_df["missingness"] = missingness
        metrics_df["missingness_params"] = params
        metrics_df["missingness_label"] = missingness_label
        metrics_df["imputer"] = imputer
        metrics_df["baseline_label"] = f"{missingness_label}+{imputer}"
        metrics_df["param_value"] = _param_value(missingness, params)

        metrics_rows.append(metrics_df)

        # run manifest
        available_models = sorted(metrics_df["model"].unique())
        manifest_rows.append({
            "run_id": meta.get("run_id"),
            "dataset": meta.get("dataset"),
            "freq": meta.get("freq"),
            "missingness": missingness,
            "missingness_params": params,
            "missingness_label": missingness_label,
            "imputer": imputer,
            "available_models": ",".join(available_models),
        })

        # 缺失形态特征（可选）
        if args.with_missingness_features:
            processed_dir = args.reports_dir.parent / "data" / "processed" / run_dir.name
            mask_path = processed_dir / "mask.csv"
            meta_path = processed_dir / "metadata.json"
            if mask_path.exists():
                try:
                    import json
                    from experiments_scheduling_suite.src.posthoc.derive_features import compute_missingness_features

                    mask_df = pd.read_csv(mask_path)
                    target_col = None
                    if meta_path.exists():
                        meta_json = json.loads(meta_path.read_text())
                        target_col = meta_json.get("target_col")
                    feats = compute_missingness_features(mask_df, target_col=target_col)
                    feats["run_id"] = run_dir.name
                    missingness_rows.append(feats)
                except Exception as exc:
                    warnings.append(f"Missingness feature failed for {run_dir.name}: {exc}")

    if metrics_rows:
        metrics_long = pd.concat(metrics_rows, ignore_index=True)
        # 统一输出为长表 (metric/value)
        if "metric" not in metrics_long.columns:
            value_cols = [c for c in ["rmse", "mae", "mape"] if c in metrics_long.columns]
            id_cols = [c for c in metrics_long.columns if c not in value_cols]
            metrics_long = metrics_long.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
        metrics_long.to_csv(out_tables / "metrics_long.csv", index=False)
        # 记录极端离群值，便于核查
        try:
            rmse = metrics_long[metrics_long["metric"] == "rmse"].copy()
            if not rmse.empty:
                threshold = rmse["value"].quantile(0.99)
                outliers = rmse[rmse["value"] >= threshold].sort_values("value", ascending=False).head(50)
                outliers.to_csv(out_logs / "outlier_metrics_rmse.csv", index=False)
        except Exception as exc:
            warnings.append(f"Outlier scan failed: {exc}")
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_tables / "run_manifest.csv", index=False)
    if missingness_rows:
        pd.DataFrame(missingness_rows).to_csv(out_tables / "missingness_features.csv", index=False)

    if warnings:
        (out_logs / "collect_warnings.txt").write_text("\n".join(warnings))

    print(f"Saved metrics_long to {out_tables / 'metrics_long.csv'}")
    print(f"Saved run_manifest to {out_tables / 'run_manifest.csv'}")


if __name__ == "__main__":
    main()
