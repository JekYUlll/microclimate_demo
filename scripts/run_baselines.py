from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.simple import (
    ar_predict,
    build_tabular_features,
    fit_tabular_regressor,
    naive_persistence,
    seasonal_naive,
    train_ar_coeffs,
    predict_tabular,
)
from src.data.normalize import load_scaler
from src.eval.metrics import metrics_overall, metrics_by_horizon
from src.train.train_tcn import train_tcn
from src.train.train_tft_pinn import train_tft
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML for pipeline baselines.")
    parser.add_argument("--skip-tabular", action="store_true", help="Skip tabular GBRT baseline.")
    parser.add_argument("--skip-tcn", action="store_true", help="Skip lightweight TCN baseline.")
    parser.add_argument("--skip-tft", action="store_true", help="Skip TFT baseline.")
    # Legacy Darts args (used only when --config is not provided)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "AntAWS" / "3_hourly" / "Taishan_3h.csv")
    parser.add_argument("--target-col", type=str, default="Temperature(Ąć)")
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=6)
    return parser.parse_args()


def write_preds(path: Path, t_ref, y_true, y_pred, target_cols, horizons):
    rows = []
    for i, ts in enumerate(t_ref):
        for h_idx, h in enumerate(horizons):
            row = {"timestamp": str(ts), "horizon": int(h)}
            for t_idx, col in enumerate(target_cols):
                row[f"y_true_{col}"] = float(y_true[i, h_idx, t_idx])
                row[f"y_pred_{col}"] = float(y_pred[i, h_idx, t_idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_metrics(path: Path, y_true: np.ndarray, y_pred: np.ndarray, target_cols, horizons, model_name: str):
    overall = metrics_overall(y_true, y_pred)
    per_h = metrics_by_horizon(y_true, y_pred, target_cols, horizons)
    overall_df = pd.DataFrame([{**overall, "model": model_name}])
    per_h["model"] = model_name
    overall_df.to_csv(path, index=False)
    per_h.to_csv(path.with_name(f"{model_name}_metrics_by_horizon.csv"), index=False)


def run_pipeline_baselines(cfg: dict, skip_tabular: bool, skip_tcn: bool, skip_tft: bool) -> None:
    ensure_dirs(cfg)
    train = np.load(cfg["processed_dir"] / "train.npz", allow_pickle=True)
    test = np.load(cfg["processed_dir"] / "test.npz", allow_pickle=True)
    scaler = load_scaler(cfg["processed_dir"] / "scaler.json")

    X_train = train["X"]
    Y_train = train["Y"]
    X_test = test["X"]
    Y_test = test["Y"]
    t_ref = test["t_ref"]
    feature_cols = list(train["feature_cols"])
    target_cols = list(train["target_cols"])
    horizons = list(train["horizons"])

    feat_mean = scaler.feature_mean
    feat_std = scaler.feature_std
    X_train_denorm = X_train * feat_std + feat_mean
    X_test_denorm = X_test * feat_std + feat_mean
    Y_train_denorm = scaler.inverse_transform_y(Y_train)
    Y_test_denorm = scaler.inverse_transform_y(Y_test)

    preds_dir = cfg["reports_dir"] / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Naive persistence
    naive_pred = naive_persistence(X_test_denorm, feature_cols, target_cols, horizons)
    for t_idx in range(len(target_cols)):
        if np.isnan(naive_pred[:, :, t_idx]).any():
            mean_val = Y_train_denorm[:, :, t_idx].mean()
            naive_pred[:, :, t_idx] = np.nan_to_num(naive_pred[:, :, t_idx], nan=mean_val)
    write_preds(preds_dir / "naive_persistence.csv", t_ref, Y_test_denorm, naive_pred, target_cols, horizons)
    save_metrics(cfg["reports_dir"] / "tables" / "naive_persistence_metrics.csv", Y_test_denorm, naive_pred, target_cols, horizons, "naive_persistence")

    # Seasonal naive
    season_len = int(cfg.get("baseline", {}).get("season_length", 8))
    seasonal_pred = seasonal_naive(X_test_denorm, feature_cols, target_cols, horizons, season_length=season_len)
    for t_idx in range(len(target_cols)):
        if np.isnan(seasonal_pred[:, :, t_idx]).any():
            mean_val = Y_train_denorm[:, :, t_idx].mean()
            seasonal_pred[:, :, t_idx] = np.nan_to_num(seasonal_pred[:, :, t_idx], nan=mean_val)
    write_preds(preds_dir / "naive_seasonal.csv", t_ref, Y_test_denorm, seasonal_pred, target_cols, horizons)
    save_metrics(cfg["reports_dir"] / "tables" / "naive_seasonal_metrics.csv", Y_test_denorm, seasonal_pred, target_cols, horizons, "naive_seasonal")

    # AR baseline
    ar_coeffs = train_ar_coeffs(X_train_denorm, Y_train_denorm, feature_cols, target_cols, p=5)
    ar_pred = ar_predict(X_test_denorm, feature_cols, target_cols, horizons, ar_coeffs)
    for t_idx in range(len(target_cols)):
        if np.isnan(ar_pred[:, :, t_idx]).any():
            mean_val = Y_train_denorm[:, :, t_idx].mean()
            ar_pred[:, :, t_idx] = np.nan_to_num(ar_pred[:, :, t_idx], nan=mean_val)
    write_preds(preds_dir / "arima.csv", t_ref, Y_test_denorm, ar_pred, target_cols, horizons)
    save_metrics(cfg["reports_dir"] / "tables" / "arima_metrics.csv", Y_test_denorm, ar_pred, target_cols, horizons, "arima")

    if not skip_tabular:
        # Tabular regressor
        X_tab_train = build_tabular_features(X_train)
        X_tab_test = build_tabular_features(X_test)
        tab_cfg = cfg.get("models", {}).get("tabular", {})
        tab_pred_norm = np.zeros_like(Y_test)
        for h_idx in range(Y_train.shape[1]):
            for t_idx in range(Y_train.shape[2]):
                model = fit_tabular_regressor(X_tab_train, Y_train[:, h_idx, t_idx], tab_cfg)
                tab_pred_norm[:, h_idx, t_idx] = predict_tabular(model, X_tab_test)
        tab_pred = scaler.inverse_transform_y(tab_pred_norm)
        write_preds(preds_dir / "tabular_gbrt.csv", t_ref, Y_test_denorm, tab_pred, target_cols, horizons)
        save_metrics(cfg["reports_dir"] / "tables" / "tabular_gbrt_metrics.csv", Y_test_denorm, tab_pred, target_cols, horizons, "tabular_gbrt")

    if not skip_tcn:
        # TCN baseline
        try:
            tcn_res = train_tcn(cfg)
            write_preds(preds_dir / "tcn.csv", tcn_res["t_ref"], tcn_res["y_true"], tcn_res["y_pred"], tcn_res["target_cols"], tcn_res["horizons"])
            save_metrics(cfg["reports_dir"] / "tables" / "tcn_metrics.csv", tcn_res["y_true"], tcn_res["y_pred"], tcn_res["target_cols"], tcn_res["horizons"], "tcn")
        except Exception as exc:
            print(f"TCN baseline skipped: {exc}")

    if not skip_tft:
        # TFT baseline (lambda_phys=0)
        try:
            tft_res = train_tft(cfg, lambda_phys=0.0, model_name="tft")
            write_preds(preds_dir / "tft.csv", tft_res["t_ref"], tft_res["y_true"], tft_res["y_pred"], tft_res["target_cols"], tft_res["horizons"])
            save_metrics(cfg["reports_dir"] / "tables" / "tft_metrics.csv", tft_res["y_true"], tft_res["y_pred"], tft_res["target_cols"], tft_res["horizons"], "tft")
        except Exception as exc:
            print(f"TFT baseline skipped: {exc}")


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        run_pipeline_baselines(cfg, args.skip_tabular, args.skip_tcn, args.skip_tft)
        return

    # Legacy Darts path
    try:
        from darts import TimeSeries
        from src.baselines.common import SeriesConfig, log_header, log_message, prepare_series
        from src.baselines.darts_baselines import BaselineConfig, evaluate_all
    except Exception as exc:
        raise SystemExit("Darts dependencies not available; use --config for pipeline baselines") from exc

    cfg = SeriesConfig(
        data_path=args.data,
        target_col=args.target_col,
        encoding=args.encoding,
        freq=args.freq,
        log_path=None,
        verbose=True,
    )
    log_header(cfg)
    df, values = prepare_series(cfg)
    target_col = df.columns[1]
    series = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols=target_col)

    base_cfg = BaselineConfig(
        models=["naive_mean", "naive_seasonal", "arima"],
        horizon=args.horizon,
        input_chunk_length=24,
        train_ratio=0.8,
        stride=1,
        max_windows=None,
        season_length=8,
        arima_order=(2, 1, 2),
        epochs=10,
        tft_hidden_size=32,
        tft_num_heads=4,
        tft_dropout=0.1,
        pl_trainer_kwargs=None,
    )

    results = evaluate_all(series, values, base_cfg)
    table = pd.DataFrame(results).sort_values("rmse")
    with pd.option_context("display.max_columns", None, "display.width", 120, "display.float_format", lambda x: f"{x:.4f}"):
        log_message(cfg, table.to_string(index=False))


if __name__ == "__main__":
    main()
