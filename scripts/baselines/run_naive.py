from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from darts import TimeSeries
from darts.models import NaiveMean, NaiveSeasonal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import SeriesConfig, log_header, log_message, prepare_series  # noqa: E402
from src.utils.config import load_config, ensure_dirs, resolve_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Naive baselines (mean/seasonal).")
    parser.add_argument("--config", type=Path, default=None, help="Optional config YAML for defaults.")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment id (overrides config).")
    parser.add_argument("--data", type=Path, required=True, help="Path to the station CSV file.")
    parser.add_argument("--target-col", type=str, required=True, help="Target column name in the CSV.")
    parser.add_argument("--encoding", type=str, default=None, help="Optional file encoding.")
    parser.add_argument("--freq", type=str, default=None, help="Optional resample frequency, e.g. 3H.")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for rolling forecasts.")
    parser.add_argument("--season-length", type=int, default=8, help="Season length for seasonal naive.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit evaluation windows.")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional path to save RMSE/MAE bar chart.")
    return parser.parse_args()


def evaluate(series: TimeSeries, values, horizon: int, stride: int, train_ratio: float, max_windows, season_length: int):
    split_idx = int(len(values) * train_ratio)
    split_idx = max(split_idx, horizon)

    results = []
    for name, model in [
        ("naive_mean", NaiveMean()),
        ("naive_seasonal", NaiveSeasonal(K=season_length)),
    ]:
        forecasts = model.historical_forecasts(
            series,
            start=split_idx,
            forecast_horizon=horizon,
            stride=stride,
            retrain=True,
            last_points_only=False,
        )
        preds = [ts.values().squeeze(-1) for ts in forecasts]
        actuals = []
        for idx in range(split_idx, len(values) - horizon + 1, stride):
            actuals.append(values[idx : idx + horizon])
        if max_windows is not None:
            preds = preds[-max_windows:]
            actuals = actuals[-max_windows:]
        if not preds or not actuals:
            continue
        import numpy as np

        preds_arr = np.stack(preds)
        actuals_arr = np.stack(actuals)
        diff = preds_arr - actuals_arr
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        step_diff = diff[:, 0]
        rmse_t1 = float(np.sqrt(np.mean(step_diff**2)))
        mae_t1 = float(np.mean(np.abs(step_diff)))
        results.append({
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "rmse_t1": rmse_t1,
            "mae_t1": mae_t1,
            "windows": float(len(actuals_arr)),
        })
    return results


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        if args.exp_id:
            cfg["exp_id"] = args.exp_id
            cfg = resolve_config(cfg)
        ensure_dirs(cfg)
        station_id = cfg.get("station_id_main")
        args.data = cfg["processed_dir"] / f"{station_id}.csv"
        targets = cfg.get("columns", {}).get("targets", [])
        if targets:
            args.target_col = targets[0]
        args.freq = cfg.get("freq")
        args.horizon = max(cfg.get("horizons", [args.horizon]))
        args.train_ratio = float(cfg.get("split", {}).get("train", args.train_ratio))
        args.stride = int(cfg.get("stride", args.stride))
        args.season_length = int(cfg.get("baseline", {}).get("season_length", args.season_length))
        if args.plot is None:
            args.plot = cfg["reports_dir"] / "figures" / "naive_baselines.png"
        if args.log_path is None:
            args.log_path = cfg["reports_dir"] / "tables" / "naive_baselines_log.txt"
    cfg = SeriesConfig(
        data_path=args.data,
        target_col=args.target_col,
        encoding=args.encoding,
        freq=args.freq,
        log_path=args.log_path,
        verbose=not args.quiet,
    )
    log_header(cfg)

    df, values = prepare_series(cfg)
    target_col = df.columns[1]
    series = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols=target_col)

    results = evaluate(
        series,
        values,
        horizon=args.horizon,
        stride=args.stride,
        train_ratio=args.train_ratio,
        max_windows=args.max_windows,
        season_length=args.season_length,
    )

    table = pd.DataFrame(results).sort_values("rmse")
    with pd.option_context("display.max_columns", None, "display.width", 120, "display.float_format", lambda x: f"{x:.4f}"):
        output = table.to_string(index=False)
        log_message(cfg, output)
    if args.plot:
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.bar(table["model"], table["rmse"], label="RMSE")
        plt.bar(table["model"], table["mae"], label="MAE", alpha=0.7)
        plt.ylabel("Error")
        plt.title("Naive baselines")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close()
        log_message(cfg, f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
