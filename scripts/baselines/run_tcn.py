from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Union

import pandas as pd
from darts import TimeSeries
from darts.models import TCNModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import SeriesConfig, log_header, log_message, prepare_series  # noqa: E402
from src.utils.config import load_config, ensure_dirs, resolve_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TCN baseline.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config YAML for defaults.")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment id (overrides config).")
    parser.add_argument("--data", type=Path, required=True, help="Path to the station CSV file.")
    parser.add_argument("--target-col", type=str, required=True, help="Target column name in the CSV.")
    parser.add_argument("--encoding", type=str, default=None, help="Optional file encoding.")
    parser.add_argument("--freq", type=str, default=None, help="Optional resample frequency, e.g. 3H.")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon.")
    parser.add_argument("--input-window", type=int, default=24, help="Input window length.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for rolling forecasts.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit evaluation windows.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for TCN.")
    parser.add_argument("--kernel-size", type=int, default=3, help="TCN kernel size.")
    parser.add_argument("--num-filters", type=int, default=16, help="TCN number of filters.")
    parser.add_argument("--dilation-base", type=int, default=2, help="TCN dilation base.")
    parser.add_argument("--dropout", type=float, default=0.1, help="TCN dropout.")
    parser.add_argument("--devices", type=int, default=1, help="Lightning devices (GPU count).")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional path to save RMSE/MAE bar chart.")
    return parser.parse_args()


def evaluate(series: TimeSeries, values, cfg) -> List[dict]:
    split_idx = int(len(values) * cfg.train_ratio)
    split_idx = max(split_idx, cfg.horizon)

    model = TCNModel(
        input_chunk_length=cfg.input_window,
        output_chunk_length=cfg.horizon,
        n_epochs=cfg.epochs,
        kernel_size=cfg.kernel_size,
        num_filters=cfg.num_filters,
        dilation_base=cfg.dilation_base,
        dropout=cfg.dropout,
        random_state=42,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": cfg.devices,
            "enable_progress_bar": True,
        },
    )

    model.fit(series[:split_idx])
    forecasts = model.historical_forecasts(
        series,
        start=split_idx,
        forecast_horizon=cfg.horizon,
        stride=cfg.stride,
        retrain=False,
        last_points_only=False,
    )

    def _normalize_forecasts(
        forecasts: Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
    ) -> List[TimeSeries]:
        if isinstance(forecasts, TimeSeries):
            return [forecasts]
        if not forecasts:
            return []
        if isinstance(forecasts[0], TimeSeries):  # type: ignore[index]
            return list(forecasts)  # type: ignore[return-value]
        flattened: List[TimeSeries] = []
        for group in forecasts:  # type: ignore[assignment]
            if isinstance(group, TimeSeries):
                flattened.append(group)
            else:
                flattened.extend(list(group))
        return flattened

    norm_forecasts = _normalize_forecasts(forecasts)
    preds = [ts.values().squeeze(-1) for ts in norm_forecasts]
    actuals = []
    for idx in range(split_idx, len(values) - cfg.horizon + 1, cfg.stride):
        actuals.append(values[idx : idx + cfg.horizon])
    if cfg.max_windows is not None:
        preds = preds[-cfg.max_windows:]
        actuals = actuals[-cfg.max_windows:]

    import numpy as np

    preds_arr = np.stack(preds)
    actuals_arr = np.stack(actuals)
    diff = preds_arr - actuals_arr
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    step_diff = diff[:, 0]
    rmse_t1 = float(np.sqrt(np.mean(step_diff**2)))
    mae_t1 = float(np.mean(np.abs(step_diff)))

    return [{
        "model": "tcn",
        "rmse": rmse,
        "mae": mae,
        "rmse_t1": rmse_t1,
        "mae_t1": mae_t1,
        "windows": float(len(actuals_arr)),
    }]


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
        args.input_window = int(cfg.get("window_size", args.input_window))
        args.train_ratio = float(cfg.get("split", {}).get("train", args.train_ratio))
        args.stride = int(cfg.get("stride", args.stride))
        tcn_cfg = cfg.get("models", {}).get("tcn", {})
        args.epochs = int(tcn_cfg.get("epochs", args.epochs))
        args.kernel_size = int(tcn_cfg.get("kernel_size", args.kernel_size))
        args.num_filters = int(tcn_cfg.get("channels", [args.num_filters])[-1])
        args.dropout = float(tcn_cfg.get("dropout", args.dropout))
        if args.plot is None:
            args.plot = cfg["reports_dir"] / "figures" / "tcn_baseline.png"
        if args.log_path is None:
            args.log_path = cfg["reports_dir"] / "tables" / "tcn_baseline_log.txt"
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

    class TCNCfg:
        input_window = args.input_window
        horizon = args.horizon
        train_ratio = args.train_ratio
        stride = args.stride
        max_windows = args.max_windows
        epochs = args.epochs
        kernel_size = args.kernel_size
        num_filters = args.num_filters
        dilation_base = args.dilation_base
        dropout = args.dropout
        devices = args.devices

    results = evaluate(series, values, TCNCfg())

    table = pd.DataFrame(results).sort_values("rmse")
    with pd.option_context("display.max_columns", None, "display.width", 120, "display.float_format", lambda x: f"{x:.4f}"):
        output = table.to_string(index=False)
        log_message(cfg, output)
    if args.plot:
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.bar(table["model"], table["rmse"], label="RMSE")
        plt.bar(table["model"], table["mae"], label="MAE", alpha=0.7)
        plt.ylabel("Error")
        plt.title("TCN baseline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close()
        log_message(cfg, f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
