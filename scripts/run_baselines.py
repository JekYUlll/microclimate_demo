from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from darts import TimeSeries

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import SeriesConfig, log_header, log_message, prepare_series  # noqa: E402
from src.baselines.darts_baselines import BaselineConfig, evaluate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Darts baseline models for time series forecasting.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "AntAWS" / "3_hourly" / "Taishan_3h.csv",
        help="Path to the station CSV file.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Temperature(Ąć)",
        help="Target column name in the CSV.",
    )
    parser.add_argument("--encoding", type=str, default=None, help="Optional file encoding.")
    parser.add_argument("--freq", type=str, default=None, help="Optional resample frequency, e.g. 3H.")
    parser.add_argument(
        "--models",
        type=str,
        default="naive_mean,naive_drift,naive_seasonal,theta,ets,arima",
        help="Comma-separated model names.",
    )
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon.")
    parser.add_argument("--input-window", type=int, default=24, help="Input window length (for deep models).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for rolling forecasts.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit evaluation windows.")
    parser.add_argument("--season-length", type=int, default=8, help="Season length for seasonal naive.")
    parser.add_argument("--arima", type=str, default="2,1,2", help="ARIMA order as p,d,q.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for deep models.")
    parser.add_argument("--tft-hidden-size", type=int, default=32, help="TFT hidden size.")
    parser.add_argument("--tft-num-heads", type=int, default=4, help="TFT attention heads.")
    parser.add_argument("--tft-dropout", type=float, default=0.1, help="TFT dropout.")
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="PyTorch Lightning devices for deep models (multi-GPU on one host).",
    )
    parser.add_argument("--log-path", type=Path, default=None, help="Optional log file path.")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to save a bar chart comparing RMSE/MAE across models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    arima_order = tuple(int(x) for x in args.arima.split(","))
    model_list = [name.strip() for name in args.models.split(",") if name.strip()]

    base_cfg = BaselineConfig(
        models=model_list,
        horizon=args.horizon,
        input_chunk_length=args.input_window,
        train_ratio=args.train_ratio,
        stride=args.stride,
        max_windows=args.max_windows,
        season_length=args.season_length,
        arima_order=arima_order,  # type: ignore[arg-type]
        epochs=args.epochs,
        tft_hidden_size=args.tft_hidden_size,
        tft_num_heads=args.tft_num_heads,
        tft_dropout=args.tft_dropout,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": args.devices,
            "enable_progress_bar": True,
        }
        if args.devices > 0
        else None,
    )

    results = evaluate_all(series, values, base_cfg)
    table = pd.DataFrame(results)
    table = table.sort_values("rmse")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        output = table.to_string(index=False)
        log_message(cfg, output)

    if args.plot:
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.bar(table["model"], table["rmse"], label="RMSE")
        plt.bar(table["model"], table["mae"], label="MAE", alpha=0.7)
        plt.ylabel("Error")
        plt.title("Baseline comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close()
        log_message(cfg, f"Saved baseline comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
