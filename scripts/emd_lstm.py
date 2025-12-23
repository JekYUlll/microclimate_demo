from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.EMD_LSTM.emd_lstm import ExperimentConfig, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMD-LSTM forecasting and plot results.")
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
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns (default: all numeric columns).",
    )
    parser.add_argument("--encoding", type=str, default=None, help="Optional file encoding.")
    parser.add_argument("--input-window", type=int, default=24, help="Input window length.")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per IMF.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between LSTM layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force cpu/cuda.")
    parser.add_argument(
        "--max-imfs",
        type=int,
        default=8,
        help="Maximum number of IMFs to model (use 0 for all).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PROJECT_ROOT / "plots" / "emd_lstm_taishan.png",
        help="Path to save plot.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Limit number of points on plot (use 0 for all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig(
        data_path=args.data,
        target_col=args.target_col,
        feature_cols=[col.strip() for col in args.feature_cols.split(",")] if args.feature_cols else None,
        encoding=args.encoding,
        input_window=args.input_window,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        max_imfs=None if args.max_imfs == 0 else args.max_imfs,
        max_points=None if args.max_points == 0 else args.max_points,
        plot_path=args.plot,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
