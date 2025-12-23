from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.RAW_LSTM.config import Settings
from src.RAW_LSTM.data import WindowDataset, feature_matrix, normalization_stats, select_features
from src.RAW_LSTM.model import LSTMForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM forecaster on the processed meteorological data.")
    parser.add_argument("--station", type=str, default=None, help="Train using data from the specified station only.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count defined in the config.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional path to a processed CSV file (defaults to data/processed/meteorology.csv).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--freq", type=str, default=None, help="Optional resampling frequency when loading raw data.")
    parser.add_argument("--device", type=str, default=None, help="Force training on cpu/cuda if available.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_datasets(
    values: np.ndarray,
    split_idx: int,
    target_idx: int,
    input_window: int,
    horizon: int,
) -> Tuple[WindowDataset, WindowDataset]:
    train_dataset = WindowDataset(values, target_idx, input_window, horizon, start=0, end=split_idx)
    val_start = max(split_idx - input_window - horizon + 1, 0)
    val_dataset = WindowDataset(values, target_idx, input_window, horizon, start=val_start)
    return train_dataset, val_dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    seen = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        preds = model(features)
        loss = criterion(preds, targets)
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size
    return running_loss / max(seen, 1)


def main() -> None:
    args = parse_args()
    settings = Settings()

    if args.freq:
        settings.data.freq = args.freq
    if args.epochs:
        settings.training.epochs = args.epochs
    if args.batch_size:
        settings.training.batch_size = args.batch_size
    if args.lr:
        settings.training.learning_rate = args.lr

    set_seed(settings.training.seed)

    processed_path = args.data or (settings.paths.processed_data_dir / "meteorology.csv")
    if not processed_path.exists():
        msg = (
            f"{processed_path} not found. Run `python scripts/prepare_data.py` "
            "to generate the processed dataset."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(processed_path, parse_dates=["timestamp"])
    if args.station:
        df = df[df["station"] == args.station]
    if df.empty:
        raise ValueError("No rows found for the requested configuration/station.")

    df = df.sort_values("timestamp")
    df, feature_cols = select_features(df, settings.data)
    matrix = feature_matrix(df, feature_cols)

    split_idx = int(len(matrix) * settings.data.train_ratio)
    min_required = settings.training.input_window + settings.training.forecast_horizon
    if len(matrix) < (min_required * 2):
        raise ValueError(
            "Not enough samples for the requested window and horizon. "
            "Consider lowering input_window/forecast_horizon or aggregating the data."
        )
    if split_idx < min_required:
        split_idx = min_required
    split_idx = min(split_idx, len(matrix) - min_required)
    split_idx = max(split_idx, min_required)

    train_values = matrix[:split_idx]
    mean, std = normalization_stats(train_values)
    matrix = (matrix - mean) / std

    target_idx = feature_cols.index(settings.data.target_col)

    train_dataset, val_dataset = create_datasets(
        matrix,
        split_idx=split_idx,
        target_idx=target_idx,
        input_window=settings.training.input_window,
        horizon=settings.training.forecast_horizon,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        num_workers=settings.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=settings.training.num_workers,
    )

    input_size = matrix.shape[1]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=settings.training.hidden_size,
        horizon=settings.training.forecast_horizon,
        num_layers=settings.training.num_layers,
        dropout=settings.training.dropout,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.training.learning_rate)

    best_val = float("inf")
    checkpoint_dir = settings.paths.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = settings.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, settings.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint_path = checkpoint_dir / f"lstm_{args.station or 'all'}.pt"
            payload = {
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "target_col": settings.data.target_col,
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "train_loss": train_loss,
                "val_loss": val_loss,
                "settings": asdict(settings),
            }
            torch.save(payload, checkpoint_path)
            print(f"  -> Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
