from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.normalize import load_scaler
from src.models.tcn import TCNForecaster


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def load_npz(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"],
        "Y": data["Y"],
        "t_ref": data["t_ref"],
        "feature_cols": list(data["feature_cols"]),
        "target_cols": list(data["target_cols"]),
        "horizons": list(data["horizons"]),
    }


def train_tcn(cfg: Dict) -> Dict:
    train_data = load_npz(cfg["processed_dir"] / "train.npz")
    val_data = load_npz(cfg["processed_dir"] / "val.npz")
    test_data = load_npz(cfg["processed_dir"] / "test.npz")
    scaler = load_scaler(cfg["processed_dir"] / "scaler.json")

    X_train, Y_train = train_data["X"], train_data["Y"]
    X_val, Y_val = val_data["X"], val_data["Y"]

    horizon = Y_train.shape[1]
    target_dim = Y_train.shape[2]
    input_dim = X_train.shape[2]

    tcn_cfg = cfg.get("models", {}).get("tcn", {})
    model = TCNForecaster(
        input_dim=input_dim,
        target_dim=target_dim,
        horizon=horizon,
        channels=list(tcn_cfg.get("channels", [32, 32])),
        kernel_size=int(tcn_cfg.get("kernel_size", 3)),
        dropout=float(tcn_cfg.get("dropout", 0.1)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(tcn_cfg.get("batch_size", 64))
    train_loader = DataLoader(WindowDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=float(tcn_cfg.get("lr", 1e-3)))
    loss_fn = nn.MSELoss()

    epochs = int(tcn_cfg.get("epochs", 20))
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= max(len(val_loader.dataset), 1)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}: train {train_loss:.4f} val {val_loss:.4f}")

    model.eval()
    test_X = torch.tensor(test_data["X"], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(test_X).cpu().numpy()

    y_true = test_data["Y"]
    y_pred = scaler.inverse_transform_y(preds)
    y_true_denorm = scaler.inverse_transform_y(y_true)

    return {
        "model": model,
        "history": history,
        "y_true": y_true_denorm,
        "y_pred": y_pred,
        "t_ref": test_data["t_ref"],
        "target_cols": test_data["target_cols"],
        "horizons": test_data["horizons"],
    }
