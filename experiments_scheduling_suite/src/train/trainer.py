from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    device: str


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig) -> Dict[str, list[float]]:
    """训练 PyTorch 模型，返回损失曲线。"""
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = _build_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    for _ in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

    return history


def predict_model(model: nn.Module, X: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    """模型预测。"""
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()
    loader = _build_loader(X, np.zeros((len(X), 1)), cfg.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)
