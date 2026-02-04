from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class LSTMModel(nn.Module):
    """多步预测 LSTM：使用最后时刻隐藏状态做回归。"""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, horizons: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def build_model(cfg: Dict, input_dim: int, horizons: int) -> LSTMModel:
    return LSTMModel(
        input_dim=input_dim,
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        horizons=horizons,
    )
