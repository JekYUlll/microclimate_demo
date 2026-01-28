from __future__ import annotations

from typing import List

import torch
from torch import nn


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        horizon: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_ch, ch, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.horizon = horizon
        self.target_dim = target_dim
        self.head = nn.Linear(in_ch, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        y = self.net(x)
        y = y[..., -1]  # last step
        out = self.head(y)
        return out.view(-1, self.horizon, self.target_dim)
