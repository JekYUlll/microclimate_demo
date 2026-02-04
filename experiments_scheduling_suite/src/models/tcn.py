from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class Chomp1d(nn.Module):
    """裁剪以保证因果性。"""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCNModel(nn.Module):
    """简单 TCN，多层扩张卷积。"""

    def __init__(self, input_dim: int, channels: List[int], kernel_size: int, dropout: float, horizons: int) -> None:
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        z = self.network(x.transpose(1, 2))
        last = z[:, :, -1]
        return self.head(last)


def build_model(cfg: Dict, input_dim: int, horizons: int) -> TCNModel:
    return TCNModel(
        input_dim=input_dim,
        channels=[int(x) for x in cfg.get("channels", [32, 32])],
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        horizons=horizons,
    )
