from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from experiments_scheduling_suite.src.models.transformer import PositionalEncoding


class InformerForecast(nn.Module):
    """简化版 Informer：编码器 + 逐层下采样（distilling）。"""

    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float, horizons: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
            self.layers.append(layer)
            self.downsamples.append(nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1))
        self.head = nn.Linear(d_model, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.pos(z)
        for layer, down in zip(self.layers, self.downsamples):
            z = layer(z)
            # 下采样：形状 (B, T, C) -> (B, C, T) -> (B, T', C)
            z = down(z.transpose(1, 2)).transpose(1, 2)
            if z.size(1) <= 2:
                break
        last = z[:, -1, :]
        return self.head(last)


def build_model(cfg: Dict, input_dim: int, horizons: int) -> InformerForecast:
    return InformerForecast(
        input_dim=input_dim,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_ff=int(cfg.get("dim_feedforward", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        horizons=horizons,
    )
