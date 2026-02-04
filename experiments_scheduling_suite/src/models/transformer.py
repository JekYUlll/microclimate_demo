from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecast(nn.Module):
    """编码器式 Transformer 预测模型。"""

    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float, horizons: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.pos(z)
        # 仅用编码器，取最后一个时间步做预测
        h = self.encoder(z)
        last = h[:, -1, :]
        return self.head(last)


def build_model(cfg: Dict, input_dim: int, horizons: int) -> TransformerForecast:
    return TransformerForecast(
        input_dim=input_dim,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_ff=int(cfg.get("dim_feedforward", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        horizons=horizons,
    )
