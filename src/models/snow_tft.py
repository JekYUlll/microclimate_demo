from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SnowTFT(nn.Module):
    """
    Encoder-only Transformer forecaster for multi-horizon, multi-target prediction.

    Input:  [B, window, input_dim]
    Output: [B, horizon, target_dim]
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim
        self.enc_in = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.enc_in(x)
        enc = self.pos_enc(enc)
        mem = self.encoder(enc)
        last = mem[:, -1]
        out = self.head(last)
        return out.view(-1, self.horizon, self.target_dim)
