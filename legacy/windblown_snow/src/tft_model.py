from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

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
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class SnowTFT(nn.Module):
    """
    Lightweight Transformer-based decoder for multi-target forecasting.

    Inputs:
      encoder_x: [B, window, input_dim]  (observed history)
      decoder_known: [B, horizon, known_future_dim]  (known future covariates; can be zeros)
    Output:
      preds: [B, horizon, target_dim]
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        known_future_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_in = nn.Linear(input_dim, d_model)
        self.dec_in = nn.Linear(known_future_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, target_dim)

    def forward(self, encoder_x: torch.Tensor, decoder_known: torch.Tensor) -> torch.Tensor:
        enc = self.enc_in(encoder_x)
        dec = self.dec_in(decoder_known)
        enc = self.pos_enc(enc)
        dec = self.pos_enc(dec)
        mem = self.encoder(enc)
        out = self.decoder(tgt=dec, memory=mem)
        preds = self.head(out)
        return preds
