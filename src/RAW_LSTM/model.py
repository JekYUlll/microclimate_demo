from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """A simple sequence-to-sequence baseline using stacked LSTMs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window, features)
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.regressor(last_hidden)
