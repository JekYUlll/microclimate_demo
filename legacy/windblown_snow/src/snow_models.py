from __future__ import annotations

from torch import nn


class SnowLSTM(nn.Module):
    """Simple LSTM head to forecast snow flux targets from windowed covariates."""

    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2, horizon: int = 6, target_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, target_dim)
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1:, :]
        repeated = last.repeat(1, self.horizon, 1)
        preds = self.head(repeated)
        return preds
