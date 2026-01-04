from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch


class SnowDataset(torch.utils.data.Dataset):
    """Windowed sequence dataset for snow forecasting.

    Optionally returns known future covariates for decoder-style models (e.g., TFT).
    """

    def __init__(
        self,
        feats: np.ndarray,
        tgs: np.ndarray,
        window: int,
        horizon: int,
        future_known_idx: Optional[Iterable[int]] = None,
    ):
        self.feats = feats
        self.tgs = tgs
        self.window = window
        self.horizon = horizon
        self.max_idx = len(self.feats) - window - horizon
        self.future_known_idx: Optional[List[int]] = list(future_known_idx) if future_known_idx is not None else None

    def __len__(self):
        return max(self.max_idx, 0)

    def __getitem__(self, idx):
        x = self.feats[idx: idx + self.window]
        y = self.tgs[idx + self.window: idx + self.window + self.horizon]
        if self.future_known_idx is not None:
            future_known = self.feats[idx + self.window: idx + self.window + self.horizon, self.future_known_idx]
            return torch.from_numpy(x), torch.from_numpy(future_known), torch.from_numpy(y)
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class DataLoaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader


def build_loaders(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    horizon: int,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 0,
    future_known_idx: Optional[Iterable[int]] = None,
) -> DataLoaders:
    """Split arrays and build PyTorch dataloaders."""
    split = int(len(features) * train_ratio)
    train_ds = SnowDataset(features[:split], targets[:split], window_size, horizon, future_known_idx=future_known_idx)
    val_ds = SnowDataset(
        features[split - window_size:],
        targets[split - window_size:],
        window_size,
        horizon,
        future_known_idx=future_known_idx,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return DataLoaders(train=train_loader, val=val_loader)
