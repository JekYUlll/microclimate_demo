from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def fit_scaler(df: pd.DataFrame, cols: List[str]) -> StandardScaler:
    """仅用训练集统计量拟合标准化器。"""
    mean = df[cols].mean().to_numpy()
    std = df[cols].std().replace(0, 1.0).to_numpy()
    return StandardScaler(mean=mean, std=std)


def apply_scaler(df: pd.DataFrame, cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    """应用标准化。"""
    out = df.copy()
    out[cols] = (out[cols] - scaler.mean) / scaler.std
    return out
