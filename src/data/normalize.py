from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np


@dataclass
class StandardScaler:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "StandardScaler":
        feature_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        feature_std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
        target_mean = Y.reshape(-1, Y.shape[-1]).mean(axis=0)
        target_std = Y.reshape(-1, Y.shape[-1]).std(axis=0) + 1e-6
        return cls(feature_mean, feature_std, target_mean, target_std)

    def transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xn = (X - self.feature_mean) / self.feature_std
        Yn = (Y - self.target_mean) / self.target_std
        return Xn, Yn

    def inverse_transform_y(self, Y: np.ndarray) -> np.ndarray:
        return Y * self.target_std + self.target_mean

    def to_dict(self) -> Dict:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StandardScaler":
        return cls(
            feature_mean=np.array(data["feature_mean"], dtype=float),
            feature_std=np.array(data["feature_std"], dtype=float),
            target_mean=np.array(data["target_mean"], dtype=float),
            target_std=np.array(data["target_std"], dtype=float),
        )


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.write_text(json.dumps(scaler.to_dict(), indent=2))


def load_scaler(path: Path) -> StandardScaler:
    return StandardScaler.from_dict(json.loads(path.read_text()))
