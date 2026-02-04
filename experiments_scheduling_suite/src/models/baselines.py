from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover
    MLPRegressor = None


class NaivePersistence:
    """朴素持久性：预测值等于最后时刻观测。"""

    def fit(self, X: np.ndarray, y: np.ndarray, target_idx: int = 0) -> None:
        self.horizons = y.shape[1]
        self.target_idx = target_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        last_val = X[:, -1, self.target_idx]
        return np.stack([last_val] * self.horizons, axis=1)


class MLPBaseline:
    """MLP 基线，输入为展平窗口。"""

    def __init__(self, hidden_sizes: Tuple[int, ...] = (128, 64), max_iter: int = 300) -> None:
        if MLPRegressor is None:
            raise ImportError("scikit-learn 未安装，无法使用 MLPBaseline")
        self.model = MLPRegressor(hidden_layer_sizes=hidden_sizes, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = X.reshape(X.shape[0], -1)
        self.model.fit(Xf, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = X.reshape(X.shape[0], -1)
        return self.model.predict(Xf)
