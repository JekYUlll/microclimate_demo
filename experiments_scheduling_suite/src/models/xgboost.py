from __future__ import annotations

from typing import List

import numpy as np

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


class XGBoostModel:
    """按 horizon 分别训练 XGBoost。"""

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost 未安装")
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        self.models: List[XGBRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = X.reshape(X.shape[0], -1)
        self.models = []
        for h in range(y.shape[1]):
            model = XGBRegressor(**self.params)
            model.fit(Xf, y[:, h])
            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = X.reshape(X.shape[0], -1)
        preds = [model.predict(Xf) for model in self.models]
        return np.stack(preds, axis=1)
