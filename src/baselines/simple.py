from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np


def _target_indices(feature_cols: List[str], target_cols: List[str]) -> List[int]:
    idxs = []
    for col in target_cols:
        if col in feature_cols:
            idxs.append(feature_cols.index(col))
        else:
            idxs.append(-1)
    return idxs


def naive_persistence(X: np.ndarray, feature_cols: List[str], target_cols: List[str], horizons: List[int]) -> np.ndarray:
    idxs = _target_indices(feature_cols, target_cols)
    last_step = X[:, -1, :]
    preds = np.zeros((X.shape[0], len(horizons), len(target_cols)), dtype=float)
    for t_i, feat_idx in enumerate(idxs):
        if feat_idx >= 0:
            preds[:, :, t_i] = last_step[:, feat_idx][:, None]
        else:
            preds[:, :, t_i] = np.nan
    return preds


def seasonal_naive(
    X: np.ndarray,
    feature_cols: List[str],
    target_cols: List[str],
    horizons: List[int],
    season_length: int,
) -> np.ndarray:
    idxs = _target_indices(feature_cols, target_cols)
    preds = np.zeros((X.shape[0], len(horizons), len(target_cols)), dtype=float)
    step_idx = -season_length if season_length <= X.shape[1] else -1
    for t_i, feat_idx in enumerate(idxs):
        if feat_idx >= 0:
            preds[:, :, t_i] = X[:, step_idx, feat_idx][:, None]
        else:
            preds[:, :, t_i] = np.nan
    return preds


def train_ar_coeffs(X: np.ndarray, Y: np.ndarray, feature_cols: List[str], target_cols: List[str], p: int = 5) -> Dict[str, np.ndarray]:
    coeffs: Dict[str, np.ndarray] = {}
    idxs = _target_indices(feature_cols, target_cols)
    for t_i, feat_idx in enumerate(idxs):
        if feat_idx < 0:
            continue
        p_use = min(p, X.shape[1])
        X_lags = X[:, -p_use:, feat_idx]
        X_design = np.concatenate([X_lags[:, ::-1], np.ones((X_lags.shape[0], 1))], axis=1)
        y = Y[:, 0, t_i]
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        coeffs[target_cols[t_i]] = beta
    return coeffs


def ar_predict(
    X: np.ndarray,
    feature_cols: List[str],
    target_cols: List[str],
    horizons: List[int],
    coeffs: Dict[str, np.ndarray],
) -> np.ndarray:
    preds = np.zeros((X.shape[0], len(horizons), len(target_cols)), dtype=float)
    idxs = _target_indices(feature_cols, target_cols)
    for t_i, feat_idx in enumerate(idxs):
        if feat_idx < 0 or target_cols[t_i] not in coeffs:
            preds[:, :, t_i] = np.nan
            continue
        beta = coeffs[target_cols[t_i]]
        p_use = len(beta) - 1
        for i in range(X.shape[0]):
            history = list(X[i, -p_use:, feat_idx].tolist())
            for h_idx in range(len(horizons)):
                x_vec = history[::-1] + [1.0]
                pred = float(np.dot(beta, np.array(x_vec)))
                preds[i, h_idx, t_i] = pred
                history.append(pred)
                history = history[-p_use:]
    return preds


def build_tabular_features(X: np.ndarray) -> np.ndarray:
    last_step = X[:, -1, :]
    mean_step = X.mean(axis=1)
    std_step = X.std(axis=1)
    return np.concatenate([last_step, mean_step, std_step], axis=1)


class Regressor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class LinearModel:
    beta: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_design = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X_design @ self.beta


def fit_tabular_regressor(X: np.ndarray, y: np.ndarray, cfg: Dict) -> Regressor:
    model_name = cfg.get("model", "gbrt")
    if model_name == "gbrt":
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=int(cfg.get("n_estimators", 200)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                max_depth=int(cfg.get("max_depth", 3)),
            ).fit(X, y)
        except Exception:
            pass

    # fallback: linear regression via lstsq
    X_design = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return LinearModel(beta=beta)


def predict_tabular(model: Regressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
