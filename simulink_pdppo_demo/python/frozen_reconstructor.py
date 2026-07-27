"""Fixed downstream reconstruction model for the Simulink demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "frozen_model" / "channel_reconstructor.npz"


class FrozenChannelReconstructor:
    """Small fixed model that reconstructs inactive sensor channels.

    The model is deliberately static: all coefficients are loaded from disk and
    no fitting is performed during the demo.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Frozen reconstruction model not found: {model_path}. "
                "Run python/export_frozen_demo_models.py first."
            )
        data = np.load(model_path, allow_pickle=True)
        self.channels = [str(x) for x in data["channels"].tolist()]
        self.baseline = data["baseline"].astype(float)
        self.decay = data["decay"].astype(float)
        self.coupling = data["coupling"].astype(float)
        self.event_gain = data["event_gain"].astype(float)
        self.observed_blend = data["observed_blend"].astype(float)

    def reconstruct(self, partial: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
        values = partial[self.channels].to_numpy(dtype=float)
        observed = mask[self.channels].to_numpy(dtype=float) > 0.5

        output = np.zeros_like(values, dtype=float)
        prev = self._initial_state(values, observed)

        for t in range(values.shape[0]):
            row = values[t]
            obs = observed[t] & np.isfinite(row)
            context = np.where(obs, row, prev)
            event_proxy = float(np.nanmax(context[[5, 6, 7]]))
            event_proxy = np.clip(event_proxy, 0.0, 1.0)

            structural = self.baseline + self.coupling @ (context - self.baseline)
            predicted = (
                self.decay * prev
                + (1.0 - self.decay) * structural
                + self.event_gain * event_proxy * (1.0 - self.decay)
            )
            predicted = np.clip(predicted, 0.0, 1.0)

            current = predicted.copy()
            current[obs] = (
                self.observed_blend[obs] * row[obs]
                + (1.0 - self.observed_blend[obs]) * predicted[obs]
            )
            current = np.clip(current, 0.0, 1.0)
            output[t] = current
            prev = current

        reconstructed = partial.copy()
        reconstructed[self.channels] = output
        return reconstructed

    def _initial_state(self, values: np.ndarray, observed: np.ndarray) -> np.ndarray:
        init = self.baseline.copy()
        for i in range(values.shape[1]):
            idx = np.flatnonzero(observed[:, i] & np.isfinite(values[:, i]))
            if idx.size:
                init[i] = values[idx[0], i]
        return np.clip(init, 0.0, 1.0)
