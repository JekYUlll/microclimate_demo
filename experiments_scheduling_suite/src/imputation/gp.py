from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.imputation.base import BaseImputer

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None


class GPImputer(BaseImputer):
    """按列进行 GP 插补（小规模数据可用）。"""

    def __init__(self, length_scale: float = 5.0, noise: float = 1e-3) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self._report: Dict[str, object] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn 未安装，无法使用 GP 插补")
        out = df.copy()
        value_cols = [c for c in out.columns if c != "timestamp"]
        out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")
        x = np.arange(len(out)).reshape(-1, 1)
        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.noise)
        for col in value_cols:
            y = out[col].to_numpy()
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                out[col] = pd.Series(y).interpolate(method="linear").ffill().bfill().to_numpy()
                continue
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
            gp.fit(x[mask], y[mask])
            y_pred = gp.predict(x)
            out[col] = y_pred
        self._report = {"method": "gp", "length_scale": self.length_scale, "noise": self.noise}
        return out

    def report(self) -> Dict[str, object]:
        return self._report
