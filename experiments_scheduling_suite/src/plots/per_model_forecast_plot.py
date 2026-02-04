from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int],
    out_path: Path,
    title: str,
) -> None:
    """将 h=1/2/3 预测与真实值画在同一张图中。"""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.plot(timestamps, y_true[:, i], label="true", linewidth=1.0)
        ax.plot(timestamps, y_pred[:, i], label="pred", linewidth=1.0)
        ax.set_title(f"H={h}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
