from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_heatmaps(
    metrics_long: pd.DataFrame,
    out_dir: Path,
    model: str,
    horizon: int,
    metric: str = "rmse",
    baseline_label: Optional[str] = None,
    use_delta: bool = True,
) -> Path:
    """绘制 missingness x imputer 的热力图。"""
    apply_style()
    df = metrics_long.copy()
    df = df[(df["model"] == model) & (df["horizon"] == horizon) & (df["metric"] == metric)]
    if df.empty:
        return out_dir

    pivot = df.pivot_table(index="missingness_label", columns="imputer", values="value", aggfunc="mean")

    if use_delta and baseline_label:
        base = df[df["baseline_label"] == baseline_label]
        base_val = base["value"].mean() if not base.empty else None
        if base_val is not None and np.isfinite(base_val) and base_val != 0:
            pivot = (pivot - base_val) / base_val * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{model} H={horizon} ({metric})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{model}_H{horizon}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
