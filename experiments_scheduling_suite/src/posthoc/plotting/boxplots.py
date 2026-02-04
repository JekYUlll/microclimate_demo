from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_boxplot(metrics_long: pd.DataFrame, out_dir: Path, horizon: int, metric: str = "rmse") -> Path:
    """按策略分组的箱线图。"""
    apply_style()
    df = metrics_long[(metrics_long["horizon"] == horizon) & (metrics_long["metric"] == metric)]
    if df.empty:
        return out_dir

    labels = sorted(df["missingness_label"].unique())
    data = [df[df["missingness_label"] == lab]["value"].dropna().to_numpy() for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    # jitter 点
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, alpha=0.6)
    ax.set_title(f"Boxplot H={horizon} ({metric})")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"boxplot_H{horizon}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
