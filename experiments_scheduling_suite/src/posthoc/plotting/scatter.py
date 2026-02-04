from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_scatter(
    metrics_long: pd.DataFrame,
    features: pd.DataFrame,
    out_dir: Path,
    horizon: int,
    x_col: str,
    metric: str = "rmse",
) -> Path:
    """缺失形态特征 vs 误差散点图。"""
    apply_style()
    df = metrics_long[(metrics_long["horizon"] == horizon) & (metrics_long["metric"] == metric)]
    df = df.merge(features, on="run_id", how="left")
    df = df.dropna(subset=[x_col])
    if df.empty:
        return out_dir

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, sub in df.groupby("missingness"):
        ax.scatter(sub[x_col], sub["value"], label=name, alpha=0.7, s=20)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(f"{x_col} vs {metric} (H={horizon})")
    ax.legend(fontsize=7)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scatter_{x_col}_H{horizon}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
