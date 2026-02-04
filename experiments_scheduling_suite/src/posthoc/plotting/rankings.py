from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments_scheduling_suite.src.plots.style import apply_style


def rank_correlation(metrics_long: pd.DataFrame, out_dir: Path, horizon: int, metric: str = "rmse") -> Path:
    """计算不同策略的模型排名相关性矩阵。"""
    apply_style()
    df = metrics_long[(metrics_long["horizon"] == horizon) & (metrics_long["metric"] == metric)]
    if df.empty:
        return out_dir

    strategies = sorted(df["missingness_label"].unique())
    model_scores = {
        strat: df[df["missingness_label"] == strat].groupby("model")["value"].mean() for strat in strategies
    }

    # 统一模型集合
    common_models = set.intersection(*(set(s.index) for s in model_scores.values())) if strategies else set()
    common_models = sorted(common_models)
    if not common_models:
        return out_dir

    ranks = {}
    for strat in strategies:
        scores = model_scores[strat].loc[common_models]
        ranks[strat] = scores.rank(method="average")

    corr = np.zeros((len(strategies), len(strategies)))
    for i, a in enumerate(strategies):
        for j, b in enumerate(strategies):
            rho, _ = spearmanr(ranks[a], ranks[b])
            corr[i, j] = rho

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(strategies)))
    ax.set_yticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_yticklabels(strategies)
    ax.set_title(f"Rank correlation H={horizon}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank_corr_heatmap_H{horizon}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
