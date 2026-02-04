from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_sensitivity(
    metrics_long: pd.DataFrame,
    out_dir: Path,
    missingness: str,
    horizon: int,
    metric: str = "rmse",
    top_models: Optional[int] = None,
) -> Path:
    """绘制参数敏感性曲线。"""
    apply_style()
    df = metrics_long[
        (metrics_long["missingness"] == missingness)
        & (metrics_long["horizon"] == horizon)
        & (metrics_long["metric"] == metric)
    ]
    if df.empty or "param_value" not in df.columns:
        return out_dir

    if top_models:
        # 按平均表现选前 N 模型
        model_rank = df.groupby("model")["value"].mean().sort_values().head(top_models).index.tolist()
        df = df[df["model"].isin(model_rank)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("param_value")
        ax.plot(sub["param_value"], sub["value"], marker="o", label=model)
    ax.set_xlabel("param_value")
    ax.set_ylabel(metric)
    ax.set_title(f"Sensitivity {missingness} H={horizon}")
    ax.legend(fontsize=7)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sensitivity_{missingness}_H{horizon}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
