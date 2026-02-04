from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments_scheduling_suite.src.plots.style import apply_style


def _gap_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for missing in mask:
        if missing:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def plot_missingness_heatmap(mask_df: pd.DataFrame, out_path: Path) -> None:
    """时间×变量的缺失热力图。"""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mask_df.T, aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_yticks(range(mask_df.shape[1]))
    ax.set_yticklabels(mask_df.columns, fontsize=8)
    ax.set_xlabel("time index")
    ax.set_title("Missingness heatmap (1=observed, 0=missing)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gap_distribution(mask_df: pd.DataFrame, out_path: Path) -> None:
    """缺失段长度分布（直方图）。"""
    apply_style()
    gaps: List[int] = []
    for col in mask_df.columns:
        gaps.extend(_gap_lengths(mask_df[col].to_numpy() == 0))
    fig, ax = plt.subplots()
    if gaps:
        ax.hist(gaps, bins=40, alpha=0.8)
    ax.set_xlabel("gap length (steps)")
    ax.set_ylabel("count")
    ax.set_title("Gap length distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overlay(original: pd.DataFrame, masked: pd.DataFrame, imputed: pd.DataFrame, target: str, out_path: Path) -> None:
    """原始/遮罩/插补序列叠加示例。"""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original["timestamp"], original[target], label="original", linewidth=1.0)
    ax.plot(masked["timestamp"], masked[target], label="masked", linewidth=1.0, alpha=0.7)
    ax.plot(imputed["timestamp"], imputed[target], label="imputed", linewidth=1.2)
    ax.set_title(f"Overlay - {target}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, cols: List[str], out_path: Path) -> None:
    """关键特征分布直方图。"""
    apply_style()
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", fontsize=9)
            ax.set_title(col)
            ax.set_axis_off()
            continue
        # 如果取值范围过小，避免直方图分箱失败
        vmin = float(series.min())
        vmax = float(series.max())
        if np.isclose(vmin, vmax):
            ax.axvline(vmin, color="tab:blue")
            ax.set_title(f"{col} (常数)")
            continue
        # 动态调整分箱数量，避免“Too many bins”错误
        bins = min(40, max(5, int(np.sqrt(len(series)))))
        ax.hist(series, bins=bins, alpha=0.8)
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_pretrain_viz(
    original: pd.DataFrame,
    masked: pd.DataFrame,
    imputed: pd.DataFrame,
    mask_df: pd.DataFrame,
    target: str,
    out_dir: Path,
    feature_cols: Optional[List[str]] = None,
) -> None:
    """统一入口：生成预训练可视化。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_missingness_heatmap(mask_df, out_dir / "missingness_heatmap.png")
    plot_gap_distribution(mask_df, out_dir / "gap_length_hist.png")
    plot_overlay(original, masked, imputed, target, out_dir / "overlay.png")
    if feature_cols:
        plot_feature_distributions(imputed, feature_cols, out_dir / "feature_distributions.png")
