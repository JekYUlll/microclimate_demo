from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def make_overview(figures: list[Path], out_path: Path) -> None:
    """简单拼图：将若干图片按网格拼接（占位实现）。"""
    # 这里只给出占位实现，后续可根据需要扩展
    if not figures:
        return
    cols = 2
    rows = (len(figures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.ravel()
    for ax, path in zip(axes, figures):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.name, fontsize=8)
    for ax in axes[len(figures):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
