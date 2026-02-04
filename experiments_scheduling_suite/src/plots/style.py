from __future__ import annotations

import matplotlib.pyplot as plt


def apply_style() -> None:
    """统一图表风格。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.figsize": (8, 4),
    })
