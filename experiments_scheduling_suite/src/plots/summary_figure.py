from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from experiments_scheduling_suite.src.plots.style import apply_style
from matplotlib.lines import Line2D


def _build_color_map(models: List[str]) -> Dict[str, np.ndarray]:
    """为模型生成一致的颜色映射。"""
    if not models:
        return {}
    cmap = plt.cm.get_cmap("tab20", max(1, len(models)))
    return {name: cmap(i) for i, name in enumerate(models)}


def _radar_chart(ax, labels: List[str], values: List[float], title: str, color: str, alpha: float = 0.2) -> None:
    # 简易雷达图：将指标归一化到 [0,1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]
    ax.plot(angles, values, linewidth=1.2, color=color)
    ax.fill(angles, values, alpha=alpha, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=9)


def plot_summary(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    horizons: List[int],
    out_path: Path,
) -> None:
    """生成包含主图+缩放+雷达图的总结图。"""
    apply_style()
    fig = plt.figure(figsize=(13, 8))

    # 统一模型顺序与颜色（按名称排序，保证跨图一致）
    model_names = sorted(model_preds.keys())
    color_map = _build_color_map(model_names)

    # 主图：全段 overlay
    ax_main = fig.add_subplot(2, 2, 1)
    ax_main.plot(timestamps, y_true[:, 0], label="true", linewidth=1.2, color="black")
    for name in model_names:
        pred = model_preds[name]
        ax_main.plot(timestamps, pred[:, 0], label=name, linewidth=0.8, alpha=0.9, color=color_map.get(name))
    ax_main.set_title("Overlay (H=1)")
    # 把主图图例放到图外，避免遮挡曲线
    ax_main.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    # 缩放图：取中间片段
    ax_zoom = fig.add_subplot(2, 2, 2)
    n = len(timestamps)
    start = int(n * 0.4)
    end = int(n * 0.5)
    # 在主图上标注缩放区域
    if n > 0:
        ax_main.axvspan(timestamps[start], timestamps[end - 1], color="gray", alpha=0.15, linestyle="--")
    ax_zoom.plot(timestamps[start:end], y_true[start:end, 0], label="true", linewidth=1.2, color="black")
    for name in model_names:
        pred = model_preds[name]
        ax_zoom.plot(timestamps[start:end], pred[start:end, 0], label=name, linewidth=0.8, alpha=0.9, color=color_map.get(name))
    ax_zoom.set_title("Zoom-in (H=1)")

    # 雷达图：分别对应 h=1/2/3（每个模型一条线）
    metric_labels = ["RMSE", "MAE", "MAPE"]
    for i, h in enumerate(horizons[:3]):
        ax_radar = fig.add_subplot(2, 3, 4 + i, polar=True)
        # 只保留指标齐全的模型
        valid_models = []
        for name in model_names:
            if name not in metrics:
                continue
            row = metrics[name]
            vals = [row.get(f"rmse_h{h}"), row.get(f"mae_h{h}"), row.get(f"mape_h{h}")]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                continue
            valid_models.append(name)
        if not valid_models:
            ax_radar.set_title(f"H={h} (无有效指标)", fontsize=9)
            ax_radar.set_xticks([])
            ax_radar.set_yticks([])
            continue

        # 对每个指标做最大值归一化（跨模型）
        max_rmse = max(metrics[m][f"rmse_h{h}"] for m in valid_models) or 1.0
        max_mae = max(metrics[m][f"mae_h{h}"] for m in valid_models) or 1.0
        max_mape = max(metrics[m][f"mape_h{h}"] for m in valid_models) or 1.0
        max_rmse = max(max_rmse, 1e-9)
        max_mae = max(max_mae, 1e-9)
        max_mape = max(max_mape, 1e-9)

        for name in valid_models:
            row = metrics[name]
            values = [
                row[f"rmse_h{h}"] / max_rmse,
                row[f"mae_h{h}"] / max_mae,
                row[f"mape_h{h}"] / max_mape,
            ]
            _radar_chart(ax_radar, metric_labels, values, title=f"H={h}", color=color_map.get(name, "tab:blue"), alpha=0.12)
        ax_radar.set_title(f"H={h}", fontsize=9)
        # 雷达图图例统一放到整张图右侧，避免遮挡
        if i == 0:
            handles = [
                Line2D([0], [0], color=color_map.get(name, "tab:blue"), lw=1.5)
                for name in valid_models
            ]
            fig.legend(handles, valid_models, loc="center right", bbox_to_anchor=(0.98, 0.5), fontsize=7, frameon=False)

    # 手动调整边距，给右侧图例留空间
    fig.subplots_adjust(left=0.06, right=0.82, top=0.93, bottom=0.07, hspace=0.35, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
