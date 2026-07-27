#!/usr/bin/env python3
"""Plot robust-planner schedule traces from step-trace CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


DEFAULT_RUN_DIR = Path(
    "v1/artifacts/robust_objective_sweep_44_45_20260607/"
    "seed44_oracle0_event0_task1_all"
)
DEFAULT_OUT_DIR = Path("v1/artifacts/schedule_trace_figures_20260608")

SENSOR_IDS = (
    "met_station_core",
    "radiometer_basic",
    "surface_temp_ir",
    "ultrasonic_anemometer_hd",
    "shielded_thermo_hygro",
    "snow_particle_counter",
    "laser_disdrometer",
    "fc4_flux",
)

SENSOR_LABELS = (
    "Core met",
    "Radiometer",
    "Surface IR",
    "Hi-res wind",
    "Thermo-hygro",
    "Particle ctr",
    "Laser disd.",
    "FC4 flux",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot active sensor masks and task error for a robust trace."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--split", choices=("validation", "final"), default="final")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Window start to plot. Default chooses the largest paired margin.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def mask_bits_to_array(bits: str) -> np.ndarray:
    return np.asarray([char == "1" for char in str(bits).strip()], dtype=int)


def mask_matrix(df: pd.DataFrame) -> np.ndarray:
    rows = [mask_bits_to_array(bits) for bits in df["mask_bits"].astype(str)]
    matrix = np.vstack(rows).astype(int)
    if matrix.shape[1] != len(SENSOR_IDS):
        raise ValueError(f"expected {len(SENSOR_IDS)} sensors, got {matrix.shape[1]}")
    return matrix


def choose_start(run_dir: Path, split: str) -> int:
    paired = pd.read_csv(run_dir / f"{split}_paired.csv")
    if len(paired) == 0:
        raise ValueError(f"{split}_paired.csv has no rows")
    row = paired.sort_values("margin", ascending=False).iloc[0]
    return int(row["start"])


def task_error(df: pd.DataFrame) -> np.ndarray:
    columns = [name for name in df.columns if name.startswith("norm_error_")]
    if not columns:
        return np.asarray(df["oracle_loss"], dtype=float)
    return df[columns].mean(axis=1).to_numpy(dtype=float)


def schedule_stats(matrix: np.ndarray) -> dict[str, float | int]:
    switches = np.any(np.diff(matrix, axis=0) != 0, axis=1) if len(matrix) > 1 else []
    return {
        "active_mean": float(matrix.sum(axis=1).mean()),
        "switch_rate": float(np.mean(switches)) if len(matrix) > 1 else 0.0,
        "unique_masks": int(len({tuple(row.tolist()) for row in matrix})),
    }


def policy_slice(step_trace: pd.DataFrame, *, policy: str, start: int) -> pd.DataFrame:
    rows = step_trace.loc[
        (step_trace["policy"] == policy) & (step_trace["start"].astype(int) == int(start))
    ].copy()
    if len(rows) == 0:
        raise ValueError(f"no rows for policy={policy}, start={start}")
    return rows.sort_values("relative_step").reset_index(drop=True)


def add_event_spans(ax: plt.Axes, event: np.ndarray) -> None:
    event = np.asarray(event, dtype=float)
    in_span = False
    start = 0
    for idx, value in enumerate(event):
        if value > 0.5 and not in_span:
            in_span = True
            start = idx
        if in_span and (value <= 0.5 or idx == len(event) - 1):
            end = idx if value <= 0.5 else idx + 1
            ax.axvspan(start - 0.5, end - 0.5, color="#E69F00", alpha=0.12, lw=0)
            in_span = False


def plot_trace(run_dir: Path, *, split: str, start: int, out_dir: Path, stem: str) -> Path:
    step_trace = pd.read_csv(run_dir / f"{split}_step_trace.csv")
    paired = pd.read_csv(run_dir / f"{split}_paired.csv")
    paired_row = paired.loc[paired["start"].astype(int) == int(start)]
    if len(paired_row) == 0:
        raise ValueError(f"start {start} not found in {split}_paired.csv")
    paired_row = paired_row.iloc[0]

    static = policy_slice(step_trace, policy="validation_selected_static", start=start)
    robust = policy_slice(step_trace, policy="robust_world_model_mpc", start=start)
    static_masks = mask_matrix(static)
    robust_masks = mask_matrix(robust)
    steps = np.arange(len(robust_masks))
    event = robust["event"].to_numpy(dtype=float)
    static_error = task_error(static)
    robust_error = task_error(robust)
    static_stats = schedule_stats(static_masks)
    robust_stats = schedule_stats(robust_masks)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.6, 6.5), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.34, 1.6, 1.6, 1.35],
        hspace=0.22,
        left=0.16,
        right=0.98,
        top=0.91,
        bottom=0.12,
    )
    fig.suptitle(
        (
            "Robust objective-family schedule trace "
            f"({split}, start={start}, margin={float(paired_row['margin']):+.4f})"
        ),
        fontsize=11,
        fontweight="bold",
    )

    event_ax = fig.add_subplot(grid[0])
    event_ax.imshow(
        event.reshape(1, -1),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#F7F7F5", "#E69F00"]),
        vmin=0,
        vmax=1,
    )
    event_ax.set_yticks([])
    event_ax.set_xticks([])
    event_ax.set_title(f"Event context: event duty={event.mean()*100:.1f}%")
    for spine in event_ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    cmap = ListedColormap(["#F7F7F5", "#0072B2"])
    panels = [
        (
            "Validation-selected static",
            static_masks,
            static_stats,
            static_error,
            "#009E73",
        ),
        (
            "Selected robust planner",
            robust_masks,
            robust_stats,
            robust_error,
            "#0072B2",
        ),
    ]
    for row_idx, (title, matrix, stats, _, _) in enumerate(panels, start=1):
        ax = fig.add_subplot(grid[row_idx], sharex=event_ax)
        ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(SENSOR_LABELS)))
        ax.set_yticklabels(SENSOR_LABELS)
        ax.set_xticks([])
        ax.set_title(
            (
                f"{title}: active={stats['active_mean']:.2f}, "
                f"switch={stats['switch_rate']*100:.1f}%, "
                f"unique masks={stats['unique_masks']}"
            )
        )
        ax.set_yticks(np.arange(-0.5, len(SENSOR_LABELS), 1), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=0.9)
        ax.tick_params(which="minor", left=False)
        for change in np.flatnonzero(np.any(np.diff(matrix, axis=0) != 0, axis=1)) + 0.5:
            ax.axvline(change, color="#333333", lw=0.55, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")

    err_ax = fig.add_subplot(grid[3], sharex=event_ax)
    add_event_spans(err_ax, event)
    err_ax.plot(steps, static_error, color="#009E73", lw=1.7, label="Static task error")
    err_ax.plot(steps, robust_error, color="#0072B2", lw=1.7, label="Robust task error")
    err_ax.set_xlim(-0.5, len(steps) - 0.5)
    err_ax.set_xlabel("Step within 64-step final window")
    err_ax.set_ylabel("Mean normalized task error")
    err_ax.grid(axis="y", color="#dddddd", lw=0.7)
    err_ax.legend(loc="upper right", frameon=False, ncol=2)
    err_ax.set_title(
        (
            f"Window objective: static={float(paired_row['static_objective']):.4f}, "
            f"robust={float(paired_row['planner_objective']):.4f}"
        )
    )

    legend_handles = [
        Patch(facecolor="#F7F7F5", edgecolor="#BBBBBB", label="Off"),
        Patch(facecolor="#0072B2", edgecolor="#0072B2", label="On"),
        Patch(facecolor="#E69F00", alpha=0.25, label="Event"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.02),
        frameon=False,
        ncol=3,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    start = int(args.start) if args.start is not None else choose_start(run_dir, args.split)
    stem = args.stem or f"schedule_trace_{run_dir.name}_{args.split}_start{start}"
    path = plot_trace(run_dir, split=args.split, start=start, out_dir=args.out_dir, stem=stem)
    print(path)


if __name__ == "__main__":
    main()
