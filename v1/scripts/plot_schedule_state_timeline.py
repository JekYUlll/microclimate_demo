#!/usr/bin/env python3
"""Plot v1 scheduler state timelines from rollout NPZ artifacts.

The figure mirrors the older paper's behaviour diagnostic: event context,
sensor runtime modes, and rolling oracle loss are aligned on one time axis.
For the v1 experiments we compare the validation-selected static anchor, the
deployable student, and the MPC teacher because their switching patterns are
the central current diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


DEFAULT_RUN_DIR = Path(
    "v1/artifacts/claim_suite_v6_transport_option_planner_smoke_20260604/"
    "learned_hybrid_option_planner_posguard_safe_seed44"
)
DEFAULT_OUT_DIR = Path("v1/artifacts/schedule_state_figures_20260604")
DEFAULT_SENSOR_CFG = Path("v1/configs/sensors/windblown_sensors_physical_event_v6_complex_static_break.yaml")

POLICY_SPECS = [
    ("validation_selected_static", "Validation-selected static", "#4daf4a"),
    ("forecast_aware_option_planner", "Deployable option student", "#1f78b4"),
    ("mpc_teacher", "Privileged MPC teacher", "#984ea3"),
]


def load_rollout(run_dir: Path, policy: str) -> dict[str, np.ndarray]:
    path = run_dir / f"rollout_{policy}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def nice_sensor_label(sensor_id: str) -> str:
    mapping = {
        "met_station_core": "weather core",
        "radiometer_basic": "radiometer",
        "surface_temp_ir": "surface IR",
        "ultrasonic_anemometer_hd": "hi-res wind",
        "shielded_thermo_hygro": "thermo-hygro",
        "snow_particle_counter": "particle counter",
        "laser_disdrometer": "laser disdrometer",
        "fc4_flux": "FC4 flux",
    }
    return mapping.get(sensor_id, sensor_id.replace("_", " "))


def load_warmup_steps(sensor_cfg: Path | None, sensor_ids: list[str]) -> dict[str, int]:
    if sensor_cfg is None:
        return {sensor_id: 0 for sensor_id in sensor_ids}
    path = Path(sensor_cfg)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    raw = {
        str(item.get("sensor_id")): int(item.get("warmup_steps", 0))
        for item in cfg.get("sensors", [])
        if item.get("sensor_id") is not None
    }
    return {sensor_id: max(int(raw.get(sensor_id, 0)), 0) for sensor_id in sensor_ids}


def reconstruct_execution_modes(
    *,
    selected_masks: np.ndarray,
    sensor_ids: list[str],
    warmup_steps: dict[str, int],
    step_indices: np.ndarray,
) -> np.ndarray:
    """Reconstruct per-step execution modes before observation is produced.

    Rollout artifacts store ``mode_ids_after_step``. That is correct for
    environment state recovery, but it makes one-step warmups visually disappear:
    a sensor can be WARMING during the observation part of a step and become
    ACTIVE at the end of that same step. For behaviour figures we want the
    execution semantics of the displayed step, so the first ``warmup_steps``
    selected steps of each powered run are shown as WARMING.
    """

    selected = np.asarray(selected_masks, dtype=bool)
    modes = np.zeros(selected.shape, dtype=int)
    steps = np.asarray(step_indices, dtype=int).reshape(-1)
    if selected.shape[0] != steps.shape[0]:
        raise ValueError("selected_masks and step_indices length mismatch")
    for col, sensor_id in enumerate(sensor_ids):
        warm = max(int(warmup_steps.get(sensor_id, 0)), 0)
        run_age = 0
        for row in range(selected.shape[0]):
            if row == 0 or int(steps[row]) != int(steps[row - 1]) + 1:
                run_age = 0
            if not selected[row, col]:
                modes[row, col] = 0
                run_age = 0
                continue
            modes[row, col] = 1 if run_age < warm else 2
            run_age += 1
    return modes


def choose_start(rollout: dict[str, np.ndarray], *, length: int) -> int:
    event = np.asarray(rollout["event_flags"], dtype=int)
    masks = np.asarray(rollout["selected_masks"], dtype=int)
    if len(event) <= length:
        return 0
    toggles = np.abs(np.diff(masks, axis=0)).sum(axis=1)
    best: tuple[float, int] | None = None
    for start in range(0, len(event) - length + 1, 16):
        e = event[start : start + length]
        t = toggles[start : start + length - 1]
        event_balance = min(float(e.mean()), 1.0 - float(e.mean()))
        score = 2.0 * float(np.abs(np.diff(e)).sum()) + 0.05 * float(t.sum()) + 10.0 * event_balance
        if best is None or score > best[0]:
            best = (score, start)
    return int(best[1]) if best is not None else 0


def slice_rollout(rollout: dict[str, np.ndarray], start: int, length: int) -> dict[str, np.ndarray]:
    end = min(start + length, len(rollout["event_flags"]))
    sliced: dict[str, np.ndarray] = {}
    for key, value in rollout.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == len(rollout["event_flags"]):
            sliced[key] = arr[start:end]
        else:
            sliced[key] = arr
    return sliced


def per_policy_title(rollout: dict[str, np.ndarray], label: str) -> str:
    masks = np.asarray(rollout["selected_masks"], dtype=int)
    diffs = np.abs(np.diff(masks, axis=0))
    any_switch = float((diffs.sum(axis=1) > 0).mean()) if len(diffs) else 0.0
    ge3 = float((diffs.sum(axis=1) >= 3).mean()) if len(diffs) else 0.0
    mean_active = float(masks.sum(axis=1).mean())
    return f"{label} | active={mean_active:.2f}, switch={any_switch*100:.1f}%, >=3={ge3*100:.1f}%"


def plot_schedule_state_timeline(
    *,
    run_dir: Path,
    out_dir: Path,
    sensor_cfg: Path | None,
    student_policy: str,
    start: int | None,
    length: int,
    rolling_window: int,
    stem: str,
) -> list[Path]:
    policy_specs = [
        POLICY_SPECS[0],
        (student_policy, "Deployable option student", "#1f78b4"),
        POLICY_SPECS[2],
    ]
    raw = {policy: load_rollout(run_dir, policy) for policy, _, _ in policy_specs}
    if start is None:
        start = choose_start(raw["mpc_teacher"], length=length)
    sliced = {policy: slice_rollout(data, start, length) for policy, data in raw.items()}
    length = len(next(iter(sliced.values()))["event_flags"])
    steps = np.arange(length)

    sensor_ids = [str(x) for x in sliced["mpc_teacher"]["sensor_ids"].tolist()]
    sensor_labels = [nice_sensor_label(x) for x in sensor_ids]
    warmup_steps = load_warmup_steps(sensor_cfg, sensor_ids)
    event = np.asarray(sliced["mpc_teacher"]["event_flags"], dtype=float)
    step_indices = np.asarray(sliced["mpc_teacher"]["step_indices"], dtype=int)
    boundaries = np.flatnonzero(np.diff(step_indices) != 1) + 0.5

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig = plt.figure(figsize=(7.4, 8.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[0.34, 1.35, 1.35, 1.35, 1.55],
        hspace=0.22,
        left=0.16,
        right=0.97,
        top=0.96,
        bottom=0.10,
    )

    ax_event = fig.add_subplot(gs[0])
    event_cmap = ListedColormap(["#fbfbfb", "#e66101"])
    ax_event.imshow(event[None, :], aspect="auto", interpolation="nearest", cmap=event_cmap, vmin=0, vmax=1)
    ax_event.set_yticks([])
    ax_event.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_event.set_title("Transport-rich event context")
    for boundary in boundaries:
        ax_event.axvline(boundary, color="#808080", linestyle="--", linewidth=0.7, alpha=0.55)
    for spine in ax_event.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    mode_cmap = ListedColormap(["#f7f7f5", "#9ecae1", "#08519c"])
    mode_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mode_cmap.N)
    mode_axes = []
    for row, (policy, label, _) in enumerate(policy_specs, start=1):
        ax = fig.add_subplot(gs[row], sharex=ax_event)
        mode_axes.append(ax)
        modes = reconstruct_execution_modes(
            selected_masks=np.asarray(sliced[policy]["selected_masks"], dtype=int),
            sensor_ids=sensor_ids,
            warmup_steps=warmup_steps,
            step_indices=np.asarray(sliced[policy]["step_indices"], dtype=int),
        ).T
        ax.imshow(modes, aspect="auto", interpolation="nearest", cmap=mode_cmap, norm=mode_norm)
        ax.set_yticks(np.arange(len(sensor_labels)))
        ax.set_yticklabels(sensor_labels)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.set_title(per_policy_title(sliced[policy], label))
        ax.set_yticks(np.arange(-0.5, len(sensor_labels), 1.0), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=1.0)
        ax.tick_params(axis="y", which="minor", length=0)
        for boundary in boundaries:
            ax.axvline(boundary, color="#808080", linestyle="--", linewidth=0.7, alpha=0.55)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")

    ax_loss = fig.add_subplot(gs[4], sharex=ax_event)
    all_smoothed = []
    mean_labels = []
    for policy, label, color in policy_specs:
        losses = np.asarray(sliced[policy]["oracle_losses"], dtype=float)
        smoothed = rolling_mean(np.minimum(losses, np.nanpercentile(losses, 90)), rolling_window)
        all_smoothed.append(smoothed)
        ax_loss.plot(steps, smoothed, color=color, linewidth=1.5, label=label)
        mean_labels.append((label, float(losses.mean()), color))
    y_values = np.concatenate(all_smoothed)
    lo, hi = np.nanpercentile(y_values, [2, 98])
    pad = max(0.02, 0.08 * (hi - lo))
    ax_loss.set_ylim(max(0.0, lo - pad), hi + pad)
    ax_loss.set_xlim(0, length - 1)
    ax_loss.set_xlabel(f"displayed rollout step (source start={start}, length={length})")
    ax_loss.set_ylabel("rolling clipped oracle loss")
    ax_loss.set_xticks(np.arange(0, length + 1, 128))
    ax_loss.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    for boundary in boundaries:
        ax_loss.axvline(boundary, color="#808080", linestyle="--", linewidth=0.7, alpha=0.55)
    ax_loss.legend(loc="upper left", ncol=1, frameon=False, handlelength=1.8)
    for idx, (label, value, color) in enumerate(sorted(mean_labels, key=lambda item: item[1])):
        ax_loss.text(
            0.985,
            0.96 - idx * 0.14,
            f"{label}: mean {value:.3f}",
            transform=ax_loss.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=8,
        )

    legend_handles = [
        Patch(facecolor="#f7f7f5", edgecolor="#999999", label="OFF"),
        Patch(facecolor="#9ecae1", edgecolor="#999999", label="WARMING"),
        Patch(facecolor="#08519c", edgecolor="#999999", label="ACTIVE"),
        Patch(facecolor="#e66101", edgecolor="#999999", label="event"),
        Patch(facecolor="#ffffff", edgecolor="#808080", linestyle="--", label="window boundary"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.01),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{stem}.png", out_dir / f"{stem}.svg", out_dir / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    fig.savefig(outputs[2])
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sensor-cfg", type=Path, default=DEFAULT_SENSOR_CFG)
    parser.add_argument("--student-policy", default="forecast_aware_option_planner")
    parser.add_argument("--start", type=int, default=512, help="display start; use -1 for auto")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rolling-window", type=int, default=48)
    parser.add_argument(
        "--stem",
        default="v6_transport_seed44_static_student_teacher_state_timeline",
        help="output filename stem without extension",
    )
    args = parser.parse_args()
    start = None if int(args.start) < 0 else int(args.start)
    outputs = plot_schedule_state_timeline(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        sensor_cfg=args.sensor_cfg,
        student_policy=str(args.student_policy),
        start=start,
        length=int(args.length),
        rolling_window=int(args.rolling_window),
        stem=str(args.stem),
    )
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
