#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_ROOTS = [
    "v1/artifacts/claim_suite_v6_transport_runtime_risk_denseval_20260604",
    "v1/artifacts/claim_suite_v6_transport_cost_knn_riskband_20260604",
    "v1/artifacts/claim_suite_v6_transport_macro_option_riskband_20260604",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose final-test static/teacher objective transfer for v1 "
            "forecast-aware scheduling runs without retraining."
        )
    )
    parser.add_argument("suite_roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--out-dir", default="v1/artifacts/objective_transfer_audit_v6_20260604")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.suite_roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_rows, pair_rows, seed_rows, artifact_rows = collect_rows(roots)
    if not seed_rows:
        raise FileNotFoundError(f"No auditable static/teacher rollout pairs found under {roots}")

    windows = pd.DataFrame(window_rows).sort_values(["root", "preset", "seed", "policy", "window_id"])
    pairs = pd.DataFrame(pair_rows).sort_values(["root", "preset", "seed", "window_id"])
    seeds = pd.DataFrame(seed_rows).sort_values(["root", "preset", "seed"])
    artifacts = pd.DataFrame(artifact_rows).sort_values(["root", "preset", "seed"])

    windows.to_csv(out_dir / "objective_window_rows.csv", index=False)
    pairs.to_csv(out_dir / "objective_pair_rows.csv", index=False)
    seeds.to_csv(out_dir / "objective_seed_summary.csv", index=False)
    artifacts.to_csv(out_dir / "objective_artifact_availability.csv", index=False)

    summary = summarize(seeds)
    summary.to_csv(out_dir / "objective_transfer_summary.csv", index=False)
    event_alignment = summarize_event_alignment(pairs)
    event_alignment.to_csv(out_dir / "objective_event_alignment.csv", index=False)
    report = render_report(summary, seeds, event_alignment, artifacts)
    (out_dir / "objective_transfer_audit.md").write_text(report, encoding="utf-8")
    print(report)


def collect_rows(
    roots: list[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    window_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    seen_seed_keys: set[tuple[str, int]] = set()

    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            preset, seed = parse_preset_seed(run_dir.name, manifest)
            dedupe_key = (str(run_dir.resolve()), int(seed))
            if dedupe_key in seen_seed_keys:
                continue
            seen_seed_keys.add(dedupe_key)

            task_cols = tuple(str(x) for x in manifest.get("task_error_columns", []))
            raw_scales = manifest.get("task_error_scales")
            task_scales = tuple(float(x) for x in raw_scales) if isinstance(raw_scales, list) else None
            task_weight = safe_float(manifest.get("task_error_weight"))
            event_only = bool(manifest.get("task_error_event_only", True))

            rollouts = {}
            for policy in ("validation_selected_static", "mpc_teacher"):
                path = run_dir / f"rollout_{policy}.npz"
                if path.exists():
                    rollouts[policy] = load_rollout(path)
            if "validation_selected_static" not in rollouts or "mpc_teacher" not in rollouts:
                continue

            artifact_rows.append(artifact_availability(root, preset, seed, run_dir, manifest))
            metrics = read_metrics(run_dir / "metrics_final.csv")
            per_policy_windows: dict[str, list[dict[str, object]]] = {}

            for policy, rollout in rollouts.items():
                rows = rollout_window_rows(
                    root=root,
                    preset=preset,
                    seed=seed,
                    run_dir=run_dir,
                    policy=policy,
                    rollout=rollout,
                    task_cols=task_cols,
                    task_scales=task_scales,
                    task_weight=task_weight,
                    event_only=event_only,
                )
                per_policy_windows[policy] = rows
                window_rows.extend(rows)

            static_windows = {int(row["window_id"]): row for row in per_policy_windows["validation_selected_static"]}
            teacher_windows = {int(row["window_id"]): row for row in per_policy_windows["mpc_teacher"]}
            for window_id in sorted(set(static_windows) & set(teacher_windows)):
                static = static_windows[window_id]
                teacher = teacher_windows[window_id]
                pair_rows.append(pair_window_row(root, preset, seed, run_dir, window_id, static, teacher))

            seed_rows.append(seed_summary_row(root, preset, seed, run_dir, metrics, static_windows, teacher_windows))

    return window_rows, pair_rows, seed_rows, artifact_rows


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def read_metrics(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if "policy" not in table.columns:
        return {}
    return {str(row["policy"]): row for row in table.to_dict(orient="records")}


def rollout_window_rows(
    *,
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    policy: str,
    rollout: dict[str, np.ndarray],
    task_cols: tuple[str, ...],
    task_scales: tuple[float, ...] | None,
    task_weight: float,
    event_only: bool,
) -> list[dict[str, object]]:
    step_indices = np.asarray(rollout["step_indices"], dtype=int)
    rows = []
    for window_id, start, end in window_slices(step_indices):
        rows.append(
            {
                "root": str(root),
                "preset": preset,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "policy": policy,
                "window_id": int(window_id),
                "row_start": int(start),
                "row_end": int(end),
                "step_start": int(step_indices[start]) if len(step_indices) else -1,
                "step_end": int(step_indices[end - 1]) if end > start else -1,
                **rollout_metrics(
                    rollout,
                    start=start,
                    end=end,
                    task_cols=task_cols,
                    task_scales=task_scales,
                    task_weight=task_weight,
                    event_only=event_only,
                ),
            }
        )
    return rows


def window_slices(step_indices: np.ndarray) -> list[tuple[int, int, int]]:
    steps = np.asarray(step_indices, dtype=int).reshape(-1)
    if steps.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(steps) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(steps)]])
    return [(idx, int(start), int(end)) for idx, (start, end) in enumerate(zip(starts, ends, strict=True))]


def rollout_metrics(
    rollout: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
    task_cols: tuple[str, ...],
    task_scales: tuple[float, ...] | None,
    task_weight: float,
    event_only: bool,
) -> dict[str, object]:
    oracle = finite_mean(np.asarray(rollout.get("oracle_losses", []), dtype=float)[start:end])
    task = task_error_metrics(
        rollout,
        start=start,
        end=end,
        task_cols=task_cols,
        task_scales=task_scales,
        event_only=event_only,
    )
    task_error = safe_float(task.get("task_error_mean"))
    objective = oracle + float(task_weight) * task_error if np.isfinite(task_error) else oracle
    masks = np.asarray(rollout.get("selected_masks", []), dtype=int)[start:end]
    diffs = np.abs(np.diff(masks.astype(np.int8), axis=0)).sum(axis=1) if masks.shape[0] > 1 else np.asarray([])
    event = np.asarray(rollout.get("event_flags", []), dtype=float)[start:end]
    powers = np.asarray(rollout.get("powers", []), dtype=float)[start:end]
    soc = np.asarray(rollout.get("soc", []), dtype=float)[start:end]
    warmup_deltas = np.asarray(rollout.get("warmup_abort_deltas", []), dtype=float)[start:end]
    return {
        "objective": float(objective),
        "oracle_loss_mean": oracle,
        **task,
        "power_mean": finite_mean(powers),
        "power_max": finite_max(powers),
        "event_rate": finite_mean(event),
        "soc_mean": finite_mean(soc),
        "soc_min": finite_min(soc),
        "active_count_mean": finite_mean(masks.sum(axis=1)) if masks.size else float("nan"),
        "switch_any_rate": finite_mean((diffs > 0).astype(float)) if diffs.size else 0.0,
        "switch_ge2_rate": finite_mean((diffs >= 2).astype(float)) if diffs.size else 0.0,
        "switch_ge3_rate": finite_mean((diffs >= 3).astype(float)) if diffs.size else 0.0,
        "switch_count_mean": finite_mean(diffs.astype(float)) if diffs.size else 0.0,
        "unique_masks": int(np.unique(masks, axis=0).shape[0]) if masks.size else 0,
        "warmup_abort_deltas": int(np.nansum(warmup_deltas)) if warmup_deltas.size else 0,
    }


def task_error_metrics(
    rollout: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
    task_cols: tuple[str, ...],
    task_scales: tuple[float, ...] | None,
    event_only: bool,
) -> dict[str, object]:
    if not task_cols:
        return {
            "task_error_mean": float("nan"),
            "task_error_event_mean": float("nan"),
            "task_error_all_mean": float("nan"),
        }
    state_columns = [str(x) for x in np.asarray(rollout.get("state_columns", []), dtype=str).tolist()]
    index = {name: idx for idx, name in enumerate(state_columns)}
    positions = [index[name] for name in task_cols if name in index]
    if not positions:
        return {
            "task_error_mean": float("nan"),
            "task_error_event_mean": float("nan"),
            "task_error_all_mean": float("nan"),
        }
    if task_scales is None or len(task_scales) != len(task_cols):
        scales = np.ones(len(positions), dtype=float)
    else:
        scales = np.asarray([task_scales[idx] for idx, name in enumerate(task_cols) if name in index], dtype=float)
    scales = np.maximum(scales.reshape(1, -1), 1.0e-12)
    obs = np.asarray(rollout["observations"], dtype=float)[start:end, :][:, positions]
    truth = np.asarray(rollout["truth"], dtype=float)[start:end, :][:, positions]
    err = np.abs(obs - truth) / scales
    row_err = np.mean(err, axis=1) if err.size else np.asarray([], dtype=float)
    event = np.asarray(rollout.get("event_flags", []), dtype=bool)[start:end]
    all_mean = finite_mean(row_err)
    event_mean = finite_mean(row_err[event]) if event.size and np.any(event) else float("nan")
    selected = event_mean if bool(event_only) else all_mean
    out: dict[str, object] = {
        "task_error_mean": float(selected),
        "task_error_event_mean": float(event_mean),
        "task_error_all_mean": float(all_mean),
    }
    for local_idx, name in enumerate(name for name in task_cols if name in index):
        values = err[:, local_idx]
        out[f"task_error_{name}_all_mean"] = finite_mean(values)
        out[f"task_error_{name}_event_mean"] = finite_mean(values[event]) if event.size and np.any(event) else float("nan")
    return out


def pair_window_row(
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    window_id: int,
    static: dict[str, object],
    teacher: dict[str, object],
) -> dict[str, object]:
    oracle_margin = safe_float(static.get("oracle_loss_mean")) - safe_float(teacher.get("oracle_loss_mean"))
    task_margin = safe_float(static.get("task_error_mean")) - safe_float(teacher.get("task_error_mean"))
    objective_margin = safe_float(static.get("objective")) - safe_float(teacher.get("objective"))
    task_component_margin = objective_margin - oracle_margin
    return {
        "root": str(root),
        "preset": preset,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "window_id": int(window_id),
        "step_start": static.get("step_start"),
        "step_end": static.get("step_end"),
        "objective_margin": objective_margin,
        "oracle_margin": oracle_margin,
        "task_error_margin": task_margin,
        "task_component_margin": task_component_margin,
        "teacher_window_win": bool(objective_margin > 0.0),
        "event_rate": static.get("event_rate"),
        "teacher_power_delta": safe_float(teacher.get("power_mean")) - safe_float(static.get("power_mean")),
        "teacher_switch_delta": safe_float(teacher.get("switch_any_rate")) - safe_float(static.get("switch_any_rate")),
        "teacher_active_count_delta": safe_float(teacher.get("active_count_mean")) - safe_float(static.get("active_count_mean")),
        "static_objective": static.get("objective"),
        "teacher_objective": teacher.get("objective"),
        "static_oracle_loss_mean": static.get("oracle_loss_mean"),
        "teacher_oracle_loss_mean": teacher.get("oracle_loss_mean"),
        "static_task_error_mean": static.get("task_error_mean"),
        "teacher_task_error_mean": teacher.get("task_error_mean"),
        "static_power_mean": static.get("power_mean"),
        "teacher_power_mean": teacher.get("power_mean"),
        "static_switch_any_rate": static.get("switch_any_rate"),
        "teacher_switch_any_rate": teacher.get("switch_any_rate"),
    }


def seed_summary_row(
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    metrics: dict[str, dict[str, object]],
    static_windows: dict[int, dict[str, object]],
    teacher_windows: dict[int, dict[str, object]],
) -> dict[str, object]:
    static_final = metrics.get("validation_selected_static", {})
    teacher_final = metrics.get("mpc_teacher", {})
    static_objective = first_finite(
        safe_float(static_final.get("objective_loss_mean")),
        finite_mean([row["objective"] for row in static_windows.values()]),
    )
    teacher_objective = first_finite(
        safe_float(teacher_final.get("objective_loss_mean")),
        finite_mean([row["objective"] for row in teacher_windows.values()]),
    )
    static_oracle = first_finite(
        safe_float(static_final.get("oracle_loss_mean")),
        finite_mean([row["oracle_loss_mean"] for row in static_windows.values()]),
    )
    teacher_oracle = first_finite(
        safe_float(teacher_final.get("oracle_loss_mean")),
        finite_mean([row["oracle_loss_mean"] for row in teacher_windows.values()]),
    )
    static_task = first_finite(
        safe_float(static_final.get("task_error_mean")),
        finite_mean([row["task_error_mean"] for row in static_windows.values()]),
    )
    teacher_task = first_finite(
        safe_float(teacher_final.get("task_error_mean")),
        finite_mean([row["task_error_mean"] for row in teacher_windows.values()]),
    )
    objective_margin = static_objective - teacher_objective
    oracle_margin = static_oracle - teacher_oracle
    task_error_margin = static_task - teacher_task
    task_component_margin = objective_margin - oracle_margin
    pair_ids = sorted(set(static_windows) & set(teacher_windows))
    window_margins = [
        safe_float(static_windows[idx].get("objective")) - safe_float(teacher_windows[idx].get("objective"))
        for idx in pair_ids
    ]
    return {
        "root": str(root),
        "preset": preset,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "n_windows": int(len(pair_ids)),
        "teacher_window_wins": int(np.sum(np.asarray(window_margins, dtype=float) > 0.0)) if window_margins else 0,
        "objective_margin": float(objective_margin),
        "oracle_margin": float(oracle_margin),
        "task_error_margin": float(task_error_margin),
        "task_component_margin": float(task_component_margin),
        "task_component_share": float(task_component_margin / objective_margin)
        if np.isfinite(objective_margin) and abs(float(objective_margin)) > 1.0e-12
        else float("nan"),
        "static_objective": float(static_objective),
        "teacher_objective": float(teacher_objective),
        "static_oracle_loss_mean": float(static_oracle),
        "teacher_oracle_loss_mean": float(teacher_oracle),
        "static_task_error_mean": float(static_task),
        "teacher_task_error_mean": float(teacher_task),
        "static_power_mean": first_finite(safe_float(static_final.get("power_mean")), finite_mean([row["power_mean"] for row in static_windows.values()])),
        "teacher_power_mean": first_finite(safe_float(teacher_final.get("power_mean")), finite_mean([row["power_mean"] for row in teacher_windows.values()])),
        "static_warmup_abort_count": safe_int(static_final.get("warmup_abort_count")),
        "teacher_warmup_abort_count": safe_int(teacher_final.get("warmup_abort_count")),
        "window_margin_min": finite_min(window_margins),
        "window_margin_q25": finite_quantile(window_margins, 0.25),
        "window_margin_mean": finite_mean(window_margins),
    }


def artifact_availability(
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    truth_path = Path(str(manifest.get("truth_csv", "")))
    probability_cols = []
    forecast = manifest.get("learned_event_forecast", {})
    if isinstance(forecast, dict):
        raw = forecast.get("probability_columns", [])
        if isinstance(raw, list):
            probability_cols = [str(x) for x in raw]
    truth_has_probability_cols = False
    if truth_path.exists() and probability_cols:
        try:
            columns = pd.read_csv(truth_path, nrows=0).columns
            truth_has_probability_cols = all(col in columns for col in probability_cols)
        except Exception:
            truth_has_probability_cols = False
    teacher_path = run_dir / "teacher_dataset.npz"
    teacher_has_feature_names = False
    if teacher_path.exists():
        try:
            with np.load(teacher_path, allow_pickle=True) as data:
                teacher_has_feature_names = "feature_names" in data.files
        except Exception:
            teacher_has_feature_names = False
    return {
        "root": str(root),
        "preset": preset,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "truth_csv": str(truth_path),
        "learned_event_forecast_enabled": bool(forecast),
        "learned_event_probability_columns": "|".join(probability_cols),
        "truth_has_learned_event_probability_columns": bool(truth_has_probability_cols),
        "teacher_dataset_has_feature_names": bool(teacher_has_feature_names),
    }


def summarize(seeds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in seeds.groupby(["root", "preset"], sort=True):
        root, preset = key
        objective_margin = group["objective_margin"].to_numpy(dtype=float)
        oracle_margin = group["oracle_margin"].to_numpy(dtype=float)
        task_margin = group["task_error_margin"].to_numpy(dtype=float)
        task_component = group["task_component_margin"].to_numpy(dtype=float)
        power_delta = group["teacher_power_mean"].to_numpy(dtype=float) - group["static_power_mean"].to_numpy(dtype=float)
        objective_mean = finite_mean(objective_margin)
        task_component_mean = finite_mean(task_component)
        rows.append(
            {
                "root": root,
                "preset": preset,
                "n": int(len(group)),
                "teacher_seed_wins": int(np.sum(objective_margin > 0.0)),
                "teacher_window_wins": int(group["teacher_window_wins"].sum()),
                "total_windows": int(group["n_windows"].sum()),
                "objective_margin_mean": objective_mean,
                "oracle_margin_mean": finite_mean(oracle_margin),
                "task_error_margin_mean": finite_mean(task_margin),
                "task_component_margin_mean": task_component_mean,
                "task_component_share_mean": float(task_component_mean / objective_mean)
                if np.isfinite(objective_mean) and abs(float(objective_mean)) > 1.0e-12
                else float("nan"),
                "teacher_power_delta_mean": finite_mean(power_delta),
                "static_warmup_abort_mean": finite_mean(group["static_warmup_abort_count"].to_numpy(dtype=float)),
                "teacher_warmup_abort_mean": finite_mean(group["teacher_warmup_abort_count"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["root", "preset"])


def summarize_event_alignment(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if pairs.empty:
        return pd.DataFrame(rows)
    for key, group in pairs.groupby(["root", "preset"], sort=True):
        root, preset = key
        add_corr(rows, root, preset, group, "event_rate", "objective_margin")
        add_corr(rows, root, preset, group, "event_rate", "oracle_margin")
        add_corr(rows, root, preset, group, "event_rate", "task_error_margin")
        add_corr(rows, root, preset, group, "teacher_power_delta", "objective_margin")
        add_corr(rows, root, preset, group, "teacher_switch_delta", "objective_margin")
    return pd.DataFrame(rows)


def add_corr(
    rows: list[dict[str, object]],
    root: str,
    preset: str,
    group: pd.DataFrame,
    x: str,
    y: str,
) -> None:
    data = group[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(data))
    rows.append(
        {
            "root": root,
            "preset": preset,
            "x": x,
            "y": y,
            "n": n,
            "pearson": float(data[x].corr(data[y], method="pearson")) if n >= 2 else float("nan"),
            "spearman": float(data[x].corr(data[y], method="spearman")) if n >= 2 else float("nan"),
        }
    )


def render_report(
    summary: pd.DataFrame,
    seeds: pd.DataFrame,
    event_alignment: pd.DataFrame,
    artifacts: pd.DataFrame,
) -> str:
    lines = ["# Objective Transfer Audit", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Seed-Level Static vs Teacher Decomposition")
    lines.append("")
    seed_cols = [
        "root",
        "preset",
        "seed",
        "teacher_window_wins",
        "n_windows",
        "objective_margin",
        "oracle_margin",
        "task_error_margin",
        "task_component_margin",
        "task_component_share",
        "static_power_mean",
        "teacher_power_mean",
        "static_warmup_abort_count",
        "teacher_warmup_abort_count",
        "window_margin_min",
    ]
    lines.append(markdown_table(seeds[seed_cols]))
    lines.append("")
    lines.append("## Event/Behavior Alignment")
    lines.append("")
    lines.append(markdown_table(event_alignment))
    lines.append("")
    lines.append("## Artifact Availability")
    lines.append("")
    artifact_cols = [
        "root",
        "preset",
        "seed",
        "learned_event_forecast_enabled",
        "truth_has_learned_event_probability_columns",
        "teacher_dataset_has_feature_names",
    ]
    lines.append(markdown_table(artifacts[artifact_cols]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Positive `objective_margin` means the privileged teacher beats the validation-selected static anchor."
    )
    lines.append(
        "- `oracle_margin` and `task_error_margin` separate the two objective terms; the configured scalar objective is "
        "`oracle_loss + task_error_weight * task_error`, so use the raw task margin together with the manifest weight."
    )
    lines.append(
        "- Missing learned-event probability columns mean current artifacts cannot audit whether the deployable forecast "
        "probabilities are calibrated to teacher-improvement windows. Future runs should save augmented truth columns or "
        "feature names."
    )
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    body = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in body)) for idx in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        out = float(value)
        if not np.isfinite(out):
            return "nan"
        return f"{out:.6g}"
    return str(value)


def parse_preset_seed(name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    return name, safe_int(manifest.get("seed"))


def first_finite(*values: float) -> float:
    for value in values:
        if np.isfinite(float(value)):
            return float(value)
    return float("nan")


def finite_mean(values: object) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_min(values: object) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def finite_max(values: object) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def finite_quantile(values: object, q: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, float(q))) if arr.size else float("nan")


def safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


if __name__ == "__main__":
    main()
