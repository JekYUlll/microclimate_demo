#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.event_forecaster import (  # noqa: E402
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    train_event_forecaster,
)


DEFAULT_ROOTS = [
    "v1/artifacts/claim_suite_v6_transport_macro_option_riskband_20260604",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether learned event probabilities align with per-step "
            "teacher-improvement margins over the static anchor."
        )
    )
    parser.add_argument("suite_roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--out-dir", default="v1/artifacts/teacher_improvement_alignment_v6_20260604")
    parser.add_argument("--window-grid", nargs="*", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.suite_roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows, summaries, availability = audit_run(
                root=root,
                run_dir=run_dir,
                out_dir=out_dir,
                window_grid=tuple(int(x) for x in args.window_grid),
                device=str(args.device),
            )
            step_rows.extend(rows)
            summary_rows.extend(summaries)
            availability_rows.append(availability)

    if not summary_rows:
        raise FileNotFoundError(f"No auditable runs found under {roots}")

    steps = pd.DataFrame(step_rows).sort_values(["root", "preset", "seed", "window_steps", "row_idx"])
    summary = pd.DataFrame(summary_rows).sort_values(["root", "preset", "seed", "window_steps"])
    availability = pd.DataFrame(availability_rows).sort_values(["root", "preset", "seed"])

    steps.to_csv(out_dir / "teacher_improvement_step_rows.csv", index=False)
    summary.to_csv(out_dir / "teacher_improvement_alignment_summary.csv", index=False)
    availability.to_csv(out_dir / "teacher_improvement_artifact_availability.csv", index=False)

    report = render_report(summary, availability)
    (out_dir / "teacher_improvement_alignment_audit.md").write_text(report, encoding="utf-8")
    print(report)


def audit_run(
    *,
    root: Path,
    run_dir: Path,
    out_dir: Path,
    window_grid: tuple[int, ...],
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preset, seed = parse_preset_seed(run_dir.name, manifest)
    static_path = run_dir / "rollout_validation_selected_static.npz"
    teacher_path = run_dir / "rollout_mpc_teacher.npz"
    if not static_path.exists() or not teacher_path.exists():
        return [], [], {}
    static = load_rollout(static_path)
    teacher = load_rollout(teacher_path)
    truth, prob_cols, availability = load_or_reconstruct_augmented_truth(
        manifest=manifest,
        run_dir=run_dir,
        out_dir=out_dir,
        preset=preset,
        seed=seed,
        device=device,
    )
    task_cols = tuple(str(x) for x in manifest.get("task_error_columns", []))
    raw_scales = manifest.get("task_error_scales")
    task_scales = tuple(float(x) for x in raw_scales) if isinstance(raw_scales, list) else None
    task_weight = safe_float(manifest.get("task_error_weight"))
    event_only = bool(manifest.get("task_error_event_only", True))
    base_rows = build_step_rows(
        root=root,
        preset=preset,
        seed=seed,
        run_dir=run_dir,
        truth=truth,
        prob_cols=prob_cols,
        static=static,
        teacher=teacher,
        task_cols=task_cols,
        task_scales=task_scales,
        task_weight=task_weight,
        event_only=event_only,
    )
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for window_steps in window_grid:
        windowed = apply_window(base_rows, int(window_steps))
        rows.extend(windowed)
        summaries.append(summarize_alignment(root, preset, seed, run_dir, windowed, int(window_steps)))
    availability.update({"root": str(root), "preset": preset, "seed": int(seed), "run_dir": str(run_dir)})
    return rows, summaries, availability


def load_or_reconstruct_augmented_truth(
    *,
    manifest: dict[str, object],
    run_dir: Path,
    out_dir: Path,
    preset: str,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    forecast_summary = manifest.get("learned_event_forecast", {})
    if not isinstance(forecast_summary, dict):
        raise ValueError(f"{run_dir} has no learned_event_forecast manifest block")
    prob_cols = tuple(str(x) for x in forecast_summary.get("probability_columns", []))
    augmented_value = str(forecast_summary.get("augmented_truth_csv", ""))
    augmented_path = Path(augmented_value) if augmented_value else Path("__missing_augmented_truth__")
    if augmented_value and augmented_path.is_file() and prob_cols:
        truth = pd.read_csv(augmented_path)
        if all(col in truth.columns for col in prob_cols):
            return truth, prob_cols, {
                "source": "existing_augmented_truth",
                "augmented_truth_csv": str(augmented_path),
                "probability_columns": "|".join(prob_cols),
            }

    truth_path = Path(str(manifest.get("truth_csv", "")))
    if not truth_path.exists():
        truth_path = ROOT / truth_path
    if not truth_path.exists():
        raise FileNotFoundError(f"truth_csv not found: {manifest.get('truth_csv')}")
    truth = pd.read_csv(truth_path)
    run_args = manifest.get("run_args", {})
    if not isinstance(run_args, dict):
        run_args = {}
    feature_cols = tuple(str(x) for x in forecast_summary.get("feature_columns", []))
    if not feature_cols:
        feature_cols = tuple(str(x) for x in manifest.get("state_columns", []))
    feature_cols = tuple(col for col in feature_cols if col in truth.columns)
    if not feature_cols:
        raise ValueError(f"No event-forecast feature columns available for {run_dir}")
    bounds_raw = forecast_summary.get("train_bounds")
    if isinstance(bounds_raw, list) and len(bounds_raw) == 2:
        train_bounds = (int(bounds_raw[0]), int(bounds_raw[1]))
    else:
        bounds = manifest.get("bounds", {})
        if not isinstance(bounds, dict):
            raise ValueError(f"No train bounds available for {run_dir}")
        pre = bounds.get("oracle_pretrain", [0, 0])
        train = bounds.get("rl_train", [0, 0])
        train_bounds = (int(pre[0]), int(train[1]))
    horizon = len(prob_cols) if prob_cols else int(run_args.get("horizon", 8))
    prefix = str(run_args.get("event_forecast_probability_prefix", "learned_event_p"))
    cfg = EventForecasterTrainingConfig(
        horizon=int(horizon),
        lookback=int(run_args.get("event_forecast_lookback", 8)),
        event_column=str(run_args.get("event_column", "event_flag")),
        hidden_dim=int(run_args.get("event_forecast_hidden_dim", 128)),
        epochs=int(run_args.get("event_forecast_epochs", 40)),
        batch_size=int(run_args.get("event_forecast_batch_size", 256)),
        learning_rate=float(run_args.get("event_forecast_learning_rate", 1.0e-3)),
        weight_decay=float(run_args.get("event_forecast_weight_decay", 1.0e-4)),
        seed=int(manifest.get("seed", seed)),
        device=str(device),
        probability_prefix=prefix,
        period_steps=max(1, int(round(86400.0 / max(float(run_args.get("freq_s", 10800)), 1.0)))),
    )
    dataset = build_event_forecast_dataset(
        truth,
        bounds=train_bounds,
        feature_columns=feature_cols,
        event_column=str(cfg.event_column),
        cfg=cfg,
    )
    bundle = train_event_forecaster(dataset, cfg)
    augmented, learned_cols = augment_truth_with_event_forecasts(truth, bundle)
    target = out_dir / f"reconstructed_truth_{preset}_seed{seed}.csv"
    augmented.to_csv(target, index=False)
    return augmented, tuple(str(x) for x in learned_cols), {
        "source": "reconstructed_event_forecaster",
        "augmented_truth_csv": str(target),
        "probability_columns": "|".join(str(x) for x in learned_cols),
        "final_brier": float(bundle.history["brier"][-1]) if bundle.history.get("brier") else float("nan"),
        "final_loss": float(bundle.history["loss"][-1]) if bundle.history.get("loss") else float("nan"),
    }


def build_step_rows(
    *,
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    truth: pd.DataFrame,
    prob_cols: tuple[str, ...],
    static: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    task_cols: tuple[str, ...],
    task_scales: tuple[float, ...] | None,
    task_weight: float,
    event_only: bool,
) -> list[dict[str, object]]:
    static_steps = np.asarray(static["step_indices"], dtype=int)
    teacher_steps = np.asarray(teacher["step_indices"], dtype=int)
    n = min(len(static_steps), len(teacher_steps))
    rows: list[dict[str, object]] = []
    for row_idx in range(n):
        if int(static_steps[row_idx]) != int(teacher_steps[row_idx]):
            continue
        step_idx = int(static_steps[row_idx])
        if step_idx < 0 or step_idx >= len(truth):
            continue
        prob = truth.loc[step_idx, list(prob_cols)].astype(float).to_numpy(dtype=float) if prob_cols else np.asarray([])
        static_oracle = safe_float(np.asarray(static["oracle_losses"], dtype=float)[row_idx])
        teacher_oracle = safe_float(np.asarray(teacher["oracle_losses"], dtype=float)[row_idx])
        static_task = step_task_error(static, row_idx, task_cols=task_cols, task_scales=task_scales)
        teacher_task = step_task_error(teacher, row_idx, task_cols=task_cols, task_scales=task_scales)
        is_event = bool(np.asarray(static.get("event_flags", []), dtype=bool)[row_idx])
        task_component_margin = (
            float(task_weight) * (static_task - teacher_task)
            if (not event_only or is_event) and np.isfinite(static_task) and np.isfinite(teacher_task)
            else 0.0
        )
        oracle_margin = static_oracle - teacher_oracle
        margin = oracle_margin + task_component_margin
        rows.append(
            {
                "root": str(root),
                "preset": preset,
                "seed": int(seed),
                "run_dir": str(run_dir),
                "row_idx": int(row_idx),
                "step_idx": int(step_idx),
                "window_steps": 1,
                "event_prob_mean": finite_mean(prob),
                "event_prob_max": finite_max(prob),
                "event_prob_first": safe_float(prob[0]) if prob.size else float("nan"),
                "true_event": bool(is_event),
                "oracle_margin": float(oracle_margin),
                "task_component_margin": float(task_component_margin),
                "objective_margin": float(margin),
                "teacher_improves": bool(margin > 0.0),
            }
        )
    return rows


def apply_window(rows: list[dict[str, object]], window_steps: int) -> list[dict[str, object]]:
    if window_steps <= 1:
        return [dict(row) for row in rows]
    out: list[dict[str, object]] = []
    for start in range(0, len(rows), int(window_steps)):
        chunk = rows[start : start + int(window_steps)]
        if not chunk:
            continue
        merged = dict(chunk[0])
        merged["row_idx"] = int(chunk[0]["row_idx"])
        merged["step_idx"] = int(chunk[0]["step_idx"])
        merged["window_steps"] = int(window_steps)
        for key in [
            "event_prob_mean",
            "event_prob_max",
            "event_prob_first",
            "oracle_margin",
            "task_component_margin",
            "objective_margin",
        ]:
            values = [safe_float(row.get(key)) for row in chunk]
            merged[key] = finite_mean(values)
        merged["true_event"] = bool(np.mean([float(bool(row["true_event"])) for row in chunk]) >= 0.5)
        merged["teacher_improves"] = bool(safe_float(merged["objective_margin"]) > 0.0)
        out.append(merged)
    return out


def step_task_error(
    rollout: dict[str, np.ndarray],
    row_idx: int,
    *,
    task_cols: tuple[str, ...],
    task_scales: tuple[float, ...] | None,
) -> float:
    if not task_cols:
        return float("nan")
    state_columns = tuple(str(x) for x in np.asarray(rollout.get("state_columns", []), dtype=str).tolist())
    index = {name: idx for idx, name in enumerate(state_columns)}
    positions = [index[name] for name in task_cols if name in index]
    if not positions:
        return float("nan")
    if task_scales is None or len(task_scales) != len(task_cols):
        scales = np.ones(len(positions), dtype=float)
    else:
        scales = np.asarray([task_scales[idx] for idx, name in enumerate(task_cols) if name in index], dtype=float)
    obs = np.asarray(rollout["observations"], dtype=float)[int(row_idx), positions]
    truth = np.asarray(rollout["truth"], dtype=float)[int(row_idx), positions]
    return float(np.mean(np.abs(obs - truth) / np.maximum(scales, 1.0e-12)))


def summarize_alignment(
    root: Path,
    preset: str,
    seed: int,
    run_dir: Path,
    rows: list[dict[str, object]],
    window_steps: int,
) -> dict[str, object]:
    margins = np.asarray([safe_float(row["objective_margin"]) for row in rows], dtype=float)
    scores = np.asarray([safe_float(row["event_prob_mean"]) for row in rows], dtype=float)
    labels = margins > 0.0
    valid = np.isfinite(scores) & np.isfinite(margins)
    scores = scores[valid]
    margins = margins[valid]
    labels = labels[valid]
    pos_scores = scores[labels]
    neg_scores = scores[~labels]
    u, auc, p = mann_whitney_auc(pos_scores, neg_scores)
    return {
        "root": str(root),
        "preset": preset,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "window_steps": int(window_steps),
        "n": int(scores.size),
        "n_positive": int(pos_scores.size),
        "n_negative": int(neg_scores.size),
        "positive_rate": float(np.mean(labels)) if labels.size else float("nan"),
        "event_prob_positive_mean": finite_mean(pos_scores),
        "event_prob_negative_mean": finite_mean(neg_scores),
        "event_prob_gap": finite_mean(pos_scores) - finite_mean(neg_scores),
        "mannwhitney_u": float(u),
        "auc_positive_over_negative": float(auc),
        "mannwhitney_p_approx": float(p),
        "spearman_prob_vs_margin": spearman(scores, margins),
        "pearson_prob_vs_margin": pearson(scores, margins),
        "margin_mean": finite_mean(margins),
        "margin_q25": finite_quantile(margins, 0.25),
        "margin_min": finite_min(margins),
    }


def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n1, n0 = int(pos.size), int(neg.size)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), float("nan")
    values = np.concatenate([pos, neg])
    ranks = rankdata(values)
    rank_sum_pos = float(np.sum(ranks[:n1]))
    u = rank_sum_pos - n1 * (n1 + 1) / 2.0
    auc = u / float(n1 * n0)
    mu = n1 * n0 / 2.0
    sigma = sqrt(max(n1 * n0 * (n1 + n0 + 1) / 12.0, 1.0e-12))
    z = (u - mu) / sigma
    p = erfc(abs(z) / sqrt(2.0))
    return float(u), float(auc), float(p)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return float("nan")
    return pearson(rankdata(x[valid]), rankdata(y[valid]))


def render_report(summary: pd.DataFrame, availability: pd.DataFrame) -> str:
    lines = ["# Teacher-Improvement Alignment Audit", ""]
    lines.append("## Summary")
    lines.append("")
    display = summary[
        [
            "preset",
            "seed",
            "window_steps",
            "n",
            "n_positive",
            "n_negative",
            "event_prob_gap",
            "auc_positive_over_negative",
            "mannwhitney_p_approx",
            "spearman_prob_vs_margin",
            "margin_mean",
        ]
    ]
    lines.append(markdown_table(display))
    lines.append("")
    lines.append("## Artifact Availability")
    lines.append("")
    lines.append(markdown_table(availability))
    lines.append("")
    lines.append("## Decision Signal")
    lines.append("")
    best = summary.loc[summary["window_steps"].eq(1)].copy()
    if best.empty:
        lines.append("- No step-level rows were available.")
    else:
        finite_auc = best["auc_positive_over_negative"].replace([np.inf, -np.inf], np.nan).dropna()
        finite_gap = best["event_prob_gap"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite_auc.empty and float(finite_auc.mean()) >= 0.60 and int((finite_auc >= 0.60).sum()) >= 2:
            lines.append("- Learned-event probabilities align with teacher-improvement labels strongly enough to try Branch F.")
        elif not finite_gap.empty and float(finite_gap.mean()) > 0.0:
            lines.append("- Learned-event probabilities have weak positive alignment; Branch F is possible but should use a guarded smoke only.")
        else:
            lines.append("- Learned-event probabilities do not align with teacher-improvement labels; prioritize transport-aware features.")
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


def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def parse_preset_seed(name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    return name, safe_int(manifest.get("seed"))


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
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


if __name__ == "__main__":
    main()
