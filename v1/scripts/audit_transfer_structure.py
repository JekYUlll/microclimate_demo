#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether v1 validation-to-final failures are protocol/regime "
            "shift or dynamic-policy instability."
        )
    )
    parser.add_argument("suite_roots", nargs="+", help="One or more claim-suite roots.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--focus-seeds", nargs="*", type=int, default=[41, 44, 52, 53, 55])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.suite_roots]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif len(roots) == 1:
        out_dir = roots[0] / "aggregate"
    else:
        raise SystemExit("--out-dir is required when auditing multiple suite roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows, static_rows, focus_rows = collect_rows(roots, focus_seeds=set(args.focus_seeds))
    if not candidate_rows and not static_rows:
        raise FileNotFoundError(f"No auditable rows found under {roots}")

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["root", "preset", "seed", "candidate_policy"] if candidate_rows else []
    )
    static = pd.DataFrame(static_rows).sort_values(["root", "preset", "seed"] if static_rows else [])
    focus = pd.DataFrame(focus_rows).sort_values(["root", "preset", "seed", "policy"] if focus_rows else [])

    candidates.to_csv(out_dir / "transfer_candidate_rows.csv", index=False)
    static.to_csv(out_dir / "transfer_static_rows.csv", index=False)
    focus.to_csv(out_dir / "seed_focus_rollout_summary.csv", index=False)

    correlations = build_correlations(static, candidates)
    correlations.to_csv(out_dir / "transfer_structure_correlations.csv", index=False)

    report = render_report(static, candidates, correlations, focus)
    (out_dir / "transfer_structure_audit.md").write_text(report, encoding="utf-8")
    print(report)


def collect_rows(
    roots: list[Path],
    *,
    focus_seeds: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    static_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    seen_static: set[str] = set()
    seen_candidates: set[tuple[str, str]] = set()

    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            gate_path = run_dir / "gate_summary.json"
            metrics_path = run_dir / "metrics_final.csv"
            if not gate_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            preset, seed = parse_preset_seed(run_dir.name, manifest)
            static_key = str(run_dir.resolve())
            final_metrics = read_final_metrics(metrics_path)
            static_final = safe_float(gate.get("validation_selected_static_objective"))
            if not np.isfinite(static_final):
                static_final = safe_float(final_metrics.get("validation_selected_static", {}).get("objective_loss_mean"))
            selected_static = manifest.get("selected_static", {})
            if not isinstance(selected_static, dict):
                selected_static = {}
            selected_policy = str(gate.get("best_deployable_policy") or "")
            teacher_obj = safe_float(gate.get("teacher_reference_objective"))
            if static_key not in seen_static:
                seen_static.add(static_key)
                static_rows.append(
                    {
                        "root": str(root),
                        "preset": preset,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "static_action_idx": safe_int(selected_static.get("action_idx")),
                        "static_sensor_ids": str(selected_static.get("sensor_ids", "")),
                        "static_validation_objective": safe_float(selected_static.get("validation_objective")),
                        "static_final_objective": static_final,
                        "teacher_margin": static_final - teacher_obj,
                        "teacher_win": bool(gate.get("teacher_beats_static", False)),
                        **event_diagnostics(manifest),
                    }
                )

            for row in validation_rows(manifest):
                policy = str(row.get("policy", ""))
                final_row = final_metrics.get(policy, {})
                candidate_final = safe_float(final_row.get("objective_loss_mean"))
                if not np.isfinite(candidate_final):
                    continue
                key = (static_key, policy)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                validation_margin = safe_float(row.get("objective_margin_mean"))
                final_margin = static_final - candidate_final
                candidate_rows.append(
                    {
                        "root": str(root),
                        "preset": preset,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "candidate_policy": policy,
                        "is_selected": policy == selected_policy,
                        "static_action_idx": safe_int(selected_static.get("action_idx")),
                        "static_sensor_ids": str(selected_static.get("sensor_ids", "")),
                        "candidate_validation_objective": safe_float(row.get("objective")),
                        "candidate_final_objective": candidate_final,
                        "validation_margin_mean": validation_margin,
                        "validation_margin_median": safe_float(row.get("objective_margin_median")),
                        "validation_margin_q25": safe_float(row.get("objective_margin_q25")),
                        "validation_margin_min": safe_float(row.get("objective_margin_min")),
                        "validation_negative_starts": safe_int(row.get("negative_start_count")),
                        "validation_guard_pass": bool(row.get("static_margin_guard_pass", False)),
                        "validation_positive_center": bool(row.get("static_margin_positive_center", False)),
                        "final_margin": final_margin,
                        "final_win": bool(final_margin > 0.0),
                        "transfer_gap": final_margin - validation_margin,
                        "final_power_mean": safe_float(final_row.get("power_mean")),
                        "final_warmup_abort_count": safe_int(final_row.get("warmup_abort_count")),
                        **event_diagnostics(manifest),
                        **candidate_policy_fields(row, manifest),
                    }
                )

            if seed in focus_seeds:
                focus_rows.extend(rollout_summaries(root, preset, seed, run_dir))

    return candidate_rows, static_rows, focus_rows


def validation_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    selection = manifest.get("deployable_selection", {})
    if not isinstance(selection, dict):
        return []
    rows = selection.get("validation_rows", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def read_final_metrics(metrics_path: Path) -> dict[str, dict[str, object]]:
    if not metrics_path.exists():
        return {}
    try:
        table = pd.read_csv(metrics_path)
    except Exception:
        return {}
    if "policy" not in table.columns:
        return {}
    return {str(row.get("policy")): row for row in table.to_dict(orient="records")}


def event_diagnostics(manifest: dict[str, object]) -> dict[str, float]:
    train = start_diag(manifest, "train")
    validation = start_diag(manifest, "validation")
    final = start_diag(manifest, "final_test")
    validation_mean = safe_float(validation.get("mean"))
    final_mean = safe_float(final.get("mean"))
    return {
        "train_event_mean": safe_float(train.get("mean")),
        "validation_event_mean": validation_mean,
        "validation_event_min": safe_float(validation.get("min")),
        "validation_event_max": safe_float(validation.get("max")),
        "validation_event_std": safe_float(validation.get("std")),
        "final_event_mean": final_mean,
        "final_event_min": safe_float(final.get("min")),
        "final_event_max": safe_float(final.get("max")),
        "final_event_std": safe_float(final.get("std")),
        "final_minus_validation_event": final_mean - validation_mean,
    }


def candidate_policy_fields(row: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    policy = str(row.get("policy", ""))
    calibration = {}
    if "event_threshold" in policy:
        calibration = calibration_row(manifest, "event_threshold_policy")
    elif "contextual_duty" in policy:
        calibration = calibration_row(manifest, "contextual_duty_policy")
    fields: dict[str, object] = {
        "candidate_action_idx": first_valid_int(row.get("action_idx"), calibration.get("action_idx")),
        "candidate_threshold": first_valid_float(row.get("threshold"), calibration.get("threshold")),
        "candidate_aggregation": first_nonempty(row.get("aggregation"), calibration.get("aggregation")),
    }
    if "contextual_duty" in policy:
        block = manifest.get("contextual_duty_policy", {})
        if isinstance(block, dict):
            fields.update(
                {
                    "contextual_blend": safe_float(block.get("blend")),
                    "contextual_deficit_weight": safe_float(block.get("deficit_weight")),
                    "contextual_freshness_weight": safe_float(block.get("freshness_weight")),
                    "contextual_power_weight": safe_float(block.get("power_weight")),
                }
            )
    return fields


def calibration_row(manifest: dict[str, object], key: str) -> dict[str, object]:
    block = manifest.get(key, {})
    if not isinstance(block, dict):
        return {}
    row = block.get("calibration_row", {})
    return row if isinstance(row, dict) else {}


def start_diag(manifest: dict[str, object], split: str) -> dict[str, float]:
    starts = manifest.get("starts", {})
    if not isinstance(starts, dict):
        return {}
    block = starts.get(split, {})
    if not isinstance(block, dict):
        return {}
    diag = block.get("diagnostics", {})
    if not isinstance(diag, dict):
        return {}
    rates = diag.get("selected_event_rates", [])
    if isinstance(rates, list) and rates:
        arr = np.asarray([safe_float(value) for value in rates], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
            }
    mean = safe_float(diag.get("selected_event_rate_mean"))
    return {"mean": mean, "min": mean, "max": mean, "std": 0.0}


def rollout_summaries(root: Path, preset: str, seed: int, run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("rollout_*.npz")):
        policy = path.stem.replace("rollout_", "")
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue
        masks = np.asarray(data.get("selected_masks", []), dtype=int)
        if masks.size == 0:
            continue
        powers = np.asarray(data.get("powers", []), dtype=float)
        event_flags = np.asarray(data.get("event_flags", []), dtype=float)
        soc = np.asarray(data.get("soc", []), dtype=float)
        rows.append(
            {
                "root": str(root),
                "preset": preset,
                "seed": seed,
                "run_dir": str(run_dir),
                "policy": policy,
                "steps": int(masks.shape[0]),
                "event_rate": finite_mean(event_flags),
                "power_mean": finite_mean(powers),
                "power_max": finite_max(powers),
                "soc_mean": finite_mean(soc),
                "soc_min": finite_min(soc),
                "soc_q10": finite_quantile(soc, 0.10),
                "soc_final": finite_last(soc),
                "top_masks": top_mask_summary(masks, data),
            }
        )
    return rows


def top_mask_summary(masks: np.ndarray, data: np.lib.npyio.NpzFile) -> str:
    sensor_ids = [str(value) for value in data.get("sensor_ids", [])]
    unique, counts = np.unique(masks, axis=0, return_counts=True)
    order = np.argsort(-counts)[:5]
    parts: list[str] = []
    for idx in order:
        mask = unique[idx].astype(int)
        if sensor_ids and len(sensor_ids) == len(mask):
            names = [sensor_ids[pos] for pos, active in enumerate(mask) if int(active)]
            label = "+".join(names) if names else "none"
        else:
            label = "".join(str(int(value)) for value in mask)
        parts.append(f"{int(counts[idx])}:{label}")
    return "; ".join(parts)


def build_correlations(static: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unique_static = prefer_finite(
        static,
        keys=["seed", "static_action_idx", "static_validation_objective", "static_final_objective"],
        score_cols=["teacher_margin"],
    )
    unique_candidates = prefer_finite(
        candidates,
        keys=["seed", "candidate_policy", "static_action_idx", "candidate_final_objective", "final_margin"],
        score_cols=[
            "validation_margin_mean",
            "validation_margin_q25",
            "validation_negative_starts",
            "candidate_action_idx",
        ],
    )
    add_corr(
        rows,
        static,
        label="static_validation_vs_final_objective",
        x="static_validation_objective",
        y="static_final_objective",
        subset="all_static_rows",
    )
    add_corr(
        rows,
        unique_static,
        label="static_validation_vs_final_objective",
        x="static_validation_objective",
        y="static_final_objective",
        subset="unique_static_rows",
    )
    if not candidates.empty:
        add_corr(
            rows,
            candidates,
            label="candidate_validation_vs_final_objective",
            x="candidate_validation_objective",
            y="candidate_final_objective",
            subset="all_candidate_rows",
        )
        add_corr(
            rows,
            unique_candidates,
            label="candidate_validation_vs_final_objective",
            x="candidate_validation_objective",
            y="candidate_final_objective",
            subset="unique_candidate_rows",
        )
        add_corr(
            rows,
            candidates,
            label="validation_margin_vs_final_margin",
            x="validation_margin_mean",
            y="final_margin",
            subset="all_candidate_rows",
        )
        add_corr(
            rows,
            unique_candidates,
            label="validation_margin_vs_final_margin",
            x="validation_margin_mean",
            y="final_margin",
            subset="unique_candidate_rows",
        )
        add_corr(
            rows,
            candidates.loc[candidates["is_selected"].astype(bool)],
            label="selected_validation_margin_vs_final_margin",
            x="validation_margin_mean",
            y="final_margin",
            subset="selected_candidate_rows",
        )
        selected_unique = unique_candidates.loc[unique_candidates["is_selected"].astype(bool)]
        add_corr(
            rows,
            selected_unique,
            label="selected_validation_margin_vs_final_margin",
            x="validation_margin_mean",
            y="final_margin",
            subset="unique_selected_candidate_rows",
        )
        for column in [
            "validation_margin_q25",
            "validation_negative_starts",
            "validation_event_mean",
            "final_minus_validation_event",
            "final_event_mean",
            "final_power_mean",
        ]:
            add_corr(
                rows,
                candidates,
                label=f"{column}_vs_final_margin",
                x=column,
                y="final_margin",
                subset="all_candidate_rows",
            )
            add_corr(
                rows,
                unique_candidates,
                label=f"{column}_vs_final_margin",
                x=column,
                y="final_margin",
                subset="unique_candidate_rows",
            )
    return pd.DataFrame(rows)


def prefer_finite(frame: pd.DataFrame, *, keys: list[str], score_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    existing_keys = [key for key in keys if key in frame.columns]
    if not existing_keys:
        return frame.copy()
    scored = frame.copy()
    score = np.zeros(len(scored), dtype=int)
    for col in score_cols:
        if col in scored.columns:
            score += np.isfinite(pd.to_numeric(scored[col], errors="coerce").to_numpy(dtype=float)).astype(int)
    scored["_finite_score"] = score
    scored = scored.sort_values(existing_keys + ["_finite_score"], ascending=[True] * len(existing_keys) + [False])
    return scored.drop_duplicates(existing_keys, keep="first").drop(columns=["_finite_score"])


def add_corr(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    label: str,
    x: str,
    y: str,
    subset: str,
) -> None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return
    data = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(data))
    pearson = float(data[x].corr(data[y], method="pearson")) if n >= 2 else float("nan")
    spearman = float(data[x].corr(data[y], method="spearman")) if n >= 2 else float("nan")
    rows.append(
        {
            "subset": subset,
            "relationship": label,
            "x": x,
            "y": y,
            "n": n,
            "pearson": pearson,
            "spearman": spearman,
        }
    )


def render_report(
    static: pd.DataFrame,
    candidates: pd.DataFrame,
    correlations: pd.DataFrame,
    focus: pd.DataFrame,
) -> str:
    lines = ["# Transfer Structure Audit", ""]
    lines.append("## Correlations")
    lines.append("")
    lines.append(markdown_table(correlations))
    lines.append("")
    lines.append("## Candidate Rows")
    lines.append("")
    if candidates.empty:
        lines.append("No candidate rows with final metrics.")
    else:
        display_cols = [
            "root",
            "preset",
            "seed",
            "candidate_policy",
            "is_selected",
            "static_action_idx",
            "candidate_action_idx",
            "validation_event_mean",
            "final_event_mean",
            "final_minus_validation_event",
            "validation_margin_mean",
            "validation_margin_q25",
            "validation_negative_starts",
            "final_margin",
            "final_win",
        ]
        lines.append(markdown_table(candidates[display_cols]))
    lines.append("")
    lines.append("## Static Rows")
    lines.append("")
    static_cols = [
        "root",
        "preset",
        "seed",
        "static_action_idx",
        "validation_event_mean",
        "final_event_mean",
        "static_validation_objective",
        "static_final_objective",
        "teacher_margin",
        "teacher_win",
    ]
    lines.append(markdown_table(static[static_cols] if not static.empty else static))
    lines.append("")
    lines.append("## Focus Rollout Summaries")
    lines.append("")
    if focus.empty:
        lines.append("No focus rollouts found.")
    else:
        focus_cols = [
            "root",
            "preset",
            "seed",
            "policy",
            "event_rate",
            "power_mean",
            "soc_mean",
            "soc_min",
            "soc_q10",
            "soc_final",
            "top_masks",
        ]
        lines.append(markdown_table(focus[focus_cols]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Use `static_validation_vs_final_objective` as the protocol-shift check: "
        "low static correlation means validation and final windows differ even for "
        "a fixed static anchor."
    )
    lines.append(
        "- Use `validation_margin_vs_final_margin` and `selected_validation_margin_vs_final_margin` "
        "as the dynamic-policy transfer check: weak or negative correlation means "
        "aggregate validation margins cannot select deployable dynamic policies safely."
    )
    lines.append(
        "- Focus rollouts show whether failure seeds are event-sparse or whether the "
        "action/SOC pattern itself is incompatible with the final regime."
    )
    return "\n".join(lines)


def parse_preset_seed(name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    return name, safe_int(manifest.get("seed"))


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


def first_valid_int(*values: object) -> int:
    for value in values:
        out = safe_int(value)
        if out >= 0:
            return out
    return -1


def first_valid_float(*values: object) -> float:
    for value in values:
        out = safe_float(value)
        if np.isfinite(out):
            return out
    return float("nan")


def first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "")
        if text:
            return text
    return ""


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_min(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def finite_last(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr[-1]) if arr.size else float("nan")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    body = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in body))
        for idx in range(len(headers))
    ]
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
        if np.isnan(out):
            return "nan"
        return f"{out:.6g}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


if __name__ == "__main__":
    main()
