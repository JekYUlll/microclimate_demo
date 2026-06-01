#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v1 multi-seed claim-suite outputs.")
    parser.add_argument("suite_roots", nargs="+")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--main-preset", default="main")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_roots = [Path(value) for value in args.suite_roots]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif len(suite_roots) == 1:
        out_dir = suite_roots[0] / "aggregate"
    else:
        raise SystemExit("--out-dir is required when aggregating multiple suite roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows, policy_rows = collect_suite_roots(suite_roots)
    if not run_rows:
        raise FileNotFoundError(f"No completed gate_summary.json files under {suite_roots}")
    runs = pd.DataFrame(run_rows).sort_values(["preset", "seed", "run_dir"])
    policies = pd.DataFrame(policy_rows).sort_values(["preset", "seed", "policy"]) if policy_rows else pd.DataFrame()
    runs.to_csv(out_dir / "claim_runs.csv", index=False)
    if not policies.empty:
        policies.to_csv(out_dir / "claim_policy_metrics.csv", index=False)

    summary_rows = [summarize_group(group) for _, group in runs.groupby("preset", sort=True)]
    summary = pd.DataFrame(summary_rows).sort_values("preset")
    summary.to_csv(out_dir / "claim_summary.csv", index=False)
    assessment = assess_claim(
        runs,
        main_preset=str(args.main_preset),
        min_seeds=int(args.min_seeds),
        min_win_rate=float(args.min_win_rate),
    )
    (out_dir / "claim_assessment.json").write_text(
        json.dumps(assessment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "claim_assessment.md").write_text(render_markdown(summary, assessment), encoding="utf-8")
    print(json.dumps(assessment, indent=2, ensure_ascii=False))


def collect_suite_roots(suite_roots: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    for suite_root in suite_roots:
        root_runs, root_policies = collect_runs(suite_root)
        run_rows.extend(root_runs)
        policy_rows.extend(root_policies)
    return run_rows, policy_rows


def collect_runs(suite_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    for summary_path in sorted(suite_root.glob("*_seed*/gate_summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        preset, seed = parse_preset_seed(run_dir.name, manifest)
        event_support_cycle = manifest.get("event_support_cycle_policy", {})
        if not isinstance(event_support_cycle, dict):
            event_support_cycle = {}
        teacher_rate = manifest.get("teacher_rate_policy", {})
        if not isinstance(teacher_rate, dict):
            teacher_rate = {}
        teacher_cycle = manifest.get("teacher_cycle_policy", {})
        if not isinstance(teacher_cycle, dict):
            teacher_cycle = {}
        rollout_value = manifest.get("rollout_value_policy", {})
        if not isinstance(rollout_value, dict):
            rollout_value = {}
        static_objective = float(summary["validation_selected_static_objective"])
        teacher_objective = nullable_float(summary.get("teacher_reference_objective"))
        deployable_objective = nullable_float(summary.get("best_deployable_objective"))
        run_rows.append(
            {
                "preset": preset,
                "seed": seed,
                "run_dir": str(run_dir),
                "objective_metric": str(summary.get("objective_metric", "")),
                "static_objective": static_objective,
                "teacher_objective": teacher_objective,
                "deployable_objective": deployable_objective,
                "teacher_margin": static_objective - teacher_objective if teacher_objective is not None else np.nan,
                "deployable_margin": static_objective - deployable_objective
                if deployable_objective is not None
                else np.nan,
                "teacher_beats_static": bool(summary.get("teacher_beats_static", False)),
                "gate_pass": bool(summary.get("gate_pass", False)),
                "best_deployable_policy": str(summary.get("best_deployable_policy", "")),
                "event_support_cycle_selection_mode": str(event_support_cycle.get("selection_mode", "")),
                "event_support_cycle_threshold": nullable_float(event_support_cycle.get("threshold")),
                "event_support_cycle_aggregation": str(event_support_cycle.get("aggregation", "")),
                "event_support_cycle_period": event_support_cycle.get("cycle_period", np.nan),
                "event_support_cycle_validation_objective": nullable_float(
                    event_support_cycle.get("validation_objective")
                ),
                "teacher_rate_blend": nullable_float(teacher_rate.get("blend")),
                "teacher_rate_freshness_weight": nullable_float(teacher_rate.get("freshness_weight")),
                "teacher_rate_power_weight": nullable_float(teacher_rate.get("power_weight")),
                "teacher_rate_validation_objective": nullable_float(teacher_rate.get("validation_objective")),
                "teacher_cycle_included": bool(teacher_cycle.get("included", False)),
                "teacher_cycle_max_lookahead": teacher_cycle.get("max_lookahead", np.nan),
                "rollout_value_included": bool(rollout_value.get("included", False)),
                "rollout_value_support_top_k": rollout_value.get("support_top_k", np.nan),
                "rollout_value_depth": rollout_value.get("planning_depth", np.nan),
                "rollout_value_threshold": nullable_float(rollout_value.get("advantage_threshold")),
                "rollout_value_validation_objective": nullable_float(rollout_value.get("validation_objective")),
            }
        )
        metrics_path = run_dir / "metrics_final.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            for row in metrics.to_dict(orient="records"):
                out = {
                    "preset": preset,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "policy": str(row.get("policy", "")),
                }
                for key in (
                    "objective_loss_mean",
                    "oracle_loss_mean",
                    "task_error_mean",
                    "task_error_event_mean",
                    "weighted_normalized_mae",
                    "power_mean",
                    "warmup_abort_count",
                    "warmup_abort_rate",
                    "steady_violation_rate",
                    "peak_violation_rate",
                ):
                    if key in row:
                        out[key] = row[key]
                policy_rows.append(out)
    return run_rows, policy_rows


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    deployable_margin = group["deployable_margin"].astype(float).to_numpy()
    teacher_margin = group["teacher_margin"].astype(float).to_numpy()
    n = int(len(group))
    wins = int(np.sum(deployable_margin > 0.0))
    teacher_wins = int(np.sum(teacher_margin > 0.0))
    return {
        "preset": str(group["preset"].iloc[0]),
        "n": n,
        "deployable_wins": wins,
        "deployable_win_rate": wins / n if n else np.nan,
        "deployable_margin_mean": float(np.nanmean(deployable_margin)),
        "deployable_margin_std": float(np.nanstd(deployable_margin, ddof=1)) if n > 1 else 0.0,
        "deployable_margin_median": float(np.nanmedian(deployable_margin)),
        "teacher_wins": teacher_wins,
        "teacher_win_rate": teacher_wins / n if n else np.nan,
        "teacher_margin_mean": float(np.nanmean(teacher_margin)),
        "sign_test_two_sided_p": exact_sign_test_two_sided(wins, n),
        "all_gate_pass": bool(np.all(group["gate_pass"].astype(bool))),
    }


def assess_claim(
    runs: pd.DataFrame,
    *,
    main_preset: str,
    min_seeds: int,
    min_win_rate: float,
) -> dict[str, object]:
    main = runs.loc[runs["preset"] == main_preset].copy()
    if main.empty:
        return {
            "claim_pass": False,
            "reason": f"No completed runs for main preset {main_preset!r}",
        }
    n = int(len(main))
    margins = main["deployable_margin"].astype(float).to_numpy()
    wins = int(np.sum(margins > 0.0))
    teacher_wins = int(np.sum(main["teacher_margin"].astype(float).to_numpy() > 0.0))
    win_rate = wins / n if n else 0.0
    mean_margin = float(np.nanmean(margins))
    required_wins = int(np.ceil(float(min_win_rate) * float(n))) if n else 0
    pass_reasons = []
    fail_reasons = []
    if n >= int(min_seeds):
        pass_reasons.append(f"n={n} >= {min_seeds}")
    else:
        fail_reasons.append(f"n={n} < {min_seeds}")
    if wins >= required_wins:
        pass_reasons.append(f"deployable wins={wins} >= {required_wins}")
    else:
        fail_reasons.append(f"deployable wins={wins} < {required_wins}")
    if mean_margin > 0.0:
        pass_reasons.append(f"mean deployable margin={mean_margin:.6g} > 0")
    else:
        fail_reasons.append(f"mean deployable margin={mean_margin:.6g} <= 0")
    if teacher_wins >= required_wins:
        pass_reasons.append(f"teacher wins={teacher_wins} >= {required_wins}")
    else:
        fail_reasons.append(f"teacher wins={teacher_wins} < {required_wins}")
    return {
        "claim_pass": not fail_reasons,
        "main_preset": main_preset,
        "n": n,
        "min_seeds": int(min_seeds),
        "min_win_rate": float(min_win_rate),
        "required_wins": required_wins,
        "deployable_wins": wins,
        "deployable_win_rate": win_rate,
        "deployable_margin_mean": mean_margin,
        "deployable_margin_median": float(np.nanmedian(margins)),
        "teacher_wins": teacher_wins,
        "sign_test_two_sided_p": exact_sign_test_two_sided(wins, n),
        "pass_reasons": pass_reasons,
        "fail_reasons": fail_reasons,
    }


def render_markdown(summary: pd.DataFrame, assessment: dict[str, object]) -> str:
    lines = [
        "# v1 Claim Suite Assessment",
        "",
        f"- Claim pass: `{assessment.get('claim_pass')}`",
        f"- Main preset: `{assessment.get('main_preset', '')}`",
        f"- Deployable wins: `{assessment.get('deployable_wins', 0)}/{assessment.get('n', 0)}`",
        f"- Mean deployable margin: `{float(assessment.get('deployable_margin_mean', float('nan'))):.6f}`",
        f"- Sign-test two-sided p: `{float(assessment.get('sign_test_two_sided_p', float('nan'))):.6f}`",
        "",
        "## Reasons",
        "",
    ]
    for item in assessment.get("pass_reasons", []):
        lines.append(f"- PASS: {item}")
    for item in assessment.get("fail_reasons", []):
        lines.append(f"- FAIL: {item}")
    lines.extend(["", "## Preset Summary", ""])
    if not summary.empty:
        display = summary[
            [
                "preset",
                "n",
                "deployable_wins",
                "deployable_margin_mean",
                "teacher_wins",
                "teacher_margin_mean",
                "sign_test_two_sided_p",
            ]
        ]
        lines.append(markdown_table(display))
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(item) for item in df.columns]
    body = [[format_cell(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in body)) if body else len(headers[idx])
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
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def parse_preset_seed(name: str, manifest: dict[str, object]) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if match:
        return str(match.group("preset")), int(match.group("seed"))
    seed = int(manifest.get("seed", -1))
    return name, seed


def nullable_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def exact_sign_test_two_sided(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    k = min(int(wins), int(n) - int(wins))
    tail = sum(comb(int(n), i) for i in range(k + 1)) / float(2**int(n))
    return float(min(1.0, 2.0 * tail))


if __name__ == "__main__":
    main()
