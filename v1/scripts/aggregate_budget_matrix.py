#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v1 budget-matrix claim-suite outputs.")
    parser.add_argument("matrix_root")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--min-seeds-per-budget", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.matrix_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    if not rows:
        raise FileNotFoundError(f"No gate_summary.json files found under {root}")
    df = pd.DataFrame(rows).sort_values(["budget", "preset", "seed"])
    if args.preset:
        df = df[df["preset"] == str(args.preset)].copy()
    if df.empty:
        raise ValueError("No rows left after preset filtering")
    df.to_csv(out_dir / "budget_matrix_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "budget_matrix_summary.csv", index=False)
    assessment = assess(summary, min_seeds=int(args.min_seeds_per_budget), min_win_rate=float(args.min_win_rate))
    (out_dir / "budget_matrix_assessment.json").write_text(
        json.dumps(assessment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "budget_matrix_assessment.md").write_text(render_markdown(summary, assessment), encoding="utf-8")
    print(json.dumps(assessment, indent=2, ensure_ascii=False))


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("budget*/**/gate_summary.json")):
        run_dir = path.parent
        budget = parse_budget(path)
        preset, seed = parse_preset_seed(run_dir.name)
        summary = json.loads(path.read_text(encoding="utf-8"))
        static = float(summary["validation_selected_static_objective"])
        deploy = nullable_float(summary.get("best_deployable_objective"))
        teacher = nullable_float(summary.get("teacher_reference_objective"))
        rows.append(
            {
                "budget": budget,
                "preset": preset,
                "seed": seed,
                "run_dir": str(run_dir),
                "static_objective": static,
                "deployable_objective": deploy,
                "teacher_objective": teacher,
                "deployable_margin": static - deploy if deploy is not None else np.nan,
                "teacher_margin": static - teacher if teacher is not None else np.nan,
                "gate_pass": bool(summary.get("gate_pass", False)),
                "best_deployable_policy": str(summary.get("best_deployable_policy", "")),
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (budget, preset), group in df.groupby(["budget", "preset"], sort=True):
        deploy_margin = group["deployable_margin"].astype(float).to_numpy()
        teacher_margin = group["teacher_margin"].astype(float).to_numpy()
        n = int(len(group))
        wins = int(np.sum(deploy_margin > 0.0))
        teacher_wins = int(np.sum(teacher_margin > 0.0))
        rows.append(
            {
                "budget": float(budget),
                "preset": str(preset),
                "n": n,
                "deployable_wins": wins,
                "deployable_win_rate": wins / n if n else np.nan,
                "deployable_margin_mean": float(np.nanmean(deploy_margin)),
                "deployable_margin_median": float(np.nanmedian(deploy_margin)),
                "teacher_wins": teacher_wins,
                "teacher_win_rate": teacher_wins / n if n else np.nan,
                "teacher_margin_mean": float(np.nanmean(teacher_margin)),
                "all_gate_pass": bool(np.all(group["gate_pass"].astype(bool))),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget", "preset"])


def assess(summary: pd.DataFrame, *, min_seeds: int, min_win_rate: float) -> dict[str, object]:
    required_wins = int(np.ceil(float(min_win_rate) * float(min_seeds)))
    failures: list[str] = []
    passes: list[str] = []
    for row in summary.to_dict(orient="records"):
        label = f"budget={float(row['budget']):.2f} preset={row['preset']}"
        if int(row["n"]) < int(min_seeds):
            failures.append(f"{label}: n={int(row['n'])} < {min_seeds}")
        else:
            passes.append(f"{label}: n={int(row['n'])} >= {min_seeds}")
        if int(row["deployable_wins"]) < required_wins:
            failures.append(f"{label}: deployable wins={int(row['deployable_wins'])} < {required_wins}")
        else:
            passes.append(f"{label}: deployable wins={int(row['deployable_wins'])} >= {required_wins}")
        if float(row["deployable_margin_mean"]) <= 0.0:
            failures.append(f"{label}: mean margin={float(row['deployable_margin_mean']):.6g} <= 0")
        else:
            passes.append(f"{label}: mean margin={float(row['deployable_margin_mean']):.6g} > 0")
        if int(row["teacher_wins"]) < required_wins:
            failures.append(f"{label}: teacher wins={int(row['teacher_wins'])} < {required_wins}")
        else:
            passes.append(f"{label}: teacher wins={int(row['teacher_wins'])} >= {required_wins}")
    return {
        "matrix_pass": not failures,
        "budgets": sorted(float(x) for x in summary["budget"].unique()),
        "required_wins": required_wins,
        "min_seeds": int(min_seeds),
        "min_win_rate": float(min_win_rate),
        "pass_reasons": passes,
        "fail_reasons": failures,
    }


def render_markdown(summary: pd.DataFrame, assessment: dict[str, object]) -> str:
    lines = [
        "# Budget Matrix Assessment",
        "",
        f"- Matrix pass: `{assessment['matrix_pass']}`",
        f"- Budgets: `{assessment['budgets']}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- PASS: {item}" for item in assessment["pass_reasons"])
    lines.extend(f"- FAIL: {item}" for item in assessment["fail_reasons"])
    lines.append("")
    return "\n".join(lines)


def parse_budget(path: Path) -> float:
    for part in path.parts:
        if re.fullmatch(r"budget\d+(?:p\d+)?", part):
            return float(part.replace("budget", "").replace("p", "."))
    return float("nan")


def parse_preset_seed(name: str) -> tuple[str, int]:
    marker = "_seed"
    if marker not in name:
        return name, -1
    preset, seed_text = name.rsplit(marker, 1)
    return preset, int(seed_text)


def nullable_float(value: object) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


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


if __name__ == "__main__":
    main()
