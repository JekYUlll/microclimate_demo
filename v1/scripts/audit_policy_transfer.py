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
        description="Audit validation-to-final transfer for v1 deployable policy candidates."
    )
    parser.add_argument("suite_roots", nargs="+", help="One or more claim-suite roots.")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.suite_roots]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif len(roots) == 1:
        out_dir = roots[0] / "aggregate"
    else:
        raise SystemExit("--out-dir is required when auditing multiple roots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_transfer_rows(roots)
    if not rows:
        raise FileNotFoundError(f"No validation/final transfer rows found under {roots}")
    transfer = pd.DataFrame(rows).sort_values(["root", "preset", "seed", "policy"])
    transfer.to_csv(out_dir / "policy_transfer_rows.csv", index=False)

    summary = summarize_transfer(transfer, by=("root", "policy"))
    summary.to_csv(out_dir / "policy_transfer_summary.csv", index=False)

    overall = summarize_transfer(transfer, by=("policy",))
    overall.to_csv(out_dir / "policy_transfer_summary_overall.csv", index=False)

    selected = transfer.loc[transfer["is_selected"].astype(bool)].copy()
    selected.to_csv(out_dir / "policy_transfer_selected.csv", index=False)

    report = render_report(summary, selected, overall)
    (out_dir / "policy_transfer_audit.md").write_text(report, encoding="utf-8")
    print(report)


def collect_transfer_rows(roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        for manifest_path in sorted(root.glob("*_seed*/manifest.json")):
            run_dir = manifest_path.parent
            metrics_path = run_dir / "metrics_final.csv"
            if not metrics_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            preset, seed = parse_preset_seed(run_dir.name, manifest)
            selection = manifest.get("deployable_selection", {})
            if not isinstance(selection, dict):
                continue
            validation_rows = selection.get("validation_rows", [])
            if not isinstance(validation_rows, list):
                continue
            final_metrics = pd.read_csv(metrics_path)
            if "policy" not in final_metrics.columns or "objective_loss_mean" not in final_metrics.columns:
                continue
            final_by_policy = {
                str(row["policy"]): row
                for row in final_metrics.to_dict(orient="records")
                if isinstance(row, dict)
            }
            static_row = final_by_policy.get("validation_selected_static")
            if static_row is None:
                continue
            static_final = safe_float(static_row.get("objective_loss_mean"))
            selected_policy = str(selection.get("selected_policy", ""))
            for row in validation_rows:
                if not isinstance(row, dict):
                    continue
                policy = str(row.get("policy", ""))
                final_row = final_by_policy.get(policy)
                if final_row is None:
                    continue
                validation_margin = safe_float(row.get("objective_margin_mean"))
                final_objective = safe_float(final_row.get("objective_loss_mean"))
                final_margin = static_final - final_objective
                rows.append(
                    {
                        "root": str(root),
                        "preset": preset,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "policy": policy,
                        "is_selected": policy == selected_policy,
                        "validation_objective": safe_float(row.get("objective")),
                        "validation_margin_mean": validation_margin,
                        "validation_margin_median": safe_float(row.get("objective_margin_median")),
                        "validation_margin_q25": safe_float(row.get("objective_margin_q25")),
                        "validation_margin_min": safe_float(row.get("objective_margin_min")),
                        "validation_negative_starts": safe_int(row.get("negative_start_count")),
                        "validation_guard_pass": bool(row.get("static_margin_guard_pass", False)),
                        "validation_positive_center": bool(row.get("static_margin_positive_center", False)),
                        "final_objective": final_objective,
                        "static_final_objective": static_final,
                        "final_margin": final_margin,
                        "final_win": bool(final_margin > 0.0),
                        "transfer_gap": final_margin - validation_margin,
                        "final_power_mean": safe_float(final_row.get("power_mean")),
                        "final_warmup_abort_count": safe_int(final_row.get("warmup_abort_count")),
                    }
                )
    return rows


def summarize_transfer(transfer: pd.DataFrame, *, by: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in transfer.groupby(list(by), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(by, key, strict=True))
        final_wins = int(group["final_win"].astype(bool).sum())
        selected = group.loc[group["is_selected"].astype(bool)]
        selected_wins = int(selected["final_win"].astype(bool).sum()) if not selected.empty else 0
        rows.append(
            {
                **key_values,
                "n_final_observed": int(len(group)),
                "n_validation_guard_pass": int(group["validation_guard_pass"].astype(bool).sum()),
                "n_selected": int(len(selected)),
                "final_wins_all": final_wins,
                "final_win_rate_all": final_wins / len(group) if len(group) else np.nan,
                "final_wins_selected": selected_wins,
                "final_win_rate_selected": selected_wins / len(selected) if len(selected) else np.nan,
                "n_validation_positive_center": int(group["validation_positive_center"].astype(bool).sum()),
                "validation_margin_mean": float(np.nanmean(group["validation_margin_mean"].to_numpy(dtype=float))),
                "validation_margin_median_mean": float(
                    np.nanmean(group["validation_margin_median"].to_numpy(dtype=float))
                ),
                "validation_margin_q25_mean": float(
                    np.nanmean(group["validation_margin_q25"].to_numpy(dtype=float))
                ),
                "final_margin_mean": float(np.nanmean(group["final_margin"].to_numpy(dtype=float))),
                "transfer_gap_mean": float(np.nanmean(group["transfer_gap"].to_numpy(dtype=float))),
                "transfer_gap_median": float(np.nanmedian(group["transfer_gap"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(list(by))


def render_report(summary: pd.DataFrame, selected: pd.DataFrame, overall: pd.DataFrame) -> str:
    lines = ["# Policy Transfer Audit", ""]
    lines.append("## Selected Final Outcomes")
    lines.append("")
    if selected.empty:
        lines.append("No selected deployable policies with final metrics were found.")
    else:
        display = selected[
            [
                "root",
                "preset",
                "seed",
                "policy",
                "validation_margin_mean",
                "validation_margin_median",
                "validation_margin_q25",
                "validation_positive_center",
                "final_margin",
                "transfer_gap",
                "final_win",
            ]
        ]
        lines.append(markdown_table(display))
    lines.extend(["", "## Overall Summary By Policy", ""])
    lines.append(markdown_table(overall))
    lines.extend(["", "## Summary By Root And Policy", ""])
    lines.append(markdown_table(summary))
    lines.append("")
    return "\n".join(lines)


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
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, (np.floating,)):
        if np.isnan(float(value)):
            return "nan"
        return f"{float(value):.6g}"
    return str(value)


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


if __name__ == "__main__":
    main()
