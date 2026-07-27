#!/usr/bin/env python3
"""Audit window-level opportunity in intervention-effect rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate paired intervention-effect rows to seed/start windows "
            "and report oracle-safe opportunity ceilings."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "robust_intervention_effect_train_multiseed_rows.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "effect_window_ceiling_20260607"
        ),
    )
    parser.add_argument("--min-window-rows", type=int, default=3)
    return parser.parse_args()


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def aggregate_windows(df: pd.DataFrame, scope: str, min_window_rows: int) -> pd.DataFrame:
    if scope == "selected_dynamic":
        selected_dynamic = ~boolish(df["selected_is_anchor"])
        scoped = df.loc[selected_dynamic].copy()
    elif scope == "all_raw":
        scoped = df.copy()
    else:
        raise ValueError(f"unknown scope: {scope}")

    if len(scoped) == 0:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, object]] = {
        "row_count": ("effect_margin", "count"),
        "effect_mean": ("effect_margin", "mean"),
        "effect_q25": ("effect_margin", lambda x: float(np.quantile(x, 0.25))),
        "effect_min": ("effect_margin", "min"),
        "positive_rate": ("effect_margin", lambda x: float(np.mean(np.asarray(x) > 0.0))),
    }
    for column in [
        "predicted_anchor_minus_raw",
        "component_task_margin_mean",
        "component_total_margin_mean",
        "raw_anchor_hamming",
        "raw_active_count",
        "anchor_active_count",
    ]:
        if column in scoped.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_max"] = (column, "max")
            aggregations[f"{column}_q25"] = (column, lambda x: float(np.quantile(x, 0.25)))

    event_columns = [c for c in scoped.columns if c.startswith("context_learned_event_p_")]
    for column in event_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_max"] = (column, "max")

    grouped = scoped.groupby(["seed", "start"]).agg(**aggregations).reset_index()
    grouped.insert(0, "scope", scope)
    grouped["safe_window"] = (
        (grouped["row_count"] >= int(min_window_rows))
        & (grouped["effect_mean"] > 0.0)
        & (grouped["effect_q25"] >= 0.0)
    )
    return grouped


def summarize(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scope, seed), group in windows.groupby(["scope", "seed"], dropna=False):
        safe = group["safe_window"].astype(bool)
        rows.append(
            {
                "scope": scope,
                "seed": int(seed),
                "windows": int(len(group)),
                "safe_windows": int(safe.sum()),
                "safe_rate": float(safe.mean()) if len(group) else np.nan,
                "rows": int(group["row_count"].sum()),
                "mean_window_effect_mean": float(group["effect_mean"].mean())
                if len(group)
                else np.nan,
                "worst_window_q25": float(group["effect_q25"].min()) if len(group) else np.nan,
                "best_window_mean": float(group["effect_mean"].max()) if len(group) else np.nan,
                "best_window_q25": float(group["effect_q25"].max()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for value in row.tolist():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(output: Path, summary: pd.DataFrame, windows: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Effect Window Ceiling Audit")
    lines.append("")
    lines.append("## Per-Seed Summary")
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Safe Windows")
    safe_cols = [
        "scope",
        "seed",
        "start",
        "row_count",
        "effect_mean",
        "effect_q25",
        "positive_rate",
    ]
    safe = windows.loc[windows["safe_window"], safe_cols].sort_values(["scope", "seed", "start"])
    if len(safe):
        lines.append(markdown_table(safe))
    else:
        lines.append("No safe windows under the configured criterion.")
    lines.append("")
    lines.append("## Source-Oracle Ceiling")
    source_oracle = build_source_oracle(windows)
    source_summary = (
        source_oracle.groupby("seed", as_index=False)
        .agg(
            windows=("start", "count"),
            source_oracle_safe_windows=("source_oracle_safe", "sum"),
        )
        .sort_values("seed")
    )
    lines.append(markdown_table(source_summary))
    lines.append("")
    label_summary = (
        source_oracle.groupby(["seed", "source_oracle_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    lines.append(markdown_table(label_summary))
    lines.append("")
    lines.append("## Decision")
    selected = summary.loc[summary["scope"] == "selected_dynamic"]
    all_raw = summary.loc[summary["scope"] == "all_raw"]
    selected_missing = selected.loc[selected["safe_windows"] == 0, "seed"].astype(int).tolist()
    all_raw_missing = all_raw.loc[all_raw["safe_windows"] == 0, "seed"].astype(int).tolist()
    lines.append(
        f"- `selected_dynamic` has zero oracle-safe train windows for seeds: "
        f"{selected_missing}."
    )
    lines.append(
        f"- `all_raw` has zero oracle-safe train windows for seeds: {all_raw_missing}."
    )
    lines.append(
        "- A single row-level rejection boundary is therefore structurally weak: "
        "the current planner-selected action stream and the broader raw stream "
        "expose different opportunity regimes."
    )
    if bool(source_summary["source_oracle_safe_windows"].gt(0).all()):
        lines.append(
            "- Source-oracle labels have at least one safe train window in every "
            "seed, so a window-level source selector is a plausible next "
            "prototype."
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_source_oracle(windows: pd.DataFrame) -> pd.DataFrame:
    pivot = windows.pivot_table(
        index=["seed", "start"],
        columns="scope",
        values="safe_window",
        aggfunc="first",
    )
    for column in ("selected_dynamic", "all_raw"):
        if column not in pivot.columns:
            pivot[column] = False
        pivot[column] = pivot[column].where(pivot[column].notna(), False).astype(bool)
    pivot = pivot.reset_index()
    pivot["source_oracle_safe"] = pivot["selected_dynamic"] | pivot["all_raw"]

    def label(row: pd.Series) -> str:
        if bool(row["selected_dynamic"]):
            return "selected_dynamic"
        if bool(row["all_raw"]):
            return "raw_bypass"
        return "anchor"

    pivot["source_oracle_label"] = pivot.apply(label, axis=1)
    return pivot


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"seed", "start", "effect_margin", "selected_is_anchor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    windows = pd.concat(
        [
            aggregate_windows(df, "selected_dynamic", args.min_window_rows),
            aggregate_windows(df, "all_raw", args.min_window_rows),
        ],
        ignore_index=True,
    )
    summary = summarize(windows)
    args.output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output / "effect_window_ceiling_windows.csv", index=False)
    summary.to_csv(args.output / "effect_window_ceiling_summary.csv", index=False)
    source_oracle = build_source_oracle(windows)
    source_oracle.to_csv(
        args.output / "effect_window_source_oracle.csv",
        index=False,
    )
    payload = {
        "input": str(args.input),
        "min_window_rows": int(args.min_window_rows),
        "scopes": sorted(windows["scope"].unique().tolist()),
        "selected_dynamic_zero_safe_seeds": summary.loc[
            (summary["scope"] == "selected_dynamic") & (summary["safe_windows"] == 0),
            "seed",
        ].astype(int).tolist(),
        "all_raw_zero_safe_seeds": summary.loc[
            (summary["scope"] == "all_raw") & (summary["safe_windows"] == 0),
            "seed",
        ].astype(int).tolist(),
        "source_oracle_safe_windows_by_seed": {
            str(int(seed)): int(group["source_oracle_safe"].sum())
            for seed, group in source_oracle.groupby("seed")
        },
    }
    (args.output / "effect_window_ceiling_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(args.output / "effect_window_ceiling_summary.md", summary, windows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
