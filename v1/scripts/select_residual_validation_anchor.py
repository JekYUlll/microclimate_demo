#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a residual policy anchor from locked validation candidates."
    )
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.sweep_root)
    summaries = []
    for path in sorted(root.glob("anchor_*/mean_risk_gate_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["summary_path"] = str(path)
        summaries.append(summary)
    if not summaries:
        raise ValueError(f"No validation summaries under {root}")
    comparator_indices = {
        int(summary["comparator_anchor_idx"]) for summary in summaries
    }
    if len(comparator_indices) != 1:
        raise ValueError("Validation candidates use different comparators")
    valid = [
        summary
        for summary in summaries
        if bool(summary["validation_gate_pass"])
    ]
    selected = None
    if valid:
        selected = max(
            valid,
            key=lambda summary: (
                float(summary["validation"]["margin_q25"]),
                float(summary["validation"]["margin_mean"]),
                float(summary["validation"]["margin_min"]),
                -int(summary["validation"]["negative_starts"]),
                int(summary["validation"]["dynamic_windows"]),
                -int(summary["policy_anchor_idx"]),
            ),
        )
    result = {
        "candidate_count": len(summaries),
        "passing_candidate_count": len(valid),
        "validation_selection_pass": selected is not None,
        "comparator_anchor_idx": next(iter(comparator_indices)),
        "selected_policy_anchor_idx": (
            int(selected["policy_anchor_idx"])
            if selected is not None
            else None
        ),
        "selected_validation": (
            selected["validation"] if selected is not None else None
        ),
        "candidates": [
            {
                "policy_anchor_idx": int(summary["policy_anchor_idx"]),
                "validation_gate_pass": bool(
                    summary["validation_gate_pass"]
                ),
                **dict(summary["validation"]),
            }
            for summary in summaries
        ],
        "selection_rule": (
            "gate pass, then q25/mean/min/fewer negatives/more dynamic windows"
        ),
    }
    output = (
        resolve_project_path(args.out)
        if args.out is not None
        else root / "residual_validation_anchor_selection.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
