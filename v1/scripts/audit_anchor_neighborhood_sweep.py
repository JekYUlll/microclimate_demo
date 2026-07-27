#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an anchor-neighborhood controller on fit and gate it on calibration."
    )
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No rows in {path}")
    frame = pd.DataFrame(rows)
    frame["max_anchor_hamming"] = frame["controller_config"].map(
        lambda value: int(value["max_anchor_hamming"])
    )
    frame["base_controller"] = frame["controller_id"].str.split(
        "_h", n=1
    ).str[0]
    return frame


def controller_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller_id, group in frame.groupby("controller_id", sort=True):
        start_margin = group.groupby("start", sort=True)["margin"].mean()
        rows.append(
            {
                "controller_id": str(controller_id),
                "base_controller": str(group["base_controller"].iloc[0]),
                "max_anchor_hamming": int(
                    group["max_anchor_hamming"].iloc[0]
                ),
                "rows": int(len(group)),
                "starts": int(group["start"].nunique()),
                "margin_mean": float(group["margin"].mean()),
                "margin_q25": float(group["margin"].quantile(0.25)),
                "margin_min": float(group["margin"].min()),
                "negative_rows": int((group["margin"] < 0.0).sum()),
                "nonzero_margin_rows": int(
                    (group["margin"].abs() > 1.0e-12).sum()
                ),
                "negative_start_means": int((start_margin < 0.0).sum()),
                "hard_constraint_violations": int(
                    group["constraint_violation_count"].sum()
                ),
                "warmup_abort_count": int(group["warmup_abort_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["margin_q25", "margin_mean", "margin_min", "controller_id"],
        ascending=[False, False, False, True],
    )


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.data_root)
    fit = load_rows(root / "risk_fit" / "window_risk_rows.jsonl")
    calibration = load_rows(
        root / "risk_calibration" / "window_risk_rows.jsonl"
    )
    fit_summary = controller_summary(fit)
    calibration_summary = controller_summary(calibration)
    selected_id = str(fit_summary.iloc[0]["controller_id"])
    selected_fit = fit_summary[
        fit_summary["controller_id"] == selected_id
    ].iloc[0]
    selected_calibration = calibration_summary[
        calibration_summary["controller_id"] == selected_id
    ].iloc[0]
    gate = bool(
        float(selected_calibration["margin_mean"]) > 0.0
        and float(selected_calibration["margin_q25"]) >= 0.0
        and int(selected_calibration["negative_start_means"]) <= 1
        and int(selected_calibration["nonzero_margin_rows"]) > 0
        and int(selected_calibration["hard_constraint_violations"]) == 0
    )
    fit_summary.to_csv(
        root / "anchor_neighborhood_fit_summary.csv", index=False
    )
    calibration_summary.to_csv(
        root / "anchor_neighborhood_calibration_summary.csv", index=False
    )
    result = {
        "selected_controller_id": selected_id,
        "selected_fit": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in selected_fit.to_dict().items()
        },
        "selected_calibration": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in selected_calibration.to_dict().items()
        },
        "calibration_gate_pass": gate,
        "selection_rule": (
            "fit max margin_q25, then mean, min, controller_id; "
            "calibration requires nonzero dynamic effect"
        ),
        "max_anchor_hamming_values": sorted(
            int(x) for x in fit["max_anchor_hamming"].unique()
        ),
        "base_controllers": sorted(
            str(x) for x in fit["base_controller"].unique()
        ),
    }
    (root / "anchor_neighborhood_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
