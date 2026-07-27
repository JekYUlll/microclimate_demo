#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate residual deployment thresholds on train-only risk calibration rows."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-negative-starts", type=int, default=1)
    parser.add_argument(
        "--anchor-action-idx",
        type=int,
        default=None,
        help="Restrict calibration to one train-prequalified policy anchor.",
    )
    parser.add_argument(
        "--audit-per-anchor",
        action="store_true",
        help=(
            "Audit anchor-conditioned thresholds with leave-one-start-out "
            "selection; does not change the global deployment calibration."
        ),
    )
    parser.add_argument("--audit-jobs", type=int, default=1)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def threshold_grid(values: pd.Series) -> tuple[float, ...]:
    quantiles = np.quantile(
        values.to_numpy(dtype=float),
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    return tuple(sorted({float(np.round(value, 9)) for value in quantiles}))


def evaluate_thresholds(
    rows: pd.DataFrame,
    *,
    min_lower: float,
    max_negative_probability: float,
    min_mean: float,
) -> dict[str, object]:
    selected = []
    for (start, anchor_idx), group in rows.groupby(
        ["start", "anchor_action_idx"],
        sort=True,
    ):
        safe = group[
            (group["risk_lower_bound"] >= float(min_lower))
            & (
                group["negative_probability"]
                <= float(max_negative_probability)
            )
            & (group["mean_margin_pred"] >= float(min_mean))
        ]
        if safe.empty:
            margin = 0.0
            controller_id = "static_fallback"
            dynamic = False
        else:
            chosen = safe.sort_values(
                [
                    "mean_margin_pred",
                    "risk_lower_bound",
                    "controller_id",
                ],
                ascending=[False, False, True],
            ).iloc[0]
            margin = float(chosen["margin"])
            controller_id = str(chosen["controller_id"])
            dynamic = True
        selected.append(
            {
                "start": int(start),
                "anchor_action_idx": int(anchor_idx),
                "controller_id": controller_id,
                "margin": margin,
                "dynamic": dynamic,
            }
        )
    selected_frame = pd.DataFrame(selected)
    start_summary = selected_frame.groupby("start", sort=True).agg(
        margin=("margin", "mean"),
        dynamic_groups=("dynamic", "sum"),
    )
    return {
        "min_risk_lower_bound": float(min_lower),
        "max_negative_probability": float(max_negative_probability),
        "min_predicted_mean_margin": float(min_mean),
        "margin_mean": float(start_summary["margin"].mean()),
        "margin_q25": float(start_summary["margin"].quantile(0.25)),
        "margin_min": float(start_summary["margin"].min()),
        "negative_starts": int((start_summary["margin"] < 0.0).sum()),
        "dynamic_groups": int(selected_frame["dynamic"].sum()),
        "dynamic_starts": int((start_summary["dynamic_groups"] > 0).sum()),
        "selected_rows": selected,
    }


def selection_priority(row: dict[str, object]) -> tuple[object, ...]:
    """Prefer risk-safe thresholds that transfer dynamic use across starts."""
    return (
        int(row["dynamic_starts"]),
        float(row["margin_q25"]),
        float(row["margin_mean"]),
        float(row["margin_min"]),
        -int(row["negative_starts"]),
        int(row["dynamic_groups"]),
        float(row["min_risk_lower_bound"]),
        -float(row["max_negative_probability"]),
        float(row["min_predicted_mean_margin"]),
    )


def candidate_thresholds(
    rows: pd.DataFrame,
    *,
    lower_grid: tuple[float, ...],
    negative_grid: tuple[float, ...],
    mean_grid: tuple[float, ...],
) -> list[dict[str, object]]:
    return [
        evaluate_thresholds(
            rows,
            min_lower=lower,
            max_negative_probability=negative,
            min_mean=mean,
        )
        for lower in lower_grid
        for negative in negative_grid
        for mean in mean_grid
    ]


def valid_thresholds(
    candidates: list[dict[str, object]],
    *,
    max_negative_starts: int,
) -> list[dict[str, object]]:
    return [
        row
        for row in candidates
        if float(row["margin_mean"]) > 0.0
        and float(row["margin_q25"]) >= 0.0
        and int(row["negative_starts"]) <= int(max_negative_starts)
        and int(row["dynamic_groups"]) > 0
    ]


def select_thresholds(
    rows: pd.DataFrame,
    *,
    lower_grid: tuple[float, ...],
    negative_grid: tuple[float, ...],
    mean_grid: tuple[float, ...],
    max_negative_starts: int,
) -> tuple[dict[str, object] | None, int]:
    valid = valid_thresholds(
        candidate_thresholds(
            rows,
            lower_grid=lower_grid,
            negative_grid=negative_grid,
            mean_grid=mean_grid,
        ),
        max_negative_starts=max_negative_starts,
    )
    return (
        max(valid, key=selection_priority) if valid else None,
        len(valid),
    )


def audit_one_anchor_leave_one_start_out(
    anchor_idx: int,
    group: pd.DataFrame,
    *,
    lower_grid: tuple[float, ...],
    negative_grid: tuple[float, ...],
    mean_grid: tuple[float, ...],
    max_negative_starts: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    starts = tuple(sorted(int(value) for value in group["start"].unique()))
    full_selected, full_valid_count = select_thresholds(
        group,
        lower_grid=lower_grid,
        negative_grid=negative_grid,
        mean_grid=mean_grid,
        max_negative_starts=max_negative_starts,
    )
    anchor_heldout: list[dict[str, object]] = []
    selected_triplets: list[tuple[float, float, float]] = []
    for heldout_start in starts:
        train = group[group["start"] != int(heldout_start)]
        test = group[group["start"] == int(heldout_start)]
        selected, valid_count = select_thresholds(
            train,
            lower_grid=lower_grid,
            negative_grid=negative_grid,
            mean_grid=mean_grid,
            max_negative_starts=max_negative_starts,
        )
        if selected is None:
            evaluation = {
                "margin_mean": 0.0,
                "dynamic_groups": 0,
            }
            triplet = None
        else:
            triplet = (
                float(selected["min_risk_lower_bound"]),
                float(selected["max_negative_probability"]),
                float(selected["min_predicted_mean_margin"]),
            )
            selected_triplets.append(triplet)
            evaluation = evaluate_thresholds(
                test,
                min_lower=triplet[0],
                max_negative_probability=triplet[1],
                min_mean=triplet[2],
            )
        anchor_heldout.append(
            {
                "anchor_action_idx": int(anchor_idx),
                "heldout_start": int(heldout_start),
                "train_starts": int(train["start"].nunique()),
                "train_valid_candidate_count": int(valid_count),
                "threshold_selected": bool(selected is not None),
                "min_risk_lower_bound": (
                    float(triplet[0]) if triplet is not None else float("nan")
                ),
                "max_negative_probability": (
                    float(triplet[1]) if triplet is not None else float("nan")
                ),
                "min_predicted_mean_margin": (
                    float(triplet[2]) if triplet is not None else float("nan")
                ),
                "heldout_margin": float(evaluation["margin_mean"]),
                "heldout_dynamic": bool(int(evaluation["dynamic_groups"]) > 0),
                "heldout_negative": bool(
                    float(evaluation["margin_mean"]) < 0.0
                ),
            }
        )
    heldout_frame = pd.DataFrame(anchor_heldout)
    heldout_margins = heldout_frame["heldout_margin"].to_numpy(dtype=float)
    selected_folds = int(heldout_frame["threshold_selected"].sum())
    dynamic_starts = int(heldout_frame["heldout_dynamic"].sum())
    negative_starts = int(heldout_frame["heldout_negative"].sum())
    heldout_mean = float(np.mean(heldout_margins))
    heldout_q25 = float(np.quantile(heldout_margins, 0.25))
    heldout_min = float(np.min(heldout_margins))
    loso_gate = bool(
        heldout_mean > 0.0
        and heldout_q25 >= 0.0
        and negative_starts <= int(max_negative_starts)
        and dynamic_starts > 0
    )
    anchor_summary = {
        "anchor_action_idx": int(anchor_idx),
        "rows": int(len(group)),
        "starts": int(len(starts)),
        "actions_per_start_min": int(group.groupby("start").size().min()),
        "actions_per_start_max": int(group.groupby("start").size().max()),
        "full_valid_candidate_count": int(full_valid_count),
        "full_gate_pass": bool(full_selected is not None),
        "full_dynamic_starts": (
            int(full_selected["dynamic_starts"])
            if full_selected is not None
            else 0
        ),
        "loso_selected_folds": selected_folds,
        "loso_unique_threshold_triplets": int(len(set(selected_triplets))),
        "loso_dynamic_starts": dynamic_starts,
        "loso_margin_mean": heldout_mean,
        "loso_margin_q25": heldout_q25,
        "loso_margin_min": heldout_min,
        "loso_negative_starts": negative_starts,
        "loso_gate_pass": loso_gate,
    }
    return anchor_summary, anchor_heldout


def audit_anchor_leave_one_start_out(
    rows: pd.DataFrame,
    *,
    lower_grid: tuple[float, ...],
    negative_grid: tuple[float, ...],
    mean_grid: tuple[float, ...],
    max_negative_starts: int,
    jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [
        (int(anchor_idx), group.copy())
        for anchor_idx, group in rows.groupby("anchor_action_idx", sort=True)
    ]
    results = Parallel(n_jobs=max(1, int(jobs)))(
        delayed(audit_one_anchor_leave_one_start_out)(
            anchor_idx,
            group,
            lower_grid=lower_grid,
            negative_grid=negative_grid,
            mean_grid=mean_grid,
            max_negative_starts=max_negative_starts,
        )
        for anchor_idx, group in groups
    )
    anchor_rows = [summary for summary, _ in results]
    heldout_rows = [
        row
        for _, anchor_heldout in results
        for row in anchor_heldout
    ]
    return pd.DataFrame(anchor_rows), pd.DataFrame(heldout_rows)


def main() -> None:
    args = parse_args()
    model_dir = resolve_project_path(args.model_dir)
    rows = pd.read_csv(
        model_dir / "window_risk_calibration_predictions.csv"
    )
    if args.anchor_action_idx is not None:
        rows = rows[
            rows["anchor_action_idx"] == int(args.anchor_action_idx)
        ].copy()
        if rows.empty:
            raise ValueError(
                f"No calibration rows for anchor {args.anchor_action_idx}"
            )
    lower_grid = threshold_grid(rows["risk_lower_bound"])
    negative_grid = threshold_grid(rows["negative_probability"])
    mean_grid = tuple(
        sorted(
            {
                0.0,
                *threshold_grid(rows["mean_margin_pred"]),
            }
        )
    )
    candidates = candidate_thresholds(
        rows,
        lower_grid=lower_grid,
        negative_grid=negative_grid,
        mean_grid=mean_grid,
    )
    valid = valid_thresholds(
        candidates,
        max_negative_starts=int(args.max_negative_starts),
    )
    selected = max(valid, key=selection_priority) if valid else None
    result = {
        "model_dir": str(model_dir),
        "anchor_action_idx": (
            int(args.anchor_action_idx)
            if args.anchor_action_idx is not None
            else None
        ),
        "calibration_rows": int(len(rows)),
        "calibration_starts": int(rows["start"].nunique()),
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(len(valid)),
        "calibration_gate_pass": selected is not None,
        "selected": selected,
        "selection_rule": (
            "positive mean, non-negative q25, bounded negative starts, "
            "dynamic use; then maximize dynamic-start coverage before "
            "q25/mean/min/negative-start/dynamic-group/conservatism tie-break"
        ),
        "grids": {
            "min_risk_lower_bound": list(lower_grid),
            "max_negative_probability": list(negative_grid),
            "min_predicted_mean_margin": list(mean_grid),
        },
    }
    output = (
        resolve_project_path(args.out)
        if args.out is not None
        else (
            model_dir
            / (
                "residual_deployment_calibration.json"
                if args.anchor_action_idx is None
                else (
                    "residual_deployment_calibration_anchor_"
                    f"{int(args.anchor_action_idx):03d}.json"
                )
            )
        )
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.audit_per_anchor:
        anchor_audit, heldout_audit = audit_anchor_leave_one_start_out(
            rows,
            lower_grid=lower_grid,
            negative_grid=negative_grid,
            mean_grid=mean_grid,
            max_negative_starts=int(args.max_negative_starts),
            jobs=int(args.audit_jobs),
        )
        anchor_csv = output.with_name("residual_anchor_calibration_audit.csv")
        heldout_csv = output.with_name(
            "residual_anchor_calibration_loso_rows.csv"
        )
        audit_json = output.with_name("residual_anchor_calibration_audit.json")
        anchor_audit.to_csv(anchor_csv, index=False)
        heldout_audit.to_csv(heldout_csv, index=False)
        audit_result = {
            "anchors": int(len(anchor_audit)),
            "full_gate_pass_anchors": int(anchor_audit["full_gate_pass"].sum()),
            "loso_gate_pass_anchors": int(anchor_audit["loso_gate_pass"].sum()),
            "loso_any_dynamic_anchors": int(
                (anchor_audit["loso_dynamic_starts"] > 0).sum()
            ),
            "loso_all_folds_calibratable_anchors": int(
                (
                    anchor_audit["loso_selected_folds"]
                    == anchor_audit["starts"]
                ).sum()
            ),
            "anchor_summary_csv": str(anchor_csv),
            "heldout_rows_csv": str(heldout_csv),
            "selection_protocol": (
                "For each anchor and heldout start, select one threshold "
                "triplet using only the other calibration starts, then apply "
                "it once to the heldout start. Aggregate the eight heldout "
                "outcomes with the unchanged deployment risk gate."
            ),
        }
        audit_json.write_text(
            json.dumps(audit_result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result["per_anchor_audit"] = audit_result
    print(json.dumps(result, indent=2, sort_keys=True))
    if selected is None and not args.audit_per_anchor:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
