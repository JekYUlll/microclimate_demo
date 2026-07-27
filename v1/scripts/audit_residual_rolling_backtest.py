#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.window_risk import (  # noqa: E402
    build_window_risk_dataset,
    filter_exact_anchor_boundaries,
    load_window_risk_records,
)
from forecast_cmdp.window_risk_model import (  # noqa: E402
    WindowRiskTrainingConfig,
    train_window_risk_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit-only chronological backtest for rolling residual-risk training."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "fit_starts": int(metrics["fit_independent_starts"]),
        "target_starts": int(metrics["calibration_independent_starts"]),
        "mean_spearman": float(metrics["mean_spearman"]),
        "q25_pinball_improvement": float(metrics["q25_pinball_improvement"]),
        "negative_brier_improvement": float(
            metrics["negative_brier_improvement"]
        ),
        "raw_q25_coverage": float(metrics["raw_q25_coverage"]),
        "model_gate_pass": bool(metrics["model_gate"]["pass"]),
    }


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.data_root)
    dataset, exact_filter = filter_exact_anchor_boundaries(
        build_window_risk_dataset(
            load_window_risk_records(
                root / "risk_fit" / "window_risk_rows.jsonl"
            )
        )
    )
    starts = np.asarray(sorted(np.unique(dataset.starts)), dtype=np.int64)
    if starts.size < 128:
        raise ValueError("Rolling backtest requires at least 128 fit starts")
    quarters = tuple(tuple(int(x) for x in part) for part in np.array_split(starts, 4))

    def subset(selected_starts: tuple[int, ...]):
        selected = set(int(x) for x in selected_starts)
        return build_window_risk_dataset(
            [
                record
                for record in dataset.records
                if int(record.start) in selected
            ]
        )

    protocols = (
        ("q3_expanding", quarters[0] + quarters[1], quarters[2]),
        ("q3_recent", quarters[1], quarters[2]),
        (
            "q4_expanding",
            quarters[0] + quarters[1] + quarters[2],
            quarters[3],
        ),
        ("q4_recent", quarters[1] + quarters[2], quarters[3]),
    )
    config = WindowRiskTrainingConfig(
        model_family="hist_gbdt",
        n_estimators=250,
        learning_rate=0.04,
        max_leaf_nodes=7,
        min_samples_leaf=32,
        l2_regularization=1.0,
        seed=int(args.seed),
    )
    results: dict[str, dict[str, object]] = {}
    for name, train_starts, target_starts in protocols:
        _, metrics = train_window_risk_models(
            subset(train_starts),
            subset(target_starts),
            cfg=config,
        )
        results[name] = metric_summary(metrics)

    comparisons = {}
    protocol_passes = []
    for target in ("q3", "q4"):
        expanding = results[f"{target}_expanding"]
        recent = results[f"{target}_recent"]
        comparison = {
            "q25_improvement_delta": float(
                recent["q25_pinball_improvement"]
                - expanding["q25_pinball_improvement"]
            ),
            "brier_improvement_delta": float(
                recent["negative_brier_improvement"]
                - expanding["negative_brier_improvement"]
            ),
            "spearman_delta": float(
                recent["mean_spearman"] - expanding["mean_spearman"]
            ),
        }
        comparison["recent_pass"] = bool(
            comparison["q25_improvement_delta"] > 0.0
            and comparison["brier_improvement_delta"] > 0.0
            and comparison["spearman_delta"] >= -0.05
        )
        comparisons[target] = comparison
        protocol_passes.append(bool(comparison["recent_pass"]))
    result = {
        "data_root": str(root),
        "exact_anchor_filter": exact_filter,
        "quarter_starts": [list(part) for part in quarters],
        "protocols": results,
        "comparisons": comparisons,
        "rolling_protocol_gate_pass": bool(all(protocol_passes)),
        "selection_rule": (
            "Recent history must improve q25 and Brier over expanding history "
            "in both Q3 and Q4 backtests, with Spearman delta >= -0.05."
        ),
    }
    output = (
        resolve_project_path(args.out)
        if args.out is not None
        else root / "fit_rolling_backtest.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
