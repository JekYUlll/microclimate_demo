#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.archived_v2 import load_v2_helpers  # noqa: E402
from forecast_cmdp.continuous_forecaster import (  # noqa: E402
    ContinuousForecasterTrainingConfig,
)
from forecast_cmdp.probabilistic_world_model import (  # noqa: E402
    ProbabilisticWorldModelTrainingConfig,
    save_probabilistic_world_model,
    train_probabilistic_world_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and audit a chronological probabilistic world model without "
            "using validation or final-test outcomes."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--truth-csv", default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.85)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--period-steps", type=int, default=10800)
    parser.add_argument("--min-skill-vs-persistence", type=float, default=0.0)
    parser.add_argument("--min-interval-coverage", type=float, default=0.60)
    parser.add_argument("--max-interval-coverage", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run).resolve()
    manifest_path = source_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_path = (
        Path(args.truth_csv).resolve()
        if args.truth_csv
        else source_run / "truth_with_learned_event_forecast.csv"
    )
    if not truth_path.exists():
        truth_path = Path(manifest["truth_csv"])
        if not truth_path.is_absolute():
            truth_path = (PROJECT_ROOT / truth_path).resolve()
    truth = pd.read_csv(truth_path)
    helpers = load_v2_helpers()
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    bounds = manifest["bounds"]
    train_bounds = (
        int(bounds["oracle_pretrain"][0]),
        int(bounds["rl_train"][1]),
    )
    forecaster_cfg = ContinuousForecasterTrainingConfig(
        horizon=int(args.horizon),
        lookback=int(args.lookback),
        target_columns=state_columns,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        prediction_prefix="world_model",
        period_steps=max(1, int(args.period_steps)),
    )
    training_cfg = ProbabilisticWorldModelTrainingConfig(
        member_count=int(args.members),
        fit_fraction=float(args.fit_fraction),
        calibration_fraction=float(args.calibration_fraction),
        bootstrap_fraction=float(args.bootstrap_fraction),
        residual_scale=float(args.residual_scale),
        seed=int(args.seed),
        forecaster=forecaster_cfg,
    )
    model = train_probabilistic_world_model(
        truth,
        bounds=train_bounds,
        state_columns=state_columns,
        cfg=training_cfg,
    )
    metrics = dict(model.audit_metrics)
    skill = float(metrics["rmse_skill_vs_persistence"])
    coverage = float(metrics["interval_80_coverage"])
    gate_pass = bool(
        skill > float(args.min_skill_vs_persistence)
        and coverage >= float(args.min_interval_coverage)
        and coverage <= float(args.max_interval_coverage)
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "probabilistic_world_model.pt"
    save_probabilistic_world_model(model, model_path)
    summary = {
        "role": "causal_probabilistic_world_model_audit",
        "source_run": str(source_run),
        "source_manifest": str(manifest_path),
        "truth_csv": str(truth_path),
        "state_columns": list(state_columns),
        "train_bounds": list(train_bounds),
        "validation_or_final_used": False,
        "config": {
            **asdict(training_cfg),
            "forecaster": asdict(forecaster_cfg),
        },
        "metrics": metrics,
        "gate": {
            "min_skill_vs_persistence": float(args.min_skill_vs_persistence),
            "min_interval_coverage": float(args.min_interval_coverage),
            "max_interval_coverage": float(args.max_interval_coverage),
            "pass": gate_pass,
        },
        "model_path": str(model_path),
    }
    (output / "world_model_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
