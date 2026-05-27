#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.dataset import TeacherDataset
from forecast_cmdp.features import ForecastContextConfig
from forecast_cmdp.policy import BCTrainingConfig, save_bc_policy_checkpoint, train_bc_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a forecast-aware BC policy from an MPC-teacher dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-checkpoint", default="v1/artifacts/bc_policy.pt")
    parser.add_argument("--out-metadata", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--event-column", default="event_flag")
    parser.add_argument("--truth-future-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TeacherDataset.load_npz(str(args.dataset))
    train_cfg = BCTrainingConfig(
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
    )
    forecast_cfg = ForecastContextConfig(
        horizon=int(args.forecast_horizon),
        event_column=str(args.event_column),
        truth_future=bool(args.truth_future_features),
    )
    model, history = train_bc_classifier(
        dataset.features,
        dataset.labels,
        dataset.action_masks,
        cfg=train_cfg,
    )
    out_checkpoint = Path(args.out_checkpoint)
    save_bc_policy_checkpoint(
        out_checkpoint,
        model=model,
        candidate_masks=dataset.candidate_masks,
        forecast_cfg=forecast_cfg,
        train_cfg=train_cfg,
        history=history,
    )
    metadata_path = Path(args.out_metadata) if args.out_metadata else out_checkpoint.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "role": "v1_forecast_cmdp_bc_policy",
        "dataset": str(args.dataset),
        "checkpoint": str(out_checkpoint),
        "sample_count": int(dataset.features.shape[0]),
        "feature_dim": int(dataset.features.shape[1]),
        "candidate_count": int(dataset.candidate_masks.shape[0]),
        "event_rate": float(np.mean(dataset.event_flags)),
        "train_cfg": train_cfg.__dict__,
        "forecast_cfg": forecast_cfg.__dict__,
        "final_loss": float(history["loss"][-1]) if history.get("loss") else float("nan"),
        "final_accuracy": float(history["accuracy"][-1]) if history.get("accuracy") else float("nan"),
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(out_checkpoint),
                "metadata": str(metadata_path),
                "sample_count": int(dataset.features.shape[0]),
                "final_loss": metadata["final_loss"],
                "final_accuracy": metadata["final_accuracy"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
