#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a v1 protocol gate output directory.")
    parser.add_argument("out_dir")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)
    summary_path = out_dir / "gate_summary.json"
    if summary_path.exists():
        print("## gate_summary")
        print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    metrics_path = out_dir / "metrics_final.csv"
    if metrics_path.exists():
        print("\n## metrics_final")
        df = pd.read_csv(metrics_path)
        cols = [
            name
            for name in (
                "policy",
                "objective_loss_mean",
                "oracle_loss_mean",
                "mae",
                "power_mean",
                "event_rate",
                "warmup_abort_count",
            )
            if name in df.columns
        ]
        print(df[cols].to_string(index=False))
    dataset_path = out_dir / "teacher_dataset.npz"
    if dataset_path.exists():
        print("\n## teacher_labels")
        data = np.load(dataset_path, allow_pickle=False)
        labels = np.asarray(data["labels"], dtype=int)
        candidates = np.asarray(data["candidate_masks"], dtype=bool)
        unique, counts = np.unique(labels, return_counts=True)
        for idx in np.argsort(-counts)[: int(args.top_k)]:
            action = int(unique[idx])
            print(f"{int(counts[idx]):5d} action={action:3d} mask={candidates[action].astype(int).tolist()}")
    rollouts = sorted(out_dir.glob("rollout_*.npz"))
    if rollouts:
        print("\n## rollout_masks")
    for path in rollouts:
        name = path.stem.removeprefix("rollout_")
        data = np.load(path, allow_pickle=False)
        losses = np.asarray(data["oracle_losses"], dtype=float)
        powers = np.asarray(data["powers"], dtype=float)
        masks = np.asarray(data["selected_masks"], dtype=int)
        unique_masks, counts = np.unique(masks, axis=0, return_counts=True)
        print(
            f"\n{name}: loss={float(np.nanmean(losses)):.6f} "
            f"clip_rate={float(np.mean(losses >= 9.999)):.3f} "
            f"power={float(np.nanmean(powers)):.6f}"
        )
        for idx in np.argsort(-counts)[: int(args.top_k)]:
            print(f"{int(counts[idx]):5d} mask={unique_masks[idx].astype(int).tolist()}")


if __name__ == "__main__":
    main()
