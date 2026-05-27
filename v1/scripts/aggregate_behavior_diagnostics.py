#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sensor-behavior diagnostics from v1 claim-suite rollouts.")
    parser.add_argument("suite_root")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root)
    out_dir = Path(args.out_dir) if args.out_dir else suite_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, object]] = []
    sensor_rows: list[dict[str, object]] = []
    for run_dir in sorted(suite_root.glob("*_seed*")):
        if not run_dir.is_dir():
            continue
        preset, seed = parse_preset_seed(run_dir.name)
        for rollout_path in sorted(run_dir.glob("rollout_*.npz")):
            policy, policy_summary, sensor_summary = summarize_rollout(rollout_path)
            policy_summary.update({"preset": preset, "seed": seed, "policy": policy, "run_dir": str(run_dir)})
            policy_rows.append(policy_summary)
            for row in sensor_summary:
                row.update({"preset": preset, "seed": seed, "policy": policy, "run_dir": str(run_dir)})
                sensor_rows.append(row)

    if not policy_rows:
        raise FileNotFoundError(f"No rollout_*.npz files found under {suite_root}")
    policy_df = pd.DataFrame(policy_rows).sort_values(["preset", "seed", "policy"])
    sensor_df = pd.DataFrame(sensor_rows).sort_values(["preset", "seed", "policy", "sensor_id"])
    policy_df.to_csv(out_dir / "behavior_policy.csv", index=False)
    sensor_df.to_csv(out_dir / "behavior_sensor.csv", index=False)

    policy_summary = (
        policy_df.groupby(["preset", "policy"], as_index=False)
        .agg(
            n=("seed", "count"),
            power_mean=("power_mean", "mean"),
            soc_min_mean=("soc_min", "mean"),
            warmup_abort_count_mean=("warmup_abort_count", "mean"),
            switch_rate_mean=("switches_per_step", "mean"),
            event_rate_mean=("event_rate", "mean"),
        )
        .sort_values(["preset", "policy"])
    )
    sensor_summary = (
        sensor_df.groupby(["preset", "policy", "sensor_id"], as_index=False)
        .agg(
            active_rate_mean=("active_rate", "mean"),
            event_active_rate_mean=("event_active_rate", "mean"),
            non_event_active_rate_mean=("non_event_active_rate", "mean"),
            event_non_event_ratio_mean=("event_non_event_ratio", "mean"),
            switches_per_step_mean=("switches_per_step", "mean"),
        )
        .sort_values(["preset", "policy", "sensor_id"])
    )
    policy_summary.to_csv(out_dir / "behavior_policy_summary.csv", index=False)
    sensor_summary.to_csv(out_dir / "behavior_sensor_summary.csv", index=False)
    print(f"wrote behavior diagnostics to {out_dir}")


def summarize_rollout(path: Path) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    data = np.load(path, allow_pickle=False)
    policy = str(np.asarray(data["policy"]).reshape(-1)[0]) if "policy" in data.files else path.stem.removeprefix("rollout_")
    masks = np.asarray(data["selected_masks"], dtype=float)
    events = np.asarray(data["event_flags"], dtype=bool).reshape(-1)
    powers = np.asarray(data["powers"], dtype=float).reshape(-1)
    soc = np.asarray(data["soc"], dtype=float).reshape(-1) if "soc" in data.files else np.asarray([], dtype=float)
    aborts = np.asarray(data["warmup_abort_deltas"], dtype=float).reshape(-1)
    sensor_ids = [str(item) for item in np.asarray(data["sensor_ids"]).reshape(-1)]
    switches = np.abs(np.diff(masks, axis=0)) if masks.shape[0] > 1 else np.zeros((0, masks.shape[1]))
    policy_summary = {
        "steps": int(masks.shape[0]),
        "event_rate": float(np.mean(events)) if events.size else np.nan,
        "power_mean": float(np.nanmean(powers)) if powers.size else np.nan,
        "power_max": float(np.nanmax(powers)) if powers.size else np.nan,
        "soc_min": float(np.nanmin(soc)) if soc.size else np.nan,
        "soc_mean": float(np.nanmean(soc)) if soc.size else np.nan,
        "warmup_abort_count": int(np.nansum(aborts)) if aborts.size else 0,
        "switches_per_step": float(np.sum(switches) / max(1, masks.shape[0])),
    }
    sensor_rows: list[dict[str, object]] = []
    for idx, sensor_id in enumerate(sensor_ids):
        active = masks[:, idx].astype(bool)
        event_active = active[events] if np.any(events) else np.asarray([], dtype=bool)
        non_event_active = active[~events] if np.any(~events) else np.asarray([], dtype=bool)
        event_rate = float(np.mean(event_active)) if event_active.size else np.nan
        non_event_rate = float(np.mean(non_event_active)) if non_event_active.size else np.nan
        if np.isfinite(event_rate) and np.isfinite(non_event_rate) and non_event_rate > 0.0:
            ratio = event_rate / non_event_rate
        else:
            ratio = np.nan
        sensor_rows.append(
            {
                "sensor_id": sensor_id,
                "active_rate": float(np.mean(active)),
                "event_active_rate": event_rate,
                "non_event_active_rate": non_event_rate,
                "event_non_event_ratio": ratio,
                "switches_per_step": float(np.sum(switches[:, idx]) / max(1, masks.shape[0])),
            }
        )
    return policy, policy_summary, sensor_rows


def parse_preset_seed(name: str) -> tuple[str, int]:
    match = re.match(r"(?P<preset>.+)_seed(?P<seed>\d+)$", name)
    if not match:
        return name, -1
    return str(match.group("preset")), int(match.group("seed"))


if __name__ == "__main__":
    main()
