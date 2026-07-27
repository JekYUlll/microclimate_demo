#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect no-warmup split DQN diagnostic results.")
    parser.add_argument("--framework-root", default="../rl_sensor_scheduling_framework")
    parser.add_argument("--dqn-root", default="reports/v31_no_warmup_dqn_split_diagnostic")
    parser.add_argument("--source-root", default="reports/v31_split_protocol_no_warmup")
    parser.add_argument("--out-dir", default="results/tables")
    return parser.parse_args()


def read_policy_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["policy"]: row for row in csv.DictReader(handle)}


def loss(data: dict[str, dict[str, str]], policy: str) -> float | None:
    row = data.get(policy)
    if not row:
        return None
    value = row.get("oracle_loss_mean", "")
    return float(value) if value != "" else None


def fvalue(data: dict[str, dict[str, str]], policy: str, key: str) -> float | None:
    row = data.get(policy)
    if not row:
        return None
    value = row.get(key, "")
    return float(value) if value != "" else None


def main() -> None:
    args = parse_args()
    framework_root = Path(args.framework_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((framework_root / args.dqn_root / "raw").glob("budget*_seed*/v2_dqn_split_metrics.csv")):
        match = re.search(r"budget([0-9p]+)_seed(\d+)", str(path))
        if not match:
            continue
        budget = float(match.group(1).replace("p", "."))
        seed = int(match.group(2))
        label = f"budget{match.group(1)}_seed{seed}"
        dqn_data = read_policy_csv(path)
        source_path = framework_root / args.source_root / "raw" / label / "v2_custom_ppo_metrics.csv"
        source_data = read_policy_csv(source_path) if source_path.exists() else {}
        dqn_loss = loss(dqn_data, "dqn")
        if dqn_loss is None:
            continue
        ppo_loss = loss(source_data, "custom_ppo")
        fair = {
            policy: value
            for policy in dqn_data
            if policy not in {"dqn", "full_open_unconstrained"}
            for value in [loss(dqn_data, policy)]
            if value is not None
        }
        best_policy, best_loss = min(fair.items(), key=lambda item: item[1])
        rows.append(
            {
                "budget": budget,
                "seed": seed,
                "dqn_loss": dqn_loss,
                "source_ppo_loss": ppo_loss,
                "validation_static_loss": loss(dqn_data, "validation_selected_static"),
                "round_robin_loss": loss(dqn_data, "round_robin"),
                "aoi_loss": loss(dqn_data, "aoi"),
                "random_loss": loss(dqn_data, "random"),
                "best_non_dqn_policy": best_policy,
                "best_non_dqn_loss": best_loss,
                "win_source_ppo": ppo_loss is not None and dqn_loss < ppo_loss,
                "win_validation_static": (
                    loss(dqn_data, "validation_selected_static") is not None
                    and dqn_loss < loss(dqn_data, "validation_selected_static")
                ),
                "win_round_robin": loss(dqn_data, "round_robin") is not None and dqn_loss < loss(dqn_data, "round_robin"),
                "win_aoi": loss(dqn_data, "aoi") is not None and dqn_loss < loss(dqn_data, "aoi"),
                "win_best_non_dqn": dqn_loss < best_loss,
                "always_on_sensor_count": fvalue(dqn_data, "dqn", "always_on_sensor_count"),
                "always_off_sensor_count": fvalue(dqn_data, "dqn", "always_off_sensor_count"),
                "mid_duty_sensor_count": fvalue(dqn_data, "dqn", "mid_duty_sensor_count"),
                "switches_per_step": fvalue(dqn_data, "dqn", "switches_per_step"),
            }
        )

    fields = list(rows[0]) if rows else [
        "budget",
        "seed",
        "dqn_loss",
        "source_ppo_loss",
        "validation_static_loss",
        "round_robin_loss",
        "aoi_loss",
        "random_loss",
        "best_non_dqn_policy",
        "best_non_dqn_loss",
        "win_source_ppo",
        "win_validation_static",
        "win_round_robin",
        "win_aoi",
        "win_best_non_dqn",
        "always_on_sensor_count",
        "always_off_sensor_count",
        "mid_duty_sensor_count",
        "switches_per_step",
    ]
    with (out_dir / "dqn_split_diagnostic_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# DQN Split Diagnostic Summary", ""]
    if rows:
        lines.extend(
            [
                f"- Runs: {len(rows)}",
                f"- Wins source PD-PPO: {sum(row['win_source_ppo'] for row in rows)}/{len(rows)}",
                f"- Wins validation/static: {sum(row['win_validation_static'] for row in rows)}/{len(rows)}",
                f"- Wins round-robin: {sum(row['win_round_robin'] for row in rows)}/{len(rows)}",
                f"- Wins AoI: {sum(row['win_aoi'] for row in rows)}/{len(rows)}",
                f"- Wins best non-DQN: {sum(row['win_best_non_dqn'] for row in rows)}/{len(rows)}",
                f"- Mean mid-duty sensors: {mean(row['mid_duty_sensor_count'] for row in rows if row['mid_duty_sensor_count'] is not None):.3f}",
                f"- Mean always-on sensors: {mean(row['always_on_sensor_count'] for row in rows if row['always_on_sensor_count'] is not None):.3f}",
                f"- Mean always-off sensors: {mean(row['always_off_sensor_count'] for row in rows if row['always_off_sensor_count'] is not None):.3f}",
            ]
        )
    else:
        lines.append("- Runs: 0")
    (out_dir / "dqn_split_diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} DQN diagnostic rows")


if __name__ == "__main__":
    main()
