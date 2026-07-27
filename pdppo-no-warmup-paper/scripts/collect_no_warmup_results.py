#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class RunRow:
    source: str
    budget: float
    seed: int
    custom_ppo_loss: float
    validation_static_loss: float | None
    feasible_static_loss: float | None
    round_robin_loss: float | None
    aoi_loss: float | None
    random_loss: float | None
    best_original_fair_policy: str
    best_original_fair_loss: float
    best_constrained_policy: str
    best_constrained_loss: float | None
    win_validation_static: bool
    win_feasible_static: bool
    win_best_original_fair: bool
    win_best_constrained: bool | None
    always_on_sensor_count: int | None
    always_off_sensor_count: int | None
    mid_duty_sensor_count: int | None
    switches_per_step: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate no-warmup PD-PPO result CSVs without running experiments."
    )
    parser.add_argument(
        "--framework-root",
        default="../rl_sensor_scheduling_framework",
        help="Path to rl_sensor_scheduling_framework containing reports/.",
    )
    parser.add_argument("--out-dir", default="results/tables")
    return parser.parse_args()


def read_policy_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["policy"]: row for row in csv.DictReader(handle)}


def maybe_float(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key, "")
    return float(value) if value != "" else None


def maybe_int(row: dict[str, str] | None, key: str) -> int | None:
    value = maybe_float(row, key)
    return int(value) if value is not None else None


def policy_loss(data: dict[str, dict[str, str]], policy: str) -> float | None:
    return maybe_float(data.get(policy), "oracle_loss_mean")


def merge_eval_fallback(data: dict[str, dict[str, str]], eval_path: Path) -> dict[str, dict[str, str]]:
    if not eval_path.exists():
        return data
    fallback = read_policy_csv(eval_path)
    merged: dict[str, dict[str, str]] = {}
    for policy in set(data) | set(fallback):
        row = dict(fallback.get(policy, {}))
        row.update({key: value for key, value in data.get(policy, {}).items() if value != ""})
        merged[policy] = row
    return merged


def iter_result_files(root: Path, rel: str) -> Iterable[Path]:
    raw = root / rel / "raw"
    if raw.exists():
        yield from sorted(raw.glob("budget*_seed*/v2_custom_ppo_metrics.csv"))


def parse_result(path: Path, *, source: str) -> RunRow | None:
    match = re.search(r"budget([0-9p]+)_seed(\d+)", str(path))
    if not match:
        return None
    budget = float(match.group(1).replace("p", "."))
    seed = int(match.group(2))
    data = merge_eval_fallback(read_policy_csv(path), path.parent / "evaluation" / "v2_eval_overall.csv")
    if "custom_ppo" not in data:
        return None

    custom = data["custom_ppo"]
    custom_loss = policy_loss(data, "custom_ppo")
    if custom_loss is None:
        return None

    original_fair = {
        policy: loss
        for policy in data
        if policy not in {"custom_ppo", "full_open_unconstrained"}
        and not policy.startswith("duty_constrained_")
        for loss in [policy_loss(data, policy)]
        if loss is not None
    }
    if not original_fair:
        return None
    best_original_policy, best_original_loss = min(original_fair.items(), key=lambda item: item[1])

    constrained = {
        policy: loss
        for policy in data
        if policy.startswith("duty_constrained_")
        for loss in [policy_loss(data, policy)]
        if loss is not None
    }
    if constrained:
        best_constrained_policy, best_constrained_loss = min(constrained.items(), key=lambda item: item[1])
    else:
        best_constrained_policy, best_constrained_loss = "", None

    validation_static = policy_loss(data, "validation_selected_static")
    feasible_static = policy_loss(data, "feasible_static_projected")

    return RunRow(
        source=source,
        budget=budget,
        seed=seed,
        custom_ppo_loss=custom_loss,
        validation_static_loss=validation_static,
        feasible_static_loss=feasible_static,
        round_robin_loss=policy_loss(data, "round_robin"),
        aoi_loss=policy_loss(data, "aoi"),
        random_loss=policy_loss(data, "random"),
        best_original_fair_policy=best_original_policy,
        best_original_fair_loss=best_original_loss,
        best_constrained_policy=best_constrained_policy,
        best_constrained_loss=best_constrained_loss,
        win_validation_static=validation_static is not None and custom_loss < validation_static,
        win_feasible_static=feasible_static is not None and custom_loss < feasible_static,
        win_best_original_fair=custom_loss < best_original_loss,
        win_best_constrained=(
            custom_loss < best_constrained_loss if best_constrained_loss is not None else None
        ),
        always_on_sensor_count=maybe_int(custom, "always_on_sensor_count"),
        always_off_sensor_count=maybe_int(custom, "always_off_sensor_count"),
        mid_duty_sensor_count=maybe_int(custom, "mid_duty_sensor_count"),
        switches_per_step=maybe_float(custom, "switches_per_step"),
    )


def write_csv(path: Path, rows: list[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def yes_count(rows: list[RunRow], field: str) -> int:
    return sum(1 for row in rows if getattr(row, field) is True)


def valid_values(rows: list[RunRow], field: str) -> list[float]:
    values = [getattr(row, field) for row in rows]
    return [float(value) for value in values if value is not None]


def write_budget_summary(path: Path, rows: list[RunRow]) -> None:
    fieldnames = [
        "source",
        "budget",
        "n",
        "win_validation_static",
        "win_feasible_static",
        "win_best_original_fair",
        "win_best_constrained",
        "mean_custom_ppo_loss",
        "mean_mid_duty",
        "mean_always_on",
        "mean_always_off",
        "mean_switches_per_step",
    ]
    groups: dict[tuple[str, float], list[RunRow]] = {}
    for row in rows:
        groups.setdefault((row.source, row.budget), []).append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (source, budget), group in sorted(groups.items()):
            constrained = [row for row in group if row.win_best_constrained is not None]
            writer.writerow(
                {
                    "source": source,
                    "budget": budget,
                    "n": len(group),
                    "win_validation_static": f"{yes_count(group, 'win_validation_static')}/{len(group)}",
                    "win_feasible_static": f"{yes_count(group, 'win_feasible_static')}/{len(group)}",
                    "win_best_original_fair": f"{yes_count(group, 'win_best_original_fair')}/{len(group)}",
                    "win_best_constrained": (
                        f"{yes_count(constrained, 'win_best_constrained')}/{len(constrained)}"
                        if constrained
                        else ""
                    ),
                    "mean_custom_ppo_loss": f"{mean(row.custom_ppo_loss for row in group):.6f}",
                    "mean_mid_duty": mean_or_blank(valid_values(group, "mid_duty_sensor_count")),
                    "mean_always_on": mean_or_blank(valid_values(group, "always_on_sensor_count")),
                    "mean_always_off": mean_or_blank(valid_values(group, "always_off_sensor_count")),
                    "mean_switches_per_step": mean_or_blank(valid_values(group, "switches_per_step")),
                }
            )


def mean_or_blank(values: list[float]) -> str:
    return f"{mean(values):.6f}" if values else ""


def write_claim_gate(path: Path, rows: list[RunRow]) -> None:
    lines = ["# Claim Gate Summary", ""]
    for source in sorted({row.source for row in rows}):
        group = [row for row in rows if row.source == source]
        constrained = [row for row in group if row.win_best_constrained is not None]
        lines.extend(
            [
                f"## {source}",
                "",
                f"- Runs: {len(group)}",
                f"- Static wins: {yes_count(group, 'win_validation_static')}/{len(group)}",
                f"- Best original fair wins: {yes_count(group, 'win_best_original_fair')}/{len(group)}",
                (
                    f"- Best constrained wins: {yes_count(constrained, 'win_best_constrained')}/{len(constrained)}"
                    if constrained
                    else "- Best constrained wins: not available"
                ),
                f"- Mean mid-duty sensors: {mean_or_blank(valid_values(group, 'mid_duty_sensor_count'))}",
                f"- Mean always-on sensors: {mean_or_blank(valid_values(group, 'always_on_sensor_count'))}",
                f"- Mean always-off sensors: {mean_or_blank(valid_values(group, 'always_off_sensor_count'))}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    framework_root = Path(args.framework_root).resolve()
    out_dir = Path(args.out_dir)

    specs = [
        ("base_no_warmup", "reports/v31_split_protocol_no_warmup"),
        ("hard_duty_reduced", "reports/v31_split_protocol_no_warmup_hguard_reduced"),
        (
            "hard_duty_envdwell12_reduced",
            "reports/v31_split_protocol_no_warmup_hguard_envdwell12_reduced",
        ),
    ]
    rows: list[RunRow] = []
    for source, rel in specs:
        for path in iter_result_files(framework_root, rel):
            row = parse_result(path, source=source)
            if row is not None:
                rows.append(row)

    write_csv(out_dir / "no_warmup_runs.csv", rows)
    write_budget_summary(out_dir / "no_warmup_budget_summary.csv", rows)
    write_claim_gate(out_dir / "claim_gate_summary.md", rows)
    print(f"wrote {len(rows)} runs to {out_dir}")


if __name__ == "__main__":
    main()
