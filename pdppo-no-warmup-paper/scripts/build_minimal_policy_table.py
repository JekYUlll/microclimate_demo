#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable


POLICY_KEEP = {
    "custom_ppo",
    "dqn",
    "validation_selected_static",
    "feasible_static_projected",
    "round_robin",
    "aoi",
    "random",
    "duty_constrained_feasible_static_projected",
    "duty_constrained_round_robin",
    "duty_constrained_aoi",
    "duty_constrained_random",
    "custom_ppo_dwell6",
    "custom_ppo_dwell12",
    "dwell6_round_robin",
    "dwell6_aoi",
    "dwell6_random",
    "dwell12_round_robin",
    "dwell12_aoi",
    "dwell12_random",
    "duty_dwell6_round_robin",
    "duty_dwell6_aoi",
    "duty_dwell6_random",
    "duty_dwell12_round_robin",
    "duty_dwell12_aoi",
    "duty_dwell12_random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact no-warmup policy/run tables from existing CSV artifacts."
    )
    parser.add_argument("--framework-root", default="../rl_sensor_scheduling_framework")
    parser.add_argument("--out-dir", default="results/tables")
    return parser.parse_args()


def read_policy_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value != "" else None


def sfloat(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def sint(value: float | None) -> str:
    return "" if value is None else str(int(round(value)))


def parse_budget_seed(path: Path) -> tuple[float | None, int | None]:
    match = re.search(r"budget([0-9p]+)_seed(\d+)", str(path))
    if match:
        return float(match.group(1).replace("p", ".")), int(match.group(2))
    match = re.search(r"seed(\d+)", str(path))
    return None, int(match.group(1)) if match else None


def metadata_for(path: Path) -> dict:
    meta = path.with_name("v2_ppo_metadata.json")
    if not meta.exists():
        meta = path.with_name("v2_dqn_split_metadata.json")
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8"))


def budget_from_metadata(meta: dict) -> float | None:
    constraints = meta.get("constraints") or {}
    value = constraints.get("per_step_budget")
    return float(value) if value is not None else None


def source_run_from_metadata(meta: dict) -> str:
    return str(meta.get("source_run_dir") or "")


def policy_family(policy: str) -> str:
    if policy in {"custom_ppo", "dqn", "custom_ppo_dwell6", "custom_ppo_dwell12"}:
        return "learned"
    if policy in {"validation_selected_static", "feasible_static_projected"}:
        return "static"
    if "static" in policy:
        return "static_projected"
    if policy.startswith("duty_") or policy.startswith("dwell"):
        return "operational_dynamic"
    if policy in {"round_robin", "aoi"}:
        return "dynamic_heuristic"
    if policy == "random":
        return "random"
    return "other"


def duty_pass(row: dict[str, str]) -> bool | None:
    always_on = f(row, "always_on_sensor_count")
    always_off = f(row, "always_off_sensor_count")
    mid = f(row, "mid_duty_sensor_count")
    if always_on is None or always_off is None or mid is None:
        return None
    return always_on == 0 and always_off == 0 and mid >= 6


def iter_specs(root: Path) -> Iterable[tuple[str, str, Path]]:
    specs = [
        (
            "base_no_warmup",
            "split_train_eval",
            root / "reports/v31_split_protocol_no_warmup/raw",
        ),
        (
            "hard_duty_reduced",
            "split_train_eval",
            root / "reports/v31_split_protocol_no_warmup_hguard_reduced/raw",
        ),
        (
            "hard_duty_envdwell12_reduced",
            "split_train_eval",
            root / "reports/v31_split_protocol_no_warmup_hguard_envdwell12_reduced/raw",
        ),
        (
            "dqn_split_diagnostic",
            "split_train_eval",
            root / "reports/v31_no_warmup_dqn_split_diagnostic/raw",
        ),
    ]
    for track, evidence_type, raw in specs:
        if not raw.exists():
            continue
        for csv_path in sorted(raw.glob("budget*_seed*/v2_custom_ppo_metrics.csv")):
            yield track, evidence_type, csv_path
        for csv_path in sorted(raw.glob("budget*_seed*/v2_dqn_split_metrics.csv")):
            yield track, evidence_type, csv_path

    replay_specs = [
        ("env_dwell6_replay", root / "reports/v31_env_dwell6_operational_eval"),
        ("env_dwell12_replay", root / "reports/v31_env_dwell12_operational_eval"),
        ("switch_limited_replay", root / "reports/v31_switch_limited_operational_eval"),
    ]
    for track, directory in replay_specs:
        if not directory.exists():
            continue
        for csv_path in sorted(directory.glob("no_warmup_hguard_seed*/v2_custom_ppo_metrics.csv")):
            yield track, "operational_replay", csv_path


def build_policy_rows(framework_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for track, evidence_type, csv_path in iter_specs(framework_root):
        budget, seed = parse_budget_seed(csv_path)
        meta = metadata_for(csv_path)
        if budget is None:
            budget = budget_from_metadata(meta)
        if seed is None:
            seed = meta.get("seed")
        rows = [row for row in read_policy_csv(csv_path) if row.get("policy") in POLICY_KEEP]
        rows = [row for row in rows if f(row, "oracle_loss_mean") is not None]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda row: f(row, "oracle_loss_mean") or float("inf"))
        ranks = {row["policy"]: idx for idx, row in enumerate(ranked, start=1)}
        best_loss = f(ranked[0], "oracle_loss_mean")
        learned = next(
            (
                f(row, "oracle_loss_mean")
                for row in rows
                if row.get("policy") in {"custom_ppo", "dqn"}
            ),
            None,
        )
        for row in rows:
            loss = f(row, "oracle_loss_mean")
            out.append(
                {
                    "track": track,
                    "evidence_type": evidence_type,
                    "budget": sfloat(budget),
                    "seed": "" if seed is None else str(int(seed)),
                    "policy": row["policy"],
                    "policy_family": policy_family(row["policy"]),
                    "oracle_loss_mean": sfloat(loss),
                    "rank_by_loss": str(ranks[row["policy"]]),
                    "delta_to_best": sfloat(None if loss is None or best_loss is None else loss - best_loss),
                    "delta_to_learned": sfloat(None if loss is None or learned is None else loss - learned),
                    "switches_per_step": sfloat(f(row, "switches_per_step")),
                    "always_on_sensor_count": sint(f(row, "always_on_sensor_count")),
                    "always_off_sensor_count": sint(f(row, "always_off_sensor_count")),
                    "mid_duty_sensor_count": sint(f(row, "mid_duty_sensor_count")),
                    "duty_pass": "" if duty_pass(row) is None else str(duty_pass(row)),
                    "source_csv": str(csv_path.relative_to(framework_root)),
                    "source_run_dir": source_run_from_metadata(meta),
                }
            )
    return out


def run_gate_rows(policy_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in policy_rows:
        key = (row["track"], row["evidence_type"], row["budget"], row["seed"])
        groups.setdefault(key, []).append(row)

    out: list[dict[str, str]] = []
    for (track, evidence_type, budget, seed), rows in sorted(groups.items()):
        learned_rows = [row for row in rows if row["policy"] in {"custom_ppo", "dqn"}]
        if not learned_rows:
            continue
        learned = min(learned_rows, key=lambda row: float(row["oracle_loss_mean"]))
        learned_loss = float(learned["oracle_loss_mean"])
        lookup = {row["policy"]: row for row in rows}
        non_learned = [row for row in rows if row is not learned]
        best_non = min(non_learned, key=lambda row: float(row["oracle_loss_mean"])) if non_learned else None

        def loss(policy: str) -> float | None:
            row = lookup.get(policy)
            return float(row["oracle_loss_mean"]) if row else None

        def win(policy: str) -> str:
            value = loss(policy)
            return "" if value is None else str(learned_loss < value)

        out.append(
            {
                "track": track,
                "evidence_type": evidence_type,
                "budget": budget,
                "seed": seed,
                "learned_policy": learned["policy"],
                "learned_loss": learned["oracle_loss_mean"],
                "learned_rank": learned["rank_by_loss"],
                "learned_switches_per_step": learned["switches_per_step"],
                "learned_always_on": learned["always_on_sensor_count"],
                "learned_always_off": learned["always_off_sensor_count"],
                "learned_mid_duty": learned["mid_duty_sensor_count"],
                "learned_duty_pass": learned["duty_pass"],
                "validation_selected_static": sfloat(loss("validation_selected_static")),
                "feasible_static_projected": sfloat(loss("feasible_static_projected")),
                "round_robin": sfloat(loss("round_robin")),
                "aoi": sfloat(loss("aoi")),
                "random": sfloat(loss("random")),
                "best_non_learned_policy": "" if best_non is None else best_non["policy"],
                "best_non_learned_loss": "" if best_non is None else best_non["oracle_loss_mean"],
                "win_validation_selected_static": win("validation_selected_static"),
                "win_feasible_static_projected": win("feasible_static_projected"),
                "win_round_robin": win("round_robin"),
                "win_aoi": win("aoi"),
                "win_random": win("random"),
                "win_best_non_learned": "" if best_non is None else str(learned_loss < float(best_non["oracle_loss_mean"])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, gate_rows: list[dict[str, str]]) -> None:
    lines = ["# No-Warmup Minimal Gate Summary", ""]
    for track in sorted({row["track"] for row in gate_rows}):
        group = [row for row in gate_rows if row["track"] == track]
        split = [row for row in group if row["evidence_type"] == "split_train_eval"]
        replay = [row for row in group if row["evidence_type"] == "operational_replay"]
        rows = split or replay or group
        lines.extend([f"## {track}", "", f"- Runs: {len(rows)}"])
        for field, label in [
            ("win_validation_selected_static", "wins validation/static"),
            ("win_feasible_static_projected", "wins feasible static"),
            ("win_round_robin", "wins round-robin"),
            ("win_aoi", "wins AoI"),
            ("win_best_non_learned", "wins best non-learned"),
        ]:
            valid = [row for row in rows if row[field] != ""]
            wins = sum(row[field] == "True" for row in valid)
            lines.append(f"- {label}: {wins}/{len(valid)}")
        duty_valid = [row for row in rows if row["learned_duty_pass"] != ""]
        duty_wins = sum(row["learned_duty_pass"] == "True" for row in duty_valid)
        lines.append(f"- learned duty pass: {duty_wins}/{len(duty_valid)}")
        switches = [float(row["learned_switches_per_step"]) for row in rows if row["learned_switches_per_step"]]
        if switches:
            lines.append(f"- mean learned switches/step: {mean(switches):.6f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    framework_root = Path(args.framework_root).resolve()
    out_dir = Path(args.out_dir)
    policy_rows = build_policy_rows(framework_root)
    gate_rows = run_gate_rows(policy_rows)
    write_csv(out_dir / "no_warmup_minimal_policy_table.csv", policy_rows)
    write_csv(out_dir / "no_warmup_minimal_run_gate_table.csv", gate_rows)
    write_summary(out_dir / "no_warmup_minimal_gate_summary.md", gate_rows)
    print(f"wrote {len(policy_rows)} policy rows and {len(gate_rows)} run gates")


if __name__ == "__main__":
    main()
