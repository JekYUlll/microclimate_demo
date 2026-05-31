#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CLAIM_SUITE = ROOT / "v1" / "scripts" / "run_claim_suite.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a v1 preset across multiple budget constraints.")
    parser.add_argument("--out-root", default="v1/artifacts/strong_claim_budget_matrix")
    parser.add_argument("--preset", default="learned_ensemble_value_safe")
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.05, 1.20, 1.35])
    parser.add_argument(
        "--startup-peak-budgets",
        nargs="*",
        type=float,
        default=[],
        help="Optional per-budget startup peak limits. Defaults to budget + 0.40.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--gpu-ids", nargs="*", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--train-rollouts", type=int, default=None)
    parser.add_argument("--static-selection-steps", type=int, default=None)
    parser.add_argument("--static-selection-rollouts", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--eval-rollouts", type=int, default=None)
    parser.add_argument("--input-root", default="rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov")
    parser.add_argument("--input-budget-tag", default="budget1p20")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-rule-baselines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    startups = list(args.startup_peak_budgets)
    if startups and len(startups) != len(args.budgets):
        raise ValueError("--startup-peak-budgets must be empty or match --budgets length")
    if not startups:
        startups = [float(budget) + 0.40 for budget in args.budgets]
    commands = [
        build_command(args, budget=float(budget), startup=float(startup))
        for budget, startup in zip(args.budgets, startups, strict=True)
    ]
    for command in commands:
        print(format_command(command), flush=True)
        if not bool(args.dry_run):
            subprocess.run(command, cwd=str(ROOT), check=True)


def build_command(args: argparse.Namespace, *, budget: float, startup: float) -> list[str]:
    out_root = Path(args.out_root) / f"budget{budget_tag(budget)}"
    command = [
        sys.executable,
        str(CLAIM_SUITE),
        "--input-root",
        str(args.input_root),
        "--budget-tag",
        str(args.input_budget_tag),
        "--out-root",
        str(out_root),
        "--presets",
        str(args.preset),
        "--seeds",
        *[str(int(seed)) for seed in args.seeds],
        "--budget",
        f"{float(budget):.6g}",
        "--startup-peak-budget",
        f"{float(startup):.6g}",
        "--max-parallel",
        str(int(args.max_parallel)),
    ]
    if args.gpu_ids:
        command.extend(["--gpu-ids", *[str(x) for x in args.gpu_ids]])
    for arg_name, flag in (
        ("train_steps", "--train-steps"),
        ("train_rollouts", "--train-rollouts"),
        ("static_selection_steps", "--static-selection-steps"),
        ("static_selection_rollouts", "--static-selection-rollouts"),
        ("eval_steps", "--eval-steps"),
        ("eval_rollouts", "--eval-rollouts"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            command.extend([flag, str(int(value))])
    if bool(args.continue_on_error):
        command.append("--continue-on-error")
    else:
        command.append("--no-continue-on-error")
    if bool(args.include_rule_baselines):
        command.append("--include-rule-baselines")
    else:
        command.append("--no-include-rule-baselines")
    return command


def budget_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def format_command(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)


if __name__ == "__main__":
    main()
