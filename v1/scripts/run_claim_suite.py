#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "v1" / "scripts" / "run_protocol_gate.py"

TASK_COLUMNS = [
    "snow_mass_flux_kg_m2_s",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
]
TASK_SCALES = ["1e-4", "0.2", "5.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v1 multi-seed claim suite.")
    parser.add_argument(
        "--input-root",
        default="rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov",
        help="Directory containing per-seed inputs as <budget-tag>_seed<seed>/.",
    )
    parser.add_argument("--budget-tag", default="budget1p20")
    parser.add_argument("--out-root", default="v1/artifacts/claim_suite_semimarkov_n5")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=[
            "main",
            "main_safe",
            "safe_dagger3",
            "knn_safe",
            "knn_safe_dagger3",
            "cost_safe",
            "support1_safe",
            "support2_safe",
            "support3_safe",
            "support4_safe",
            "support5_safe",
            "support6_safe",
            "support8_safe",
            "support12_safe",
            "support_calib_safe",
            "mask_safe",
            "mask_anchor_safe",
            "hybrid_val_safe",
            "residual_safe",
            "oracle_context_safe",
            "value_residual_safe",
            "value_residual_no_dagger",
            "value_residual_oracle_objective",
            "learned_value_residual_safe",
            "learned_ensemble_value_safe",
            "learned_advantage_residual_safe",
            "learned_advantage_residual_calib_safe",
            "cost_support6_safe",
            "no_dagger",
            "oracle_objective",
            "no_anchor_guard",
        ],
        default=["main"],
    )
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--bc-device", default="cpu")
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--static-selection-steps", type=int, default=256)
    parser.add_argument("--static-selection-rollouts", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--gpu-ids", nargs="*", default=[])
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-rule-baselines", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    if args.dry_run:
        for item in commands:
            print(format_command(item.command, env=item.env))
        return
    run_commands(commands, max_parallel=resolve_parallelism(args), continue_on_error=bool(args.continue_on_error))


class CommandItem:
    def __init__(self, command: list[str], out_dir: Path, log_path: Path, env: dict[str, str]) -> None:
        self.command = command
        self.out_dir = out_dir
        self.log_path = log_path
        self.env = env


def build_commands(args: argparse.Namespace) -> list[CommandItem]:
    commands: list[CommandItem] = []
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    gpu_ids = [str(item) for item in args.gpu_ids]
    job_index = 0
    for preset in args.presets:
        for seed in args.seeds:
            seed_dir = input_root / f"{args.budget_tag}_seed{int(seed)}"
            truth_csv = seed_dir / "truth_energy_split.csv"
            oracle_path = seed_dir / "v2_tcn_oracle.pt"
            if not truth_csv.exists() or not oracle_path.exists():
                raise FileNotFoundError(f"Missing seed inputs under {seed_dir}")
            out_dir = out_root / f"{preset}_seed{int(seed)}"
            if bool(args.skip_existing) and (out_dir / "gate_summary.json").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[job_index % len(gpu_ids)]
            command = base_command(
                seed=int(seed),
                truth_csv=truth_csv,
                oracle_path=oracle_path,
                out_dir=out_dir,
                preset=str(preset),
                sensor_cfg=str(args.sensor_cfg),
                oracle_device=str(args.oracle_device),
                bc_device=str(args.bc_device),
                budget=float(args.budget),
                startup_peak_budget=float(args.startup_peak_budget),
                energy_capacity=float(args.energy_capacity),
                harvest_per_step=float(args.harvest_per_step),
                train_steps=int(args.train_steps),
                train_rollouts=int(args.train_rollouts),
                static_selection_steps=int(args.static_selection_steps),
                static_selection_rollouts=int(args.static_selection_rollouts),
                eval_steps=int(args.eval_steps),
                eval_rollouts=int(args.eval_rollouts),
                include_rule_baselines=bool(args.include_rule_baselines),
            )
            commands.append(CommandItem(command=command, out_dir=out_dir, log_path=out_dir / "run.log", env=env))
            job_index += 1
    return commands


def base_command(
    *,
    seed: int,
    truth_csv: Path,
    oracle_path: Path,
    out_dir: Path,
    preset: str,
    sensor_cfg: str,
    oracle_device: str,
    bc_device: str,
    budget: float,
    startup_peak_budget: float,
    energy_capacity: float,
    harvest_per_step: float,
    train_steps: int,
    train_rollouts: int,
    static_selection_steps: int,
    static_selection_rollouts: int,
    eval_steps: int,
    eval_rollouts: int,
    include_rule_baselines: bool,
) -> list[str]:
    common = [
        sys.executable,
        str(RUNNER),
        "--truth-csv",
        str(truth_csv),
        "--sensor-cfg",
        sensor_cfg,
        "--oracle-path",
        str(oracle_path),
        "--oracle-type",
        "tcn",
        "--oracle-device",
        oracle_device,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--freq-s",
        "10800",
        "--split-ratios",
        "0.30",
        "0.45",
        "0.125",
        "0.125",
        "--selection",
        "event_rich",
        "--selection-stride",
        "64",
        "--lookback",
        "20",
        "--horizon",
        "8",
        "--train-steps",
        str(int(train_steps)),
        "--train-rollouts",
        str(int(train_rollouts)),
        "--static-selection-steps",
        str(int(static_selection_steps)),
        "--static-selection-rollouts",
        str(int(static_selection_rollouts)),
        "--eval-steps",
        str(int(eval_steps)),
        "--eval-rollouts",
        str(int(eval_rollouts)),
        "--max-active",
        "4",
        "--budget",
        f"{float(budget):.6g}",
        "--startup-peak-budget",
        f"{float(startup_peak_budget):.6g}",
        "--energy-account",
        "--energy-capacity",
        f"{float(energy_capacity):.6g}",
        "--initial-energy",
        "180",
        "--harvest-per-step",
        f"{float(harvest_per_step):.6g}",
        "--reserve-energy",
        "20",
        "--lambda-warmup-abort",
        "0.08",
        "--planning-horizon",
        "3",
        "--beam-width",
        "4",
        "--max-branch",
        "8",
        "--teacher-lambda-warmup-abort",
        "0.16",
        "--candidate-prior-weight",
        "0.5",
        "--candidate-prefilter-top-k",
        "24",
        "--teacher-anchor-source",
        "validation_best",
        "--anchor-improvement-margin",
        "0.002",
        "--bc-epochs",
        "100",
        "--bc-hidden-dim",
        "256",
        "--bc-device",
        bc_device,
        "--bc-batch-size",
        "128",
    ]
    if include_rule_baselines:
        common.append("--include-rule-baselines")
    else:
        common.append("--no-include-rule-baselines")

    if preset in {
        "main",
        "main_safe",
        "safe_dagger3",
        "knn_safe",
        "knn_safe_dagger3",
        "cost_safe",
        "support1_safe",
        "support2_safe",
        "support3_safe",
        "support4_safe",
        "support5_safe",
        "support6_safe",
        "support8_safe",
        "support12_safe",
        "support_calib_safe",
        "mask_safe",
        "mask_anchor_safe",
        "hybrid_val_safe",
        "residual_safe",
        "oracle_context_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "cost_support6_safe",
        "no_dagger",
        "no_anchor_guard",
    }:
        common.extend(
            [
                "--objective-mode",
                "task_composite",
                "--task-error-weight",
                "0.2",
                "--task-error-columns",
                *TASK_COLUMNS,
                "--task-error-scales",
                *TASK_SCALES,
            ]
        )
    elif preset in {"oracle_objective", "value_residual_oracle_objective"}:
        common.extend(["--objective-mode", "oracle", "--task-error-weight", "0.0"])
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if preset == "no_anchor_guard":
        common.append("--no-anchor-regret-guard")
    else:
        common.append("--anchor-regret-guard")

    if preset in {
        "main_safe",
        "safe_dagger3",
        "knn_safe",
        "knn_safe_dagger3",
        "cost_safe",
        "support1_safe",
        "support2_safe",
        "support3_safe",
        "support4_safe",
        "support5_safe",
        "support6_safe",
        "support8_safe",
        "support12_safe",
        "support_calib_safe",
        "mask_safe",
        "mask_anchor_safe",
        "hybrid_val_safe",
        "residual_safe",
        "oracle_context_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "cost_support6_safe",
    }:
        common.append("--bc-preserve-warming")
    else:
        common.append("--no-bc-preserve-warming")

    support_top_k = {
        "support1_safe": 1,
        "support2_safe": 2,
        "support3_safe": 3,
        "support4_safe": 4,
        "support5_safe": 5,
        "support6_safe": 6,
        "support8_safe": 8,
        "support12_safe": 12,
        "cost_support6_safe": 6,
        "oracle_context_safe": 5,
        "value_residual_safe": 5,
        "value_residual_no_dagger": 5,
        "value_residual_oracle_objective": 5,
        "learned_value_residual_safe": 5,
        "learned_ensemble_value_safe": 8,
        "learned_advantage_residual_safe": 6,
        "learned_advantage_residual_calib_safe": 6,
    }.get(preset, 0)
    common.extend(["--bc-action-support-top-k", str(support_top_k)])
    if preset == "support_calib_safe":
        common.extend(["--bc-action-support-grid", "0", "4", "6", "8", "12"])
    if preset in {
        "mask_safe",
        "mask_anchor_safe",
        "residual_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
    }:
        common.append("--no-include-bc-policy")
    else:
        common.append("--include-bc-policy")

    if preset in {"knn_safe", "knn_safe_dagger3"}:
        common.extend(["--include-knn-policy", "--knn-k", "7"])
    else:
        common.append("--no-include-knn-policy")
    if preset in {"mask_safe", "mask_anchor_safe", "hybrid_val_safe"}:
        common.append("--include-mask-bc-policy")
        if preset in {"mask_anchor_safe", "hybrid_val_safe"}:
            common.extend(["--mask-bc-anchor-bias", "0.25"])
    else:
        common.append("--no-include-mask-bc-policy")
    if preset == "hybrid_val_safe":
        common.extend(["--deployable-selection", "validation"])
    if preset == "residual_safe":
        common.extend(
            [
                "--include-residual-bc-policy",
                "--residual-bc-support-top-k",
                "5",
                "--residual-deviation-threshold-grid",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.65",
                "0.8",
                "0.9",
                "0.98",
            ]
        )
    else:
        common.append("--no-include-residual-bc-policy")
    if preset == "oracle_context_safe":
        common.append("--forecast-truth-future")
    if preset in {
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
    }:
        common.extend(
            [
                "--include-value-residual-policy",
                "--value-residual-support-top-k",
                "5",
                "--value-residual-advantage-grid",
                "-1.0",
                "-0.5",
                "-0.2",
                "-0.1",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
                "1.0",
            ]
        )
    else:
        common.append("--no-include-value-residual-policy")
    if preset == "learned_ensemble_value_safe":
        common.extend(
            [
                "--include-ensemble-value-policy",
                "--ensemble-value-support-top-k",
                "8",
                "--ensemble-value-size",
                "5",
                "--ensemble-value-beta-grid",
                "0.0",
                "0.25",
                "0.5",
                "1.0",
                "--ensemble-value-advantage-grid",
                "-0.5",
                "-0.2",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
            ]
        )
    else:
        common.append("--no-include-ensemble-value-policy")
    if preset in {"learned_advantage_residual_safe", "learned_advantage_residual_calib_safe"}:
        common.extend(
            [
                "--include-advantage-residual-policy",
                "--advantage-residual-support-top-k",
                "6",
                "--advantage-residual-grid",
                "-0.2",
                "-0.1",
                "0.0",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.8",
                "1.0",
            ]
        )
        if preset == "learned_advantage_residual_calib_safe":
            common.extend(["--advantage-residual-support-grid", "3", "5", "6", "8", "12"])
    else:
        common.append("--no-include-advantage-residual-policy")
    if preset in {
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
    }:
        common.extend(
            [
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
            ]
        )
    if preset in {"cost_safe", "cost_support6_safe"}:
        common.extend(["--include-cost-policy", "--cost-epochs", "50", "--cost-hidden-dim", "256"])
    else:
        common.append("--no-include-cost-policy")

    if preset in {"no_dagger", "value_residual_no_dagger"}:
        common.extend(["--dagger-iters", "0"])
    elif preset in {"safe_dagger3", "knn_safe_dagger3"}:
        common.extend(["--dagger-iters", "3"])
    else:
        common.extend(["--dagger-iters", "1"])
    return common


def run_commands(commands: list[CommandItem], *, max_parallel: int, continue_on_error: bool) -> None:
    if not commands:
        print("[run_claim_suite] no pending jobs")
        return
    pending = list(commands)
    running: list[tuple[subprocess.Popen[bytes], CommandItem, object]] = []
    failures: list[tuple[CommandItem, int]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            item = pending.pop(0)
            item.out_dir.mkdir(parents=True, exist_ok=True)
            log_file = item.log_path.open("wb")
            log_file.write((format_command(item.command, env=item.env) + "\n\n").encode("utf-8"))
            log_file.flush()
            print(f"[run_claim_suite] start {item.out_dir}")
            process = subprocess.Popen(
                item.command,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=item.env,
            )
            running.append((process, item, log_file))
        time.sleep(5.0)
        still_running: list[tuple[subprocess.Popen[bytes], CommandItem, object]] = []
        for process, item, log_file in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, item, log_file))
                continue
            log_file.close()
            print(f"[run_claim_suite] done {item.out_dir} rc={return_code}")
            if return_code != 0:
                failures.append((item, int(return_code)))
                if not continue_on_error:
                    for other, _, other_log in still_running:
                        other.terminate()
                        other_log.close()
                    raise SystemExit(f"Job failed: {item.out_dir} rc={return_code}")
        running = still_running
    if failures:
        for item, code in failures:
            print(f"[run_claim_suite] failed {item.out_dir} rc={code}")
        raise SystemExit(1)


def resolve_parallelism(args: argparse.Namespace) -> int:
    if args.max_parallel is not None:
        return max(1, int(args.max_parallel))
    if args.gpu_ids:
        return max(1, len(args.gpu_ids))
    return 1


def format_command(command: list[str], *, env: dict[str, str]) -> str:
    prefix = ""
    if "CUDA_VISIBLE_DEVICES" in env:
        prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
    return prefix + " ".join(shell_quote(part) for part in command)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._/-:+=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
