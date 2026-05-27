#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


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
            "no_dagger",
            "oracle_objective",
            "no_anchor_guard",
        ],
        default=["main"],
    )
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--bc-device", default="cpu")
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
        "128",
        "--train-rollouts",
        "4",
        "--static-selection-steps",
        "256",
        "--static-selection-rollouts",
        "4",
        "--eval-steps",
        "256",
        "--eval-rollouts",
        "4",
        "--max-active",
        "4",
        "--budget",
        "1.20",
        "--startup-peak-budget",
        "1.60",
        "--energy-account",
        "--energy-capacity",
        "180",
        "--initial-energy",
        "180",
        "--harvest-per-step",
        "0.92",
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
    elif preset == "oracle_objective":
        common.extend(["--objective-mode", "oracle", "--task-error-weight", "0.0"])
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if preset == "no_anchor_guard":
        common.append("--no-anchor-regret-guard")
    else:
        common.append("--anchor-regret-guard")

    if preset in {"main_safe", "safe_dagger3", "knn_safe", "knn_safe_dagger3"}:
        common.append("--bc-preserve-warming")
    else:
        common.append("--no-bc-preserve-warming")

    if preset in {"knn_safe", "knn_safe_dagger3"}:
        common.extend(["--include-knn-policy", "--knn-k", "7"])
    else:
        common.append("--no-include-knn-policy")

    if preset == "no_dagger":
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
