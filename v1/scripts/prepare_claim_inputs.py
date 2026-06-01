#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RLSF_ROOT = ROOT / "rl_sensor_scheduling_framework"
RLSF_SRC = RLSF_ROOT / "src"
if str(RLSF_SRC) not in sys.path:
    sys.path.insert(0, str(RLSF_SRC))

TRUTH_BUILDER = RLSF_ROOT / "scripts" / "20_build_public_weather_truth.py"
HELPERS_PATH = RLSF_ROOT / "scripts" / "23_v2_train_ppo.py"
SPLIT_PATH = RLSF_ROOT / "scripts" / "61_energy_account_split_protocol_run.py"

TASK_TARGET_WEIGHTS = (0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 12.0, 8.0, 8.0)
TASK_TARGET_SCALES = (5.0, 5.0, 5.0, 1.0, 1.0, 100.0, 0.00005, 0.2, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare v1 claim-suite inputs without training archived PPO: "
            "per-seed truth CSV, split manifest, and frozen TCN oracle."
        )
    )
    parser.add_argument("--out-root", default="v1/artifacts/claim_inputs_semimarkov")
    parser.add_argument("--budget-tag", default="budget1p20")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--antaws-root", default="data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--truth-steps", type=int, default=90000)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.30, 0.45, 0.125, 0.125])
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--curriculum-context-steps", type=int, default=1024)
    parser.add_argument("--curriculum-rollouts", type=int, default=6)
    parser.add_argument("--static-selection-steps", type=int, default=1024)
    parser.add_argument("--static-selection-rollouts", type=int, default=6)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--final-selection", choices=["event_rich", "uniform"], default="event_rich")

    parser.add_argument("--event-coverage", type=float, default=0.30)
    parser.add_argument("--event-model", default="semi_markov")
    parser.add_argument("--min-duration", type=int, default=10)
    parser.add_argument("--max-duration", type=int, default=30)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--lead-steps", type=int, default=5)
    parser.add_argument("--wind-margin-ms", type=float, default=1.5)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.6)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.8)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.05)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=1.2)

    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)

    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="tcn")
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-event-fraction", type=float, default=0.50)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-learning-rate", type=float, default=1e-3)
    parser.add_argument("--oracle-channels", type=int, default=64)
    parser.add_argument("--oracle-levels", type=int, default=3)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-disable-mask-channels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helpers = load_module("v1_v2_train_helpers", HELPERS_PATH)
    split = load_module("v1_split_helpers", SPLIT_PATH)
    out_root = Path(args.out_root)
    for seed in args.seeds:
        prepare_seed(args, helpers=helpers, split=split, out_dir=out_root / f"{args.budget_tag}_seed{int(seed)}", seed=int(seed))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_seed(args: argparse.Namespace, *, helpers: ModuleType, split: ModuleType, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = out_dir / "truth_energy_split.csv"
    oracle_path = out_dir / ("v2_tcn_oracle.pt" if args.oracle_type == "tcn" else "v2_linear_oracle.npz")
    manifest_path = out_dir / "split_protocol_manifest.json"

    if bool(args.dry_run):
        print(f"[prepare_claim_inputs] seed={seed} out_dir={out_dir}")
        print(" ".join(build_truth_command(args, truth_path=truth_path, seed=seed)))
        print(f"[prepare_claim_inputs] oracle_path={oracle_path}")
        return

    if bool(args.force) or not truth_path.exists():
        run_truth_builder(args, truth_path=truth_path, seed=seed)

    truth = pd.read_csv(truth_path)
    if len(truth) != int(args.truth_steps):
        raise ValueError(f"{truth_path} has {len(truth)} rows, expected {args.truth_steps}")

    bounds = split.partition_bounds(int(args.truth_steps), tuple(float(value) for value in args.split_ratios))
    starts = select_protocol_starts(args, split=split, truth=truth, bounds=bounds, seed=seed)
    write_manifest(
        args,
        manifest_path=manifest_path,
        truth_path=truth_path,
        oracle_path=oracle_path,
        bounds=bounds,
        starts=starts,
        seed=seed,
    )

    if bool(args.force) or not oracle_path.exists():
        train_and_save_oracle(args, helpers=helpers, truth=truth, bounds=bounds, oracle_path=oracle_path, seed=seed)
    print(json.dumps({"seed": int(seed), "out_dir": str(out_dir), "truth_csv": str(truth_path), "oracle_path": str(oracle_path)}))


def build_truth_command(args: argparse.Namespace, *, truth_path: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(TRUTH_BUILDER),
        "--antaws-root",
        resolve_antaws_root(str(args.antaws_root)),
        "--stations",
        *[str(station) for station in args.stations],
        "--steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--seed",
        str(int(seed)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        str(args.event_model),
        "--blowing-snow-min-duration-steps",
        str(int(args.min_duration)),
        "--blowing-snow-max-duration-steps",
        str(int(args.max_duration)),
        "--blowing-snow-min-gap-steps",
        str(int(args.min_gap)),
        "--blowing-snow-lead-steps",
        str(int(args.lead_steps)),
        "--blowing-snow-wind-margin-ms",
        str(float(args.wind_margin_ms)),
        "--cred-hysteresis-on",
        str(float(args.cred_hysteresis_on)),
        "--cred-hysteresis-off",
        str(float(args.cred_hysteresis_off)),
        "--flux-wind-exponent",
        str(float(args.flux_wind_exponent)),
        "--event-microstructure-sigma",
        str(float(args.event_microstructure_sigma)),
        "--event-microstructure-alpha",
        str(float(args.event_microstructure_alpha)),
        "--event-microstructure-diameter-scale",
        str(float(args.event_microstructure_diameter_scale)),
        "--event-microstructure-velocity-scale",
        str(float(args.event_microstructure_velocity_scale)),
        "--out",
        str(truth_path),
        "--report-dir",
        str(truth_path.parent / "dataset_validation"),
    ]


def run_truth_builder(args: argparse.Namespace, *, truth_path: Path, seed: int) -> None:
    if truth_path.exists():
        truth_path.unlink()
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_truth_command(args, truth_path=truth_path, seed=seed), cwd=str(ROOT), check=True)


def select_protocol_starts(
    args: argparse.Namespace,
    *,
    split: ModuleType,
    truth: pd.DataFrame,
    bounds: dict[str, tuple[int, int]],
    seed: int,
) -> dict[str, object]:
    curriculum_starts, curriculum_diag = split.event_rich_non_overlapping_starts(
        truth,
        bounds=bounds["rl_train"],
        window_steps=int(args.curriculum_context_steps),
        horizon=int(args.horizon),
        count=int(args.curriculum_rollouts),
        stride=int(args.selection_stride),
    )
    validation_starts, validation_diag = split.event_rich_non_overlapping_starts(
        truth,
        bounds=bounds["validation"],
        window_steps=int(args.static_selection_steps),
        horizon=int(args.horizon),
        count=int(args.static_selection_rollouts),
        stride=int(args.selection_stride),
    )
    if str(args.final_selection) == "event_rich":
        final_starts, final_diag = split.event_rich_non_overlapping_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=int(args.horizon),
            count=int(args.eval_rollouts),
            stride=int(args.selection_stride),
        )
    else:
        final_starts = split.random_non_overlapping_starts(
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=int(args.horizon),
            count=int(args.eval_rollouts),
            seed=int(seed) + 1777,
        )
        flags = truth["event_flag"].astype(bool).to_numpy()
        final_diag = {
            "selection": "uniform_random_non_overlapping_within_declared_partition",
            "selected_event_rates": [
                float(np.mean(flags[value : value + int(args.eval_steps)])) for value in final_starts
            ],
        }
    return {
        "curriculum_starts": tuple(int(value) for value in curriculum_starts),
        "curriculum_diag": curriculum_diag,
        "validation_starts": tuple(int(value) for value in validation_starts),
        "validation_diag": validation_diag,
        "final_starts": tuple(int(value) for value in final_starts),
        "final_diag": final_diag,
    }


def write_manifest(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    truth_path: Path,
    oracle_path: Path,
    bounds: dict[str, tuple[int, int]],
    starts: dict[str, object],
    seed: int,
) -> None:
    manifest = {
        "protocol": "v1_claim_input_forecast_oracle_only",
        "evidence_role": "v1_scheduler_claim_input_no_archived_ppo_training",
        "truth_csv": str(truth_path),
        "oracle_path": str(oracle_path),
        "truth_steps": int(args.truth_steps),
        "seed": int(seed),
        "split_ratios": [float(value) for value in args.split_ratios],
        "partitions": {name: [int(start), int(end)] for name, (start, end) in bounds.items()},
        "oracle_pretrain": {"range": [int(x) for x in bounds["oracle_pretrain"]]},
        "rl_train": {
            "range": [int(x) for x in bounds["rl_train"]],
            "normalization_range": [int(x) for x in bounds["rl_train"]],
            "curriculum_starts": [int(x) for x in starts["curriculum_starts"]],
            "context_selection_steps": int(args.curriculum_context_steps),
            **dict(starts["curriculum_diag"]),
        },
        "validation": {
            "static_selection_starts": [int(x) for x in starts["validation_starts"]],
            "static_selection_steps": int(args.static_selection_steps),
            **dict(starts["validation_diag"]),
        },
        "final_test": {
            "eval_starts": [int(x) for x in starts["final_starts"]],
            "eval_steps": int(args.eval_steps),
            **dict(starts["final_diag"]),
        },
        "energy_account": {
            "budget": float(args.budget),
            "startup_peak_budget": float(args.startup_peak_budget),
            "capacity": float(args.energy_capacity),
            "initial_energy": float(args.initial_energy),
            "harvest_per_step": float(args.harvest_per_step),
            "reserve_energy": float(args.reserve_energy),
        },
        "truth_event_design": {
            "blowing_snow_event_coverage": float(args.event_coverage),
            "blowing_snow_event_model": str(args.event_model),
            "blowing_snow_min_duration_steps": int(args.min_duration),
            "blowing_snow_max_duration_steps": int(args.max_duration),
            "blowing_snow_min_gap_steps": int(args.min_gap),
            "blowing_snow_lead_steps": int(args.lead_steps),
            "blowing_snow_wind_margin_ms": float(args.wind_margin_ms),
            "cred_hysteresis_on": float(args.cred_hysteresis_on),
            "cred_hysteresis_off": float(args.cred_hysteresis_off),
            "flux_wind_exponent": float(args.flux_wind_exponent),
            "event_microstructure_sigma": float(args.event_microstructure_sigma),
            "event_microstructure_alpha": float(args.event_microstructure_alpha),
            "event_microstructure_diameter_scale": float(args.event_microstructure_diameter_scale),
            "event_microstructure_velocity_scale": float(args.event_microstructure_velocity_scale),
        },
        "oracle": {
            "type": str(args.oracle_type),
            "lookback": int(args.lookback),
            "horizon": int(args.horizon),
            "rollout_steps": int(args.oracle_rollout_steps),
            "rollouts_per_policy": int(args.oracle_rollouts_per_policy),
            "event_fraction": float(args.oracle_event_fraction),
            "full_open_repeat": int(args.oracle_full_open_repeat),
            "epochs": int(args.oracle_epochs),
            "batch_size": int(args.oracle_batch_size),
            "learning_rate": float(args.oracle_learning_rate),
            "channels": int(args.oracle_channels),
            "levels": int(args.oracle_levels),
            "loss_clip": float(args.oracle_loss_clip),
            "use_mask_channels": not bool(args.oracle_disable_mask_channels),
            "target_weights": list(TASK_TARGET_WEIGHTS),
            "target_scales": list(TASK_TARGET_SCALES),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_and_save_oracle(
    args: argparse.Namespace,
    *,
    helpers: ModuleType,
    truth: pd.DataFrame,
    bounds: dict[str, tuple[int, int]],
    oracle_path: Path,
    seed: int,
) -> None:
    sensor_cfg = resolve_sensor_cfg(str(args.sensor_cfg))
    sensors = helpers.load_sensor_specs(sensor_cfg)
    constraints = helpers.PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=("met_station_core",),
        coverage_groups=(),
    )
    start, end = bounds["oracle_pretrain"]
    oracle_truth = truth.iloc[int(start) : int(end)].reset_index(drop=True)
    oracle = helpers.train_oracle(
        oracle_truth,
        sensors,
        constraints,
        oracle_type=str(args.oracle_type),
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        rollout_steps=int(args.oracle_rollout_steps),
        tcn_epochs=int(args.oracle_epochs),
        tcn_batch_size=int(args.oracle_batch_size),
        tcn_lr=float(args.oracle_learning_rate),
        tcn_channels=int(args.oracle_channels),
        tcn_levels=int(args.oracle_levels),
        tcn_device=str(args.oracle_device),
        tcn_loss_clip=float(args.oracle_loss_clip),
        tcn_use_mask_channels=not bool(args.oracle_disable_mask_channels),
        target_weights=tuple(float(x) for x in TASK_TARGET_WEIGHTS),
        target_scales=tuple(float(x) for x in TASK_TARGET_SCALES),
        rollouts_per_policy=int(args.oracle_rollouts_per_policy),
        event_fraction=float(args.oracle_event_fraction),
        full_open_repeat=int(args.oracle_full_open_repeat),
        base_freq_s=int(args.freq_s),
        seed=int(seed),
    )
    oracle.save(str(oracle_path))


def resolve_sensor_cfg(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path)
    candidate = RLSF_ROOT / value
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Cannot resolve sensor config: {value}")


def resolve_antaws_root(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    candidates = [
        ROOT / value,
        RLSF_ROOT / value,
        RLSF_ROOT / ".." / value,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Cannot resolve AntAWS root: {value}")


if __name__ == "__main__":
    main()
