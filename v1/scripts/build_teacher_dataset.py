#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.archived_v2 import (
    build_warmup_env,
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
    resolve_archive_path,
)
from forecast_cmdp.dataset import collect_teacher_dataset
from forecast_cmdp.features import ForecastContextConfig
from forecast_cmdp.mpc_teacher import MpcTeacherConfig, enumerate_action_masks
from forecast_cmdp.protocol import choose_non_overlapping_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forecast-aware MPC-teacher labels using archived v2 env.")
    parser.add_argument("--truth-csv", required=True)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--oracle-path", default=None)
    parser.add_argument("--oracle-type", choices=["tcn", "linear", "none"], default="tcn")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--out-npz", default="v1/artifacts/teacher_dataset.npz")
    parser.add_argument("--out-metadata", default=None)

    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--steps-per-start", type=int, default=256)
    parser.add_argument("--start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--split-start", type=int, default=None)
    parser.add_argument("--split-end", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--selection", choices=["event_rich", "uniform"], default="event_rich")
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--event-column", default="event_flag")

    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--energy-account", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--normalization-start-idx", type=int, default=None)
    parser.add_argument("--normalization-end-idx", type=int, default=None)

    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-branch", type=int, default=16)
    parser.add_argument("--event-weight-alpha", type=float, default=1.0)
    parser.add_argument("--teacher-lambda-warmup-abort", type=float, default=0.16)
    parser.add_argument("--teacher-lambda-switch", type=float, default=0.002)
    parser.add_argument("--teacher-lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--saturated-loss-threshold", type=float, default=9.999)
    parser.add_argument("--saturated-coverage-bonus", type=float, default=0.25)

    parser.add_argument("--forecast-horizon", type=int, default=None)
    parser.add_argument(
        "--truth-future-features",
        action="store_true",
        help="Use only for teacher diagnostics; BC policy inputs should normally stay causal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helpers = load_v2_helpers()

    truth_path = resolve_archive_path(args.truth_csv)
    truth = pd.read_csv(truth_path)
    sensors = load_archived_sensor_specs(args.sensor_cfg)
    oracle = load_archived_oracle(args.oracle_path, oracle_type=str(args.oracle_type), device=str(args.oracle_device))

    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=args.normalization_start_idx,
        end_idx=args.normalization_end_idx,
    )
    constraints = make_constraints(
        max_active=int(args.max_active),
        budget=float(args.budget),
        startup_peak_budget=float(args.startup_peak_budget),
    )
    env_cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(args.lookback),
        episode_len=int(args.steps_per_start),
        seed=int(args.seed),
        freq_s=int(args.freq_s),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
        event_reward_multiplier=float(args.event_reward_multiplier),
        energy_account=bool(args.energy_account),
        energy_capacity=float(args.energy_capacity),
        initial_energy=float(args.initial_energy),
        harvest_per_step=float(args.harvest_per_step),
        reserve_energy=float(args.reserve_energy),
        lambda_energy_deficit=float(args.lambda_energy_deficit),
        soc_soft_penalty_buffer=float(args.soc_soft_penalty_buffer),
        lambda_soc_soft_penalty=float(args.lambda_soc_soft_penalty),
    )
    env = build_warmup_env(truth=truth, sensors=sensors, constraints=constraints, cfg=env_cfg, oracle=oracle)
    candidate_masks = enumerate_action_masks(len(sensors), max_active=int(args.max_active))
    starts, selection_diag = choose_starts(args, truth)
    teacher_cfg = MpcTeacherConfig(
        planning_horizon=int(args.planning_horizon),
        beam_width=int(args.beam_width),
        max_branch=int(args.max_branch),
        event_weight_alpha=float(args.event_weight_alpha),
        lambda_warmup_abort=float(args.teacher_lambda_warmup_abort),
        lambda_switch=float(args.teacher_lambda_switch),
        lambda_energy_deficit=float(args.teacher_lambda_energy_deficit),
        saturated_loss_threshold=float(args.saturated_loss_threshold),
        saturated_coverage_bonus=float(args.saturated_coverage_bonus),
    )
    forecast_cfg = ForecastContextConfig(
        horizon=int(args.forecast_horizon or args.horizon),
        event_column=str(args.event_column),
        truth_future=bool(args.truth_future_features),
    )
    dataset = collect_teacher_dataset(
        env,
        candidate_masks,
        start_indices=starts,
        steps_per_start=int(args.steps_per_start),
        teacher_cfg=teacher_cfg,
        forecast_cfg=forecast_cfg,
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_npz(str(out_npz))
    metadata_path = Path(args.out_metadata) if args.out_metadata else out_npz.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "role": "v1_forecast_cmdp_teacher_dataset",
        "truth_csv": str(truth_path),
        "sensor_cfg": str(resolve_archive_path(args.sensor_cfg)),
        "oracle_path": str(resolve_archive_path(args.oracle_path)) if args.oracle_path else None,
        "oracle_type": str(args.oracle_type),
        "sample_count": int(dataset.features.shape[0]),
        "feature_dim": int(dataset.features.shape[1]),
        "candidate_count": int(dataset.candidate_masks.shape[0]),
        "candidate_masks": dataset.candidate_masks.astype(int).tolist(),
        "start_indices": [int(x) for x in starts],
        "selection": selection_diag,
        "steps_per_start": int(args.steps_per_start),
        "event_rate": float(np.mean(dataset.event_flags)),
        "constraints": {
            "max_active": int(args.max_active),
            "budget": float(args.budget),
            "startup_peak_budget": float(args.startup_peak_budget),
        },
        "teacher_cfg": teacher_cfg.__dict__,
        "forecast_cfg": forecast_cfg.__dict__,
        "deployed_policy_feature_warning": (
            "truth_future_features is diagnostic-only and should not be used for deployment"
            if bool(args.truth_future_features)
            else "features use causal forecast heuristic"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_npz": str(out_npz),
                "metadata": str(metadata_path),
                "sample_count": int(dataset.features.shape[0]),
                "feature_dim": int(dataset.features.shape[1]),
                "candidate_count": int(dataset.candidate_masks.shape[0]),
                "event_rate": float(np.mean(dataset.event_flags)),
            },
            ensure_ascii=False,
        )
    )


def choose_starts(args: argparse.Namespace, truth: pd.DataFrame) -> tuple[tuple[int, ...], dict[str, object]]:
    if args.start_indices:
        starts = tuple(int(x) for x in args.start_indices)
        return starts, {"selection": "explicit", "count": len(starts)}
    start = int(args.split_start if args.split_start is not None else 0)
    end = int(args.split_end if args.split_end is not None else len(truth))
    if end <= start:
        raise ValueError(f"Invalid split [{start}, {end})")
    selected = choose_non_overlapping_starts(
        truth,
        bounds=(start, end),
        window_steps=int(args.steps_per_start),
        horizon=int(args.horizon),
        count=int(args.rollouts),
        selection=str(args.selection),
        stride=int(args.selection_stride),
        event_column=str(args.event_column),
        seed=int(args.seed),
    )
    return tuple(int(x) for x in selected.starts), dict(selected.diagnostics)


if __name__ == "__main__":
    main()
