#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "v1"
sys.path.insert(0, str(V1_ROOT))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    make_env_config,
    normalization_stats,
)
from forecast_cmdp.probabilistic_world_model import (  # noqa: E402
    load_probabilistic_world_model,
)
from forecast_cmdp.robust_planner import (  # noqa: E402
    CausalWorldModelContext,
    build_causal_world_model_context,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a world model on complete truth histories and actual "
            "static-anchor rollout histories within its train-only audit range."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    source_run = resolve_path(args.source_run)
    manifest = json.loads(
        (source_run / "manifest.json").read_text(encoding="utf-8")
    )
    run_args = dict(manifest["run_args"])
    truth_path = source_run / "truth_with_learned_event_forecast.csv"
    if not truth_path.exists():
        truth_path = resolve_path(str(manifest["truth_csv"]))
    truth = pd.read_csv(truth_path)
    helpers = load_v2_helpers()
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(
        str(name) for name in helpers.REWARD_TARGET_COLUMNS
    )
    sensors = load_archived_sensor_specs(
        resolve_path(str(manifest["sensor_cfg"]))
    )
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    normalization_bounds = tuple(
        int(value) for value in manifest["normalization_bounds"]
    )
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=normalization_bounds[0],
        end_idx=normalization_bounds[1],
    )
    model = load_probabilistic_world_model(
        resolve_path(args.world_model),
        device=str(args.model_device),
    )
    audit_start, audit_end = (
        int(value) for value in model.audit_metrics["bounds"]["audit"]
    )
    horizon = int(model.cfg.forecaster.horizon)
    cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(run_args["lookback"]),
        episode_len=int(audit_end - audit_start),
        seed=int(manifest["seed"]) + 991,
        freq_s=int(run_args["freq_s"]),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        lambda_warmup_abort=float(run_args["lambda_warmup_abort"]),
        lambda_switch=float(run_args["lambda_switch"]),
        event_reward_multiplier=float(run_args["event_reward_multiplier"]),
        energy_account=bool(run_args["energy_account"]),
        energy_capacity=float(run_args["energy_capacity"]),
        initial_energy=float(run_args["initial_energy"]),
        harvest_per_step=float(run_args["harvest_per_step"]),
        reserve_energy=float(run_args["reserve_energy"]),
        lambda_energy_deficit=float(run_args["lambda_energy_deficit"]),
        soc_soft_penalty_buffer=float(run_args["soc_soft_penalty_buffer"]),
        lambda_soc_soft_penalty=float(run_args["lambda_soc_soft_penalty"]),
        common_random_numbers=True,
    )
    env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        cfg,
        oracle=None,
    )
    env.reset(start_idx=audit_start)
    anchor = np.asarray(manifest["selected_static"]["mask"], dtype=bool)
    stride = max(1, int(args.sample_stride))
    burn_in = max(0, int(args.burn_in))
    max_samples = max(1, int(args.max_samples))
    target_scale = np.asarray(
        model.members[0].target_std,
        dtype=float,
    ).reshape(horizon, len(state_columns))
    target_scale = np.maximum(target_scale, 1.0e-6)
    scheduler_errors: list[np.ndarray] = []
    truth_errors: list[np.ndarray] = []
    persistence_errors: list[np.ndarray] = []
    indices: list[int] = []

    while env.current_idx + horizon < audit_end and len(indices) < max_samples:
        relative = int(env.current_idx - audit_start)
        if relative >= burn_in and (relative - burn_in) % stride == 0:
            idx = int(env.current_idx)
            target = env.truth_values[idx + 1 : idx + 1 + horizon]
            scheduler_context = build_causal_world_model_context(env)
            scheduler_prediction = np.mean(
                model.predict_members(scheduler_context),
                axis=0,
            )
            lookback = int(run_args["lookback"])
            start = max(0, idx - lookback + 1)
            truth_history = env.truth_values[start : idx + 1]
            truth_context = CausalWorldModelContext(
                current_idx=idx,
                state_columns=state_columns,
                history=truth_history.copy(),
                mask_history=np.ones_like(truth_history, dtype=float),
                last_observation=env.truth_values[idx].copy(),
                observed_mask=np.ones(len(state_columns), dtype=float),
                event_probabilities=np.zeros(0, dtype=float),
            )
            truth_prediction = np.mean(
                model.predict_members(truth_context),
                axis=0,
            )
            persistence = np.repeat(
                env.last_observation.reshape(1, -1),
                horizon,
                axis=0,
            )
            scheduler_errors.append((scheduler_prediction - target) / target_scale)
            truth_errors.append((truth_prediction - target) / target_scale)
            persistence_errors.append((persistence - target) / target_scale)
            indices.append(idx)
        env.step_mask(anchor)

    scheduler_error = np.stack(scheduler_errors)
    truth_error = np.stack(truth_errors)
    persistence_error = np.stack(persistence_errors)
    scheduler_rmse = float(np.sqrt(np.mean(scheduler_error**2)))
    truth_rmse = float(np.sqrt(np.mean(truth_error**2)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_error**2)))
    summary = {
        "role": "train_only_world_model_rollout_history_shift_audit",
        "source_run": str(source_run),
        "world_model": str(resolve_path(args.world_model)),
        "audit_bounds": [audit_start, audit_end],
        "sample_count": len(indices),
        "sample_stride": stride,
        "burn_in": burn_in,
        "indices": indices,
        "metrics": {
            "truth_history_normalized_rmse": truth_rmse,
            "scheduler_history_normalized_rmse": scheduler_rmse,
            "scheduler_persistence_normalized_rmse": persistence_rmse,
            "scheduler_skill_vs_persistence": float(
                1.0 - scheduler_rmse / max(persistence_rmse, 1.0e-12)
            ),
            "scheduler_to_truth_history_rmse_ratio": float(
                scheduler_rmse / max(truth_rmse, 1.0e-12)
            ),
            "target_scheduler_normalized_rmse": {
                name: float(np.sqrt(np.mean(scheduler_error[:, :, idx] ** 2)))
                for idx, name in enumerate(state_columns)
            },
            "target_truth_history_normalized_rmse": {
                name: float(np.sqrt(np.mean(truth_error[:, :, idx] ** 2)))
                for idx, name in enumerate(state_columns)
            },
        },
        "validation_or_final_used": False,
    }
    output = resolve_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rollout_history_shift_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
