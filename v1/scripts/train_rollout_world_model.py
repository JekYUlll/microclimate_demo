#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from forecast_cmdp.mpc_teacher import MpcTeacherConfig  # noqa: E402
from forecast_cmdp.rollout_world_model import (  # noqa: E402
    RolloutWorldModelTrainingConfig,
    build_rollout_world_model_dataset,
    save_rollout_world_model,
    train_rollout_world_model,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import StaticMaskPolicy, V2Policy  # noqa: E402
from v2.rollout import run_policy_rollout  # noqa: E402


class CyclicMaskPolicy(V2Policy):
    def __init__(self, masks: np.ndarray, *, dwell: int, name: str) -> None:
        self.masks = np.asarray(masks, dtype=bool)
        self.dwell = max(1, int(dwell))
        self.name = str(name)
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def act_mask(self, env: object) -> np.ndarray:
        del env
        idx = (self.t // self.dwell) % self.masks.shape[0]
        self.t += 1
        return self.masks[int(idx)].copy()

    def act_scores(self, env: object) -> np.ndarray:
        return np.where(self.act_mask(env), 1.0, -1.0)


class RandomBlockMaskPolicy(V2Policy):
    def __init__(self, masks: np.ndarray, *, dwell: int, seed: int, name: str) -> None:
        self.masks = np.asarray(masks, dtype=bool)
        self.dwell = max(1, int(dwell))
        self.seed = int(seed)
        self.name = str(name)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.current = self.masks[0].copy()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.current = self.masks[0].copy()

    def act_mask(self, env: object) -> np.ndarray:
        del env
        if self.t % self.dwell == 0:
            self.current = self.masks[int(self.rng.integers(0, self.masks.shape[0]))].copy()
        self.t += 1
        return self.current.copy()

    def act_scores(self, env: object) -> np.ndarray:
        return np.where(self.act_mask(env), 1.0, -1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a mask-aware rollout world model from split-compliant "
            "scheduler rollouts."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.85)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--support-top-k", type=int, default=16)
    parser.add_argument("--dwell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--period-steps", type=int, default=10800)
    parser.add_argument("--min-skill-vs-persistence", type=float, default=0.0)
    parser.add_argument("--min-interval-coverage", type=float, default=0.60)
    parser.add_argument("--max-interval-coverage", type=float, default=0.98)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def source_truth_path(source_run: Path, manifest: dict[str, object]) -> Path:
    candidate = source_run / "truth_with_learned_event_forecast.csv"
    if candidate.exists():
        return candidate
    return resolve_path(str(manifest["truth_csv"]))


def main() -> None:
    args = parse_args()
    source_run = resolve_path(args.source_run)
    manifest_path = source_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_args = dict(manifest["run_args"])
    truth_path = source_truth_path(source_run, manifest)
    truth = pd.read_csv(truth_path)
    helpers = load_v2_helpers()
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    sensors = load_archived_sensor_specs(resolve_path(str(manifest["sensor_cfg"])))
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    norm_bounds = tuple(int(value) for value in manifest["normalization_bounds"])
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=norm_bounds[0],
        end_idx=norm_bounds[1],
    )
    env_cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(args.lookback),
        episode_len=1,
        seed=int(args.seed),
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
    with np.load(resolve_path(str(manifest["teacher_dataset"])), allow_pickle=False) as teacher:
        candidate_masks = np.asarray(teacher["candidate_masks"], dtype=bool)
    anchor_idx = int(manifest["selected_static"]["action_idx"])
    anchor_mask = np.asarray(manifest["selected_static"]["mask"], dtype=bool)
    support = select_support(
        candidate_masks,
        MpcTeacherConfig(**dict(manifest["teacher_cfg"])),
        anchor_idx=anchor_idx,
        top_k=int(args.support_top_k),
    )
    support_masks = candidate_masks[list(support)]
    event_columns = tuple(
        str(name)
        for name in dict(manifest["forecast_cfg"]).get("learned_event_probability_columns", ())
        if str(name) in truth.columns
    )
    event_values = truth.loc[:, event_columns].to_numpy(dtype=np.float32) if event_columns else None
    cfg = RolloutWorldModelTrainingConfig(
        horizon=int(args.horizon),
        lookback=int(args.lookback),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        member_count=int(args.members),
        bootstrap_fraction=float(args.bootstrap_fraction),
        residual_scale=float(args.residual_scale),
        seed=int(args.seed),
        device=str(args.device),
        period_steps=max(1, int(args.period_steps)),
        event_probability_horizon=len(event_columns),
    )
    train_start = int(manifest["bounds"]["oracle_pretrain"][0])
    train_end = int(manifest["bounds"]["rl_train"][1])
    fit_end = train_start + int((train_end - train_start) * float(args.fit_fraction))
    calibration_end = fit_end + int((train_end - train_start) * float(args.calibration_fraction))
    split_bounds = {
        "fit": (train_start, fit_end),
        "calibration": (fit_end, calibration_end),
        "audit": (calibration_end, train_end),
    }
    policies = make_rollout_policies(
        anchor_mask=anchor_mask,
        support_masks=support_masks,
        dwell=int(args.dwell),
        seed=int(args.seed),
    )
    split_rollouts = {
        name: collect_rollouts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=env_cfg,
            policies=policies,
            bounds=bounds,
            seed=int(args.seed) + offset * 10_000,
        )
        for offset, (name, bounds) in enumerate(split_bounds.items())
    }
    fit = build_rollout_world_model_dataset(
        split_rollouts["fit"],
        state_columns=state_columns,
        cfg=cfg,
        event_probability_values=event_values,
    )
    calibration = build_rollout_world_model_dataset(
        split_rollouts["calibration"],
        state_columns=state_columns,
        cfg=cfg,
        event_probability_values=event_values,
        normalization=fit,
    )
    audit = build_rollout_world_model_dataset(
        split_rollouts["audit"],
        state_columns=state_columns,
        cfg=cfg,
        event_probability_values=event_values,
        normalization=fit,
    )
    model = train_rollout_world_model(
        fit_dataset=fit,
        calibration_dataset=calibration,
        audit_dataset=audit,
        cfg=cfg,
    )
    metrics = dict(model.audit_metrics)
    skill = float(metrics["rmse_skill_vs_persistence"])
    coverage = float(metrics["interval_80_coverage"])
    gate_pass = bool(
        skill > float(args.min_skill_vs_persistence)
        and coverage >= float(args.min_interval_coverage)
        and coverage <= float(args.max_interval_coverage)
    )
    output = resolve_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "rollout_world_model.pt"
    save_rollout_world_model(model, model_path)
    summary = {
        "role": "mask_aware_rollout_world_model_audit",
        "source_run": str(source_run),
        "source_manifest": str(manifest_path),
        "truth_csv": str(truth_path),
        "state_columns": list(state_columns),
        "event_probability_columns": list(event_columns),
        "train_bounds": [train_start, train_end],
        "split_bounds": {key: list(value) for key, value in split_bounds.items()},
        "rollout_policies": [policy.name for policy in policies],
        "support_indices": list(support),
        "validation_or_final_used": False,
        "config": asdict(cfg),
        "dataset_rows": {
            "fit": int(fit.features.shape[0]),
            "calibration": int(calibration.features.shape[0]),
            "audit": int(audit.features.shape[0]),
        },
        "metrics": metrics,
        "gate": {
            "min_skill_vs_persistence": float(args.min_skill_vs_persistence),
            "min_interval_coverage": float(args.min_interval_coverage),
            "max_interval_coverage": float(args.max_interval_coverage),
            "pass": gate_pass,
        },
        "model_path": str(model_path),
    }
    (output / "rollout_world_model_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def select_support(
    candidate_masks: np.ndarray,
    cfg: MpcTeacherConfig,
    *,
    anchor_idx: int,
    top_k: int,
) -> tuple[int, ...]:
    masks = np.asarray(candidate_masks, dtype=bool)
    if cfg.candidate_prior_costs is None:
        selected = list(range(min(max(1, int(top_k)), masks.shape[0])))
    else:
        costs = np.asarray(cfg.candidate_prior_costs, dtype=float)
        selected = np.argsort(np.where(np.isfinite(costs), costs, np.inf), kind="stable")[: max(1, int(top_k))].tolist()
    selected.append(int(anchor_idx))
    return tuple(dict.fromkeys(int(index) for index in selected))


def make_rollout_policies(
    *,
    anchor_mask: np.ndarray,
    support_masks: np.ndarray,
    dwell: int,
    seed: int,
) -> tuple[V2Policy, ...]:
    support = np.asarray(support_masks, dtype=bool)
    if support.shape[0] == 0:
        raise ValueError("support mask set must not be empty")
    return (
        StaticMaskPolicy(tuple(bool(value) for value in anchor_mask), name="static_anchor"),
        CyclicMaskPolicy(support, dwell=int(dwell), name="support_cycle"),
        CyclicMaskPolicy(support[::-1], dwell=int(dwell), name="support_cycle_reverse"),
        RandomBlockMaskPolicy(support, dwell=int(dwell), seed=int(seed) + 17, name="support_random_block"),
    )


def collect_rollouts(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    cfg: object,
    policies: tuple[V2Policy, ...],
    bounds: tuple[int, int],
    seed: int,
) -> tuple[object, ...]:
    start, end = int(bounds[0]), int(bounds[1])
    steps = max(1, end - start)
    rows = []
    for offset, policy in enumerate(policies):
        env_cfg = replace(cfg, episode_len=steps, seed=int(seed) + int(offset))
        env = WarmupSchedulingEnv(truth, sensors, constraints, env_cfg, oracle=None)
        rows.append(run_policy_rollout(env, policy, steps=steps, start_idx=start))
    return tuple(rows)


if __name__ == "__main__":
    main()
