#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict no-warmup split-protocol DQN diagnostic by reusing an "
            "existing no-warmup PD-PPO truth/oracle/manifest."
        )
    )
    parser.add_argument("--framework-root", default="../rl_sensor_scheduling_framework")
    parser.add_argument("--source-root", default="reports/v31_split_protocol_no_warmup")
    parser.add_argument("--out-root", default="reports/v31_no_warmup_dqn_split_diagnostic")
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced_no_warmup.yaml")
    parser.add_argument("--total-timesteps", type=int, default=60000)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--exploration-fraction", type=float, default=0.30)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--oracle-prefill-steps", type=int, default=0)
    parser.add_argument("--oracle-prefill-lookahead-steps", type=int, default=2)
    parser.add_argument("--event-start-prob", type=float, default=0.67)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--save-rollouts", action="store_true")
    return parser.parse_args()


def budget_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_training_log(history: list[dict[str, float | int]], path: Path) -> None:
    rows = []
    for item in history:
        rows.append(
            {
                "step": int(item.get("timesteps", 0)),
                "loss": float(item.get("loss", float("nan"))),
                "epsilon": float(item.get("epsilon", float("nan"))),
                "reward_mean": float(item.get("reward_mean", float("nan"))),
                "episode_return_mean": float(item.get("episode_return_mean", float("nan"))),
                "unique_actions": int(item.get("unique_actions", 0)),
                "replay_size": int(item.get("replay_size", 0)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def read_source_paths(framework_root: Path, source_root: str, *, budget: float, seed: int) -> dict[str, Path]:
    label = f"budget{budget_tag(budget)}_seed{int(seed)}"
    source = framework_root / source_root / "raw" / label
    paths = {
        "source": source,
        "truth": source / "truth_v31_split.csv",
        "oracle": source / "v2_tcn_oracle.pt",
        "manifest": source / "split_protocol_manifest.json",
        "validation_static": source / "validation_static_candidates.csv",
        "ppo_metrics": source / "v2_custom_ppo_metrics.csv",
    }
    missing = [name for name, path in paths.items() if name != "ppo_metrics" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts for {label}: {missing} in {source}")
    return paths


def selected_static_mask(path: Path, candidate_masks: np.ndarray) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not path.exists():
        return None, None
    table = pd.read_csv(path)
    if table.empty:
        return None, None
    row = table.iloc[0]
    action_idx = int(row["action_idx"])
    return np.asarray(candidate_masks[action_idx], dtype=bool), {
        "action_idx": action_idx,
        "oracle_loss_mean": float(row["oracle_loss_mean"]),
        "sensor_ids": str(row.get("sensor_ids", "")),
    }


def main() -> None:
    args = parse_args()
    framework_root = Path(args.framework_root).resolve()
    sys.path.insert(0, str(framework_root / "src"))

    helpers = load_module(framework_root / "scripts" / "23_v2_train_ppo.py", "_dqn_split_helpers")
    dqn_module = load_module(framework_root / "src" / "v2" / "dqn.py", "_dqn_split_dqn")

    from v2.env import WarmupEnvConfig
    from v2.policies import FullOpenUnconstrainedScorePolicy, StaticMaskPolicy
    from v2.power_projector import PowerConstraintsV2
    from v2.rollout import save_rollout_npz
    from v2.sensor_spec import load_sensor_specs
    from v2.tcn_oracle import TCNFrozenForecastOracle

    paths = read_source_paths(framework_root, str(args.source_root), budget=float(args.budget), seed=int(args.seed))
    source_label = f"budget{budget_tag(float(args.budget))}_seed{int(args.seed)}"
    out_dir = framework_root / str(args.out_root) / "raw" / source_label
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    final_starts = tuple(int(value) for value in manifest["final_test"]["eval_starts"])
    eval_steps = int(manifest["final_test"]["eval_steps"])
    train_min = int(manifest["rl_train"]["ppo_start_min"])
    train_max = int(manifest["rl_train"]["ppo_start_max"])
    freq_s = int(manifest["truth_steps"] and 3600)
    if "freq_s" in manifest:
        freq_s = int(manifest["freq_s"])

    truth = pd.read_csv(paths["truth"])
    sensors = load_sensor_specs(str(framework_root / str(args.sensor_cfg)))
    constraints = PowerConstraintsV2(
        max_active=4,
        per_step_budget=float(args.budget),
        startup_peak_budget=3.2,
        required_sensor_ids=(),
        coverage_groups=helpers.DEFAULT_COVERAGE_GROUPS,
    )
    candidate_masks = helpers.build_projected_candidate_masks(sensors, constraints)
    oracle = TCNFrozenForecastOracle.load(paths["oracle"], device=str(args.oracle_inference_device))

    class SplitDQNTrainer(dqn_module.DQNTrainer):
        def __init__(self, *trainer_args, train_start_min: int, train_start_max: int, **trainer_kwargs) -> None:
            self.train_start_min = int(train_start_min)
            self.train_start_max = int(train_start_max)
            super().__init__(*trainer_args, **trainer_kwargs)

        def _sample_start_idx(self, steps: int, *, seed_offset: int) -> int:
            horizon = int(getattr(self.oracle.cfg, "horizon", 1))
            low = int(self.train_start_min)
            high = min(int(self.train_start_max), len(self.truth_df) - int(steps) - horizon - 1)
            if high <= low:
                return max(0, low)
            rng = np.random.default_rng(int(self.cfg.seed) + int(seed_offset) + 71_239)
            event_flags = (
                self.truth_df[self.env_cfg.event_column].astype(bool).to_numpy()
                if self.env_cfg.event_column in self.truth_df.columns
                else np.zeros(len(self.truth_df), dtype=bool)
            )
            event_lo = low
            event_hi = min(len(event_flags), high + int(steps))
            event_indices = np.flatnonzero(event_flags[event_lo:event_hi]) + event_lo
            if event_indices.size and rng.random() < float(self.cfg.event_start_prob):
                event_idx = int(rng.choice(event_indices))
                return int(np.clip(event_idx - int(steps) // 3, low, high))
            return int(rng.integers(low, high + 1))

    train_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=20,
        episode_len=512,
        seed=int(args.seed),
        base_freq_s=freq_s,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    trainer = SplitDQNTrainer(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=train_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        cfg=dqn_module.DQNConfig(
            total_timesteps=int(args.total_timesteps),
            replay_size=int(args.replay_size),
            learning_starts=int(args.learning_starts),
            batch_size=int(args.batch_size),
            train_freq=int(args.train_freq),
            gradient_steps=int(args.gradient_steps),
            target_update_interval=int(args.target_update_interval),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            n_step_return=int(args.n_step_return),
            hidden_dim=int(args.hidden_dim),
            exploration_fraction=float(args.exploration_fraction),
            exploration_final_eps=float(args.exploration_final_eps),
            oracle_prefill_steps=int(args.oracle_prefill_steps),
            oracle_prefill_lookahead_steps=int(args.oracle_prefill_lookahead_steps),
            event_start_prob=float(args.event_start_prob),
            device=str(args.device),
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            history_path=str(out_dir / "dqn_training_history_live.json"),
        ),
        train_start_min=train_min,
        train_start_max=train_max,
    )
    trainer.train()
    trainer.save(out_dir / "dqn.pt")
    trainer.save_history(out_dir / "dqn_training_history.json")
    write_training_log(trainer.history, out_dir / "dqn_training_log.csv")

    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=20,
        episode_len=eval_steps,
        seed=int(args.seed) + 9000,
        base_freq_s=freq_s,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    rows: list[dict[str, float | str | int]] = []
    dqn_result, dqn_metrics = dqn_module.evaluate_dqn(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=eval_steps,
        start_indices=final_starts,
    )
    rows.append(dqn_metrics)
    if bool(args.save_rollouts):
        save_rollout_npz(out_dir / "rollout_dqn.npz", dqn_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    full_open_result, full_open_metrics = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=eval_steps,
        start_indices=final_starts,
    )
    rows.append(full_open_metrics)
    if bool(args.save_rollouts):
        save_rollout_npz(out_dir / "rollout_full_open_unconstrained.npz", full_open_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    static_mask, static_info = selected_static_mask(paths["validation_static"], candidate_masks)
    if static_mask is not None:
        static_policy = StaticMaskPolicy(mask=tuple(bool(value) for value in static_mask), name="validation_selected_static")
        static_result, static_metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=static_policy,
            steps=eval_steps,
            start_indices=final_starts,
        )
        rows.append(static_metrics)
        if bool(args.save_rollouts):
            save_rollout_npz(out_dir / "rollout_validation_selected_static.npz", static_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 100):
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=eval_steps,
            start_indices=final_starts,
        )
        rows.append(metrics)
        if bool(args.save_rollouts):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.policy_name))
            save_rollout_npz(out_dir / f"rollout_{safe_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    metrics = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    metrics.to_csv(out_dir / "v2_dqn_split_metrics.csv", index=False)
    metadata = {
        "protocol": "no_warmup_split_dqn_diagnostic",
        "source_run_dir": str(paths["source"]),
        "source_truth_csv": str(paths["truth"]),
        "source_oracle": str(paths["oracle"]),
        "source_manifest": str(paths["manifest"]),
        "source_ppo_metrics": str(paths["ppo_metrics"]) if paths["ppo_metrics"].exists() else "",
        "budget": float(args.budget),
        "seed": int(args.seed),
        "train_start_min": train_min,
        "train_start_max": train_max,
        "final_eval_starts": list(final_starts),
        "eval_steps": eval_steps,
        "selected_static": static_info,
        "dqn": {**asdict(trainer.cfg), "candidate_count": int(candidate_masks.shape[0])},
    }
    (out_dir / "v2_dqn_split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_dqn_split_metrics.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
