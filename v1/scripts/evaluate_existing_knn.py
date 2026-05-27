#!/usr/bin/env python
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v1"))

from forecast_cmdp.archived_v2 import (  # noqa: E402
    load_archived_oracle,
    load_archived_sensor_specs,
    load_v2_helpers,
    make_constraints,
    normalization_stats,
)
from forecast_cmdp.dataset import TeacherDataset  # noqa: E402
from forecast_cmdp.features import ForecastContextConfig  # noqa: E402
from forecast_cmdp.policy import ForecastAwareCyclePolicy, ForecastAwareKNNPolicy  # noqa: E402
from forecast_cmdp.protocol import (  # noqa: E402
    evaluate_policy_over_starts,
    final_objective,
    rich_metrics,
    save_rollout,
    task_focus_metrics,
)
from run_protocol_gate import make_common_env_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posthoc-evaluate KNN deployable policy in an existing v1 run dir.")
    parser.add_argument("run_dir")
    parser.add_argument("--policy", choices=["knn", "cycle"], default="knn")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--preserve-warming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_args = Namespace(**dict(manifest["run_args"]))
    helpers = load_v2_helpers()
    truth = pd.read_csv(manifest["truth_csv"])
    sensors = load_archived_sensor_specs(manifest["sensor_cfg"])
    sensor_ids = tuple(str(sensor.sensor_id) for sensor in sensors)
    state_columns = tuple(str(name) for name in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(name) for name in helpers.REWARD_TARGET_COLUMNS)
    norm_start, norm_end = manifest["normalization_bounds"]
    norm_mean, norm_std = normalization_stats(truth, state_columns, start_idx=int(norm_start), end_idx=int(norm_end))
    oracle = load_archived_oracle(manifest["oracle_path"], oracle_type=str(manifest["oracle_type"]), device=str(run_args.oracle_device))
    constraints = make_constraints(
        max_active=int(run_args.max_active),
        budget=float(run_args.budget),
        startup_peak_budget=float(run_args.startup_peak_budget),
    )
    eval_cfg = make_common_env_config(
        run_args,
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        episode_len=int(run_args.eval_steps),
        seed=int(run_args.seed) + 20_000,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    dataset = TeacherDataset.load_npz(run_dir / "teacher_dataset.npz")
    forecast_cfg = ForecastContextConfig(**dict(manifest["forecast_cfg"]))
    if str(args.policy) == "knn":
        policy = ForecastAwareKNNPolicy(
            features=dataset.features,
            labels=dataset.labels,
            candidate_masks=dataset.candidate_masks,
            forecast_cfg=forecast_cfg,
            k=int(args.k),
            preserve_warming=bool(args.preserve_warming),
        )
        policy_name = "forecast_aware_knn_posthoc"
    else:
        policy = ForecastAwareCyclePolicy(
            labels=dataset.labels,
            candidate_masks=dataset.candidate_masks,
            preserve_warming=bool(args.preserve_warming),
        )
        policy_name = "forecast_aware_cycle_posthoc"
    result, simple_metrics = evaluate_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        policy=policy,
        steps=int(run_args.eval_steps),
        start_indices=tuple(int(x) for x in manifest["starts"]["final_test"]["starts"]),
    )
    metrics = rich_metrics(
        result,
        sensor_ids=sensor_ids,
        state_columns=state_columns,
        per_step_budget=float(run_args.budget),
        startup_peak_budget=float(run_args.startup_peak_budget),
    )
    metrics.update(
        task_focus_metrics(
            result,
            state_columns=state_columns,
            task_error_columns=tuple(str(x) for x in run_args.task_error_columns),
            task_error_scales=tuple(float(x) for x in run_args.task_error_scales) if run_args.task_error_scales else None,
            event_only=bool(run_args.task_error_event_only),
        )
    )
    metrics.update({f"rollout_{key}": value for key, value in simple_metrics.items() if key not in metrics})
    metrics["objective_loss_mean"] = final_objective(
        metrics,
        mode=str(run_args.objective_mode),
        task_error_weight=float(run_args.task_error_weight),
    )
    pd.DataFrame([metrics]).to_csv(run_dir / f"metrics_{args.policy}_posthoc.csv", index=False)
    save_rollout(
        run_dir / f"rollout_forecast_aware_{args.policy}_posthoc.npz",
        result,
        sensor_ids=sensor_ids,
        state_columns=state_columns,
    )
    static_objective = float(manifest["gate_summary"]["validation_selected_static_objective"])
    summary = {
        "policy": policy_name,
        "k": int(args.k),
        "preserve_warming": bool(args.preserve_warming),
        "static_objective": static_objective,
        "objective": float(metrics["objective_loss_mean"]),
        "margin": static_objective - float(metrics["objective_loss_mean"]),
        "gate_pass": bool(float(metrics["objective_loss_mean"]) < static_objective),
    }
    (run_dir / f"gate_summary_{args.policy}_posthoc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
