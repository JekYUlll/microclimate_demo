#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_pinball_loss

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
from forecast_cmdp.features import ForecastContextConfig  # noqa: E402
from forecast_cmdp.mean_risk_policy import build_residual_risk_feature  # noqa: E402
from forecast_cmdp.window_risk import (  # noqa: E402
    ControllerSpec,
    build_window_risk_dataset,
    load_window_risk_records,
)
from v2.env import WarmupSchedulingEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare causal and privileged residual-risk context without changing formal artifacts."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cv_module():
    path = V1_ROOT / "scripts" / "audit_window_risk_grouped_cv.py"
    spec = importlib.util.spec_from_file_location("residual_cv_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import grouped-CV helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild_features(
    records,
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: object,
    env_cfg: object,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
):
    rebuilt = []
    for record in records:
        env = WarmupSchedulingEnv(
            truth,
            sensors,
            constraints,
            env_cfg,
            oracle=None,
        )
        env.reset(start_idx=int(record.start))
        feature, feature_names = build_residual_risk_feature(
            env=env,
            forecast_cfg=forecast_cfg,
            candidate_masks=candidate_masks,
            anchor_idx=int(record.anchor_action_idx),
            controller=ControllerSpec(
                controller_id=str(record.controller_id),
                parameters=dict(record.controller_config),
            ),
        )
        rebuilt.append(
            replace(
                record,
                feature_vector=tuple(float(x) for x in feature),
                feature_names=tuple(str(x) for x in feature_names),
            )
        )
    return build_window_risk_dataset(rebuilt)


def chronological_metrics(
    cv_module,
    family: str,
    *,
    fit_features: np.ndarray,
    fit_margins: np.ndarray,
    calibration_features: np.ndarray,
    calibration_margins: np.ndarray,
    seed: int,
) -> dict[str, float]:
    mean_model, quantile_model, negative_model = cv_module.model_family(
        family,
        alpha=0.25,
        seed=int(seed),
    )
    mean_model.fit(fit_features, fit_margins)
    quantile_model.fit(fit_features, fit_margins)
    negative_model.fit(fit_features, (fit_margins < 0.0).astype(np.int8))
    mean_pred = np.asarray(mean_model.predict(calibration_features), dtype=float)
    q25_pred = np.asarray(
        quantile_model.predict(calibration_features), dtype=float
    )
    negative_pred = np.asarray(
        negative_model.predict_proba(calibration_features), dtype=float
    )[:, 1]
    constant_mean = np.full(
        calibration_margins.shape,
        float(np.mean(fit_margins)),
        dtype=float,
    )
    constant_q25 = np.full(
        calibration_margins.shape,
        float(np.quantile(fit_margins, 0.25)),
        dtype=float,
    )
    constant_negative = np.full(
        calibration_margins.shape,
        float(np.mean(fit_margins < 0.0)),
        dtype=float,
    )
    mean_mae = float(mean_absolute_error(calibration_margins, mean_pred))
    constant_mean_mae = float(
        mean_absolute_error(calibration_margins, constant_mean)
    )
    q25_pinball = float(
        mean_pinball_loss(calibration_margins, q25_pred, alpha=0.25)
    )
    constant_q25_pinball = float(
        mean_pinball_loss(calibration_margins, constant_q25, alpha=0.25)
    )
    negative_brier = float(
        brier_score_loss(
            (calibration_margins < 0.0).astype(np.int8),
            negative_pred,
        )
    )
    constant_negative_brier = float(
        brier_score_loss(
            (calibration_margins < 0.0).astype(np.int8),
            constant_negative,
        )
    )
    return {
        "mean_spearman": cv_module.safe_spearman(
            mean_pred, calibration_margins
        ),
        "mean_mae_improvement": (
            (constant_mean_mae - mean_mae) / constant_mean_mae
        ),
        "q25_pinball_improvement": (
            (constant_q25_pinball - q25_pinball)
            / constant_q25_pinball
        ),
        "negative_brier_improvement": (
            (constant_negative_brier - negative_brier)
            / constant_negative_brier
        ),
    }


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.data_root)
    protocol = json.loads(
        (root / "window_risk_protocol.json").read_text(encoding="utf-8")
    )
    source_run = resolve_project_path(protocol["source_run"])
    manifest = json.loads(
        (source_run / "manifest.json").read_text(encoding="utf-8")
    )
    run_args = dict(manifest["run_args"])
    truth = pd.read_csv(resolve_project_path(protocol["truth_csv"]))
    helpers = load_v2_helpers()
    state_columns = tuple(str(x) for x in helpers.STATE_COLUMNS)
    reward_target_columns = tuple(str(x) for x in helpers.REWARD_TARGET_COLUMNS)
    sensors = load_archived_sensor_specs(
        resolve_project_path(manifest["sensor_cfg"])
    )
    constraints = make_constraints(
        max_active=int(run_args["max_active"]),
        budget=float(run_args["budget"]),
        startup_peak_budget=float(run_args["startup_peak_budget"]),
    )
    normalization_bounds = tuple(int(x) for x in manifest["normalization_bounds"])
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns,
        start_idx=normalization_bounds[0],
        end_idx=normalization_bounds[1],
    )
    env_cfg = make_env_config(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(run_args["lookback"]),
        episode_len=int(protocol["window_steps"]),
        seed=int(manifest["seed"]),
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
    )
    with np.load(
        resolve_project_path(manifest["teacher_dataset"]),
        allow_pickle=False,
    ) as teacher:
        candidate_masks = np.asarray(teacher["candidate_masks"], dtype=bool)
    fit_records = load_window_risk_records(
        root / "risk_fit" / "window_risk_rows.jsonl"
    )
    calibration_records = load_window_risk_records(
        root / "risk_calibration" / "window_risk_rows.jsonl"
    )
    causal_cfg = ForecastContextConfig(
        **dict(protocol["effective_forecast_cfg"])
    )
    contexts = {
        "learned_causal": causal_cfg,
        "privileged_future": replace(
            causal_cfg,
            truth_future=True,
            continuous_truth_future=True,
            continuous_current_source="truth",
        ),
    }
    cv_module = load_cv_module()
    results: dict[str, object] = {}
    for context_name, forecast_cfg in contexts.items():
        fit = rebuild_features(
            fit_records,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            env_cfg=env_cfg,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
        )
        calibration = rebuild_features(
            calibration_records,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            env_cfg=env_cfg,
            candidate_masks=candidate_masks,
            forecast_cfg=forecast_cfg,
        )
        x_fit = np.asarray(fit.features, dtype=np.float32)
        x_cal = np.asarray(calibration.features, dtype=np.float32)
        varying = np.var(x_fit, axis=0) > 1.0e-12
        x_fit = x_fit[:, varying]
        x_cal = x_cal[:, varying]
        y_fit = np.asarray(fit.margins, dtype=float)
        y_cal = np.asarray(calibration.margins, dtype=float)
        families = {}
        for family in ("hist_gbdt", "xgboost"):
            families[family] = {
                "fit_grouped_cv": cv_module.evaluate_family(
                    family,
                    features=x_fit,
                    margins=y_fit,
                    groups=np.asarray(fit.starts, dtype=np.int64),
                    folds=int(args.folds),
                    alpha=0.25,
                    seed=int(args.seed),
                ),
                "chronological_calibration": chronological_metrics(
                    cv_module,
                    family,
                    fit_features=x_fit,
                    fit_margins=y_fit,
                    calibration_features=x_cal,
                    calibration_margins=y_cal,
                    seed=int(args.seed),
                ),
            }
        results[context_name] = {
            "input_features": int(fit.features.shape[1]),
            "varying_features": int(np.sum(varying)),
            "families": families,
        }
    output = (
        resolve_project_path(args.out)
        if args.out is not None
        else root / "residual_context_ablation.json"
    )
    output.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
