from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_pinball_loss

from .window_risk import WindowRiskDataset


@dataclass(frozen=True)
class WindowRiskTrainingConfig:
    model_family: str = "gbdt"
    quantile_alpha: float = 0.25
    n_estimators: int = 200
    learning_rate: float = 0.03
    max_depth: int = 3
    max_leaf_nodes: int = 7
    min_samples_leaf: int = 8
    l2_regularization: float = 1.0
    negative_prevalence_min: float = 0.05
    negative_prevalence_max: float = 0.95
    seed: int = 41


@dataclass
class WindowRiskModelBundle:
    mean_model: Any
    quantile_model: Any
    negative_model: Any | None
    feature_names: tuple[str, ...]
    quantile_alpha: float
    conformal_correction: float
    negative_prevalence: float

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("Window-risk features do not match the trained schema")
        mean = np.asarray(self.mean_model.predict(x), dtype=float).reshape(-1)
        q25 = np.asarray(self.quantile_model.predict(x), dtype=float).reshape(-1)
        lower = q25 - float(self.conformal_correction)
        if self.negative_model is None:
            negative = np.full(x.shape[0], float("nan"), dtype=float)
        else:
            probabilities = np.asarray(self.negative_model.predict_proba(x), dtype=float)
            classes = np.asarray(self.negative_model.classes_, dtype=int)
            if 1 in classes:
                negative = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
            else:
                negative = np.full(x.shape[0], float(self.negative_prevalence), dtype=float)
        return {
            "mean_margin": mean,
            "q25_margin": q25,
            "risk_lower_bound": lower,
            "negative_probability": np.clip(negative, 0.0, 1.0),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "WindowRiskModelBundle":
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(bundle).__name__}")
        return bundle


def train_window_risk_models(
    fit: WindowRiskDataset,
    calibration: WindowRiskDataset,
    *,
    cfg: WindowRiskTrainingConfig,
    out_dir: str | Path | None = None,
) -> tuple[WindowRiskModelBundle, dict[str, object]]:
    _validate_datasets(fit, calibration)
    alpha = float(cfg.quantile_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("quantile_alpha must be between zero and one")
    x_fit = np.asarray(fit.features, dtype=np.float32)
    y_fit = np.asarray(fit.margins, dtype=float)
    x_cal = np.asarray(calibration.features, dtype=np.float32)
    y_cal = np.asarray(calibration.margins, dtype=float)

    family = str(cfg.model_family)
    if family == "gbdt":
        mean_model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=int(cfg.n_estimators),
            learning_rate=float(cfg.learning_rate),
            max_depth=int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            random_state=int(cfg.seed),
        )
        quantile_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=int(cfg.n_estimators),
            learning_rate=float(cfg.learning_rate),
            max_depth=int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            random_state=int(cfg.seed) + 1,
        )
    elif family == "hist_gbdt":
        common = {
            "learning_rate": float(cfg.learning_rate),
            "max_iter": int(cfg.n_estimators),
            "max_leaf_nodes": int(cfg.max_leaf_nodes),
            "min_samples_leaf": int(cfg.min_samples_leaf),
            "l2_regularization": float(cfg.l2_regularization),
        }
        mean_model = HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=int(cfg.seed),
            **common,
        )
        quantile_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=alpha,
            random_state=int(cfg.seed) + 1,
            **common,
        )
    elif family == "xgboost":
        from xgboost import XGBRegressor

        common = {
            "n_estimators": int(cfg.n_estimators),
            "learning_rate": float(cfg.learning_rate),
            "max_depth": int(cfg.max_depth),
            "min_child_weight": float(cfg.min_samples_leaf),
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
            "n_jobs": 8,
        }
        mean_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=int(cfg.seed),
            **common,
        )
        quantile_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            random_state=int(cfg.seed) + 1,
            **common,
        )
    else:
        raise ValueError(f"Unsupported window-risk model family: {family}")
    mean_model.fit(x_fit, y_fit)
    quantile_model.fit(x_fit, y_fit)

    fit_negative = (y_fit < 0.0).astype(np.int8)
    prevalence = float(np.mean(fit_negative))
    negative_enabled = (
        float(cfg.negative_prevalence_min)
        <= prevalence
        <= float(cfg.negative_prevalence_max)
        and np.unique(fit_negative).size == 2
    )
    negative_model = None
    if negative_enabled:
        if family == "xgboost":
            from xgboost import XGBClassifier

            negative_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=int(cfg.n_estimators),
                learning_rate=float(cfg.learning_rate),
                max_depth=int(cfg.max_depth),
                min_child_weight=float(cfg.min_samples_leaf),
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=10.0,
                n_jobs=8,
                random_state=int(cfg.seed) + 2,
            )
        elif family == "hist_gbdt":
            negative_model = HistGradientBoostingClassifier(
                learning_rate=float(cfg.learning_rate),
                max_iter=int(cfg.n_estimators),
                max_leaf_nodes=int(cfg.max_leaf_nodes),
                min_samples_leaf=int(cfg.min_samples_leaf),
                l2_regularization=float(cfg.l2_regularization),
                random_state=int(cfg.seed) + 2,
            )
        else:
            negative_model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=max(8, int(cfg.min_samples_leaf)),
                l2_regularization=1.0e-3,
                random_state=int(cfg.seed) + 2,
            )
        negative_model.fit(x_fit, fit_negative)

    mean_pred = np.asarray(mean_model.predict(x_cal), dtype=float)
    q25_pred = np.asarray(quantile_model.predict(x_cal), dtype=float)
    correction = one_sided_conformal_correction(
        q25_pred,
        y_cal,
        alpha=alpha,
        groups=calibration.starts,
    )
    bundle = WindowRiskModelBundle(
        mean_model=mean_model,
        quantile_model=quantile_model,
        negative_model=negative_model,
        feature_names=fit.feature_names,
        quantile_alpha=alpha,
        conformal_correction=float(correction),
        negative_prevalence=prevalence,
    )
    predictions = bundle.predict(x_cal)
    metrics = evaluate_window_risk_models(
        fit=fit,
        calibration=calibration,
        predictions=predictions,
        raw_q25_predictions=q25_pred,
        negative_model_enabled=negative_enabled,
        alpha=alpha,
    )
    metrics["training_config"] = {
        "model_family": family,
        "quantile_alpha": alpha,
        "n_estimators": int(cfg.n_estimators),
        "learning_rate": float(cfg.learning_rate),
        "max_depth": int(cfg.max_depth),
        "max_leaf_nodes": int(cfg.max_leaf_nodes),
        "min_samples_leaf": int(cfg.min_samples_leaf),
        "l2_regularization": float(cfg.l2_regularization),
        "negative_prevalence_min": float(cfg.negative_prevalence_min),
        "negative_prevalence_max": float(cfg.negative_prevalence_max),
        "seed": int(cfg.seed),
    }
    if out_dir is not None:
        write_window_risk_model_artifacts(
            out_dir,
            bundle=bundle,
            fit=fit,
            calibration=calibration,
            predictions=predictions,
            metrics=metrics,
        )
    return bundle, metrics


def one_sided_conformal_correction(
    quantile_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    groups: np.ndarray | None = None,
) -> float:
    predictions = np.asarray(quantile_predictions, dtype=float).reshape(-1)
    values = np.asarray(targets, dtype=float).reshape(-1)
    if predictions.shape != values.shape or values.size == 0:
        raise ValueError("Conformal predictions and targets must have matching nonempty shapes")
    residuals = predictions - values
    if groups is not None:
        group_values = np.asarray(groups).reshape(-1)
        if group_values.shape != values.shape:
            raise ValueError("Conformal groups must match predictions and targets")
        residuals = np.asarray(
            [
                float(np.max(residuals[group_values == group]))
                for group in np.unique(group_values)
            ],
            dtype=float,
        )
    level = min(
        1.0,
        np.ceil((residuals.size + 1) * (1.0 - float(alpha))) / float(residuals.size),
    )
    return max(0.0, float(np.quantile(residuals, level, method="higher")))


def evaluate_window_risk_models(
    *,
    fit: WindowRiskDataset,
    calibration: WindowRiskDataset,
    predictions: dict[str, np.ndarray],
    raw_q25_predictions: np.ndarray,
    negative_model_enabled: bool,
    alpha: float,
) -> dict[str, object]:
    y_fit = np.asarray(fit.margins, dtype=float)
    y_cal = np.asarray(calibration.margins, dtype=float)
    mean_pred = np.asarray(predictions["mean_margin"], dtype=float)
    q25_pred = np.asarray(raw_q25_predictions, dtype=float)
    lower = np.asarray(predictions["risk_lower_bound"], dtype=float)
    negative_probability = np.asarray(predictions["negative_probability"], dtype=float)
    train_q25 = float(np.quantile(y_fit, float(alpha)))
    constant_q25 = np.full(y_cal.shape, train_q25, dtype=float)
    model_pinball = float(mean_pinball_loss(y_cal, q25_pred, alpha=float(alpha)))
    baseline_pinball = float(mean_pinball_loss(y_cal, constant_q25, alpha=float(alpha)))
    pinball_improvement = (
        float((baseline_pinball - model_pinball) / baseline_pinball)
        if baseline_pinball > 1.0e-12
        else float("nan")
    )
    negative_targets = (y_cal < 0.0).astype(np.int8)
    prevalence = float(np.mean(y_fit < 0.0))
    evaluated_negative_probability = np.where(
        np.isfinite(negative_probability),
        negative_probability,
        prevalence,
    )
    baseline_negative_probability = np.full(y_cal.shape, prevalence, dtype=float)
    model_brier = float(
        brier_score_loss(negative_targets, evaluated_negative_probability, pos_label=1)
    )
    baseline_brier = float(
        brier_score_loss(negative_targets, baseline_negative_probability, pos_label=1)
    )
    brier_improvement = (
        float((baseline_brier - model_brier) / baseline_brier)
        if baseline_brier > 1.0e-12
        else float("nan")
    )
    bins = risk_bin_diagnostics(lower, y_cal)
    controller_calibration = controller_margin_summary(calibration)
    data_gate = data_sufficiency_gate(fit, calibration, controller_calibration)
    model_gate = {
        "pinball_improvement_at_least_10pct": bool(
            np.isfinite(pinball_improvement) and pinball_improvement >= 0.10
        ),
        "raw_q25_coverage_in_band": bool(0.15 <= float(np.mean(y_cal <= q25_pred)) <= 0.35),
        "risk_bins_tail_monotonic": bool(
            float(bins["q25_monotonic_fraction"]) >= 0.5
            and float(bins["negative_rate_monotonic_fraction"]) >= 0.5
        ),
        "negative_classifier_gate": bool(
            not negative_model_enabled
            or (np.isfinite(brier_improvement) and brier_improvement > 0.0)
        ),
    }
    model_gate["pass"] = bool(all(model_gate.values()))
    return {
        "fit_rows": int(y_fit.size),
        "calibration_rows": int(y_cal.size),
        "fit_independent_starts": int(np.unique(fit.starts).size),
        "calibration_independent_starts": int(np.unique(calibration.starts).size),
        "conformal_calibration_groups": int(np.unique(calibration.starts).size),
        "fit_margin_mean": float(np.mean(y_fit)),
        "fit_margin_q25": float(np.quantile(y_fit, 0.25)),
        "fit_positive_rate": float(np.mean(y_fit > 0.0)),
        "fit_negative_rate": float(np.mean(y_fit < 0.0)),
        "calibration_margin_mean": float(np.mean(y_cal)),
        "calibration_margin_q25": float(np.quantile(y_cal, 0.25)),
        "calibration_positive_rate": float(np.mean(y_cal > 0.0)),
        "mean_mae": float(mean_absolute_error(y_cal, mean_pred)),
        "mean_spearman": safe_spearman(mean_pred, y_cal),
        "q25_pinball": model_pinball,
        "constant_q25_pinball": baseline_pinball,
        "q25_pinball_improvement": pinball_improvement,
        "raw_q25_coverage": float(np.mean(y_cal <= q25_pred)),
        "conformal_lower_coverage": float(np.mean(y_cal < lower)),
        "conformal_correction": float(q25_pred[0] - lower[0]) if lower.size else float("nan"),
        "negative_classifier_enabled": bool(negative_model_enabled),
        "negative_prevalence_fit": prevalence,
        "negative_brier": model_brier,
        "negative_constant_brier": baseline_brier,
        "negative_brier_improvement": brier_improvement,
        "risk_bins": bins,
        "controller_calibration": controller_calibration,
        "data_gate": data_gate,
        "model_gate": model_gate,
        "pilot_gate_pass": bool(data_gate["pass"] and model_gate["pass"]),
    }


def data_sufficiency_gate(
    fit: WindowRiskDataset,
    calibration: WindowRiskDataset,
    controller_calibration: list[dict[str, object]],
) -> dict[str, object]:
    fit_positive_controllers = {
        record.controller_id
        for record in fit.records
        if float(record.margin) > 0.0
    }
    positive_calibration_controllers = [
        str(row["controller_id"])
        for row in controller_calibration
        if float(row["margin_mean"]) > 0.0
    ]
    hard_violations = int(
        sum(record.constraint_violation_count for record in (*fit.records, *calibration.records))
    )
    checks = {
        "fit_starts_at_least_32": int(np.unique(fit.starts).size) >= 32,
        "calibration_starts_at_least_12": int(np.unique(calibration.starts).size) >= 12,
        "fit_positive_rate_at_least_10pct": float(np.mean(fit.margins > 0.0)) >= 0.10,
        "fit_positive_controllers_at_least_2": len(fit_positive_controllers) >= 2,
        "calibration_positive_controller_exists": bool(positive_calibration_controllers),
        "hard_constraint_violations_zero": hard_violations == 0,
    }
    return {
        **checks,
        "fit_positive_controller_count": len(fit_positive_controllers),
        "calibration_positive_controllers": positive_calibration_controllers,
        "hard_constraint_violation_count": hard_violations,
        "pass": bool(all(checks.values())),
    }


def controller_margin_summary(dataset: WindowRiskDataset) -> list[dict[str, object]]:
    frame = pd.DataFrame(
        {
            "controller_id": dataset.controller_ids,
            "margin": np.asarray(dataset.margins, dtype=float),
        }
    )
    rows = []
    for controller_id, group in frame.groupby("controller_id", sort=True):
        values = group["margin"].to_numpy(dtype=float)
        rows.append(
            {
                "controller_id": str(controller_id),
                "rows": int(values.size),
                "margin_mean": float(np.mean(values)),
                "margin_q25": float(np.quantile(values, 0.25)),
                "margin_min": float(np.min(values)),
                "negative_count": int(np.sum(values < 0.0)),
            }
        )
    return rows


def risk_bin_diagnostics(predicted_lower: np.ndarray, margins: np.ndarray, *, bins: int = 4) -> dict[str, object]:
    prediction = np.asarray(predicted_lower, dtype=float).reshape(-1)
    values = np.asarray(margins, dtype=float).reshape(-1)
    if prediction.shape != values.shape or values.size == 0:
        raise ValueError("Risk-bin predictions and margins must have matching nonempty shapes")
    frame = pd.DataFrame({"prediction": prediction, "margin": values})
    unique = int(frame["prediction"].nunique())
    bin_count = min(max(1, int(bins)), unique)
    if bin_count <= 1:
        rows = [
            {
                "bin": 0,
                "rows": int(values.size),
                "predicted_lower_mean": float(np.mean(prediction)),
                "margin_mean": float(np.mean(values)),
                "margin_q25": float(np.quantile(values, 0.25)),
                "negative_rate": float(np.mean(values < 0.0)),
            }
        ]
        return {
            "rows": rows,
            "mean_monotonic_fraction": 0.0,
            "q25_monotonic_fraction": 0.0,
            "negative_rate_monotonic_fraction": 0.0,
            "monotonic_fraction": 0.0,
        }
    frame["bin"] = pd.qcut(frame["prediction"], q=bin_count, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.groupby("bin", sort=True, observed=True):
        rows.append(
            {
                "bin": int(bin_id),
                "rows": int(len(group)),
                "predicted_lower_mean": float(group["prediction"].mean()),
                "margin_mean": float(group["margin"].mean()),
                "margin_q25": float(group["margin"].quantile(0.25)),
                "negative_rate": float(np.mean(group["margin"].to_numpy(dtype=float) < 0.0)),
            }
        )
    mean_comparisons = [
        float(rows[idx + 1]["margin_mean"]) >= float(rows[idx]["margin_mean"])
        for idx in range(len(rows) - 1)
    ]
    q25_comparisons = [
        float(rows[idx + 1]["margin_q25"]) >= float(rows[idx]["margin_q25"])
        for idx in range(len(rows) - 1)
    ]
    negative_rate_comparisons = [
        float(rows[idx + 1]["negative_rate"])
        <= float(rows[idx]["negative_rate"])
        for idx in range(len(rows) - 1)
    ]
    mean_fraction = (
        float(np.mean(mean_comparisons)) if mean_comparisons else 0.0
    )
    q25_fraction = (
        float(np.mean(q25_comparisons)) if q25_comparisons else 0.0
    )
    negative_fraction = (
        float(np.mean(negative_rate_comparisons))
        if negative_rate_comparisons
        else 0.0
    )
    return {
        "rows": rows,
        "mean_monotonic_fraction": mean_fraction,
        "q25_monotonic_fraction": q25_fraction,
        "negative_rate_monotonic_fraction": negative_fraction,
        "monotonic_fraction": min(q25_fraction, negative_fraction),
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "left": np.asarray(left, dtype=float).reshape(-1),
            "right": np.asarray(right, dtype=float).reshape(-1),
        }
    )
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def write_window_risk_model_artifacts(
    out_dir: str | Path,
    *,
    bundle: WindowRiskModelBundle,
    fit: WindowRiskDataset,
    calibration: WindowRiskDataset,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, object],
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle.save(output / "window_risk_model.joblib")
    rows = []
    for idx, record in enumerate(calibration.records):
        rows.append(
            {
                "seed": int(record.seed),
                "start": int(record.start),
                "anchor_action_idx": int(record.anchor_action_idx),
                "controller_id": str(record.controller_id),
                "margin": float(record.margin),
                "mean_margin_pred": float(predictions["mean_margin"][idx]),
                "q25_margin_pred": float(predictions["q25_margin"][idx]),
                "risk_lower_bound": float(predictions["risk_lower_bound"][idx]),
                "negative_probability": float(predictions["negative_probability"][idx]),
            }
        )
    pd.DataFrame(rows).to_csv(output / "window_risk_calibration_predictions.csv", index=False)
    (output / "window_risk_model_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "window_risk_model_schema.json").write_text(
        json.dumps(
            {
                "feature_names": list(bundle.feature_names),
                "feature_count": len(bundle.feature_names),
                "quantile_alpha": float(bundle.quantile_alpha),
                "conformal_correction": float(bundle.conformal_correction),
                "negative_classifier_enabled": bundle.negative_model is not None,
                "fit_rows": int(fit.margins.size),
                "calibration_rows": int(calibration.margins.size),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _validate_datasets(fit: WindowRiskDataset, calibration: WindowRiskDataset) -> None:
    if fit.feature_names != calibration.feature_names:
        raise ValueError("Fit and calibration feature schemas differ")
    if fit.features.ndim != 2 or calibration.features.ndim != 2:
        raise ValueError("Window-risk features must be 2D")
    if fit.features.shape[1] != calibration.features.shape[1]:
        raise ValueError("Fit and calibration feature widths differ")
    if fit.margins.size == 0 or calibration.margins.size == 0:
        raise ValueError("Fit and calibration datasets must be nonempty")
    if not np.all(np.isfinite(fit.features)) or not np.all(np.isfinite(calibration.features)):
        raise ValueError("Window-risk features must be finite")
    if not np.all(np.isfinite(fit.margins)) or not np.all(np.isfinite(calibration.margins)):
        raise ValueError("Window-risk margins must be finite")
