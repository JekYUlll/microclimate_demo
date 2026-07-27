from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .continuous_forecaster import (
    ContinuousForecastDataset,
    ContinuousForecastNet,
    ContinuousForecasterBundle,
    ContinuousForecasterTrainingConfig,
    build_continuous_forecast_dataset,
    predict_continuous_features,
    predict_continuous_from_history,
    train_continuous_forecaster,
)
from .robust_planner import CausalWorldModelContext, ScenarioBatch


@dataclass(frozen=True)
class ProbabilisticWorldModelTrainingConfig:
    member_count: int = 5
    fit_fraction: float = 0.70
    calibration_fraction: float = 0.15
    bootstrap_fraction: float = 0.85
    residual_scale: float = 1.0
    seed: int = 42
    forecaster: ContinuousForecasterTrainingConfig = (
        ContinuousForecasterTrainingConfig()
    )


@dataclass
class ProbabilisticWorldModel:
    members: tuple[ContinuousForecasterBundle, ...]
    residual_bank: np.ndarray
    state_columns: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    cfg: ProbabilisticWorldModelTrainingConfig
    audit_metrics: dict[str, Any]

    def predict_members(
        self,
        context: CausalWorldModelContext,
    ) -> np.ndarray:
        if tuple(context.state_columns) != tuple(self.state_columns):
            raise ValueError("world-model context state columns do not match")
        return np.stack(
            [
                predict_continuous_from_history(
                    context.history,
                    history_columns=context.state_columns,
                    current_idx=context.current_idx,
                    bundle=member,
                )
                for member in self.members
            ],
            axis=0,
        ).astype(float)

    def sample(
        self,
        context: CausalWorldModelContext,
        *,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ScenarioBatch:
        if tuple(context.state_columns) != tuple(self.state_columns):
            raise ValueError("world-model context state columns do not match")
        required_future = max(0, int(horizon) - 1)
        model_horizon = int(self.cfg.forecaster.horizon)
        if required_future > model_horizon:
            raise ValueError(
                f"requested future horizon {required_future} exceeds model horizon "
                f"{model_horizon}"
            )
        predictions = self.predict_members(context)
        count = max(1, int(n_scenarios))
        member_indices = rng.integers(0, len(self.members), size=count)
        residual_indices = rng.integers(0, len(self.residual_bank), size=count)
        future = (
            predictions[member_indices]
            + float(self.cfg.residual_scale) * self.residual_bank[residual_indices]
        )
        future = np.clip(
            future,
            self.lower_bounds.reshape(1, 1, -1),
            self.upper_bounds.reshape(1, 1, -1),
        )
        current = np.repeat(
            np.asarray(context.last_observation, dtype=float).reshape(1, 1, -1),
            count,
            axis=0,
        )
        values = np.concatenate([current, future[:, :required_future]], axis=1)
        event_probabilities = _scenario_event_probabilities(
            context.event_probabilities,
            horizon=int(horizon),
        )
        events = rng.random((count, int(horizon))) < event_probabilities.reshape(
            1, -1
        )
        batch = ScenarioBatch(
            values=values.astype(float),
            event_flags=events.astype(bool),
            state_columns=self.state_columns,
        )
        batch.validate(n_scenarios=count, min_horizon=int(horizon))
        return batch


def train_probabilistic_world_model(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    state_columns: tuple[str, ...],
    cfg: ProbabilisticWorldModelTrainingConfig,
) -> ProbabilisticWorldModel:
    start, end = int(bounds[0]), int(bounds[1])
    if end <= start:
        raise ValueError("world-model bounds must be increasing")
    fit_end = start + int((end - start) * float(cfg.fit_fraction))
    calibration_end = fit_end + int(
        (end - start) * float(cfg.calibration_fraction)
    )
    horizon = max(1, int(cfg.forecaster.horizon))
    minimum = horizon + max(4, int(cfg.forecaster.lookback))
    if min(fit_end - start, calibration_end - fit_end, end - calibration_end) < minimum:
        raise ValueError("world-model chronological partitions are too short")
    columns = tuple(str(name) for name in state_columns)
    fit_dataset = build_continuous_forecast_dataset(
        truth,
        bounds=(start, fit_end),
        feature_columns=columns,
        target_columns=columns,
        cfg=cfg.forecaster,
    )
    calibration_dataset = build_continuous_forecast_dataset(
        truth,
        bounds=(fit_end, calibration_end),
        feature_columns=columns,
        target_columns=columns,
        cfg=cfg.forecaster,
        normalization=fit_dataset,
    )
    audit_dataset = build_continuous_forecast_dataset(
        truth,
        bounds=(calibration_end, end),
        feature_columns=columns,
        target_columns=columns,
        cfg=cfg.forecaster,
        normalization=fit_dataset,
    )
    members = _train_bootstrap_members(fit_dataset, cfg)
    calibration_predictions = _ensemble_predictions(
        calibration_dataset.features,
        members,
    )
    calibration_truth = _physical_targets(calibration_dataset)
    residual_bank = (
        calibration_truth - np.mean(calibration_predictions, axis=0)
    ).astype(np.float32)
    raw_fit = truth.loc[start : fit_end - 1, columns].to_numpy(dtype=float)
    lower, upper = _robust_state_bounds(raw_fit)
    audit_metrics = _audit_world_model(
        audit_dataset,
        members,
        residual_bank,
    )
    audit_metrics["bounds"] = {
        "fit": [start, fit_end],
        "calibration": [fit_end, calibration_end],
        "audit": [calibration_end, end],
    }
    audit_metrics["fit_rows"] = int(fit_dataset.features.shape[0])
    audit_metrics["calibration_rows"] = int(
        calibration_dataset.features.shape[0]
    )
    audit_metrics["audit_rows"] = int(audit_dataset.features.shape[0])
    return ProbabilisticWorldModel(
        members=members,
        residual_bank=residual_bank,
        state_columns=columns,
        lower_bounds=lower,
        upper_bounds=upper,
        cfg=cfg,
        audit_metrics=audit_metrics,
    )


def save_probabilistic_world_model(
    model: ProbabilisticWorldModel,
    path: str | Path,
) -> None:
    import torch

    members: list[dict[str, Any]] = []
    for member in model.members:
        members.append(
            {
                "feature_columns": tuple(member.feature_columns),
                "target_columns": tuple(member.target_columns),
                "feature_mean": np.asarray(member.feature_mean, dtype=np.float32),
                "feature_std": np.asarray(member.feature_std, dtype=np.float32),
                "target_mean": np.asarray(member.target_mean, dtype=np.float32),
                "target_std": np.asarray(member.target_std, dtype=np.float32),
                "cfg": asdict(member.cfg),
                "history": member.history,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in member.model.state_dict().items()
                },
                "input_dim": int(member.model.input_dim),
                "output_dim": int(member.model.output_dim),
            }
        )
    payload = {
        "format_version": 1,
        "members": members,
        "residual_bank": np.asarray(model.residual_bank, dtype=np.float32),
        "state_columns": tuple(model.state_columns),
        "lower_bounds": np.asarray(model.lower_bounds, dtype=np.float32),
        "upper_bounds": np.asarray(model.upper_bounds, dtype=np.float32),
        "cfg": {
            **asdict(model.cfg),
            "forecaster": asdict(model.cfg.forecaster),
        },
        "audit_metrics": model.audit_metrics,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_probabilistic_world_model(
    path: str | Path,
    *,
    device: str = "cpu",
) -> ProbabilisticWorldModel:
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported probabilistic world-model format")
    member_bundles: list[ContinuousForecasterBundle] = []
    for item in payload["members"]:
        member_cfg = ContinuousForecasterTrainingConfig(**item["cfg"])
        network = ContinuousForecastNet(
            input_dim=int(item["input_dim"]),
            hidden_dim=int(member_cfg.hidden_dim),
            output_dim=int(item["output_dim"]),
        ).to(device)
        network.load_state_dict(item["state_dict"])
        member_bundles.append(
            ContinuousForecasterBundle(
                model=network.eval(),
                feature_columns=tuple(item["feature_columns"]),
                target_columns=tuple(item["target_columns"]),
                feature_mean=np.asarray(item["feature_mean"], dtype=np.float32),
                feature_std=np.asarray(item["feature_std"], dtype=np.float32),
                target_mean=np.asarray(item["target_mean"], dtype=np.float32),
                target_std=np.asarray(item["target_std"], dtype=np.float32),
                cfg=member_cfg,
                history={
                    str(key): [float(value) for value in values]
                    for key, values in item["history"].items()
                },
            )
        )
    config = dict(payload["cfg"])
    forecaster_cfg = ContinuousForecasterTrainingConfig(
        **config.pop("forecaster")
    )
    training_cfg = ProbabilisticWorldModelTrainingConfig(
        **config,
        forecaster=forecaster_cfg,
    )
    return ProbabilisticWorldModel(
        members=tuple(member_bundles),
        residual_bank=np.asarray(payload["residual_bank"], dtype=np.float32),
        state_columns=tuple(payload["state_columns"]),
        lower_bounds=np.asarray(payload["lower_bounds"], dtype=float),
        upper_bounds=np.asarray(payload["upper_bounds"], dtype=float),
        cfg=training_cfg,
        audit_metrics=dict(payload["audit_metrics"]),
    )


def _train_bootstrap_members(
    dataset: ContinuousForecastDataset,
    cfg: ProbabilisticWorldModelTrainingConfig,
) -> tuple[ContinuousForecasterBundle, ...]:
    count = max(1, int(cfg.member_count))
    rows = dataset.features.shape[0]
    sample_count = max(2, int(np.ceil(rows * float(cfg.bootstrap_fraction))))
    members: list[ContinuousForecasterBundle] = []
    for member_idx in range(count):
        rng = np.random.default_rng(int(cfg.seed) + 1009 * member_idx)
        indices = rng.integers(0, rows, size=sample_count)
        member_dataset = replace(
            dataset,
            features=np.asarray(dataset.features[indices], dtype=np.float32),
            targets=np.asarray(dataset.targets[indices], dtype=np.float32),
        )
        member_cfg = replace(
            cfg.forecaster,
            seed=int(cfg.seed) + member_idx,
        )
        members.append(train_continuous_forecaster(member_dataset, member_cfg))
    return tuple(members)


def _ensemble_predictions(
    features: np.ndarray,
    members: tuple[ContinuousForecasterBundle, ...],
) -> np.ndarray:
    return np.stack(
        [predict_continuous_features(features, member) for member in members],
        axis=0,
    ).astype(np.float32)


def _physical_targets(dataset: ContinuousForecastDataset) -> np.ndarray:
    rows = dataset.targets.shape[0]
    horizon = int(dataset.targets.shape[1] // len(dataset.target_columns))
    values = (
        dataset.targets * dataset.target_std.reshape(1, -1)
        + dataset.target_mean.reshape(1, -1)
    )
    return values.reshape(rows, horizon, len(dataset.target_columns)).astype(
        np.float32
    )


def _audit_world_model(
    dataset: ContinuousForecastDataset,
    members: tuple[ContinuousForecasterBundle, ...],
    residual_bank: np.ndarray,
) -> dict[str, Any]:
    predictions = _ensemble_predictions(dataset.features, members)
    mean_prediction = np.mean(predictions, axis=0)
    truth = _physical_targets(dataset)
    target_scale = np.maximum(
        np.asarray(dataset.target_std, dtype=float).reshape(
            int(dataset.target_std.size // len(dataset.target_columns)),
            len(dataset.target_columns),
        ),
        1.0e-6,
    )
    normalized_error = (mean_prediction - truth) / target_scale.reshape(
        1, *target_scale.shape
    )
    current = (
        dataset.features[:, : len(dataset.feature_columns)]
        * dataset.feature_std.reshape(1, -1)
        + dataset.feature_mean.reshape(1, -1)
    )
    persistence = np.repeat(
        current.reshape(current.shape[0], 1, current.shape[1]),
        truth.shape[1],
        axis=1,
    )
    persistence_error = (persistence - truth) / target_scale.reshape(
        1, *target_scale.shape
    )
    residual_low = np.quantile(residual_bank, 0.10, axis=0)
    residual_high = np.quantile(residual_bank, 0.90, axis=0)
    covered = (truth >= mean_prediction + residual_low) & (
        truth <= mean_prediction + residual_high
    )
    return {
        "normalized_rmse": float(np.sqrt(np.mean(normalized_error**2))),
        "persistence_normalized_rmse": float(
            np.sqrt(np.mean(persistence_error**2))
        ),
        "rmse_skill_vs_persistence": float(
            1.0
            - np.sqrt(np.mean(normalized_error**2))
            / max(np.sqrt(np.mean(persistence_error**2)), 1.0e-12)
        ),
        "interval_80_coverage": float(np.mean(covered)),
        "member_spread_normalized": float(
            np.mean(np.std(predictions, axis=0) / target_scale.reshape(1, *target_scale.shape))
        ),
        "target_normalized_rmse": {
            name: float(np.sqrt(np.mean(normalized_error[:, :, idx] ** 2)))
            for idx, name in enumerate(dataset.target_columns)
        },
    }


def _robust_state_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    lower = np.nanquantile(raw, 0.001, axis=0)
    upper = np.nanquantile(raw, 0.999, axis=0)
    span = np.maximum(upper - lower, 1.0e-6)
    return (lower - 0.10 * span).astype(float), (upper + 0.10 * span).astype(
        float
    )


def _scenario_event_probabilities(
    probabilities: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(int(horizon), dtype=float)
    if values.size >= int(horizon):
        return np.clip(values[: int(horizon)], 0.0, 1.0)
    return np.clip(
        np.pad(values, (0, int(horizon) - values.size), mode="edge"),
        0.0,
        1.0,
    )
