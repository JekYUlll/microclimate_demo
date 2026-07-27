from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_forecaster import ContinuousForecastNet
from .robust_planner import CausalWorldModelContext, ScenarioBatch
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.rollout import RolloutResult  # noqa: E402


@dataclass(frozen=True)
class RolloutWorldModelTrainingConfig:
    horizon: int = 12
    lookback: int = 20
    hidden_dim: int = 128
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    member_count: int = 5
    bootstrap_fraction: float = 0.85
    residual_scale: float = 1.0
    seed: int = 42
    device: str = "auto"
    period_steps: int = 10800
    event_probability_horizon: int = 0


@dataclass
class RolloutWorldModelDataset:
    features: np.ndarray
    targets: np.ndarray
    physical_targets: np.ndarray
    state_columns: tuple[str, ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    cfg: RolloutWorldModelTrainingConfig


@dataclass
class RolloutWorldModelMember:
    model: Any
    history: dict[str, list[float]]


@dataclass
class RolloutWorldModel:
    members: tuple[RolloutWorldModelMember, ...]
    residual_bank: np.ndarray
    state_columns: tuple[str, ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    cfg: RolloutWorldModelTrainingConfig
    audit_metrics: dict[str, Any]

    def predict_members(self, context: CausalWorldModelContext) -> np.ndarray:
        if tuple(context.state_columns) != tuple(self.state_columns):
            raise ValueError("world-model context state columns do not match")
        feature = build_rollout_world_model_feature(
            context.history,
            context.mask_history,
            current_idx=int(context.current_idx),
            state_mean=self.state_mean,
            state_std=self.state_std,
            cfg=self.cfg,
            event_probabilities=context.event_probabilities,
        ).reshape(1, -1)
        torch = _torch_module()
        predictions: list[np.ndarray] = []
        for member in self.members:
            device = next(member.model.parameters()).device
            with torch.no_grad():
                normalized = (
                    member.model(
                        torch.as_tensor(feature, dtype=torch.float32, device=device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            values = (
                normalized * self.target_std.reshape(1, -1)
                + self.target_mean.reshape(1, -1)
            )
            predictions.append(
                values.reshape(int(self.cfg.horizon), len(self.state_columns))
            )
        return np.stack(predictions, axis=0).astype(float)

    def sample(
        self,
        context: CausalWorldModelContext,
        *,
        horizon: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> ScenarioBatch:
        required_future = max(0, int(horizon) - 1)
        if required_future > int(self.cfg.horizon):
            raise ValueError(
                f"requested future horizon {required_future} exceeds model "
                f"horizon {int(self.cfg.horizon)}"
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
        probabilities = _event_probabilities(
            context.event_probabilities,
            horizon=int(horizon),
        )
        events = rng.random((count, int(horizon))) < probabilities.reshape(1, -1)
        batch = ScenarioBatch(
            values=values.astype(float),
            event_flags=events.astype(bool),
            state_columns=self.state_columns,
        )
        batch.validate(n_scenarios=count, min_horizon=int(horizon))
        return batch


def build_rollout_world_model_dataset(
    rollouts: tuple[RolloutResult, ...] | list[RolloutResult],
    *,
    state_columns: tuple[str, ...],
    cfg: RolloutWorldModelTrainingConfig,
    event_probability_values: np.ndarray | None = None,
    normalization: RolloutWorldModelDataset | None = None,
) -> RolloutWorldModelDataset:
    if not rollouts:
        raise ValueError("rollout world-model dataset requires at least one rollout")
    horizon = max(1, int(cfg.horizon))
    lookback = max(1, int(cfg.lookback))
    state_columns = tuple(str(name) for name in state_columns)
    if normalization is None:
        observed = np.concatenate([result.observations for result in rollouts], axis=0)
        targets_raw = np.concatenate([result.truth for result in rollouts], axis=0)
        state_mean = np.nanmean(observed, axis=0).astype(np.float32)
        state_std = np.nanstd(observed, axis=0).astype(np.float32)
        target_base = _collect_physical_targets(
            rollouts,
            horizon=horizon,
            lookback=lookback,
        )
        target_flat = target_base.reshape(target_base.shape[0], -1)
        target_mean = np.nanmean(target_flat, axis=0).astype(np.float32)
        target_std = np.nanstd(target_flat, axis=0).astype(np.float32)
        state_mean = np.where(np.isfinite(state_mean), state_mean, 0.0).astype(np.float32)
        state_std = np.where(state_std > 1.0e-6, state_std, 1.0).astype(np.float32)
        target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0).astype(np.float32)
        target_std = np.where(target_std > 1.0e-6, target_std, 1.0).astype(np.float32)
    else:
        if tuple(normalization.state_columns) != state_columns:
            raise ValueError("normalization state columns do not match")
        state_mean = np.asarray(normalization.state_mean, dtype=np.float32)
        state_std = np.asarray(normalization.state_std, dtype=np.float32)
        target_mean = np.asarray(normalization.target_mean, dtype=np.float32)
        target_std = np.asarray(normalization.target_std, dtype=np.float32)

    features: list[np.ndarray] = []
    physical_targets: list[np.ndarray] = []
    for result in rollouts:
        obs = np.asarray(result.observations, dtype=np.float32)
        masks = np.asarray(result.masks, dtype=np.float32)
        truth = np.asarray(result.truth, dtype=np.float32)
        steps = np.asarray(result.step_indices, dtype=int)
        if obs.shape != masks.shape or obs.shape != truth.shape:
            raise ValueError("rollout observations/masks/truth must have matching shapes")
        for pos in range(lookback - 1, obs.shape[0] - horizon):
            idx = int(steps[pos])
            event_probabilities = _event_lookup(
                event_probability_values,
                idx=idx,
                horizon=int(cfg.event_probability_horizon),
            )
            features.append(
                build_rollout_world_model_feature(
                    obs[pos - lookback + 1 : pos + 1],
                    masks[pos - lookback + 1 : pos + 1],
                    current_idx=idx,
                    state_mean=state_mean,
                    state_std=state_std,
                    cfg=cfg,
                    event_probabilities=event_probabilities,
                )
            )
            physical_targets.append(truth[pos + 1 : pos + 1 + horizon])
    if not features:
        raise ValueError("rollouts are too short for the requested lookback/horizon")
    physical = np.stack(physical_targets, axis=0).astype(np.float32)
    target_flat = physical.reshape(physical.shape[0], -1)
    normalized_targets = ((target_flat - target_mean.reshape(1, -1)) / target_std.reshape(1, -1)).astype(np.float32)
    return RolloutWorldModelDataset(
        features=np.vstack(features).astype(np.float32),
        targets=np.where(np.isfinite(normalized_targets), normalized_targets, 0.0).astype(np.float32),
        physical_targets=physical,
        state_columns=state_columns,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std,
        cfg=cfg,
    )


def train_rollout_world_model(
    *,
    fit_dataset: RolloutWorldModelDataset,
    calibration_dataset: RolloutWorldModelDataset,
    audit_dataset: RolloutWorldModelDataset,
    cfg: RolloutWorldModelTrainingConfig,
) -> RolloutWorldModel:
    members = _train_members(fit_dataset, cfg)
    calibration_predictions = _predict_dataset_members(
        calibration_dataset.features,
        members,
        horizon=int(cfg.horizon),
        n_state=len(fit_dataset.state_columns),
        target_mean=fit_dataset.target_mean,
        target_std=fit_dataset.target_std,
    )
    calibration_truth = calibration_dataset.physical_targets
    residual_bank = (
        calibration_truth - np.mean(calibration_predictions, axis=0)
    ).astype(np.float32)
    audit_metrics = audit_rollout_world_model(
        audit_dataset,
        members=members,
        residual_bank=residual_bank,
    )
    raw_fit = fit_dataset.physical_targets.reshape(-1, len(fit_dataset.state_columns))
    lower, upper = _robust_state_bounds(raw_fit)
    return RolloutWorldModel(
        members=members,
        residual_bank=residual_bank,
        state_columns=fit_dataset.state_columns,
        state_mean=fit_dataset.state_mean,
        state_std=fit_dataset.state_std,
        target_mean=fit_dataset.target_mean,
        target_std=fit_dataset.target_std,
        lower_bounds=lower,
        upper_bounds=upper,
        cfg=cfg,
        audit_metrics=audit_metrics,
    )


def audit_rollout_world_model(
    dataset: RolloutWorldModelDataset,
    *,
    members: tuple[RolloutWorldModelMember, ...],
    residual_bank: np.ndarray,
) -> dict[str, Any]:
    predictions = _predict_dataset_members(
        dataset.features,
        members,
        horizon=int(dataset.cfg.horizon),
        n_state=len(dataset.state_columns),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
    )
    mean_prediction = np.mean(predictions, axis=0)
    truth = dataset.physical_targets
    horizon = int(dataset.cfg.horizon)
    target_scale = np.maximum(
        dataset.target_std.reshape(horizon, len(dataset.state_columns)),
        1.0e-6,
    )
    error = (mean_prediction - truth) / target_scale.reshape(1, *target_scale.shape)
    current = _current_values_from_features(dataset)
    persistence = np.repeat(current.reshape(current.shape[0], 1, current.shape[1]), horizon, axis=1)
    persistence_error = (persistence - truth) / target_scale.reshape(1, *target_scale.shape)
    residual_low = np.quantile(residual_bank, 0.10, axis=0)
    residual_high = np.quantile(residual_bank, 0.90, axis=0)
    covered = (truth >= mean_prediction + residual_low) & (
        truth <= mean_prediction + residual_high
    )
    rmse = float(np.sqrt(np.mean(error**2)))
    persistence_rmse = float(np.sqrt(np.mean(persistence_error**2)))
    return {
        "normalized_rmse": rmse,
        "persistence_normalized_rmse": persistence_rmse,
        "rmse_skill_vs_persistence": float(
            1.0 - rmse / max(persistence_rmse, 1.0e-12)
        ),
        "interval_80_coverage": float(np.mean(covered)),
        "target_normalized_rmse": {
            name: float(np.sqrt(np.mean(error[:, :, idx] ** 2)))
            for idx, name in enumerate(dataset.state_columns)
        },
    }


def build_rollout_world_model_feature(
    history: np.ndarray,
    mask_history: np.ndarray,
    *,
    current_idx: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cfg: RolloutWorldModelTrainingConfig,
    event_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float32)
    masks = np.asarray(mask_history, dtype=np.float32)
    if values.shape != masks.shape or values.ndim != 2:
        raise ValueError("history and mask_history must be matching 2D arrays")
    lookback = max(1, int(cfg.lookback))
    if values.shape[0] < lookback:
        pad = lookback - values.shape[0]
        values = np.vstack([np.repeat(values[:1], pad, axis=0), values])
        masks = np.vstack([np.zeros((pad, masks.shape[1]), dtype=np.float32), masks])
    elif values.shape[0] > lookback:
        values = values[-lookback:]
        masks = masks[-lookback:]
    mean = np.asarray(state_mean, dtype=np.float32).reshape(1, -1)
    std = np.maximum(np.asarray(state_std, dtype=np.float32).reshape(1, -1), 1.0e-6)
    normalized = np.where(np.isfinite(values), values, mean)
    normalized = ((normalized - mean) / std).astype(np.float32)
    current_mask = np.clip(masks[-1], 0.0, 1.0).astype(np.float32)
    observed_ratio = np.mean(np.clip(masks, 0.0, 1.0), axis=0).astype(np.float32)
    ages = []
    for column in range(masks.shape[1]):
        observed = np.flatnonzero(masks[:, column] > 0.5)
        age = lookback if observed.size == 0 else lookback - 1 - int(observed[-1])
        ages.append(float(age) / float(max(1, lookback)))
    event = _event_lookup_from_context(
        event_probabilities,
        horizon=int(cfg.event_probability_horizon),
    )
    phase = _phase_features(int(current_idx), max(1, int(cfg.period_steps)))
    return np.concatenate(
        [
            normalized.reshape(-1),
            np.clip(masks, 0.0, 1.0).reshape(-1),
            current_mask,
            observed_ratio,
            np.asarray(ages, dtype=np.float32),
            event,
            np.asarray(phase, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def save_rollout_world_model(model: RolloutWorldModel, path: str | Path) -> None:
    import torch

    members = []
    for member in model.members:
        members.append(
            {
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
        "model_type": "rollout_world_model",
        "members": members,
        "residual_bank": np.asarray(model.residual_bank, dtype=np.float32),
        "state_columns": tuple(model.state_columns),
        "state_mean": np.asarray(model.state_mean, dtype=np.float32),
        "state_std": np.asarray(model.state_std, dtype=np.float32),
        "target_mean": np.asarray(model.target_mean, dtype=np.float32),
        "target_std": np.asarray(model.target_std, dtype=np.float32),
        "lower_bounds": np.asarray(model.lower_bounds, dtype=np.float32),
        "upper_bounds": np.asarray(model.upper_bounds, dtype=np.float32),
        "cfg": asdict(model.cfg),
        "audit_metrics": model.audit_metrics,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_rollout_world_model(path: str | Path, *, device: str = "cpu") -> RolloutWorldModel:
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != "rollout_world_model":
        raise ValueError("not a rollout world-model checkpoint")
    cfg = RolloutWorldModelTrainingConfig(**payload["cfg"])
    members = []
    for item in payload["members"]:
        model = ContinuousForecastNet(
            input_dim=int(item["input_dim"]),
            hidden_dim=int(cfg.hidden_dim),
            output_dim=int(item["output_dim"]),
        ).to(device)
        model.load_state_dict(item["state_dict"])
        members.append(
            RolloutWorldModelMember(
                model=model.eval(),
                history={
                    str(key): [float(value) for value in values]
                    for key, values in item["history"].items()
                },
            )
        )
    return RolloutWorldModel(
        members=tuple(members),
        residual_bank=np.asarray(payload["residual_bank"], dtype=np.float32),
        state_columns=tuple(payload["state_columns"]),
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
        state_std=np.asarray(payload["state_std"], dtype=np.float32),
        target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
        target_std=np.asarray(payload["target_std"], dtype=np.float32),
        lower_bounds=np.asarray(payload["lower_bounds"], dtype=float),
        upper_bounds=np.asarray(payload["upper_bounds"], dtype=float),
        cfg=cfg,
        audit_metrics=dict(payload["audit_metrics"]),
    )


def _train_members(
    dataset: RolloutWorldModelDataset,
    cfg: RolloutWorldModelTrainingConfig,
) -> tuple[RolloutWorldModelMember, ...]:
    torch = _torch_module()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = _select_device(torch, str(cfg.device))
    x = np.asarray(dataset.features, dtype=np.float32)
    y = np.asarray(dataset.targets, dtype=np.float32)
    rows = x.shape[0]
    sample_count = max(2, int(np.ceil(rows * float(cfg.bootstrap_fraction))))
    members: list[RolloutWorldModelMember] = []
    for member_idx in range(max(1, int(cfg.member_count))):
        rng = np.random.default_rng(int(cfg.seed) + 1009 * member_idx)
        indices = rng.integers(0, rows, size=sample_count)
        torch.manual_seed(int(cfg.seed) + member_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed) + member_idx)
        model = ContinuousForecastNet(
            input_dim=x.shape[1],
            hidden_dim=int(cfg.hidden_dim),
            output_dim=y.shape[1],
        ).to(device)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(x[indices], dtype=torch.float32),
                torch.as_tensor(y[indices], dtype=torch.float32),
            ),
            batch_size=max(1, int(cfg.batch_size)),
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        loss_fn = nn.SmoothL1Loss()
        history = {"loss": [], "rmse": []}
        for _ in range(max(1, int(cfg.epochs))):
            losses: list[float] = []
            rmses: list[float] = []
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                with torch.no_grad():
                    rmses.append(float(torch.sqrt(torch.mean((pred - yb) ** 2)).detach().cpu().item()))
            history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
            history["rmse"].append(float(np.mean(rmses)) if rmses else float("nan"))
        members.append(RolloutWorldModelMember(model=model.eval(), history=history))
    return tuple(members)


def _predict_dataset_members(
    features: np.ndarray,
    members: tuple[RolloutWorldModelMember, ...],
    *,
    horizon: int,
    n_state: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    torch = _torch_module()
    x = np.asarray(features, dtype=np.float32)
    predictions: list[np.ndarray] = []
    for member in members:
        device = next(member.model.parameters()).device
        with torch.no_grad():
            pred = member.model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        physical = (
            pred.astype(np.float32) * np.asarray(target_std, dtype=np.float32).reshape(1, -1)
            + np.asarray(target_mean, dtype=np.float32).reshape(1, -1)
        )
        predictions.append(physical.astype(np.float32))
    stacked = np.stack(predictions, axis=0)
    return stacked.reshape(len(members), x.shape[0], int(horizon), int(n_state))


def _current_values_from_features(dataset: RolloutWorldModelDataset) -> np.ndarray:
    d = len(dataset.state_columns)
    lookback = int(dataset.cfg.lookback)
    history_flat = dataset.features[:, : lookback * d]
    current_z = history_flat.reshape(dataset.features.shape[0], lookback, d)[:, -1, :]
    return current_z * dataset.state_std.reshape(1, -1) + dataset.state_mean.reshape(1, -1)


def _collect_physical_targets(
    rollouts: tuple[RolloutResult, ...] | list[RolloutResult],
    *,
    horizon: int,
    lookback: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for result in rollouts:
        truth = np.asarray(result.truth, dtype=np.float32)
        for pos in range(max(0, int(lookback) - 1), truth.shape[0] - int(horizon)):
            rows.append(truth[pos + 1 : pos + 1 + int(horizon)])
    if not rows:
        raise ValueError("rollouts are too short to collect physical targets")
    return np.stack(rows, axis=0).astype(np.float32)


def _event_lookup(
    values: np.ndarray | None,
    *,
    idx: int,
    horizon: int,
) -> np.ndarray:
    if values is None or int(horizon) <= 0:
        return np.zeros(max(0, int(horizon)), dtype=np.float32)
    table = np.asarray(values, dtype=np.float32)
    if table.ndim == 1:
        table = table.reshape(-1, 1)
    if idx < 0 or idx >= table.shape[0]:
        return np.zeros(int(horizon), dtype=np.float32)
    return _event_lookup_from_context(table[int(idx)], horizon=int(horizon))


def _event_lookup_from_context(values: np.ndarray | None, *, horizon: int) -> np.ndarray:
    if int(horizon) <= 0:
        return np.zeros(0, dtype=np.float32)
    if values is None:
        return np.zeros(int(horizon), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(int(horizon), dtype=np.float32)
    if arr.size >= int(horizon):
        return np.clip(arr[: int(horizon)], 0.0, 1.0).astype(np.float32)
    return np.clip(np.pad(arr, (0, int(horizon) - arr.size), mode="edge"), 0.0, 1.0).astype(np.float32)


def _event_probabilities(values: np.ndarray, *, horizon: int) -> np.ndarray:
    return _event_lookup_from_context(values, horizon=int(horizon))


def _phase_features(idx: int, period_steps: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * float(int(idx) % max(1, int(period_steps))) / float(max(1, int(period_steps)))
    return float(np.sin(angle)), float(np.cos(angle))


def _robust_state_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    lower = np.nanquantile(raw, 0.001, axis=0)
    upper = np.nanquantile(raw, 0.999, axis=0)
    span = np.maximum(upper - lower, 1.0e-6)
    return (lower - 0.10 * span).astype(float), (upper + 0.10 * span).astype(float)


def _torch_module() -> Any:
    import torch

    return torch


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
