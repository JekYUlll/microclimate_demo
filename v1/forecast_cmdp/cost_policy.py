from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import ForecastContextConfig, append_event_forecast, build_event_forecast
from .mpc_teacher import (
    MpcTeacherConfig,
    _rollout_repeated_mask_cost,
    beam_search_first_action_costs,
    beam_search_teacher_action,
    restore_env,
    snapshot_env,
)
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.custom_ppo import feasible_candidate_mask  # noqa: E402
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class ActionCostTrainingConfig:
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    device: str = "auto"
    ensemble_size: int = 1
    bootstrap_fraction: float = 0.85


@dataclass
class ActionCostDataset:
    inputs: np.ndarray
    costs: np.ndarray
    feature_dim: int
    n_sensors: int


@dataclass
class AnchorAdvantageDataset:
    inputs: np.ndarray
    advantages: np.ndarray
    feature_dim: int
    n_sensors: int
    anchor_idx: int


@dataclass
class FeatureTransitionDataset:
    inputs: np.ndarray
    deltas: np.ndarray
    feature_dim: int
    n_sensors: int


def collect_action_cost_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
) -> ActionCostDataset:
    masks = np.asarray(candidate_masks, dtype=bool)
    rows: list[np.ndarray] = []
    costs_out: list[float] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(env._state().astype(np.float32), forecast)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            finite = np.flatnonzero(np.isfinite(costs))
            if finite.size:
                finite_costs = costs[finite].astype(float)
                center = float(np.min(finite_costs))
                spread = float(np.std(finite_costs))
                scale = spread if spread > 1.0e-6 else 1.0
                for action_idx in finite:
                    action_features = masks[int(action_idx)].astype(np.float32)
                    rows.append(np.concatenate([feature, action_features], axis=0).astype(np.float32))
                    costs_out.append(float((costs[int(action_idx)] - center) / scale))
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No action-cost rows were collected")
    return ActionCostDataset(
        inputs=np.vstack(rows).astype(np.float32),
        costs=np.asarray(costs_out, dtype=np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
    )


def collect_anchor_advantage_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    anchor_mask: tuple[bool, ...] | np.ndarray,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
) -> AnchorAdvantageDataset:
    masks = np.asarray(candidate_masks, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    anchor_idx = _candidate_index(masks, anchor)
    if anchor_idx is None:
        raise ValueError("Anchor mask is not present in candidate_masks")
    anchor_features = anchor.astype(np.float32)
    rows: list[np.ndarray] = []
    advantages_out: list[float] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
            feature = append_event_forecast(env._state().astype(np.float32), forecast)
            feature_dim = int(feature.shape[0])
            costs = beam_search_first_action_costs(env, masks, teacher_cfg)
            finite = np.flatnonzero(np.isfinite(costs))
            anchor_cost = _anchor_rollout_cost(env, anchor, masks, teacher_cfg)
            if np.isfinite(anchor_cost):
                scale_values = costs[finite].astype(float) if finite.size else np.asarray([], dtype=float)
                scale_values = np.concatenate([scale_values, np.asarray([float(anchor_cost)], dtype=float)])
                spread = float(np.std(scale_values))
                scale = spread if spread > 1.0e-6 else 1.0
                for action_idx in finite:
                    action_features = masks[int(action_idx)].astype(np.float32)
                    advantage = 0.0 if int(action_idx) == int(anchor_idx) else (anchor_cost - float(costs[int(action_idx)])) / scale
                    rows.append(
                        np.concatenate(
                            [
                                feature,
                                action_features,
                                anchor_features,
                                action_features - anchor_features,
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    advantages_out.append(float(advantage))
                if not np.any(finite == int(anchor_idx)):
                    rows.append(
                        np.concatenate(
                            [
                                feature,
                                anchor_features,
                                anchor_features,
                                np.zeros_like(anchor_features),
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    advantages_out.append(0.0)
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No anchor-advantage rows were collected")
    return AnchorAdvantageDataset(
        inputs=np.vstack(rows).astype(np.float32),
        advantages=np.asarray(advantages_out, dtype=np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
        anchor_idx=int(anchor_idx),
    )


def collect_feature_transition_dataset(
    env: WarmupSchedulingEnv,
    candidate_masks: np.ndarray,
    *,
    start_indices: tuple[int, ...],
    steps_per_start: int,
    teacher_cfg: MpcTeacherConfig,
    forecast_cfg: ForecastContextConfig,
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None,
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None,
) -> FeatureTransitionDataset:
    """Collect causal one-step feature transitions for a learned planner.

    Training may use the simulator/truth split to observe next features under
    candidate actions. Deployed policies only use the fitted transition model.
    """

    masks = np.asarray(candidate_masks, dtype=bool)
    allowed = _allowed_action_mask(allowed_action_indices, masks.shape[0])
    if anchor_mask is not None:
        anchor_idx = _candidate_index(masks, np.asarray(anchor_mask, dtype=bool).reshape(-1))
        if anchor_idx is not None:
            allowed[int(anchor_idx)] = True
    rows: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    feature_dim: int | None = None
    for start_idx in start_indices:
        env.reset(start_idx=int(start_idx))
        for _ in range(int(steps_per_start)):
            feature = _current_policy_feature(env, forecast_cfg)
            feature_dim = int(feature.shape[0])
            valid = feasible_candidate_mask(env, masks) & allowed
            valid_ids = np.flatnonzero(valid)
            if valid_ids.size:
                state_snapshot = snapshot_env(env)
                for action_idx in valid_ids:
                    restore_env(env, state_snapshot)
                    _, _, _, _ = env.step_mask(masks[int(action_idx)])
                    next_feature = _current_policy_feature(env, forecast_cfg)
                    action_features = masks[int(action_idx)].astype(np.float32)
                    rows.append(np.concatenate([feature, action_features], axis=0).astype(np.float32))
                    deltas.append((next_feature - feature).astype(np.float32))
                restore_env(env, state_snapshot)
            action = beam_search_teacher_action(env, masks, teacher_cfg)
            _, _, done, _ = env.step_mask(action)
            if done:
                break
    if not rows or feature_dim is None:
        raise ValueError("No feature-transition rows were collected")
    return FeatureTransitionDataset(
        inputs=np.vstack(rows).astype(np.float32),
        deltas=np.vstack(deltas).astype(np.float32),
        feature_dim=int(feature_dim),
        n_sensors=int(masks.shape[1]),
    )


def train_action_cost_model(dataset: ActionCostDataset, cfg: ActionCostTrainingConfig) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.costs, dtype=np.float32).reshape(-1, 1)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ActionCostNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_feature_transition_model(
    dataset: FeatureTransitionDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.deltas, dtype=np.float32)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = FeatureTransitionNet(
        input_dim=x.shape[1],
        output_dim=int(dataset.feature_dim),
        hidden_dim=int(cfg.hidden_dim),
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_anchor_advantage_model(
    dataset: AnchorAdvantageDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(dataset.inputs, dtype=np.float32)
    y = np.asarray(dataset.advantages, dtype=np.float32).reshape(-1, 1)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = ActionCostNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
    return model.eval(), history


def train_action_cost_ensemble(
    dataset: ActionCostDataset,
    cfg: ActionCostTrainingConfig,
) -> tuple[list[Any], list[dict[str, list[float]]]]:
    size = max(1, int(cfg.ensemble_size))
    if size == 1:
        model, history = train_action_cost_model(dataset, cfg)
        return [model], [history]
    rng = np.random.default_rng(int(cfg.seed))
    n = int(dataset.inputs.shape[0])
    sample_size = max(1, int(round(float(cfg.bootstrap_fraction) * float(n))))
    models: list[Any] = []
    histories: list[dict[str, list[float]]] = []
    for member in range(size):
        ids = rng.integers(0, n, size=sample_size, endpoint=False)
        member_dataset = ActionCostDataset(
            inputs=np.asarray(dataset.inputs[ids], dtype=np.float32),
            costs=np.asarray(dataset.costs[ids], dtype=np.float32),
            feature_dim=int(dataset.feature_dim),
            n_sensors=int(dataset.n_sensors),
        )
        member_cfg = ActionCostTrainingConfig(
            hidden_dim=int(cfg.hidden_dim),
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            seed=int(cfg.seed) + 997 * int(member + 1),
            device=str(cfg.device),
            ensemble_size=1,
            bootstrap_fraction=float(cfg.bootstrap_fraction),
        )
        model, history = train_action_cost_model(member_dataset, member_cfg)
        models.append(model)
        histories.append(history)
    return models, histories


class ActionCostNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _ActionCostNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), 1),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _ActionCostNet()


class FeatureTransitionNet:
    def __new__(cls, *, input_dim: int, output_dim: int, hidden_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _FeatureTransitionNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.output_dim = int(output_dim)
                self.hidden_dim = int(hidden_dim)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(output_dim)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _FeatureTransitionNet()


@dataclass
class ForecastAwareCostPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    preserve_warming: bool = True
    name: str = "forecast_aware_cost"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return np.zeros(self.candidate_masks.shape[1], dtype=bool)
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            costs = self.model(x).reshape(-1).detach().cpu().numpy()
        best = int(valid_ids[int(np.argmin(costs))])
        return self.candidate_masks[best].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareValueResidualPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_value_residual"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            costs = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmin(costs))
        best_idx = int(valid_ids[best_local])
        anchor_cost = _predict_single_action_cost(
            self.model,
            feature=feature,
            action_features=self.anchor_mask_arr.astype(np.float32),
            device_obj=self.device_obj,
            torch=torch,
        )
        best_cost = float(costs[best_local])
        advantage = anchor_cost - best_cost
        if best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareEnsembleValuePolicy(V2Policy):
    models: list[Any]
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    uncertainty_beta: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_ensemble_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.models = list(self.models)
        if not self.models:
            raise ValueError("ForecastAwareEnsembleValuePolicy requires at least one model")
        for model in self.models:
            model.to(self.device_obj)
            model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            preds = [
                model(x).reshape(-1).detach().cpu().numpy().astype(float)
                for model in self.models
            ]
        pred = np.vstack(preds).astype(float)
        mean = np.mean(pred, axis=0)
        std = np.std(pred, axis=0)
        score = mean + float(self.uncertainty_beta) * std
        best_local = int(np.argmin(score))
        best_idx = int(valid_ids[best_local])
        anchor_preds = _predict_single_action_ensemble_costs(
            self.models,
            feature=feature,
            action_features=self.anchor_mask_arr.astype(np.float32),
            device_obj=self.device_obj,
            torch=torch,
        )
        anchor_score = float(np.mean(anchor_preds) + float(self.uncertainty_beta) * np.std(anchor_preds))
        best_score = float(score[best_local])
        advantage = anchor_score - best_score
        if best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareAdvantageResidualPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_advantage_residual"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        anchor_features = self.anchor_mask_arr.astype(np.float32)
        rows = []
        for action_idx in valid_ids:
            action_features = self.candidate_masks[int(action_idx)].astype(np.float32)
            rows.append(
                np.concatenate(
                    [
                        feature,
                        action_features,
                        anchor_features,
                        action_features - anchor_features,
                    ],
                    axis=0,
                )
            )
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            advantages = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmax(advantages))
        best_idx = int(valid_ids[best_local])
        if best_idx == self.anchor_idx:
            return self.anchor_mask_arr.astype(bool).copy()
        if float(advantages[best_local]) <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[best_idx].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareRolloutValuePolicy(V2Policy):
    cost_model: Any
    transition_model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    advantage_threshold: float = 0.0
    planning_depth: int = 2
    beam_width: int = 4
    max_branch: int = 6
    discount: float = 0.95
    preserve_warming: bool = True
    name: str = "forecast_aware_rollout_value"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.cost_model.to(self.device_obj)
        self.transition_model.to(self.device_obj)
        self.cost_model.eval()
        self.transition_model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        if self.anchor_idx is not None:
            self.allowed_action_mask[int(self.anchor_idx)] = True
        self.future_action_ids = np.flatnonzero(self.allowed_action_mask)
        if self.future_action_ids.size == 0:
            self.future_action_ids = np.arange(self.candidate_masks.shape[0], dtype=int)

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        feature = _current_policy_feature(env, self.forecast_cfg)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy()
        best_idx, best_score = self._plan_from_feature(feature, valid_ids)
        anchor_score = self._score_repeated_anchor(feature)
        advantage = float(anchor_score - best_score)
        if best_idx is None or best_idx == self.anchor_idx or advantage <= float(self.advantage_threshold):
            return self.anchor_mask_arr.astype(bool).copy()
        return self.candidate_masks[int(best_idx)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _plan_from_feature(self, feature: np.ndarray, valid_ids: np.ndarray) -> tuple[int | None, float]:
        first_ids = np.asarray(valid_ids, dtype=int).reshape(-1)
        beams: list[tuple[float, int | None, np.ndarray]] = [(0.0, None, np.asarray(feature, dtype=np.float32))]
        depth = max(1, int(self.planning_depth))
        for step in range(depth):
            expanded: list[tuple[float, int | None, np.ndarray]] = []
            for score_so_far, first_idx, beam_feature in beams:
                action_ids = first_ids if step == 0 else self.future_action_ids
                costs = self._predict_costs(beam_feature, action_ids)
                if costs.size == 0:
                    continue
                order = np.argsort(costs, kind="stable")[: max(1, int(self.max_branch))]
                for local_idx in order:
                    action_idx = int(action_ids[int(local_idx)])
                    action_features = self.candidate_masks[action_idx].astype(np.float32)
                    step_cost = float(costs[int(local_idx)])
                    next_feature = self._predict_next_feature(beam_feature, action_features)
                    expanded.append(
                        (
                            float(score_so_far + (float(self.discount) ** step) * step_cost),
                            int(action_idx) if first_idx is None else int(first_idx),
                            next_feature,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            beams = expanded[: max(1, int(self.beam_width))]
        completed = [beam for beam in beams if beam[1] is not None]
        if not completed:
            return None, float("inf")
        best = min(completed, key=lambda item: item[0])
        return int(best[1]), float(best[0])

    def _score_repeated_anchor(self, feature: np.ndarray) -> float:
        current = np.asarray(feature, dtype=np.float32)
        action_features = self.anchor_mask_arr.astype(np.float32)
        total = 0.0
        for step in range(max(1, int(self.planning_depth))):
            total += (float(self.discount) ** step) * self._predict_one_cost(current, action_features)
            current = self._predict_next_feature(current, action_features)
        return float(total)

    def _predict_costs(self, feature: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        ids = np.asarray(action_ids, dtype=int).reshape(-1)
        if ids.size == 0:
            return np.asarray([], dtype=float)
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            return self.cost_model(x).reshape(-1).detach().cpu().numpy().astype(float)

    def _predict_one_cost(self, feature: np.ndarray, action_features: np.ndarray) -> float:
        torch, _, _, _ = _torch_modules()
        row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            return float(self.cost_model(x).reshape(-1).detach().cpu().numpy()[0])

    def _predict_next_feature(self, feature: np.ndarray, action_features: np.ndarray) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            delta = self.transition_model(x).reshape(-1).detach().cpu().numpy().astype(np.float32)
        return (np.asarray(feature, dtype=np.float32) + delta).astype(np.float32)

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
        if not bool(self.preserve_warming):
            return np.asarray(valid, dtype=bool)
        required = np.asarray(
            [
                str(env.runtimes[sid].mode.name).lower() == "warming" and int(env.runtimes[sid].warm_remaining) > 0
                for sid in env.sensor_ids
            ],
            dtype=bool,
        )
        if not np.any(required):
            return np.asarray(valid, dtype=bool)
        keep_warming = np.all(self.candidate_masks[:, required], axis=1)
        guarded = np.asarray(valid, dtype=bool) & keep_warming
        if np.any(guarded):
            return guarded
        return np.asarray(valid, dtype=bool)

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        if bool(self.has_action_support):
            return np.zeros_like(np.asarray(valid, dtype=bool))
        return np.asarray(valid, dtype=bool)


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _anchor_rollout_cost(
    env: WarmupSchedulingEnv,
    anchor: np.ndarray,
    candidate_masks: np.ndarray,
    teacher_cfg: MpcTeacherConfig,
) -> float:
    snapshot = snapshot_env(env)
    try:
        return float(
            _rollout_repeated_mask_cost(
                env,
                np.asarray(anchor, dtype=bool).reshape(-1),
                max(1, int(teacher_cfg.planning_horizon)),
                np.asarray(candidate_masks, dtype=bool),
                teacher_cfg,
            )
        )
    finally:
        restore_env(env, snapshot)


def _predict_single_action_cost(
    model: Any,
    *,
    feature: np.ndarray,
    action_features: np.ndarray,
    device_obj: Any,
    torch: Any,
) -> float:
    row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=device_obj)
        return float(model(x).reshape(-1).detach().cpu().numpy()[0])


def _predict_single_action_ensemble_costs(
    models: list[Any],
    *,
    feature: np.ndarray,
    action_features: np.ndarray,
    device_obj: Any,
    torch: Any,
) -> np.ndarray:
    row = np.concatenate([feature, action_features], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=device_obj)
        preds = [float(model(x).reshape(-1).detach().cpu().numpy()[0]) for model in models]
    return np.asarray(preds, dtype=float)


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _allowed_action_mask(indices: tuple[int, ...] | np.ndarray | None, n_actions: int) -> np.ndarray:
    mask = np.ones(int(n_actions), dtype=bool)
    if indices is None:
        return mask
    mask[:] = False
    values = np.asarray(indices, dtype=int).reshape(-1)
    values = values[(values >= 0) & (values < int(n_actions))]
    if values.size == 0:
        mask[:] = True
    else:
        mask[values] = True
    return mask


def _candidate_index(candidates: np.ndarray, mask: np.ndarray) -> int | None:
    matches = np.all(np.asarray(candidates, dtype=bool) == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        return None
    return int(ids[0])


def _current_policy_feature(env: WarmupSchedulingEnv, forecast_cfg: ForecastContextConfig) -> np.ndarray:
    state = env._state().astype(np.float32)
    forecast = build_event_forecast(env.truth_df, int(env.current_idx), forecast_cfg)
    return append_event_forecast(state, forecast)
