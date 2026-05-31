from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import ForecastContextConfig, append_event_forecast, build_event_forecast
from .mpc_teacher import MpcTeacherConfig, beam_search_first_action_costs, beam_search_teacher_action
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


@dataclass
class ActionCostDataset:
    inputs: np.ndarray
    costs: np.ndarray
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
        anchor_valid = self.anchor_idx is not None and bool(valid[int(self.anchor_idx)])
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size == 0:
            return self.anchor_mask_arr.astype(bool).copy() if anchor_valid else np.zeros(self.candidate_masks.shape[1], dtype=bool)
        rows = [
            np.concatenate([feature, self.candidate_masks[int(action_idx)].astype(np.float32)], axis=0)
            for action_idx in valid_ids
        ]
        with torch.no_grad():
            x = torch.as_tensor(np.vstack(rows).astype(np.float32), dtype=torch.float32, device=self.device_obj)
            costs = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        best_local = int(np.argmin(costs))
        best_idx = int(valid_ids[best_local])
        if anchor_valid:
            anchor_positions = np.flatnonzero(valid_ids == int(self.anchor_idx))
            if anchor_positions.size:
                anchor_cost = float(costs[int(anchor_positions[0])])
                best_cost = float(costs[best_local])
                advantage = anchor_cost - best_cost
                if advantage <= float(self.advantage_threshold):
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
        return np.asarray(valid, dtype=bool)


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


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
