from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import ForecastContextConfig, append_event_forecast, build_event_forecast
from .reuse import ensure_archive_src

ensure_archive_src()

from v2.custom_ppo import feasible_candidate_mask  # noqa: E402
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.policies import V2Policy  # noqa: E402


@dataclass(frozen=True)
class BCTrainingConfig:
    hidden_dim: int = 128
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    device: str = "auto"


def train_bc_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    action_masks: np.ndarray,
    *,
    cfg: BCTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = np.asarray(action_masks, dtype=bool)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y.shape[0] != x.shape[0] or valid.shape[0] != x.shape[0]:
        raise ValueError("features, labels and action_masks must have matching rows")
    n_actions = int(valid.shape[1])
    if np.any(y < 0) or np.any(y >= n_actions):
        raise ValueError("labels contain action indices outside action mask width")

    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    model = MaskedBCNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim), n_actions=n_actions).to(device)
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(valid, dtype=torch.bool),
    )
    loader = DataLoader(dataset, batch_size=max(1, int(cfg.batch_size)), shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "accuracy": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        hits = 0
        total = 0
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            logits = model(xb)
            logits = logits.masked_fill(~mb, -1.0e9)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred = torch.argmax(logits, dim=1)
            hits += int((pred == yb).detach().sum().cpu().item())
            total += int(yb.numel())
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["accuracy"].append(float(hits) / max(total, 1))
    return model.eval(), history


def save_bc_policy_checkpoint(
    path: str | Path,
    *,
    model: Any,
    candidate_masks: np.ndarray,
    forecast_cfg: ForecastContextConfig,
    train_cfg: BCTrainingConfig,
    history: dict[str, list[float]] | None = None,
) -> None:
    torch, _, _, _ = _torch_modules()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": int(getattr(model, "input_dim")),
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "n_actions": int(getattr(model, "n_actions")),
        "candidate_masks": np.asarray(candidate_masks, dtype=bool),
        "forecast_cfg": asdict(forecast_cfg),
        "train_cfg": asdict(train_cfg),
        "history": history or {},
    }
    torch.save(payload, str(target))


def load_bc_policy_checkpoint(path: str | Path, *, device: str = "auto") -> ForecastAwareBCPolicy:
    torch, _, _, _ = _torch_modules()
    device_obj = _select_device(torch, str(device))
    payload = torch.load(str(path), map_location=device_obj, weights_only=False)
    model = MaskedBCNet(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        n_actions=int(payload["n_actions"]),
    )
    model.load_state_dict(payload["state_dict"])
    forecast_cfg = ForecastContextConfig(**dict(payload["forecast_cfg"]))
    return ForecastAwareBCPolicy(
        model=model,
        candidate_masks=np.asarray(payload["candidate_masks"], dtype=bool),
        forecast_cfg=forecast_cfg,
        device=str(device_obj),
    )


class MaskedBCNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int, n_actions: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _MaskedBCNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.n_actions = int(n_actions)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(n_actions)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _MaskedBCNet()


@dataclass
class ForecastAwareBCPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    fallback_mask: tuple[bool, ...] | np.ndarray | None = None
    min_confidence: float = 0.0
    min_logit_margin: float | None = None
    preserve_warming: bool = False
    name: str = "forecast_aware_bc"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.fallback_mask is not None:
            self.fallback_mask = tuple(bool(x) for x in np.asarray(self.fallback_mask, dtype=bool).reshape(-1))

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid_for_selection = self._apply_warming_preservation(env, valid)
        with torch.no_grad():
            x = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            logits = self.model(x)
            mask_t = torch.as_tensor(valid_for_selection.reshape(1, -1), dtype=torch.bool, device=self.device_obj)
            logits = logits.masked_fill(~mask_t, -1.0e9)
            action = int(torch.argmax(logits, dim=1).detach().cpu().item())
            if self.fallback_mask is not None:
                fallback = _candidate_index(self.candidate_masks, np.asarray(self.fallback_mask, dtype=bool))
                if fallback is not None and bool(valid[int(fallback)]):
                    fallback_idx = int(fallback)
                    use_fallback = False
                    min_confidence = max(0.0, float(self.min_confidence))
                    if min_confidence > 0.0:
                        probs = torch.softmax(logits, dim=1)
                        confidence = float(probs[0, action].detach().cpu().item())
                        use_fallback = use_fallback or confidence < min_confidence
                    if self.min_logit_margin is not None:
                        margin = float(logits[0, action].detach().cpu().item() - logits[0, fallback_idx].detach().cpu().item())
                        use_fallback = use_fallback or margin < float(self.min_logit_margin)
                    if use_fallback:
                        action = fallback_idx
        return self.candidate_masks[action].astype(bool).copy()

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


@dataclass
class ForecastAwareKNNPolicy(V2Policy):
    features: np.ndarray
    labels: np.ndarray
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    k: int = 5
    preserve_warming: bool = True
    name: str = "forecast_aware_knn"

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.features.ndim != 2:
            raise ValueError("features must be 2D")
        if self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("features and labels must have matching rows")
        self.mean = np.mean(self.features, axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.std(self.features, axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.where(self.std > 1.0e-6, self.std, 1.0).astype(np.float32)
        self.normalized_features = ((self.features - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)).astype(
            np.float32
        )

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, valid)
        valid_labels = valid[self.labels]
        if not np.any(valid_labels):
            valid_ids = np.flatnonzero(valid)
            if valid_ids.size == 0:
                return np.zeros(self.candidate_masks.shape[1], dtype=bool)
            return self.candidate_masks[int(valid_ids[0])].astype(bool).copy()
        query = ((feature.astype(np.float32) - self.mean) / self.std).reshape(1, -1)
        train_idx = np.flatnonzero(valid_labels)
        distances = np.sum((self.normalized_features[train_idx] - query) ** 2, axis=1)
        k = min(max(1, int(self.k)), int(train_idx.size))
        nearest_local = np.argpartition(distances, k - 1)[:k]
        nearest_idx = train_idx[nearest_local]
        nearest_distances = distances[nearest_local]
        votes: dict[int, float] = {}
        best_distance: dict[int, float] = {}
        for row_idx, distance in zip(nearest_idx, nearest_distances):
            label = int(self.labels[int(row_idx)])
            weight = 1.0 / (float(distance) + 1.0e-6)
            votes[label] = votes.get(label, 0.0) + weight
            best_distance[label] = min(best_distance.get(label, float("inf")), float(distance))
        action = min(votes, key=lambda label: (-votes[label], best_distance[label], label))
        return self.candidate_masks[int(action)].astype(bool).copy()

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


@dataclass
class ForecastAwareCyclePolicy(V2Policy):
    labels: np.ndarray
    candidate_masks: np.ndarray
    preserve_warming: bool = True
    max_lookahead: int = 32
    name: str = "forecast_aware_cycle"

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.labels.size == 0:
            raise ValueError("Cycle policy requires at least one label")
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, valid)
        for offset in range(max(1, int(self.max_lookahead))):
            label = int(self.labels[(self.cursor + offset) % self.labels.size])
            if 0 <= label < self.candidate_masks.shape[0] and bool(valid[label]):
                self.cursor = (self.cursor + offset + 1) % self.labels.size
                return self.candidate_masks[label].astype(bool).copy()
        valid_ids = np.flatnonzero(valid)
        self.cursor = (self.cursor + 1) % self.labels.size
        if valid_ids.size == 0:
            return np.zeros(self.candidate_masks.shape[1], dtype=bool)
        return self.candidate_masks[int(valid_ids[0])].astype(bool).copy()

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


def _candidate_index(candidates: np.ndarray, mask: np.ndarray) -> int | None:
    matches = np.all(np.asarray(candidates, dtype=bool) == np.asarray(mask, dtype=bool).reshape(1, -1), axis=1)
    ids = np.flatnonzero(matches)
    if ids.size == 0:
        return None
    return int(ids[0])
