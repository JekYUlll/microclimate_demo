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
        with torch.no_grad():
            x = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            logits = self.model(x)
            mask_t = torch.as_tensor(valid.reshape(1, -1), dtype=torch.bool, device=self.device_obj)
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
