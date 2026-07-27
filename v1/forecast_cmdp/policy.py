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


def train_mask_bc(
    features: np.ndarray,
    labels: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    cfg: BCTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(features, dtype=np.float32)
    y_idx = np.asarray(labels, dtype=np.int64).reshape(-1)
    masks = np.asarray(candidate_masks, dtype=bool)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y_idx.shape[0] != x.shape[0]:
        raise ValueError("features and labels must have matching rows")
    if np.any(y_idx < 0) or np.any(y_idx >= masks.shape[0]):
        raise ValueError("labels contain action indices outside candidate mask width")
    y = masks[y_idx].astype(np.float32)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = MaskBCNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim), n_sensors=int(masks.shape[1])).to(device)
    pos = np.maximum(np.sum(y, axis=0), 1.0)
    neg = np.maximum(float(y.shape[0]) - pos, 1.0)
    pos_weight = np.clip(neg / pos, 0.25, 4.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "sensor_accuracy": [], "exact_match": []}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        sensor_hits = 0
        sensor_total = 0
        exact_hits = 0
        rows = 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred = logits.detach() >= 0.0
            target = yb.detach() >= 0.5
            sensor_hits += int((pred == target).sum().cpu().item())
            sensor_total += int(target.numel())
            exact_hits += int(torch.all(pred == target, dim=1).sum().cpu().item())
            rows += int(target.shape[0])
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["sensor_accuracy"].append(float(sensor_hits) / max(sensor_total, 1))
        history["exact_match"].append(float(exact_hits) / max(rows, 1))
    return model.eval(), history


def train_sequence_mask_bc(
    features: np.ndarray,
    labels: np.ndarray,
    candidate_masks: np.ndarray,
    step_indices: np.ndarray,
    *,
    cfg: BCTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, _, _ = _torch_modules()
    x = np.asarray(features, dtype=np.float32)
    y_idx = np.asarray(labels, dtype=np.int64).reshape(-1)
    masks = np.asarray(candidate_masks, dtype=bool)
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y_idx.shape[0] != x.shape[0] or steps.shape[0] != x.shape[0]:
        raise ValueError("features, labels and step_indices must have matching rows")
    if np.any(y_idx < 0) or np.any(y_idx >= masks.shape[0]):
        raise ValueError("labels contain action indices outside candidate mask width")
    y = masks[y_idx].astype(np.float32)
    sequence_slices = _contiguous_sequence_slices(steps)
    if not sequence_slices:
        raise ValueError("No sequence rows available")

    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = SequenceMaskNet(
        feature_dim=x.shape[1],
        hidden_dim=int(cfg.hidden_dim),
        n_sensors=int(masks.shape[1]),
    ).to(device)
    pos = np.maximum(np.sum(y, axis=0), 1.0)
    neg = np.maximum(float(y.shape[0]) - pos, 1.0)
    pos_weight = np.clip(neg / pos, 0.25, 4.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    rng = np.random.default_rng(int(cfg.seed))
    history = {"loss": [], "sensor_accuracy": [], "exact_match": [], "sequence_count": [float(len(sequence_slices))]}
    for _ in range(int(cfg.epochs)):
        order = np.arange(len(sequence_slices))
        rng.shuffle(order)
        losses: list[float] = []
        sensor_hits = 0
        sensor_total = 0
        exact_hits = 0
        rows = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for seq_idx in order:
            start, end = sequence_slices[int(seq_idx)]
            xb = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[start:end], dtype=torch.float32, device=device)
            hidden = model.initial_hidden(batch_size=1, device=device)
            prev_mask = torch.zeros((1, int(masks.shape[1])), dtype=torch.float32, device=device)
            logits_seq = []
            for row_idx in range(int(xb.shape[0])):
                logits, hidden = model.forward_step(xb[row_idx : row_idx + 1], prev_mask, hidden)
                logits_seq.append(logits)
                prev_mask = yb[row_idx : row_idx + 1]
            logits_t = torch.cat(logits_seq, dim=0)
            loss = loss_fn(logits_t, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred = logits_t.detach() >= 0.0
            target = yb.detach() >= 0.5
            sensor_hits += int((pred == target).sum().cpu().item())
            sensor_total += int(target.numel())
            exact_hits += int(torch.all(pred == target, dim=1).sum().cpu().item())
            rows += int(target.shape[0])
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["sensor_accuracy"].append(float(sensor_hits) / max(sensor_total, 1))
        history["exact_match"].append(float(exact_hits) / max(rows, 1))
    return model.eval(), history


def train_deviation_gate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    anchor_idx: int,
    cfg: BCTrainingConfig,
) -> tuple[Any, dict[str, list[float]]]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x = np.asarray(features, dtype=np.float32)
    y_idx = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y_idx.shape[0] != x.shape[0]:
        raise ValueError("features and labels must have matching rows")
    y = (y_idx != int(anchor_idx)).astype(np.float32).reshape(-1, 1)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = DeviationGateNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    pos = max(float(np.sum(y)), 1.0)
    neg = max(float(y.shape[0]) - pos, 1.0)
    pos_weight = float(np.clip(neg / pos, 0.25, 8.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([pos_weight], dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "accuracy": [], "positive_rate": [float(np.mean(y))]}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        hits = 0
        rows = 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred = torch.sigmoid(logits.detach()) >= 0.5
            hits += int((pred == (yb.detach() >= 0.5)).sum().cpu().item())
            rows += int(yb.numel())
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["accuracy"].append(float(hits) / max(rows, 1))
    return model.eval(), history


def train_binary_gate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    cfg: BCTrainingConfig,
) -> tuple[Any, dict[str, list[float]], np.ndarray, np.ndarray]:
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    x_raw = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    if x_raw.ndim != 2:
        raise ValueError("features must be 2D")
    if y.shape[0] != x_raw.shape[0]:
        raise ValueError("features and labels must have matching rows")
    mean = np.mean(x_raw, axis=0).astype(np.float32)
    std = np.std(x_raw, axis=0).astype(np.float32)
    std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
    x = ((x_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
    device = _select_device(torch, str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = DeviationGateNet(input_dim=x.shape[1], hidden_dim=int(cfg.hidden_dim)).to(device)
    pos = max(float(np.sum(y)), 1.0)
    neg = max(float(y.shape[0]) - pos, 1.0)
    pos_weight = float(np.clip(neg / pos, 0.25, 8.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([pos_weight], dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    history = {"loss": [], "accuracy": [], "positive_rate": [float(np.mean(y))]}
    for _ in range(int(cfg.epochs)):
        losses: list[float] = []
        hits = 0
        rows = 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred = torch.sigmoid(logits.detach()) >= 0.5
            hits += int((pred == (yb.detach() >= 0.5)).sum().cpu().item())
            rows += int(yb.numel())
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["accuracy"].append(float(hits) / max(rows, 1))
    return model.eval(), history, mean, std


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


class MaskBCNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int, n_sensors: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _MaskBCNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.n_sensors = int(n_sensors)
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(n_sensors)),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        return _MaskBCNet()


class SequenceMaskNet:
    def __new__(cls, *, feature_dim: int, hidden_dim: int, n_sensors: int) -> Any:
        torch, nn, _, _ = _torch_modules()

        class _SequenceMaskNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_dim = int(feature_dim)
                self.hidden_dim = int(hidden_dim)
                self.n_sensors = int(n_sensors)
                self.gru = nn.GRUCell(int(feature_dim) + int(n_sensors), int(hidden_dim))
                self.head = nn.Sequential(
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(hidden_dim), int(n_sensors)),
                )

            def initial_hidden(self, *, batch_size: int, device: Any) -> Any:
                return torch.zeros((int(batch_size), int(hidden_dim)), dtype=torch.float32, device=device)

            def forward_step(self, feature: Any, prev_mask: Any, hidden: Any) -> tuple[Any, Any]:
                x = torch.cat([feature, prev_mask], dim=1)
                next_hidden = self.gru(x, hidden)
                return self.head(next_hidden), next_hidden

            def forward(self, feature: Any, prev_mask: Any, hidden: Any) -> tuple[Any, Any]:
                return self.forward_step(feature, prev_mask, hidden)

        return _SequenceMaskNet()


class DeviationGateNet:
    def __new__(cls, *, input_dim: int, hidden_dim: int) -> Any:
        _, nn, _, _ = _torch_modules()

        class _DeviationGateNet(nn.Module):
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

        return _DeviationGateNet()


@dataclass
class ForecastAwareBCPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    fallback_mask: tuple[bool, ...] | np.ndarray | None = None
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
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
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid_for_selection = self._apply_warming_preservation(env, self._apply_allowed_actions(valid))
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

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareResidualBCPolicy(V2Policy):
    bc_model: Any
    gate_model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    deviate_threshold: float = 0.5
    preserve_warming: bool = True
    name: str = "forecast_aware_residual_bc"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.bc_model.to(self.device_obj)
        self.gate_model.to(self.device_obj)
        self.bc_model.eval()
        self.gate_model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid_for_selection = self._apply_warming_preservation(env, self._apply_allowed_actions(valid))
        anchor_valid = self.anchor_idx is not None and bool(valid[int(self.anchor_idx)])
        with torch.no_grad():
            x = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            deviate_prob = float(torch.sigmoid(self.gate_model(x)).reshape(-1)[0].detach().cpu().item())
            if anchor_valid and deviate_prob < float(self.deviate_threshold):
                return self.anchor_mask_arr.astype(bool).copy()
            logits = self.bc_model(x)
            mask_t = torch.as_tensor(valid_for_selection.reshape(1, -1), dtype=torch.bool, device=self.device_obj)
            logits = logits.masked_fill(~mask_t, -1.0e9)
            action = int(torch.argmax(logits, dim=1).detach().cpu().item())
        if anchor_valid and not bool(valid_for_selection[int(action)]):
            return self.anchor_mask_arr.astype(bool).copy()
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

    def _apply_allowed_actions(self, valid: np.ndarray) -> np.ndarray:
        allowed = np.asarray(valid, dtype=bool) & self.allowed_action_mask
        if np.any(allowed):
            return allowed
        return np.asarray(valid, dtype=bool)


@dataclass
class ForecastAwareEventThresholdPolicy(V2Policy):
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    event_action_idx: int
    threshold: float
    aggregation: str = "max"
    preserve_warming: bool = True
    name: str = "forecast_aware_event_threshold"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.event_action_idx = int(self.event_action_idx)
        if self.event_action_idx < 0 or self.event_action_idx >= self.candidate_masks.shape[0]:
            raise ValueError("event_action_idx is outside candidate mask range")
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported event threshold aggregation: {self.aggregation}")

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        if event_score >= float(self.threshold):
            mask = self.candidate_masks[int(self.event_action_idx)].astype(bool).copy()
        else:
            mask = self.anchor_mask_arr.astype(bool).copy()
        if bool(self.preserve_warming):
            mask = self._preserve_warming(env, mask)
        return mask

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _preserve_warming(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareEventSupportCyclePolicy(V2Policy):
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    event_action_indices: tuple[int, ...] | np.ndarray
    threshold: float
    aggregation: str = "max"
    cycle_period: int = 1
    selection_mode: str = "time_cycle"
    preserve_warming: bool = True
    name: str = "forecast_aware_event_support_cycle"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        indices = [int(x) for x in np.asarray(self.event_action_indices, dtype=int).reshape(-1)]
        self.event_action_indices = tuple(
            int(idx) for idx in indices if 0 <= int(idx) < int(self.candidate_masks.shape[0])
        )
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported event threshold aggregation: {self.aggregation}")
        if str(self.selection_mode) not in {"time_cycle", "freshness"}:
            raise ValueError(f"Unsupported event support selection mode: {self.selection_mode}")
        self.cycle_period = max(1, int(self.cycle_period))

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        if event_score < float(self.threshold) or not self.event_action_indices:
            mask = self.anchor_mask_arr.astype(bool).copy()
        else:
            action_idx = self._cycle_action(env)
            mask = self.candidate_masks[int(action_idx)].astype(bool).copy()
        if bool(self.preserve_warming):
            mask = self._preserve_warming(env, mask)
        return mask

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _cycle_action(self, env: WarmupSchedulingEnv) -> int:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        feasible_indices = [int(idx) for idx in self.event_action_indices if bool(valid[int(idx)])]
        choices = feasible_indices if feasible_indices else list(self.event_action_indices)
        if str(self.selection_mode) == "freshness":
            freshness = np.asarray(
                [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
                dtype=float,
            )
            scores = [
                (
                    float(np.sum(freshness[self.candidate_masks[int(idx)]])),
                    -int(idx),
                    int(idx),
                )
                for idx in choices
            ]
            return int(max(scores)[2])
        offset = int(env.current_idx) // max(1, int(self.cycle_period))
        return int(choices[offset % len(choices)])

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _preserve_warming(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareOptionPlannerPolicy(V2Policy):
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    option_action_indices: tuple[int, ...] | np.ndarray
    target_rates: tuple[float, ...] | np.ndarray | None = None
    threshold: float = 0.5
    aggregation: str = "max"
    min_dwell: int = 1
    cooldown: int = 0
    target_rate_weight: float = 1.0
    rate_balance_weight: float = 0.0
    freshness_weight: float = 0.25
    transport_weight: float = 0.25
    power_weight: float = 0.03
    switch_weight: float = 0.05
    min_soc: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_option_planner"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        if self.anchor_mask_arr.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("anchor_mask must match candidate mask sensor width")
        raw_indices = np.asarray(self.option_action_indices, dtype=int).reshape(-1)
        self.option_action_indices = tuple(
            int(idx) for idx in raw_indices if 0 <= int(idx) < int(self.candidate_masks.shape[0])
        )
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported option-planner aggregation: {self.aggregation}")
        if self.target_rates is None:
            rates = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)
        else:
            rates = np.asarray(self.target_rates, dtype=float).reshape(-1)
            if rates.shape[0] != self.candidate_masks.shape[1]:
                raise ValueError("target_rates must match candidate mask sensor width")
        self.target_rates_arr = np.clip(rates, 0.0, 1.0)
        self.min_dwell = max(1, int(self.min_dwell))
        self.cooldown = max(0, int(self.cooldown))
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.active_counts = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)
        self.current_action_idx: int | None = None
        self.dwell_remaining = 0
        self.cooldown_remaining = 0
        self.prev_mask = self.anchor_mask_arr.astype(bool).copy()

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        soc_ratio = self._soc_ratio(env)
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, valid)

        if self.current_action_idx is not None and self.dwell_remaining > 0:
            idx = int(self.current_action_idx)
            if 0 <= idx < int(valid.shape[0]) and bool(valid[idx]) and soc_ratio >= float(self.min_soc):
                self.dwell_remaining -= 1
                return self._record_and_project(env, self.candidate_masks[idx])

        if event_score < float(self.threshold) or soc_ratio < float(self.min_soc):
            leaving_dynamic = self.current_action_idx is not None
            self.current_action_idx = None
            self.dwell_remaining = 0
            if leaving_dynamic:
                self.cooldown_remaining = max(self.cooldown_remaining, int(self.cooldown))
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return self._record_and_project(env, self.anchor_mask_arr)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return self._record_and_project(env, self.anchor_mask_arr)

        option_ids = [idx for idx in self.option_action_indices if bool(valid[int(idx)])]
        if not option_ids:
            self.current_action_idx = None
            self.dwell_remaining = 0
            return self._record_and_project(env, self.anchor_mask_arr)

        selected = self._select_option(env, option_ids=option_ids, event_score=float(event_score))
        if self.current_action_idx != int(selected):
            self.current_action_idx = int(selected)
            self.dwell_remaining = max(0, int(self.min_dwell) - 1)
        else:
            self.dwell_remaining = max(0, self.dwell_remaining)
        return self._record_and_project(env, self.candidate_masks[int(selected)])

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_option(self, env: WarmupSchedulingEnv, *, option_ids: list[int], event_score: float) -> int:
        elapsed = max(1, int(self.step_count))
        realized = self.active_counts / float(elapsed)
        deficit = np.maximum(self.target_rates_arr - realized, 0.0)
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        freshness = freshness / max(float(np.max(freshness)), 1.0)
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        role = self._transport_role_weights(env) * float(event_score)
        sensor_score = (
            float(self.target_rate_weight) * (self.target_rates_arr + deficit)
            + float(self.freshness_weight) * freshness
            + float(self.transport_weight) * role
            - float(self.power_weight) * power
        )
        rows: list[tuple[float, float, int, int]] = []
        previous = np.asarray(self.prev_mask, dtype=bool)
        for idx in option_ids:
            mask = self.candidate_masks[int(idx)]
            next_realized = (self.active_counts + mask.astype(float)) / float(elapsed + 1)
            rate_error = float(np.mean(np.abs(self.target_rates_arr - next_realized)))
            switch = float(np.mean(np.abs(mask.astype(float) - previous.astype(float))))
            score = (
                float(np.sum(sensor_score[mask]))
                - float(self.rate_balance_weight) * rate_error
                - float(self.switch_weight) * switch
            )
            rows.append((score, -float(np.sum(power[mask])), -int(idx), int(idx)))
        return int(max(rows)[3])

    def _record_and_project(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        desired = self._preserve_warming_mask(env, np.asarray(mask, dtype=bool).reshape(-1))
        projected = env.projector.project_mask(desired, env.runtimes)
        selected = np.asarray(projected.selected_mask, dtype=bool).copy()
        self.active_counts += selected.astype(float)
        self.step_count += 1
        self.prev_mask = selected.astype(bool).copy()
        return selected

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _soc_ratio(self, env: WarmupSchedulingEnv) -> float:
        value = env.last_info.get("soc_ratio", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        capacity = max(float(getattr(env.cfg, "energy_capacity", 0.0)), 1.0e-6)
        if not bool(getattr(env.cfg, "energy_account_enabled", False)):
            return 1.0
        return float(np.clip(float(getattr(env, "current_energy", capacity)) / capacity, 0.0, 1.0))

    def _transport_role_weights(self, env: WarmupSchedulingEnv) -> np.ndarray:
        weights = np.zeros(len(env.sensor_ids), dtype=float)
        for idx, sid in enumerate(env.sensor_ids):
            text = str(sid).lower()
            value = 0.0
            if "snow_particle" in text or "particle_counter" in text or text == "snow":
                value = max(value, 1.0)
            if "fc4" in text or "flux" in text:
                value = max(value, 0.9)
            if "ultrasonic" in text or "anemometer" in text or "wind" in text or text == "met":
                value = max(value, 0.45)
            if "surface" in text or "radiometer" in text or "thermo" in text or "hygro" in text:
                value = max(value, 0.30)
            weights[int(idx)] = value
        return weights

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

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareUtilityPlannerPolicy(V2Policy):
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    target_rates: tuple[float, ...] | np.ndarray | None = None
    event_weight: float = 1.0
    magnitude_weight: float = 1.0
    variability_weight: float = 0.5
    freshness_weight: float = 0.25
    target_rate_weight: float = 0.0
    anchor_bias: float = 0.0
    power_weight: float = 0.03
    switch_weight: float = 0.05
    min_soc: float = 0.0
    min_dwell: int = 1
    aggregation: str = "max"
    preserve_warming: bool = True
    name: str = "forecast_aware_utility_planner"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        if self.anchor_mask_arr.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("anchor_mask must match candidate mask sensor width")
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)
        if anchor_idx is not None:
            self.allowed_action_mask[int(anchor_idx)] = True
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported utility-planner aggregation: {self.aggregation}")
        if self.target_rates is None:
            target_rates = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)
        else:
            target_rates = np.asarray(self.target_rates, dtype=float).reshape(-1)
            if target_rates.shape[0] != self.candidate_masks.shape[1]:
                raise ValueError("target_rates must match candidate mask sensor width")
        self.target_rates_arr = np.clip(target_rates, 0.0, 1.0).astype(float)
        self.min_dwell = max(1, int(self.min_dwell))
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.active_counts = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)
        self.current_action_idx: int | None = None
        self.dwell_remaining = 0
        self.prev_mask = self.anchor_mask_arr.astype(bool).copy()

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = valid & self.allowed_action_mask
        valid = self._apply_warming_preservation(env, valid)
        if self._soc_ratio(env) < float(self.min_soc):
            return self._record_and_project(env, self.anchor_mask_arr)
        if self.current_action_idx is not None and self.dwell_remaining > 0:
            idx = int(self.current_action_idx)
            if 0 <= idx < int(valid.shape[0]) and bool(valid[idx]):
                self.dwell_remaining -= 1
                return self._record_and_project(env, self.candidate_masks[idx])
        candidate_ids = np.flatnonzero(valid)
        if candidate_ids.size == 0:
            self.current_action_idx = None
            self.dwell_remaining = 0
            return self._record_and_project(env, self.anchor_mask_arr)
        selected = self._select_candidate(env, candidate_ids)
        if self.current_action_idx != int(selected):
            self.current_action_idx = int(selected)
            self.dwell_remaining = max(0, int(self.min_dwell) - 1)
        return self._record_and_project(env, self.candidate_masks[int(selected)])

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_candidate(self, env: WarmupSchedulingEnv, candidate_ids: np.ndarray) -> int:
        sensor_score = self._sensor_scores(env)
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        previous = np.asarray(self.prev_mask, dtype=bool).reshape(-1)
        rows: list[tuple[float, float, int, int]] = []
        for idx in np.asarray(candidate_ids, dtype=int).reshape(-1):
            mask = self.candidate_masks[int(idx)]
            switch = float(np.mean(np.abs(mask.astype(float) - previous.astype(float))))
            score = float(np.sum(sensor_score[mask])) - float(self.switch_weight) * switch
            rows.append((score, -float(np.sum(power[mask])), -int(idx), int(idx)))
        return int(max(rows)[3])

    def _sensor_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        column_risk = self._continuous_column_risks(forecast.continuous)
        coverage = self._sensor_column_coverage(env)
        utility = coverage @ column_risk if coverage.size else np.zeros(len(env.sensor_ids), dtype=float)
        utility = utility + float(self.event_weight) * float(event_score) * self._transport_role_weights(env)
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        freshness = freshness / max(float(np.max(freshness)), 1.0)
        elapsed = max(1, int(self.step_count))
        realized = self.active_counts / float(elapsed)
        deficit = np.maximum(self.target_rates_arr - realized, 0.0)
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        return (
            utility
            + float(self.freshness_weight) * freshness
            + float(self.target_rate_weight) * deficit
            + float(self.anchor_bias) * self.anchor_mask_arr.astype(float)
            - float(self.power_weight) * power
        ).astype(float)

    def _continuous_column_risks(self, continuous: np.ndarray) -> np.ndarray:
        columns = tuple(str(x) for x in self.forecast_cfg.continuous_columns)
        if not columns:
            return np.zeros(0, dtype=float)
        values = np.asarray(continuous, dtype=float).reshape(-1)
        expected = 7 * len(columns)
        if values.size < expected:
            values = np.pad(values, (0, expected - values.size), constant_values=0.0)
        risks: list[float] = []
        for col_idx in range(len(columns)):
            stats = values[col_idx * 7 : (col_idx + 1) * 7]
            current, future_mean, future_max, future_min, future_std, future_last, future_delta = stats
            magnitude = max(abs(float(future_mean)), abs(float(future_max)), abs(float(future_min)), abs(float(future_last)))
            variation = abs(float(future_delta)) + abs(float(future_std)) + 0.25 * abs(float(future_last) - float(current))
            risks.append(float(self.magnitude_weight) * magnitude + float(self.variability_weight) * variation)
        arr = np.asarray(risks, dtype=float)
        return np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)

    def _sensor_column_coverage(self, env: WarmupSchedulingEnv) -> np.ndarray:
        columns = tuple(str(x) for x in self.forecast_cfg.continuous_columns)
        if not columns:
            return np.zeros((len(env.sensor_ids), 0), dtype=float)
        coverage = np.zeros((len(env.sensor_ids), len(columns)), dtype=float)
        for sensor_idx, spec in enumerate(env.sensor_specs):
            observed = {str(name) for name in getattr(spec, "observed_variables", ())}
            sensor_id = str(getattr(spec, "sensor_id", env.sensor_ids[sensor_idx])).lower()
            for col_idx, column in enumerate(columns):
                col = str(column)
                if col in observed:
                    coverage[int(sensor_idx), int(col_idx)] = 1.0
                    continue
                lower_col = col.lower()
                if "snow_mass_flux" in lower_col and ("fc4" in sensor_id or "flux" in sensor_id):
                    coverage[int(sensor_idx), int(col_idx)] = 1.0
                elif "snow_particle" in lower_col and ("particle" in sensor_id or "laser" in sensor_id):
                    coverage[int(sensor_idx), int(col_idx)] = 1.0
                elif "wind" in lower_col and ("met" in sensor_id or "wind" in sensor_id or "anemometer" in sensor_id):
                    coverage[int(sensor_idx), int(col_idx)] = 0.8
                elif "temperature" in lower_col and ("surface" in sensor_id or "radiometer" in sensor_id or "met" in sensor_id):
                    coverage[int(sensor_idx), int(col_idx)] = 0.7
        return coverage

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _transport_role_weights(self, env: WarmupSchedulingEnv) -> np.ndarray:
        weights = np.zeros(len(env.sensor_ids), dtype=float)
        for idx, sid in enumerate(env.sensor_ids):
            text = str(sid).lower()
            value = 0.0
            if "snow_particle" in text or "particle_counter" in text:
                value = max(value, 1.0)
            if "laser" in text or "disdrometer" in text:
                value = max(value, 0.9)
            if "fc4" in text or "flux" in text:
                value = max(value, 0.9)
            if "met" in text or "wind" in text or "anemometer" in text:
                value = max(value, 0.45)
            if "surface" in text or "radiometer" in text:
                value = max(value, 0.30)
            weights[int(idx)] = value
        return weights

    def _soc_ratio(self, env: WarmupSchedulingEnv) -> float:
        value = env.last_info.get("soc_ratio", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        capacity = max(float(getattr(env.cfg, "energy_capacity", 0.0)), 1.0e-6)
        if not bool(getattr(env.cfg, "energy_account_enabled", False)):
            return 1.0
        return float(np.clip(float(getattr(env, "current_energy", capacity)) / capacity, 0.0, 1.0))

    def _record_and_project(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        desired = self._preserve_warming_mask(env, np.asarray(mask, dtype=bool).reshape(-1))
        projected = env.projector.project_mask(desired, env.runtimes)
        selected = np.asarray(projected.selected_mask, dtype=bool).copy()
        self.active_counts += selected.astype(float)
        self.step_count += 1
        self.prev_mask = selected.astype(bool).copy()
        return selected

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

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareProxyMPCPolicy(ForecastAwareUtilityPlannerPolicy):
    """Short-horizon causal proxy planner over feasible sensor subsets.

    Unlike the one-step utility planner, this policy scores sequences against
    the static anchor by tracking which forecasted task columns would remain
    stale across a short imagined horizon.
    """

    planning_depth: int = 3
    beam_width: int = 4
    max_branch: int = 8
    age_weight: float = 0.5
    anchor_improvement_threshold: float = 0.0
    name: str = "forecast_aware_proxy_mpc"

    def __post_init__(self) -> None:
        self.planning_depth = max(1, int(self.planning_depth))
        self.beam_width = max(1, int(self.beam_width))
        self.max_branch = max(1, int(self.max_branch))
        super().__post_init__()
        self.anchor_action_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr)

    def reset(self) -> None:
        super().reset()
        self.column_age = np.zeros(len(tuple(str(x) for x in self.forecast_cfg.continuous_columns)), dtype=float)

    def _select_candidate(self, env: WarmupSchedulingEnv, candidate_ids: np.ndarray) -> int:
        candidate_ids = np.asarray(candidate_ids, dtype=int).reshape(-1)
        if candidate_ids.size == 0:
            return int(self.anchor_action_idx) if self.anchor_action_idx is not None else 0
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        column_risk = self._continuous_column_risks(forecast.continuous)
        coverage = self._sensor_column_coverage(env)
        role_weights = self._transport_role_weights(env)
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        freshness = freshness / max(float(np.max(freshness)), 1.0)
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        branch_ids = self._branch_candidates(
            candidate_ids,
            column_risk=column_risk,
            coverage=coverage,
            event_score=float(event_score),
            role_weights=role_weights,
            freshness=freshness,
            power=power,
        )
        best_idx, best_score = self._best_sequence(
            branch_ids,
            column_risk=column_risk,
            coverage=coverage,
            event_score=float(event_score),
            role_weights=role_weights,
            freshness=freshness,
            power=power,
        )
        anchor_idx = self.anchor_action_idx
        if anchor_idx is not None and int(anchor_idx) in set(int(x) for x in candidate_ids):
            _, anchor_score = self._best_sequence(
                np.asarray([int(anchor_idx)], dtype=int),
                column_risk=column_risk,
                coverage=coverage,
                event_score=float(event_score),
                role_weights=role_weights,
                freshness=freshness,
                power=power,
            )
            if float(best_score) - float(anchor_score) < float(self.anchor_improvement_threshold):
                return int(anchor_idx)
        return int(best_idx)

    def _branch_candidates(
        self,
        candidate_ids: np.ndarray,
        *,
        column_risk: np.ndarray,
        coverage: np.ndarray,
        event_score: float,
        role_weights: np.ndarray,
        freshness: np.ndarray,
        power: np.ndarray,
    ) -> np.ndarray:
        rows: list[tuple[float, int]] = []
        for idx in np.asarray(candidate_ids, dtype=int).reshape(-1):
            mask = self.candidate_masks[int(idx)]
            score = self._proxy_step_score(
                mask,
                previous=self.prev_mask,
                active_counts=self.active_counts,
                column_age=self.column_age,
                column_risk=column_risk,
                coverage=coverage,
                event_score=float(event_score),
                role_weights=role_weights,
                freshness=freshness,
                power=power,
            )
            rows.append((float(score), int(idx)))
        ranked = [idx for _, idx in sorted(rows, key=lambda item: (item[0], -item[1]), reverse=True)]
        keep: list[int] = []
        for idx in ranked[: int(self.max_branch)]:
            if idx not in keep:
                keep.append(int(idx))
        for idx in (self.anchor_action_idx, self.current_action_idx):
            if idx is not None and int(idx) in set(int(x) for x in candidate_ids) and int(idx) not in keep:
                keep.append(int(idx))
        if not keep:
            keep = [int(candidate_ids[0])]
        return np.asarray(keep, dtype=int)

    def _best_sequence(
        self,
        branch_ids: np.ndarray,
        *,
        column_risk: np.ndarray,
        coverage: np.ndarray,
        event_score: float,
        role_weights: np.ndarray,
        freshness: np.ndarray,
        power: np.ndarray,
    ) -> tuple[int, float]:
        previous = np.asarray(self.prev_mask, dtype=bool).copy()
        initial = (
            0.0,
            -float(np.sum(power[previous])) if previous.size else 0.0,
            0,
            None,
            previous,
            np.asarray(self.active_counts, dtype=float).copy(),
            np.asarray(self.column_age, dtype=float).copy(),
        )
        beams = [initial]
        elapsed = max(1, int(self.step_count))
        for depth_idx in range(int(self.planning_depth)):
            expanded = []
            for total, _, _, first_idx, prev_mask, active_counts, column_age in beams:
                for idx in np.asarray(branch_ids, dtype=int).reshape(-1):
                    mask = self.candidate_masks[int(idx)].astype(bool)
                    step = self._proxy_step_score(
                        mask,
                        previous=prev_mask,
                        active_counts=active_counts,
                        column_age=column_age,
                        column_risk=column_risk,
                        coverage=coverage,
                        event_score=float(event_score),
                        role_weights=role_weights,
                        freshness=freshness,
                        power=power,
                    )
                    next_active = active_counts + mask.astype(float)
                    next_age = self._next_column_age(mask, column_age, coverage)
                    first = int(idx) if first_idx is None else int(first_idx)
                    mean_power = -float(np.sum(power[mask]))
                    expanded.append(
                        (
                            float(total) + float(step) / float(1 + depth_idx),
                            mean_power,
                            -int(idx),
                            first,
                            mask,
                            next_active,
                            next_age,
                        )
                    )
            expanded.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            beams = expanded[: int(self.beam_width)]
        if not beams:
            fallback = int(branch_ids[0]) if len(branch_ids) else 0
            return fallback, float("-inf")
        best = max(beams, key=lambda item: (item[0], item[1], item[2]))
        return int(best[3] if best[3] is not None else int(branch_ids[0])), float(best[0])

    def _proxy_step_score(
        self,
        mask: np.ndarray,
        *,
        previous: np.ndarray,
        active_counts: np.ndarray,
        column_age: np.ndarray,
        column_risk: np.ndarray,
        coverage: np.ndarray,
        event_score: float,
        role_weights: np.ndarray,
        freshness: np.ndarray,
        power: np.ndarray,
    ) -> float:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        column_score = 0.0
        if coverage.size and column_risk.size:
            observed = np.clip(mask.astype(float) @ coverage, 0.0, 1.0)
            ages = np.asarray(column_age, dtype=float).reshape(-1)
            if ages.shape[0] != observed.shape[0]:
                ages = np.zeros_like(observed)
            age_multiplier = 1.0 + float(self.age_weight) * np.maximum(ages, 0.0)
            column_score = float(np.sum(np.asarray(column_risk, dtype=float) * observed * age_multiplier))
        elapsed = max(1, int(self.step_count))
        realized = np.asarray(active_counts, dtype=float) / float(elapsed)
        deficit = np.maximum(self.target_rates_arr - realized, 0.0)
        switch = float(np.mean(np.abs(mask.astype(float) - np.asarray(previous, dtype=float))))
        return (
            column_score
            + float(self.event_weight) * float(event_score) * float(np.sum(role_weights[mask]))
            + float(self.freshness_weight) * float(np.sum(freshness[mask]))
            + float(self.target_rate_weight) * float(np.sum(deficit[mask]))
            + float(self.anchor_bias) * float(np.sum(self.anchor_mask_arr.astype(float)[mask]))
            - float(self.power_weight) * float(np.sum(power[mask]))
            - float(self.switch_weight) * switch
        )

    def _next_column_age(self, mask: np.ndarray, column_age: np.ndarray, coverage: np.ndarray) -> np.ndarray:
        ages = np.asarray(column_age, dtype=float).reshape(-1)
        if not coverage.size:
            return ages
        observed = np.clip(np.asarray(mask, dtype=float).reshape(-1) @ coverage, 0.0, 1.0)
        if ages.shape[0] != observed.shape[0]:
            ages = np.zeros_like(observed)
        return np.where(observed > 0.0, 0.0, ages + 1.0)

    def _record_and_project(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        desired = self._preserve_warming_mask(env, np.asarray(mask, dtype=bool).reshape(-1))
        projected = env.projector.project_mask(desired, env.runtimes)
        selected = np.asarray(projected.selected_mask, dtype=bool).copy()
        coverage = self._sensor_column_coverage(env)
        self.column_age = self._next_column_age(selected, self.column_age, coverage)
        self.active_counts += selected.astype(float)
        self.step_count += 1
        self.prev_mask = selected.astype(bool).copy()
        return selected


@dataclass
class ForecastAwareMacroOptionPolicy(V2Policy):
    features: np.ndarray
    labels: np.ndarray
    candidate_masks: np.ndarray
    step_indices: np.ndarray
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    segment_len: int = 8
    snippet_stride: int = 1
    k: int = 4
    event_threshold: float = 0.5
    aggregation: str = "max"
    distance_weighting: str = "inverse"
    refresh_interval: int = 0
    max_lookahead: int = 4
    preserve_warming: bool = True
    name: str = "forecast_aware_macro_option"

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.step_indices = np.asarray(self.step_indices, dtype=np.int64).reshape(-1)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        if self.features.ndim != 2:
            raise ValueError("features must be 2D")
        if self.labels.shape[0] != self.features.shape[0] or self.step_indices.shape[0] != self.features.shape[0]:
            raise ValueError("features, labels and step_indices must have matching rows")
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        if self.anchor_mask_arr.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("anchor_mask must match candidate mask sensor width")
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported macro-option aggregation: {self.aggregation}")
        if str(self.distance_weighting) not in {"uniform", "inverse"}:
            raise ValueError(f"Unsupported macro-option distance weighting: {self.distance_weighting}")
        self.segment_len = max(1, int(self.segment_len))
        self.snippet_stride = max(1, int(self.snippet_stride))
        self.k = max(1, int(self.k))
        self.refresh_interval = max(0, int(self.refresh_interval))
        self.max_lookahead = max(1, int(self.max_lookahead))
        self.snippet_starts, self.snippet_sequences = self._build_snippets()
        if self.snippet_starts.size == 0:
            raise ValueError("Macro-option policy requires at least one valid teacher snippet")
        snippet_features = self.features[self.snippet_starts].astype(np.float32)
        self.feature_mean = np.mean(snippet_features, axis=0).astype(np.float32)
        self.feature_std = np.std(snippet_features, axis=0).astype(np.float32)
        self.feature_std = np.where(self.feature_std > 1.0e-6, self.feature_std, 1.0).astype(np.float32)
        self.normalized_snippet_features = ((snippet_features - self.feature_mean) / self.feature_std).astype(
            np.float32
        )
        self.reset()

    def reset(self) -> None:
        self.active_sequence: np.ndarray | None = None
        self.cursor = 0
        self.steps_since_selection = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        if event_score < float(self.event_threshold):
            self.active_sequence = None
            self.cursor = 0
            self.steps_since_selection = 0
            return self._anchor_action(env)

        should_refresh = (
            self.active_sequence is None
            or self.cursor >= int(self.active_sequence.shape[0])
            or (self.refresh_interval > 0 and self.steps_since_selection >= self.refresh_interval)
        )
        if should_refresh:
            self._select_sequence(env)

        if self.active_sequence is None:
            return self._anchor_action(env)
        selected = self._next_valid_label(env)
        self.steps_since_selection += 1
        if selected is None:
            return self._anchor_action(env)
        return self.candidate_masks[int(selected)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _build_snippets(self) -> tuple[np.ndarray, list[np.ndarray]]:
        starts: list[int] = []
        sequences: list[np.ndarray] = []
        for begin, end in _contiguous_sequence_slices(self.step_indices):
            for row in range(int(begin), int(end), int(self.snippet_stride)):
                stop = min(int(row) + int(self.segment_len), int(end))
                seq = self.labels[int(row) : int(stop)]
                seq = seq[(seq >= 0) & (seq < int(self.candidate_masks.shape[0]))]
                if seq.size == 0:
                    continue
                starts.append(int(row))
                sequences.append(np.asarray(seq, dtype=np.int64).copy())
        return np.asarray(starts, dtype=np.int64), sequences

    def _select_sequence(self, env: WarmupSchedulingEnv) -> None:
        feature = self._current_feature(env)
        if int(feature.shape[0]) != int(self.features.shape[1]):
            raise ValueError(
                f"feature dimension mismatch: policy={self.features.shape[1]} env={feature.shape[0]}"
            )
        query = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        distances = np.sqrt(np.sum((self.normalized_snippet_features - query.reshape(1, -1)) ** 2, axis=1))
        k = min(int(self.k), int(distances.shape[0]))
        nearest = np.argpartition(distances, kth=k - 1)[:k]
        if str(self.distance_weighting) == "inverse":
            weights = 1.0 / (distances[nearest].astype(float) + 1.0e-6)
        else:
            weights = np.ones(k, dtype=float)
        valid = self._valid_action_mask(env)
        votes: dict[int, float] = {}
        first_valid_by_snippet: dict[int, int] = {}
        for local_idx, snippet_idx in enumerate(nearest):
            label = self._first_valid_label(self.snippet_sequences[int(snippet_idx)], valid=valid)
            if label is None:
                continue
            first_valid_by_snippet[int(snippet_idx)] = int(label)
            votes[int(label)] = votes.get(int(label), 0.0) + float(weights[int(local_idx)])
        if not votes:
            self.active_sequence = None
            self.cursor = 0
            self.steps_since_selection = 0
            return
        selected_label = max(votes.items(), key=lambda item: (float(item[1]), -int(item[0])))[0]
        candidate_snippets = [
            int(idx) for idx in nearest if first_valid_by_snippet.get(int(idx), -1) == int(selected_label)
        ]
        best_snippet = min(candidate_snippets, key=lambda idx: (float(distances[int(idx)]), int(idx)))
        self.active_sequence = self.snippet_sequences[int(best_snippet)].copy()
        self.cursor = 0
        self.steps_since_selection = 0

    def _next_valid_label(self, env: WarmupSchedulingEnv) -> int | None:
        if self.active_sequence is None:
            return None
        valid = self._valid_action_mask(env)
        limit = min(int(self.active_sequence.shape[0]), int(self.cursor) + int(self.max_lookahead))
        for pos in range(int(self.cursor), int(limit)):
            label = int(self.active_sequence[int(pos)])
            if 0 <= label < int(self.candidate_masks.shape[0]) and bool(valid[int(label)]):
                self.cursor = int(pos) + 1
                return int(label)
        self.cursor = min(int(self.cursor) + 1, int(self.active_sequence.shape[0]))
        return None

    def _first_valid_label(self, sequence: np.ndarray, *, valid: np.ndarray) -> int | None:
        limit = min(int(sequence.shape[0]), int(self.max_lookahead))
        for pos in range(int(limit)):
            label = int(sequence[int(pos)])
            if 0 <= label < int(self.candidate_masks.shape[0]) and bool(valid[int(label)]):
                return int(label)
        return None

    def _current_feature(self, env: WarmupSchedulingEnv) -> np.ndarray:
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        return append_event_forecast(state, forecast).astype(np.float32)

    def _valid_action_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        if bool(self.preserve_warming):
            valid = self._apply_warming_preservation(env, valid)
        return np.asarray(valid, dtype=bool)

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = self._preserve_warming_mask(env, self.anchor_mask_arr)
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _apply_warming_preservation(self, env: WarmupSchedulingEnv, valid: np.ndarray) -> np.ndarray:
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

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareTeacherImprovementGatePolicy(V2Policy):
    gate_model: Any
    dynamic_policy: V2Policy
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float = 0.6
    preserve_warming: bool = True
    device: str = "auto"
    name: str = "forecast_aware_teacher_improvement_gate"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.gate_model.to(self.device_obj)
        self.gate_model.eval()
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        self.feature_std = np.asarray(self.feature_std, dtype=np.float32).reshape(-1)
        self.feature_std = np.where(self.feature_std > 1.0e-6, self.feature_std, 1.0).astype(np.float32)
        if self.anchor_mask_arr.ndim != 1:
            raise ValueError("anchor_mask must be one-dimensional")
        self.last_probability = 0.0

    def reset(self) -> None:
        if hasattr(self.dynamic_policy, "reset"):
            self.dynamic_policy.reset()
        self.last_probability = 0.0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        probability = self._improvement_probability(env)
        self.last_probability = float(probability)
        if float(probability) < float(self.threshold):
            if hasattr(self.dynamic_policy, "reset"):
                self.dynamic_policy.reset()
            return self._anchor_action(env)
        return np.asarray(self.dynamic_policy.act_mask(env), dtype=bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _improvement_probability(self, env: WarmupSchedulingEnv) -> float:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast).astype(np.float32)
        if int(feature.shape[0]) != int(self.feature_mean.shape[0]):
            raise ValueError(
                f"feature dimension mismatch: gate={self.feature_mean.shape[0]} env={feature.shape[0]}"
            )
        normalized = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(normalized.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            prob = torch.sigmoid(self.gate_model(x)).detach().cpu().numpy().reshape(-1)[0]
        return float(prob)

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = np.asarray(self.anchor_mask_arr, dtype=bool).copy()
        if bool(self.preserve_warming):
            for idx, sid in enumerate(env.sensor_ids):
                runtime = env.runtimes[sid]
                if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                    desired[int(idx)] = True
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()


@dataclass
class ForecastAwareRuntimeRiskGuardPolicy(V2Policy):
    dynamic_policy: V2Policy
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    threshold: float = 0.8
    aggregation: str = "max"
    window_steps: int = 8
    event_weight: float = 1.0
    freshness_weight: float = 0.25
    transport_weight: float = 0.25
    soc_weight: float = 0.0
    min_soc: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_runtime_risk_guard"

    def __post_init__(self) -> None:
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        if self.anchor_mask_arr.ndim != 1:
            raise ValueError("anchor_mask must be one-dimensional")
        if str(self.aggregation) not in {"max", "mean", "first"}:
            raise ValueError(f"Unsupported runtime-risk aggregation: {self.aggregation}")
        self.window_steps = max(1, int(self.window_steps))
        self.reset()

    def reset(self) -> None:
        if hasattr(self.dynamic_policy, "reset"):
            self.dynamic_policy.reset()
        self.window_remaining = 0
        self.gate_open = False
        self.previous_gate_open = False
        self.last_risk_score = 0.0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self.window_remaining <= 0:
            self.previous_gate_open = bool(self.gate_open)
            self.last_risk_score = self._risk_score(env)
            soc_ratio = self._soc_ratio(env)
            self.gate_open = bool(self.last_risk_score >= float(self.threshold) and soc_ratio >= float(self.min_soc))
            self.window_remaining = int(self.window_steps)
            if self.gate_open and not self.previous_gate_open and hasattr(self.dynamic_policy, "reset"):
                self.dynamic_policy.reset()
        self.window_remaining -= 1
        if bool(self.gate_open):
            return np.asarray(self.dynamic_policy.act_mask(env), dtype=bool).copy()
        return self._anchor_action(env)

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _risk_score(self, env: WarmupSchedulingEnv) -> float:
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        event_score = self._event_score(forecast.probabilities)
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        freshness = freshness / max(float(np.max(freshness)), 1.0)
        non_anchor = ~self.anchor_mask_arr.astype(bool)
        freshness_score = float(np.mean(freshness[non_anchor])) if np.any(non_anchor) else 0.0
        role = self._transport_role_weights(env)
        role_score = float(np.mean((role * freshness)[non_anchor])) if np.any(non_anchor) else 0.0
        return float(
            float(self.event_weight) * event_score
            + float(self.freshness_weight) * freshness_score
            + float(self.transport_weight) * role_score
            + float(self.soc_weight) * self._soc_ratio(env)
        )

    def _event_score(self, probabilities: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if probs.size == 0:
            return 0.0
        if str(self.aggregation) == "mean":
            return float(np.mean(probs))
        if str(self.aggregation) == "first":
            return float(probs[0])
        return float(np.max(probs))

    def _soc_ratio(self, env: WarmupSchedulingEnv) -> float:
        value = env.last_info.get("soc_ratio", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        capacity = max(float(getattr(env.cfg, "energy_capacity", 0.0)), 1.0e-6)
        if not bool(getattr(env.cfg, "energy_account_enabled", False)):
            return 1.0
        return float(np.clip(float(getattr(env, "current_energy", capacity)) / capacity, 0.0, 1.0))

    def _transport_role_weights(self, env: WarmupSchedulingEnv) -> np.ndarray:
        weights = np.zeros(len(env.sensor_ids), dtype=float)
        for idx, sid in enumerate(env.sensor_ids):
            text = str(sid).lower()
            value = 0.0
            if "snow_particle" in text or "particle_counter" in text or text == "snow":
                value = max(value, 1.0)
            if "fc4" in text or "flux" in text:
                value = max(value, 0.9)
            if "ultrasonic" in text or "anemometer" in text or "wind" in text or text == "met":
                value = max(value, 0.45)
            if "surface" in text or "radiometer" in text or "thermo" in text or "hygro" in text:
                value = max(value, 0.30)
            weights[int(idx)] = value
        return weights

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = self._preserve_warming_mask(env, self.anchor_mask_arr)
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareWindowEligibilityPolicy(V2Policy):
    memory_features: np.ndarray
    memory_margins: np.ndarray
    dynamic_policy: V2Policy
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    k: int = 3
    margin_threshold: float = 0.0
    window_steps: int = 16
    distance_weighting: str = "inverse"
    min_soc: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_window_eligibility"

    def __post_init__(self) -> None:
        features = np.asarray(self.memory_features, dtype=np.float32)
        margins = np.asarray(self.memory_margins, dtype=float).reshape(-1)
        if features.ndim != 2:
            raise ValueError("memory_features must be 2D")
        if margins.shape[0] != features.shape[0]:
            raise ValueError("memory_features and memory_margins must have matching rows")
        if features.shape[0] == 0:
            raise ValueError("window-eligibility memory cannot be empty")
        if str(self.distance_weighting) not in {"inverse", "uniform"}:
            raise ValueError(f"Unsupported window-eligibility distance weighting: {self.distance_weighting}")
        self.feature_mean = np.mean(features, axis=0).astype(np.float32)
        self.feature_std = np.std(features, axis=0).astype(np.float32)
        self.feature_std = np.where(self.feature_std > 1.0e-6, self.feature_std, 1.0).astype(np.float32)
        self.memory_features = ((features - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(
            np.float32
        )
        self.memory_margins = margins.astype(float)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.k = max(1, int(self.k))
        self.window_steps = max(1, int(self.window_steps))
        self.reset()

    def reset(self) -> None:
        if hasattr(self.dynamic_policy, "reset"):
            self.dynamic_policy.reset()
        self.window_remaining = 0
        self.gate_open = False
        self.last_predicted_margin = 0.0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self.window_remaining <= 0:
            self.last_predicted_margin = self._predicted_margin(env)
            self.gate_open = bool(
                self.last_predicted_margin >= float(self.margin_threshold)
                and self._soc_ratio(env) >= float(self.min_soc)
            )
            self.window_remaining = int(self.window_steps)
            if hasattr(self.dynamic_policy, "reset"):
                self.dynamic_policy.reset()
        self.window_remaining -= 1
        if bool(self.gate_open):
            return np.asarray(self.dynamic_policy.act_mask(env), dtype=bool).copy()
        return self._anchor_action(env)

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _predicted_margin(self, env: WarmupSchedulingEnv) -> float:
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast).astype(np.float32)
        if feature.shape[0] != self.feature_mean.shape[0]:
            raise ValueError(
                f"feature dimension mismatch: memory={self.feature_mean.shape[0]} env={feature.shape[0]}"
            )
        normalized = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        distances = np.linalg.norm(self.memory_features - normalized.reshape(1, -1), axis=1)
        order = np.argsort(distances, kind="mergesort")[: min(int(self.k), int(distances.shape[0]))]
        selected_margins = self.memory_margins[order]
        if str(self.distance_weighting) == "uniform":
            return float(np.mean(selected_margins))
        weights = 1.0 / np.maximum(distances[order], 1.0e-6)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        return float(np.sum(weights * selected_margins))

    def _soc_ratio(self, env: WarmupSchedulingEnv) -> float:
        value = env.last_info.get("soc_ratio", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        capacity = max(float(getattr(env.cfg, "energy_capacity", 0.0)), 1.0e-6)
        if not bool(getattr(env.cfg, "energy_account_enabled", False)):
            return 1.0
        return float(np.clip(float(getattr(env, "current_energy", capacity)) / capacity, 0.0, 1.0))

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = self._preserve_warming_mask(env, self.anchor_mask_arr)
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareWindowCandidatePolicy(V2Policy):
    memory_features: np.ndarray
    memory_margins: np.ndarray
    memory_candidate_ids: np.ndarray
    candidate_policies: tuple[V2Policy, ...]
    forecast_cfg: ForecastContextConfig
    anchor_mask: tuple[bool, ...] | np.ndarray
    k: int = 5
    margin_threshold: float = 0.0
    score_quantile: float = 0.25
    window_steps: int = 16
    distance_weighting: str = "inverse"
    min_soc: float = 0.0
    min_candidate_neighbors: int = 1
    preserve_warming: bool = True
    name: str = "forecast_aware_window_candidate"

    def __post_init__(self) -> None:
        features = np.asarray(self.memory_features, dtype=np.float32)
        margins = np.asarray(self.memory_margins, dtype=float).reshape(-1)
        candidate_ids = np.asarray(self.memory_candidate_ids, dtype=int).reshape(-1)
        if features.ndim != 2:
            raise ValueError("memory_features must be 2D")
        if margins.shape[0] != features.shape[0] or candidate_ids.shape[0] != features.shape[0]:
            raise ValueError("window-candidate memory arrays must have matching rows")
        if features.shape[0] == 0:
            raise ValueError("window-candidate memory cannot be empty")
        if not self.candidate_policies:
            raise ValueError("candidate_policies cannot be empty")
        if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(self.candidate_policies)):
            raise ValueError("memory_candidate_ids contain values outside candidate_policies")
        if str(self.distance_weighting) not in {"inverse", "uniform"}:
            raise ValueError(f"Unsupported window-candidate distance weighting: {self.distance_weighting}")
        self.feature_mean = np.mean(features, axis=0).astype(np.float32)
        self.feature_std = np.std(features, axis=0).astype(np.float32)
        self.feature_std = np.where(self.feature_std > 1.0e-6, self.feature_std, 1.0).astype(np.float32)
        self.memory_features = ((features - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(
            np.float32
        )
        self.memory_margins = margins.astype(float)
        self.memory_candidate_ids = candidate_ids.astype(int)
        self.anchor_mask_arr = np.asarray(self.anchor_mask, dtype=bool).reshape(-1)
        self.k = max(1, int(self.k))
        self.window_steps = max(1, int(self.window_steps))
        self.score_quantile = float(np.clip(float(self.score_quantile), 0.0, 1.0))
        self.min_candidate_neighbors = max(1, int(self.min_candidate_neighbors))
        self.reset()

    def reset(self) -> None:
        for policy in self.candidate_policies:
            if hasattr(policy, "reset"):
                policy.reset()
        self.window_remaining = 0
        self.active_candidate_id: int | None = None
        self.last_predicted_margin = 0.0
        self.last_predicted_mean = 0.0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        if self.window_remaining <= 0:
            candidate_id, score, mean_margin = self._select_candidate(env)
            self.active_candidate_id = candidate_id
            self.last_predicted_margin = float(score)
            self.last_predicted_mean = float(mean_margin)
            self.window_remaining = int(self.window_steps)
            if candidate_id is not None:
                policy = self.candidate_policies[int(candidate_id)]
                if hasattr(policy, "reset"):
                    policy.reset()
        self.window_remaining -= 1
        if self.active_candidate_id is None:
            return self._anchor_action(env)
        return np.asarray(self.candidate_policies[int(self.active_candidate_id)].act_mask(env), dtype=bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_candidate(self, env: WarmupSchedulingEnv) -> tuple[int | None, float, float]:
        if self._soc_ratio(env) < float(self.min_soc):
            return None, float("-inf"), float("-inf")
        feature = self._current_feature(env)
        normalized = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        best: tuple[int | None, float, float] = (None, float("-inf"), float("-inf"))
        for candidate_id in range(len(self.candidate_policies)):
            score, mean_margin = self._candidate_margin_stats(normalized, int(candidate_id))
            if not np.isfinite(score) or score < float(self.margin_threshold):
                continue
            if (
                best[0] is None
                or float(score) > float(best[1])
                or (float(score) == float(best[1]) and float(mean_margin) > float(best[2]))
                or (
                    float(score) == float(best[1])
                    and float(mean_margin) == float(best[2])
                    and int(candidate_id) < int(best[0])
                )
            ):
                best = (int(candidate_id), float(score), float(mean_margin))
        return best

    def _candidate_margin_stats(self, normalized_feature: np.ndarray, candidate_id: int) -> tuple[float, float]:
        rows = np.flatnonzero(self.memory_candidate_ids == int(candidate_id))
        if rows.size < int(self.min_candidate_neighbors):
            return float("-inf"), float("-inf")
        distances = np.linalg.norm(self.memory_features[rows] - normalized_feature.reshape(1, -1), axis=1)
        order = np.argsort(distances, kind="mergesort")[: min(int(self.k), int(rows.size))]
        selected = rows[order]
        selected_margins = self.memory_margins[selected].astype(float)
        if selected_margins.size == 0:
            return float("-inf"), float("-inf")
        mean_margin = float(np.mean(selected_margins))
        if str(self.distance_weighting) == "inverse":
            weights = 1.0 / np.maximum(distances[order], 1.0e-6)
            weights = weights / max(float(np.sum(weights)), 1.0e-12)
            mean_margin = float(np.sum(weights * selected_margins))
        score = float(np.quantile(selected_margins, float(self.score_quantile)))
        return score, mean_margin

    def _current_feature(self, env: WarmupSchedulingEnv) -> np.ndarray:
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast).astype(np.float32)
        if feature.shape[0] != self.feature_mean.shape[0]:
            raise ValueError(
                f"feature dimension mismatch: memory={self.feature_mean.shape[0]} env={feature.shape[0]}"
            )
        return feature

    def _soc_ratio(self, env: WarmupSchedulingEnv) -> float:
        value = env.last_info.get("soc_ratio", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        capacity = max(float(getattr(env.cfg, "energy_capacity", 0.0)), 1.0e-6)
        if not bool(getattr(env.cfg, "energy_account_enabled", False)):
            return 1.0
        return float(np.clip(float(getattr(env, "current_energy", capacity)) / capacity, 0.0, 1.0))

    def _anchor_action(self, env: WarmupSchedulingEnv) -> np.ndarray:
        desired = self._preserve_warming_mask(env, self.anchor_mask_arr)
        projected = env.projector.project_mask(desired, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def _preserve_warming_mask(self, env: WarmupSchedulingEnv, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        if not bool(self.preserve_warming):
            return out
        for idx, sid in enumerate(env.sensor_ids):
            runtime = env.runtimes[sid]
            if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                out[int(idx)] = True
        return out


@dataclass
class ForecastAwareTeacherRatePolicy(V2Policy):
    candidate_masks: np.ndarray
    target_rates: tuple[float, ...] | np.ndarray
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    freshness_weight: float = 0.0
    power_weight: float = 0.0
    preserve_warming: bool = True
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None
    name: str = "forecast_aware_teacher_rate"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        self.target_rates_arr = np.clip(np.asarray(self.target_rates, dtype=float).reshape(-1), 0.0, 1.0)
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        if self.target_rates_arr.shape[0] != self.candidate_masks.shape[1]:
            raise ValueError("target_rates must match candidate mask sensor width")
        raw_allowed = () if self.allowed_action_indices is None else self.allowed_action_indices
        allowed = tuple(int(x) for x in np.asarray(raw_allowed, dtype=int).reshape(-1))
        self.allowed_action_indices = tuple(
            int(idx) for idx in allowed if 0 <= int(idx) < int(self.candidate_masks.shape[0])
        )
        self.anchor_mask_arr = (
            np.asarray(self.anchor_mask, dtype=bool).reshape(-1) if self.anchor_mask is not None else None
        )
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.active_counts = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        if self.allowed_action_indices:
            allowed = np.zeros_like(valid, dtype=bool)
            allowed[np.asarray(self.allowed_action_indices, dtype=int)] = True
            valid = valid & allowed
        valid = self._apply_warming_preservation(env, valid)
        candidate_ids = np.flatnonzero(valid)
        if candidate_ids.size == 0:
            mask = (
                self.anchor_mask_arr.astype(bool).copy()
                if self.anchor_mask_arr is not None
                else np.zeros(self.candidate_masks.shape[1], dtype=bool)
            )
        else:
            mask = self.candidate_masks[int(self._select_candidate(env, candidate_ids))].astype(bool).copy()
        self.active_counts += mask.astype(float)
        self.step_count += 1
        return mask

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_candidate(self, env: WarmupSchedulingEnv, candidate_ids: np.ndarray) -> int:
        elapsed = max(1, int(self.step_count))
        realized = self.active_counts / float(elapsed)
        deficit = self.target_rates_arr - realized
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        sensor_score = deficit + float(self.freshness_weight) * freshness - float(self.power_weight) * power
        rows = []
        for idx in np.asarray(candidate_ids, dtype=int).reshape(-1):
            mask = self.candidate_masks[int(idx)]
            next_realized = (self.active_counts + mask.astype(float)) / float(elapsed + 1)
            rate_error = float(np.mean(np.abs(self.target_rates_arr - next_realized)))
            rows.append((float(np.sum(sensor_score[mask])), -rate_error, -int(idx), int(idx)))
        return int(max(rows)[3])

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
class ForecastAwareContextualDutyPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None
    blend: float = 1.0
    deficit_weight: float = 1.0
    freshness_weight: float = 0.0
    power_weight: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_contextual_duty"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        self.anchor_mask_arr = (
            np.asarray(self.anchor_mask, dtype=bool).reshape(-1) if self.anchor_mask is not None else None
        )
        anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr) if self.anchor_mask_arr is not None else None
        if anchor_idx is not None:
            self.allowed_action_mask[int(anchor_idx)] = True
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.active_counts = np.zeros(int(self.candidate_masks.shape[1]), dtype=float)

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        candidate_ids = np.flatnonzero(valid)
        if candidate_ids.size == 0:
            mask = (
                self.anchor_mask_arr.astype(bool).copy()
                if self.anchor_mask_arr is not None
                else np.zeros(self.candidate_masks.shape[1], dtype=bool)
            )
        else:
            target = self._target_rates(env)
            mask = self.candidate_masks[int(self._select_candidate(env, candidate_ids, target))].astype(bool).copy()
        self.active_counts += mask.astype(float)
        self.step_count += 1
        return mask

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _target_rates(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        with torch.no_grad():
            x = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            logits = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        if self.anchor_mask_arr is not None and self.anchor_mask_arr.shape[0] == probs.shape[0]:
            blend = float(np.clip(self.blend, 0.0, 1.0))
            probs = (1.0 - blend) * self.anchor_mask_arr.astype(float) + blend * probs
        return np.clip(probs, 0.0, 1.0).astype(float)

    def _select_candidate(self, env: WarmupSchedulingEnv, candidate_ids: np.ndarray, target: np.ndarray) -> int:
        elapsed = max(1, int(self.step_count))
        realized = self.active_counts / float(elapsed)
        deficit = np.asarray(target, dtype=float) - realized
        freshness = np.asarray(
            [float(env.runtimes[sid].freshness(int(env.current_idx))) for sid in env.sensor_ids],
            dtype=float,
        )
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        sensor_score = (
            np.asarray(target, dtype=float)
            + float(self.deficit_weight) * deficit
            + float(self.freshness_weight) * freshness
            - float(self.power_weight) * power
        )
        rows = []
        for idx in np.asarray(candidate_ids, dtype=int).reshape(-1):
            mask = self.candidate_masks[int(idx)]
            next_realized = (self.active_counts + mask.astype(float)) / float(elapsed + 1)
            rate_error = float(np.mean(np.abs(np.asarray(target, dtype=float) - next_realized)))
            rows.append((float(np.sum(sensor_score[mask])), -rate_error, -int(idx), int(idx)))
        return int(max(rows)[3])

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
class ForecastAwareSequenceMaskPolicy(V2Policy):
    model: Any
    candidate_masks: np.ndarray
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    allowed_action_indices: tuple[int, ...] | np.ndarray | None = None
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None
    anchor_bias: float = 0.0
    power_weight: float = 0.0
    preserve_warming: bool = True
    name: str = "forecast_aware_sequence_mask"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        if self.candidate_masks.ndim != 2:
            raise ValueError("candidate_masks must be 2D")
        self.has_action_support = self.allowed_action_indices is not None
        self.allowed_action_mask = _allowed_action_mask(self.allowed_action_indices, self.candidate_masks.shape[0])
        self.anchor_mask_arr = (
            np.asarray(self.anchor_mask, dtype=bool).reshape(-1) if self.anchor_mask is not None else None
        )
        anchor_idx = _candidate_index(self.candidate_masks, self.anchor_mask_arr) if self.anchor_mask_arr is not None else None
        if anchor_idx is not None:
            self.allowed_action_mask[int(anchor_idx)] = True
        self.reset()

    def reset(self) -> None:
        self.hidden = None
        self.prev_mask = np.zeros(int(self.candidate_masks.shape[1]), dtype=np.float32)

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        with torch.no_grad():
            feature_t = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            prev_t = torch.as_tensor(self.prev_mask.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            if self.hidden is None:
                self.hidden = self.model.initial_hidden(batch_size=1, device=self.device_obj)
            logits_t, next_hidden = self.model.forward_step(feature_t, prev_t, self.hidden)
            scores = logits_t.reshape(-1).detach().cpu().numpy().astype(float)
            self.hidden = next_hidden.detach()
        if self.anchor_mask_arr is not None and self.anchor_mask_arr.shape[0] == scores.shape[0]:
            scores = scores + float(self.anchor_bias) * np.where(self.anchor_mask_arr, 1.0, -1.0)
        action_idx = self._select_candidate(env, scores)
        if action_idx is None:
            mask = (
                self.anchor_mask_arr.astype(bool).copy()
                if self.anchor_mask_arr is not None
                else np.zeros(self.candidate_masks.shape[1], dtype=bool)
            )
        else:
            mask = self.candidate_masks[int(action_idx)].astype(bool).copy()
        projected = env.projector.project_mask(mask, env.runtimes)
        executed = np.asarray(projected.selected_mask, dtype=bool).copy()
        self.prev_mask = executed.astype(np.float32)
        return executed

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _select_candidate(self, env: WarmupSchedulingEnv, scores: np.ndarray) -> int | None:
        valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_allowed_actions(valid)
        valid = self._apply_warming_preservation(env, valid)
        candidate_ids = np.flatnonzero(valid)
        if candidate_ids.size == 0:
            return None
        power = np.asarray([float(spec.power_cost) for spec in env.sensor_specs], dtype=float)
        rows = []
        for idx in np.asarray(candidate_ids, dtype=int).reshape(-1):
            mask = self.candidate_masks[int(idx)]
            score = float(np.sum(scores[mask])) - float(self.power_weight) * float(np.sum(power[mask]))
            rows.append((score, -float(np.sum(power[mask])), -int(np.sum(mask)), -int(idx), int(idx)))
        return int(max(rows)[4])

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
class ForecastAwareMaskBCPolicy(V2Policy):
    model: Any
    forecast_cfg: ForecastContextConfig
    device: str = "auto"
    preserve_warming: bool = True
    required_sensor_indices: tuple[int, ...] | np.ndarray | None = None
    anchor_mask: tuple[bool, ...] | np.ndarray | None = None
    anchor_bias: float = 0.0
    name: str = "forecast_aware_mask_bc"

    def __post_init__(self) -> None:
        torch, _, _, _ = _torch_modules()
        self.device_obj = _select_device(torch, str(self.device))
        self.model.to(self.device_obj)
        self.model.eval()
        raw_required = () if self.required_sensor_indices is None else self.required_sensor_indices
        self.required_sensor_indices = tuple(int(x) for x in np.asarray(raw_required, dtype=int).reshape(-1))
        self.anchor_mask_arr = (
            np.asarray(self.anchor_mask, dtype=bool).reshape(-1) if self.anchor_mask is not None else None
        )

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        scores = self.act_scores(env)
        projected = env.projector.project_scores(scores, env.runtimes)
        return np.asarray(projected.selected_mask, dtype=bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        torch, _, _, _ = _torch_modules()
        state = env._state().astype(np.float32)
        forecast = build_event_forecast(env.truth_df, int(env.current_idx), self.forecast_cfg)
        feature = append_event_forecast(state, forecast)
        with torch.no_grad():
            x = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=self.device_obj)
            scores = self.model(x).reshape(-1).detach().cpu().numpy().astype(float)
        if self.anchor_mask_arr is not None and self.anchor_mask_arr.shape[0] == scores.shape[0]:
            scores = scores + float(self.anchor_bias) * np.where(self.anchor_mask_arr, 1.0, -1.0)
        for idx in self.required_sensor_indices:
            if 0 <= int(idx) < scores.shape[0]:
                scores[int(idx)] = max(float(scores[int(idx)]), 1.0e6)
        if bool(self.preserve_warming):
            for idx, sid in enumerate(env.sensor_ids):
                runtime = env.runtimes[sid]
                if str(runtime.mode.name).lower() == "warming" and int(runtime.warm_remaining) > 0:
                    scores[int(idx)] = max(float(scores[int(idx)]), 1.0e5)
        return scores


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


@dataclass
class ValidationCyclicDwellPolicy(V2Policy):
    candidate_masks: np.ndarray
    action_indices: tuple[int, ...] | np.ndarray
    dwell_steps: int = 4
    fallback_action_idx: int | None = None
    preserve_warming: bool = True
    name: str = "validation_cyclic_dwell"

    def __post_init__(self) -> None:
        self.candidate_masks = np.asarray(self.candidate_masks, dtype=bool)
        raw_indices = tuple(int(x) for x in np.asarray(self.action_indices, dtype=int).reshape(-1))
        if not raw_indices:
            raise ValueError("ValidationCyclicDwellPolicy requires at least one action index")
        for action_idx in raw_indices:
            if action_idx < 0 or action_idx >= int(self.candidate_masks.shape[0]):
                raise ValueError(f"action index outside candidate mask range: {action_idx}")
        self.action_indices = raw_indices
        self.dwell_steps = max(1, int(self.dwell_steps))
        if self.fallback_action_idx is not None:
            fallback = int(self.fallback_action_idx)
            if fallback < 0 or fallback >= int(self.candidate_masks.shape[0]):
                raise ValueError(f"fallback_action_idx outside candidate mask range: {fallback}")
            self.fallback_action_idx = fallback
        self.cursor = 0
        self.current_action_idx: int | None = None
        self.remaining_dwell = 0

    def reset(self) -> None:
        self.cursor = 0
        self.current_action_idx = None
        self.remaining_dwell = 0

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        raw_valid = feasible_candidate_mask(env, self.candidate_masks)
        valid = self._apply_warming_preservation(env, raw_valid)
        if (
            self.current_action_idx is not None
            and self.remaining_dwell > 0
            and bool(valid[int(self.current_action_idx)])
        ):
            self.remaining_dwell -= 1
            return self.candidate_masks[int(self.current_action_idx)].astype(bool).copy()

        action = self._next_valid_cycle_action(valid)
        if action is None and self.fallback_action_idx is not None and bool(valid[int(self.fallback_action_idx)]):
            action = int(self.fallback_action_idx)
        if action is None:
            valid_ids = np.flatnonzero(valid)
            if valid_ids.size == 0 and not np.array_equal(valid, raw_valid):
                valid_ids = np.flatnonzero(raw_valid)
            if valid_ids.size == 0:
                return np.zeros(self.candidate_masks.shape[1], dtype=bool)
            action = int(valid_ids[0])

        self.current_action_idx = int(action)
        self.remaining_dwell = max(0, int(self.dwell_steps) - 1)
        return self.candidate_masks[int(action)].astype(bool).copy()

    def act_scores(self, env: WarmupSchedulingEnv) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)

    def _next_valid_cycle_action(self, valid: np.ndarray) -> int | None:
        for offset in range(len(self.action_indices)):
            pos = (int(self.cursor) + int(offset)) % len(self.action_indices)
            action_idx = int(self.action_indices[pos])
            if bool(valid[action_idx]):
                self.cursor = (pos + 1) % len(self.action_indices)
                return action_idx
        return None

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


def _contiguous_sequence_slices(step_indices: np.ndarray) -> list[tuple[int, int]]:
    steps = np.asarray(step_indices, dtype=np.int64).reshape(-1)
    if steps.size == 0:
        return []
    slices: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, int(steps.size)):
        if int(steps[idx]) != int(steps[idx - 1]) + 1:
            slices.append((int(start), int(idx)))
            start = int(idx)
    slices.append((int(start), int(steps.size)))
    return [(begin, end) for begin, end in slices if end > begin]


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
