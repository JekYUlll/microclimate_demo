from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reuse import ARCHIVE_ROOT, ensure_archive_src

ensure_archive_src()

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import SensorSpecV2, load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def load_v2_helpers() -> Any:
    """Load constants and split helpers from the archived v2 training script."""

    path = ARCHIVE_ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_archived_v2_train_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load archived v2 helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_archive_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ARCHIVE_ROOT / path


def load_archived_sensor_specs(path: str | Path) -> list[SensorSpecV2]:
    return load_sensor_specs(resolve_archive_path(path))


def load_archived_oracle(
    path: str | Path | None,
    *,
    oracle_type: str,
    device: str = "cpu",
) -> LinearFrozenForecastOracle | TCNFrozenForecastOracle | None:
    if oracle_type == "none":
        return None
    if path is None:
        raise ValueError("--oracle-path is required unless --oracle-type none")
    resolved = resolve_archive_path(path)
    if oracle_type == "tcn":
        return TCNFrozenForecastOracle.load(resolved, device=device)
    if oracle_type == "linear":
        return LinearFrozenForecastOracle.load(str(resolved))
    raise ValueError(f"Unsupported oracle_type: {oracle_type}")


def normalization_stats(
    truth: pd.DataFrame,
    state_columns: tuple[str, ...],
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if start_idx is None and end_idx is None:
        return None, None
    start = int(start_idx or 0)
    end = int(end_idx or len(truth))
    if start < 0 or end <= start or end > len(truth):
        raise ValueError(f"Invalid normalization partition [{start}, {end}) for truth length {len(truth)}")
    values = truth.iloc[start:end][list(state_columns)].to_numpy(dtype=float)
    return (
        tuple(float(x) for x in np.mean(values, axis=0)),
        tuple(float(x) for x in np.maximum(np.std(values, axis=0), 1e-6)),
    )


def make_constraints(
    *,
    max_active: int | None,
    budget: float | None,
    startup_peak_budget: float | None,
) -> PowerConstraintsV2:
    return PowerConstraintsV2(
        max_active=max_active,
        per_step_budget=budget,
        startup_peak_budget=startup_peak_budget,
    )


def make_env_config(
    *,
    state_columns: tuple[str, ...],
    reward_target_columns: tuple[str, ...],
    lookback: int,
    episode_len: int,
    seed: int,
    freq_s: int,
    normalization_mean: tuple[float, ...] | None,
    normalization_std: tuple[float, ...] | None,
    lambda_warmup_abort: float,
    lambda_switch: float,
    event_reward_multiplier: float,
    energy_account: bool,
    energy_capacity: float,
    initial_energy: float,
    harvest_per_step: float,
    reserve_energy: float,
    lambda_energy_deficit: float,
    soc_soft_penalty_buffer: float,
    lambda_soc_soft_penalty: float,
) -> WarmupEnvConfig:
    return WarmupEnvConfig(
        state_columns=state_columns,
        reward_target_columns=reward_target_columns,
        lookback=int(lookback),
        episode_len=int(episode_len),
        seed=int(seed),
        base_freq_s=int(freq_s),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        lambda_warmup_abort=float(lambda_warmup_abort),
        lambda_switch=float(lambda_switch),
        event_reward_multiplier=float(event_reward_multiplier),
        energy_account_enabled=bool(energy_account),
        energy_capacity=float(energy_capacity),
        initial_energy=float(initial_energy),
        harvest_per_step=float(harvest_per_step),
        reserve_energy=float(reserve_energy),
        lambda_energy_deficit=float(lambda_energy_deficit),
        soc_soft_penalty_buffer=float(soc_soft_penalty_buffer),
        lambda_soc_soft_penalty=float(lambda_soc_soft_penalty),
    )


def build_warmup_env(
    *,
    truth: pd.DataFrame,
    sensors: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle | TCNFrozenForecastOracle | None,
) -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=cfg,
        oracle=oracle,
    )


def load_custom_ppo_policy(
    path: str | Path,
    *,
    truth: pd.DataFrame,
    sensors: list[SensorSpecV2],
    constraints: PowerConstraintsV2,
    env_cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle | TCNFrozenForecastOracle | None,
    device: str = "cpu",
) -> object:
    from dataclasses import fields

    import torch
    from v2.custom_ppo import CustomPPO, CustomPPOConfig, CustomPPOPolicy

    payload = torch.load(str(resolve_archive_path(path)), map_location=str(device), weights_only=False)
    cfg_data = dict(payload["cfg"])
    cfg_data["device"] = str(device)
    known = {field.name for field in fields(CustomPPOConfig)}
    cfg = CustomPPOConfig(**{key: value for key, value in cfg_data.items() if key in known})
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=oracle,
        candidate_masks=np.asarray(payload["candidate_masks"], dtype=bool),
        cfg=cfg,
        candidate_prior_logits=payload.get("candidate_prior_logits"),
    )
    trainer.model.load_state_dict(payload["state_dict"])
    trainer.model.eval()
    trainer.history = payload.get("history", trainer.history)
    return CustomPPOPolicy(trainer=trainer, name="custom_ppo")
