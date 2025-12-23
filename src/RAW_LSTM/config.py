from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathConfig:
    """Centralized definition of common project paths."""

    raw_data_dir: Path = PROJECT_ROOT / "data"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    checkpoint_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    artifacts_dir: Path = PROJECT_ROOT / "models" / "artifacts"


@dataclass
class DataConfig:
    """Configuration describing how to interpret the meteorological data."""

    timestamp_col: str = "时间（世界时）"
    target_col: str = "气温（℃）"
    feature_cols: List[str] = field(
        default_factory=lambda: [
            "气温（℃）",
            "气压（hPa）",
            "风速（m/s）",
            "相对湿度（%）",
            "总云量（成）",
            "低云量（成）",
        ]
    )
    freq: str = "6H"
    train_ratio: float = 0.8


@dataclass
class TrainingConfig:
    """Hyper-parameters for the LSTM training loop."""

    # 需要微调时主要调整这些参数，也可通过 CLI 传入相同名字的选项覆盖
    input_window: int = 24
    forecast_horizon: int = 6
    batch_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 25
    num_workers: int = 0
    seed: int = 42


@dataclass
class Settings:
    """Top-level container used by the scripts."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
