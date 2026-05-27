"""Forecast-aware constrained scheduling prototype."""

from .dataset import TeacherDataset, collect_teacher_dataset
from .features import EventForecast, ForecastContextConfig, build_event_forecast
from .mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, beam_search_teacher_action
from .policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    load_bc_policy_checkpoint,
    save_bc_policy_checkpoint,
    train_bc_classifier,
)

__all__ = [
    "BCTrainingConfig",
    "EventForecast",
    "ForecastContextConfig",
    "ForecastAwareBCPolicy",
    "MpcTeacherConfig",
    "MpcTeacherPolicy",
    "TeacherDataset",
    "beam_search_teacher_action",
    "build_event_forecast",
    "collect_teacher_dataset",
    "load_bc_policy_checkpoint",
    "save_bc_policy_checkpoint",
    "train_bc_classifier",
]
