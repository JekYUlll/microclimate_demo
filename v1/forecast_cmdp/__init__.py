"""Forecast-aware constrained scheduling prototype."""

from .dataset import TeacherDataset, collect_teacher_dataset
from .features import EventForecast, ForecastContextConfig, build_event_forecast
from .event_forecaster import (
    EventForecasterTrainingConfig,
    augment_truth_with_event_forecasts,
    build_event_forecast_dataset,
    train_event_forecaster,
)
from .cost_policy import (
    ActionCostTrainingConfig,
    AnchorAdvantageDataset,
    ForecastAwareAdvantageResidualPolicy,
    collect_anchor_advantage_dataset,
    train_anchor_advantage_model,
)
from .mpc_teacher import MpcTeacherConfig, MpcTeacherPolicy, beam_search_teacher_action
from .policy import (
    BCTrainingConfig,
    ForecastAwareBCPolicy,
    ForecastAwareEventThresholdPolicy,
    load_bc_policy_checkpoint,
    save_bc_policy_checkpoint,
    train_bc_classifier,
)

__all__ = [
    "BCTrainingConfig",
    "ActionCostTrainingConfig",
    "AnchorAdvantageDataset",
    "EventForecast",
    "EventForecasterTrainingConfig",
    "ForecastContextConfig",
    "ForecastAwareAdvantageResidualPolicy",
    "ForecastAwareBCPolicy",
    "ForecastAwareEventThresholdPolicy",
    "MpcTeacherConfig",
    "MpcTeacherPolicy",
    "TeacherDataset",
    "augment_truth_with_event_forecasts",
    "beam_search_teacher_action",
    "build_event_forecast_dataset",
    "build_event_forecast",
    "collect_anchor_advantage_dataset",
    "collect_teacher_dataset",
    "load_bc_policy_checkpoint",
    "save_bc_policy_checkpoint",
    "train_bc_classifier",
    "train_anchor_advantage_model",
    "train_event_forecaster",
]
