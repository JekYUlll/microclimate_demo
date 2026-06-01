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
    FeatureTransitionDataset,
    ForecastAwareAdvantageResidualPolicy,
    ForecastAwareRolloutValuePolicy,
    collect_anchor_advantage_dataset,
    collect_feature_transition_dataset,
    train_anchor_advantage_model,
    train_feature_transition_model,
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
    "FeatureTransitionDataset",
    "EventForecast",
    "EventForecasterTrainingConfig",
    "ForecastContextConfig",
    "ForecastAwareAdvantageResidualPolicy",
    "ForecastAwareBCPolicy",
    "ForecastAwareEventThresholdPolicy",
    "ForecastAwareRolloutValuePolicy",
    "MpcTeacherConfig",
    "MpcTeacherPolicy",
    "TeacherDataset",
    "augment_truth_with_event_forecasts",
    "beam_search_teacher_action",
    "build_event_forecast_dataset",
    "build_event_forecast",
    "collect_anchor_advantage_dataset",
    "collect_feature_transition_dataset",
    "collect_teacher_dataset",
    "load_bc_policy_checkpoint",
    "save_bc_policy_checkpoint",
    "train_bc_classifier",
    "train_anchor_advantage_model",
    "train_feature_transition_model",
    "train_event_forecaster",
]
