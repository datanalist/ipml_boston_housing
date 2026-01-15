"""Модуль уведомлений о результатах ML-пайплайнов."""

from src.notifications.notifier import (
    Notifier,
    NotificationChannel,
    notify_pipeline_complete,
    notify_pipeline_error,
    notify_experiment_results,
)
from src.notifications.templates import (
    NotificationTemplate,
    SuccessTemplate,
    ErrorTemplate,
    ExperimentSummaryTemplate,
)

__all__ = [
    # Notifier
    "Notifier",
    "NotificationChannel",
    "notify_pipeline_complete",
    "notify_pipeline_error",
    "notify_experiment_results",
    # Templates
    "NotificationTemplate",
    "SuccessTemplate",
    "ErrorTemplate",
    "ExperimentSummaryTemplate",
]
