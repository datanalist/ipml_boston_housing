"""Airflow callbacks для мониторинга выполнения задач.

Этот файл также должен быть скопирован в airflow/plugins/monitoring_callbacks.py
при развёртывании Docker инфраструктуры.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from airflow.models import TaskInstance
except ImportError:
    # Для использования вне Airflow окружения
    TaskInstance = None  # type: ignore

from loguru import logger

# Директория для логов уведомлений
NOTIFICATIONS_DIR = Path(os.environ.get("NOTIFICATIONS_DIR", "data/notifications"))
NOTIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _get_task_context(context: dict) -> dict[str, Any]:
    """
    Извлечение контекста задачи.

    Args:
        context: Контекст Airflow

    Returns:
        Словарь с информацией о задаче
    """
    ti = context.get("task_instance") or context.get("ti")
    dag_run = context.get("dag_run")

    if ti is None:
        return {"error": "No task instance in context"}

    return {
        "dag_id": ti.dag_id if ti else None,
        "task_id": ti.task_id if ti else None,
        "run_id": dag_run.run_id if dag_run else None,
        "execution_date": str(ti.execution_date) if ti else None,
        "try_number": ti.try_number if ti else None,
        "state": ti.state if ti else None,
        "duration": ti.duration if ti else None,
        "start_date": str(ti.start_date) if ti and ti.start_date else None,
        "end_date": str(ti.end_date) if ti and ti.end_date else None,
    }


def _save_notification(notification: dict[str, Any]) -> None:
    """
    Сохранение уведомления в файл.

    Args:
        notification: Данные уведомления
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dag_id = notification.get("task_context", {}).get("dag_id", "unknown")
    task_id = notification.get("task_context", {}).get("task_id", "unknown")
    event_type = notification.get("event_type", "unknown")

    filename = f"{dag_id}_{task_id}_{event_type}_{timestamp}.json"
    filepath = NOTIFICATIONS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(notification, f, indent=2, ensure_ascii=False, default=str)

    logger.debug(f"Notification saved: {filepath}")


def on_task_start(context: dict) -> None:
    """
    Callback при запуске задачи.

    Args:
        context: Контекст Airflow
    """
    task_context = _get_task_context(context)

    logger.info(
        f"🚀 TASK STARTED: {task_context['dag_id']}.{task_context['task_id']} "
        f"(attempt: {task_context['try_number']})"
    )

    notification = {
        "event_type": "task_started",
        "timestamp": datetime.now().isoformat(),
        "task_context": task_context,
    }

    _save_notification(notification)


def on_task_success(context: dict) -> None:
    """
    Callback при успешном завершении задачи.

    Args:
        context: Контекст Airflow
    """
    task_context = _get_task_context(context)

    duration_str = (
        f"{task_context['duration']:.2f}s" if task_context["duration"] else "N/A"
    )
    logger.success(
        f"✅ TASK SUCCESS: {task_context['dag_id']}.{task_context['task_id']} "
        f"(duration: {duration_str})"
    )

    notification = {
        "event_type": "task_success",
        "timestamp": datetime.now().isoformat(),
        "task_context": task_context,
        "metrics": {
            "duration_seconds": task_context["duration"],
        },
    }

    _save_notification(notification)


def on_task_failure(context: dict) -> None:
    """
    Callback при ошибке задачи.

    Args:
        context: Контекст Airflow
    """
    task_context = _get_task_context(context)
    exception = context.get("exception")

    logger.error(
        f"❌ TASK FAILED: {task_context['dag_id']}.{task_context['task_id']} "
        f"(attempt: {task_context['try_number']})"
    )

    if exception:
        logger.error(f"   Error: {exception!s}")

    notification = {
        "event_type": "task_failed",
        "timestamp": datetime.now().isoformat(),
        "task_context": task_context,
        "error": {
            "type": type(exception).__name__ if exception else None,
            "message": str(exception) if exception else None,
        },
    }

    _save_notification(notification)


def on_task_retry(context: dict) -> None:
    """
    Callback при повторе задачи.

    Args:
        context: Контекст Airflow
    """
    task_context = _get_task_context(context)
    exception = context.get("exception")

    logger.warning(
        f"🔄 TASK RETRY: {task_context['dag_id']}.{task_context['task_id']} "
        f"(attempt: {task_context['try_number']})"
    )

    notification = {
        "event_type": "task_retry",
        "timestamp": datetime.now().isoformat(),
        "task_context": task_context,
        "error": {
            "type": type(exception).__name__ if exception else None,
            "message": str(exception) if exception else None,
        },
    }

    _save_notification(notification)


def on_dag_start(context: dict) -> None:
    """
    Callback при запуске DAG.

    Args:
        context: Контекст Airflow
    """
    dag_run = context.get("dag_run")
    dag_id = dag_run.dag_id if dag_run else "unknown"
    run_id = dag_run.run_id if dag_run else "unknown"

    logger.info(f"🎬 DAG STARTED: {dag_id} (run_id: {run_id})")

    notification = {
        "event_type": "dag_started",
        "timestamp": datetime.now().isoformat(),
        "dag_context": {
            "dag_id": dag_id,
            "run_id": run_id,
            "execution_date": str(dag_run.execution_date) if dag_run else None,
            "conf": dict(dag_run.conf) if dag_run and dag_run.conf else {},
        },
    }

    _save_notification(notification)


def on_dag_success(context: dict) -> None:
    """
    Callback при успешном завершении DAG.

    Args:
        context: Контекст Airflow
    """
    dag_run = context.get("dag_run")
    dag_id = dag_run.dag_id if dag_run else "unknown"
    run_id = dag_run.run_id if dag_run else "unknown"

    logger.success(f"🏁 DAG SUCCESS: {dag_id} (run_id: {run_id})")

    notification = {
        "event_type": "dag_success",
        "timestamp": datetime.now().isoformat(),
        "dag_context": {
            "dag_id": dag_id,
            "run_id": run_id,
            "execution_date": str(dag_run.execution_date) if dag_run else None,
            "start_date": str(dag_run.start_date)
            if dag_run and dag_run.start_date
            else None,
            "end_date": str(dag_run.end_date) if dag_run and dag_run.end_date else None,
        },
    }

    _save_notification(notification)


def on_dag_failure(context: dict) -> None:
    """
    Callback при ошибке DAG.

    Args:
        context: Контекст Airflow
    """
    dag_run = context.get("dag_run")
    dag_id = dag_run.dag_id if dag_run else "unknown"
    run_id = dag_run.run_id if dag_run else "unknown"

    logger.error(f"💥 DAG FAILED: {dag_id} (run_id: {run_id})")

    notification = {
        "event_type": "dag_failed",
        "timestamp": datetime.now().isoformat(),
        "dag_context": {
            "dag_id": dag_id,
            "run_id": run_id,
            "execution_date": str(dag_run.execution_date) if dag_run else None,
        },
    }

    _save_notification(notification)


# ═══════════════════════════════════════════════════════════════════════════════
# Словарь callbacks для удобного использования в DAG
# ═══════════════════════════════════════════════════════════════════════════════

MONITORING_CALLBACKS = {
    "on_execute_callback": on_task_start,
    "on_success_callback": on_task_success,
    "on_failure_callback": on_task_failure,
    "on_retry_callback": on_task_retry,
}

DAG_MONITORING_CALLBACKS = {
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}
