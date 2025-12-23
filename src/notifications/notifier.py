"""Система уведомлений о результатах ML-пайплайнов."""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.notifications.templates import (
    ErrorTemplate,
    ExperimentSummaryTemplate,
    NotificationTemplate,
    SuccessTemplate,
)

# Директория для уведомлений
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports/notifications"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class NotificationChannel(Enum):
    """Каналы уведомлений."""

    FILE = "file"  # Сохранение в файл
    CONSOLE = "console"  # Вывод в консоль
    MINIO = "minio"  # Загрузка в MinIO


class Notifier:
    """
    Система уведомлений о результатах ML-пайплайнов.

    Поддерживает:
    - Файловые отчёты (JSON, Markdown, Text)
    - Консольный вывод
    - Загрузку в MinIO
    """

    def __init__(
        self,
        channels: list[NotificationChannel] | None = None,
        reports_dir: Path | str | None = None,
    ):
        """
        Инициализация нотификатора.

        Args:
            channels: Список каналов уведомлений
            reports_dir: Директория для отчётов
        """
        self.channels = channels or [
            NotificationChannel.FILE,
            NotificationChannel.CONSOLE,
        ]
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def notify(
        self,
        template: NotificationTemplate,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Отправка уведомления через все настроенные каналы.

        Args:
            template: Шаблон уведомления
            formats: Форматы файлов (json, md, txt)

        Returns:
            Словарь с результатами отправки
        """
        formats = formats or ["json", "md", "txt"]
        results: dict[str, Any] = {"success": True, "channels": {}}

        for channel in self.channels:
            try:
                if channel == NotificationChannel.FILE:
                    result = self._send_to_file(template, formats)
                elif channel == NotificationChannel.CONSOLE:
                    result = self._send_to_console(template)
                elif channel == NotificationChannel.MINIO:
                    result = self._send_to_minio(template, formats)
                else:
                    result = {"success": False, "error": f"Unknown channel: {channel}"}

                results["channels"][channel.value] = result

            except Exception as e:
                logger.exception(f"Error sending notification to {channel}: {e}")
                results["channels"][channel.value] = {
                    "success": False,
                    "error": str(e),
                }
                results["success"] = False

        return results

    def _send_to_file(
        self,
        template: NotificationTemplate,
        formats: list[str],
    ) -> dict[str, Any]:
        """
        Сохранение уведомления в файлы.

        Args:
            template: Шаблон уведомления
            formats: Форматы файлов

        Returns:
            Результат сохранения
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        notification_type = template.__class__.__name__.replace("Template", "").lower()
        base_name = f"{notification_type}_{timestamp}"

        saved_files = []

        if "json" in formats:
            json_path = self.reports_dir / f"{base_name}.json"
            with open(json_path, "w") as f:
                json.dump(template.render_json(), f, indent=2, ensure_ascii=False)
            saved_files.append(str(json_path))

        if "md" in formats:
            md_path = self.reports_dir / f"{base_name}.md"
            with open(md_path, "w") as f:
                f.write(template.render_markdown())
            saved_files.append(str(md_path))

        if "txt" in formats:
            txt_path = self.reports_dir / f"{base_name}.txt"
            with open(txt_path, "w") as f:
                f.write(template.render_text())
            saved_files.append(str(txt_path))

        logger.info(f"Notification saved to: {', '.join(saved_files)}")

        return {
            "success": True,
            "files": saved_files,
        }

    def _send_to_console(self, template: NotificationTemplate) -> dict[str, Any]:
        """
        Вывод уведомления в консоль.

        Args:
            template: Шаблон уведомления

        Returns:
            Результат вывода
        """
        print("\n" + template.render_text() + "\n")
        return {"success": True}

    def _send_to_minio(
        self,
        template: NotificationTemplate,
        formats: list[str],
    ) -> dict[str, Any]:
        """
        Загрузка уведомления в MinIO.

        Args:
            template: Шаблон уведомления
            formats: Форматы файлов

        Returns:
            Результат загрузки
        """
        try:
            import boto3
            from botocore.client import Config

            endpoint_url = os.environ.get(
                "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
            )
            access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
            secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
            bucket_name = os.environ.get("NOTIFICATIONS_BUCKET", "notifications")

            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
            )

            # Создаём bucket если не существует
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except Exception:
                s3_client.create_bucket(Bucket=bucket_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            notification_type = template.__class__.__name__.replace(
                "Template", ""
            ).lower()
            base_key = f"notifications/{notification_type}_{timestamp}"

            uploaded_keys = []

            if "json" in formats:
                key = f"{base_key}.json"
                content = json.dumps(
                    template.render_json(), indent=2, ensure_ascii=False
                )
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=content.encode("utf-8"),
                    ContentType="application/json",
                )
                uploaded_keys.append(key)

            if "md" in formats:
                key = f"{base_key}.md"
                content = template.render_markdown()
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=content.encode("utf-8"),
                    ContentType="text/markdown",
                )
                uploaded_keys.append(key)

            logger.info(f"Notification uploaded to MinIO: {', '.join(uploaded_keys)}")

            return {
                "success": True,
                "bucket": bucket_name,
                "keys": uploaded_keys,
            }

        except ImportError:
            return {
                "success": False,
                "error": "boto3 not installed",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Удобные функции для использования
# ═══════════════════════════════════════════════════════════════════════════════


def notify_pipeline_complete(
    pipeline_name: str,
    run_id: str,
    duration_seconds: float,
    metrics: dict[str, float] | None = None,
    best_model: str | None = None,
    stages_completed: int = 0,
    stages_total: int = 0,
    artifacts: list[str] | None = None,
    channels: list[NotificationChannel] | None = None,
) -> dict[str, Any]:
    """
    Отправка уведомления об успешном завершении пайплайна.

    Args:
        pipeline_name: Название пайплайна
        run_id: ID запуска
        duration_seconds: Время выполнения
        metrics: Метрики
        best_model: Лучшая модель
        stages_completed: Количество завершённых этапов
        stages_total: Общее количество этапов
        artifacts: Список артефактов
        channels: Каналы уведомлений

    Returns:
        Результат отправки
    """
    template = SuccessTemplate(
        pipeline_name=pipeline_name,
        run_id=run_id,
        duration_seconds=duration_seconds,
        metrics=metrics or {},
        best_model=best_model,
        stages_completed=stages_completed,
        stages_total=stages_total,
        artifacts=artifacts or [],
    )

    notifier = Notifier(channels=channels)
    return notifier.notify(template)


def notify_pipeline_error(
    pipeline_name: str,
    run_id: str,
    stage_name: str,
    error_type: str,
    error_message: str,
    traceback: str | None = None,
    context: dict[str, Any] | None = None,
    channels: list[NotificationChannel] | None = None,
) -> dict[str, Any]:
    """
    Отправка уведомления об ошибке пайплайна.

    Args:
        pipeline_name: Название пайплайна
        run_id: ID запуска
        stage_name: Название этапа с ошибкой
        error_type: Тип ошибки
        error_message: Сообщение об ошибке
        traceback: Трейсбек
        context: Дополнительный контекст
        channels: Каналы уведомлений

    Returns:
        Результат отправки
    """
    template = ErrorTemplate(
        pipeline_name=pipeline_name,
        run_id=run_id,
        stage_name=stage_name,
        error_type=error_type,
        error_message=error_message,
        traceback=traceback,
        context=context or {},
    )

    notifier = Notifier(channels=channels)
    return notifier.notify(template)


def notify_experiment_results(
    experiment_name: str,
    total_experiments: int,
    successful_experiments: int,
    failed_experiments: int,
    best_model: dict[str, Any] | None = None,
    all_results: list[dict[str, Any]] | None = None,
    duration_seconds: float = 0.0,
    channels: list[NotificationChannel] | None = None,
) -> dict[str, Any]:
    """
    Отправка уведомления о результатах экспериментов.

    Args:
        experiment_name: Название эксперимента
        total_experiments: Общее количество экспериментов
        successful_experiments: Успешные эксперименты
        failed_experiments: Неудачные эксперименты
        best_model: Лучшая модель
        all_results: Все результаты
        duration_seconds: Время выполнения
        channels: Каналы уведомлений

    Returns:
        Результат отправки
    """
    template = ExperimentSummaryTemplate(
        experiment_name=experiment_name,
        total_experiments=total_experiments,
        successful_experiments=successful_experiments,
        failed_experiments=failed_experiments,
        best_model=best_model or {},
        all_results=all_results or [],
        duration_seconds=duration_seconds,
    )

    notifier = Notifier(channels=channels)
    return notifier.notify(template)
