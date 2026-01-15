"""Расширенное логирование для ML-пайплайнов."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Базовая директория для логов
LOGS_DIR = Path(os.environ.get("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class MonitoringLogger:
    """
    Расширенный логгер для мониторинга ML-пайплайнов.

    Обеспечивает:
    - Ротацию файлов логов
    - Структурированное логирование (JSON)
    - Отдельные файлы для разных компонентов
    - Консольный вывод с цветами
    """

    def __init__(
        self,
        component: str,
        log_dir: Path | None = None,
        rotation: str = "10 MB",
        retention: str = "7 days",
        json_logs: bool = True,
    ):
        """
        Инициализация логгера.

        Args:
            component: Название компонента (pipeline, training, monitoring)
            log_dir: Директория для логов
            rotation: Условие ротации логов (размер или время)
            retention: Срок хранения логов
            json_logs: Включить JSON-логирование
        """
        self.component = component
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._configure_handlers(rotation, retention, json_logs)

    def _configure_handlers(
        self,
        rotation: str,
        retention: str,
        json_logs: bool,
    ) -> None:
        """Настройка обработчиков логов."""
        # Очищаем дефолтные обработчики
        logger.remove()

        # Консольный вывод с цветами
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{self.component}</cyan> | "
            "<level>{message}</level>",
            level="INFO",
            colorize=True,
        )

        # Файл с обычными логами
        logger.add(
            self.log_dir / f"{self.component}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # JSON логи для парсинга
        if json_logs:
            logger.add(
                self.log_dir / f"{self.component}.json",
                format="{message}",
                level="INFO",
                rotation=rotation,
                retention=retention,
                serialize=True,
            )

        # Файл только с ошибками
        logger.add(
            self.log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            f"{self.component} | {{message}}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
        )

    def info(self, message: str, **kwargs: Any) -> None:
        """Информационное сообщение."""
        logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Отладочное сообщение."""
        logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Предупреждение."""
        logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Ошибка."""
        logger.error(message, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Успешное сообщение."""
        logger.success(message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Логирование исключения с трейсбеком."""
        logger.exception(message, **kwargs)

    def log_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        level: str = "info",
    ) -> None:
        """
        Логирование структурированного события.

        Args:
            event_type: Тип события
            event_data: Данные события
            level: Уровень логирования
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component,
            "event_type": event_type,
            **event_data,
        }

        log_func = getattr(logger, level, logger.info)
        log_func(f"EVENT: {event_type} | {json.dumps(event, ensure_ascii=False)}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "",
    ) -> None:
        """
        Логирование метрик.

        Args:
            metrics: Словарь метрик
            prefix: Префикс для метрик
        """
        for name, value in metrics.items():
            metric_name = f"{prefix}_{name}" if prefix else name
            logger.info(f"METRIC: {metric_name} = {value:.4f}")

    def log_stage_start(self, stage_name: str, params: dict | None = None) -> None:
        """
        Логирование начала этапа.

        Args:
            stage_name: Название этапа
            params: Параметры этапа
        """
        message = f"▶️ STAGE START: {stage_name}"
        if params:
            message += f" | params: {params}"
        logger.info(message)

    def log_stage_end(
        self,
        stage_name: str,
        duration_seconds: float,
        success: bool = True,
        metrics: dict | None = None,
    ) -> None:
        """
        Логирование завершения этапа.

        Args:
            stage_name: Название этапа
            duration_seconds: Время выполнения
            success: Успешность выполнения
            metrics: Метрики этапа
        """
        status = "✅ SUCCESS" if success else "❌ FAILED"
        message = f"⏹️ STAGE END: {stage_name} | {status} | {duration_seconds:.2f}s"
        if metrics:
            message += f" | metrics: {metrics}"

        if success:
            logger.success(message)
        else:
            logger.error(message)


def configure_logging(
    component: str = "pipeline",
    log_dir: Path | str | None = None,
    level: str = "INFO",
    json_logs: bool = True,
) -> MonitoringLogger:
    """
    Конфигурация логирования для компонента.

    Args:
        component: Название компонента
        log_dir: Директория для логов
        level: Уровень логирования
        json_logs: Включить JSON-логирование

    Returns:
        Настроенный MonitoringLogger
    """
    if log_dir:
        log_dir = Path(log_dir)

    return MonitoringLogger(
        component=component,
        log_dir=log_dir,
        json_logs=json_logs,
    )


def get_logger(component: str = "default") -> MonitoringLogger:
    """
    Получение логгера для компонента.

    Args:
        component: Название компонента

    Returns:
        MonitoringLogger
    """
    return MonitoringLogger(component=component)
