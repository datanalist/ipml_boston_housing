"""Утилиты для интеграции компонентов ML-инфраструктуры."""

import functools
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

from loguru import logger

T = TypeVar("T")


class ServiceStatus(Enum):
    """Статус сервиса."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ServiceConfig:
    """Конфигурация сервиса."""

    name: str
    host: str
    port: int
    endpoint: str = ""
    timeout: int = 10
    retries: int = 3
    extra: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Полный URL сервиса."""
        base = f"http://{self.host}:{self.port}"
        return f"{base}{self.endpoint}" if self.endpoint else base


def get_service_config(service_name: str) -> ServiceConfig:
    """
    Получение конфигурации сервиса из переменных окружения.

    Args:
        service_name: Имя сервиса (mlflow, minio, airflow, dvc)

    Returns:
        ServiceConfig с настройками сервиса
    """
    configs = {
        "mlflow": ServiceConfig(
            name="MLflow Tracking Server",
            host=os.environ.get("MLFLOW_HOST", "localhost"),
            port=int(os.environ.get("MLFLOW_PORT", "5000")),
            endpoint="/health",
            extra={
                "tracking_uri": os.environ.get(
                    "MLFLOW_TRACKING_URI", "http://localhost:5000"
                ),
                "username": os.environ.get("MLFLOW_TRACKING_USERNAME", "admin"),
                "password": os.environ.get("MLFLOW_TRACKING_PASSWORD", "admin"),
            },
        ),
        "minio": ServiceConfig(
            name="MinIO S3 Storage",
            host=os.environ.get("MINIO_HOST", "localhost"),
            port=int(os.environ.get("MINIO_PORT", "9000")),
            endpoint="/minio/health/live",
            extra={
                "access_key": os.environ.get("MINIO_ROOT_USER", "minioadmin"),
                "secret_key": os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"),
                "bucket": os.environ.get("MINIO_BUCKET", "mlflow-artifacts"),
            },
        ),
        "airflow": ServiceConfig(
            name="Apache Airflow",
            host=os.environ.get("AIRFLOW_HOST", "localhost"),
            port=int(os.environ.get("AIRFLOW_PORT", "8080")),
            endpoint="/health",
            extra={
                "username": os.environ.get("AIRFLOW_ADMIN_USERNAME", "admin"),
                "password": os.environ.get("AIRFLOW_ADMIN_PASSWORD", "admin"),
            },
        ),
        "dvc": ServiceConfig(
            name="DVC (Data Version Control)",
            host="local",
            port=0,
            extra={
                "remote": os.environ.get("DVC_REMOTE", "minio"),
            },
        ),
    }

    if service_name not in configs:
        raise ValueError(f"Неизвестный сервис: {service_name}")

    return configs[service_name]


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Декоратор для повторных попыток с экспоненциальной задержкой.

    Args:
        max_retries: Максимальное количество попыток
        base_delay: Начальная задержка в секундах
        max_delay: Максимальная задержка в секундах
        exponential_base: Основание экспоненты для увеличения задержки
        exceptions: Типы исключений для перехвата

    Returns:
        Декоратор
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        logger.warning(
                            f"Попытка {attempt + 1}/{max_retries} не удалась: {e}. "
                            f"Повтор через {delay:.1f}с..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Все {max_retries} попыток исчерпаны для {func.__name__}"
                        )

            raise last_exception  # type: ignore

        return wrapper

    return decorator


class UnifiedLogger:
    """
    Унифицированный логгер для всех компонентов ML-инфраструктуры.

    Обеспечивает единый формат логирования и автоматическое добавление
    контекста (компонент, timestamp, correlation_id).
    """

    def __init__(self, component: str):
        """
        Инициализация логгера.

        Args:
            component: Название компонента (mlflow, airflow, dvc, minio)
        """
        self.component = component
        self._correlation_id: str | None = None

    def set_correlation_id(self, correlation_id: str) -> None:
        """Установка correlation ID для трассировки."""
        self._correlation_id = correlation_id

    def _format_message(self, message: str) -> str:
        """Форматирование сообщения с контекстом."""
        prefix = f"[{self.component.upper()}]"
        if self._correlation_id:
            prefix = f"{prefix}[{self._correlation_id[:8]}]"
        return f"{prefix} {message}"

    def info(self, message: str, **kwargs: Any) -> None:
        """Информационное сообщение."""
        logger.info(self._format_message(message), **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Отладочное сообщение."""
        logger.debug(self._format_message(message), **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Предупреждение."""
        logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Сообщение об ошибке."""
        logger.error(self._format_message(message), **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Успешное сообщение."""
        logger.success(self._format_message(message), **kwargs)

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
            "correlation_id": self._correlation_id,
            **event_data,
        }

        log_func = getattr(logger, level, logger.info)
        log_func(f"{self._format_message(event_type)}: {event}")


def unified_logger(component: str) -> UnifiedLogger:
    """
    Фабрика для создания унифицированного логгера.

    Args:
        component: Название компонента

    Returns:
        UnifiedLogger для указанного компонента
    """
    return UnifiedLogger(component)


def measure_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func: Функция для измерения

    Returns:
        Декорированная функция
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.debug(f"{func.__name__} выполнена за {duration:.3f}с")
        return result

    return wrapper


def ensure_directory(path: str) -> None:
    """
    Создание директории если она не существует.

    Args:
        path: Путь к директории
    """
    import os

    os.makedirs(path, exist_ok=True)
