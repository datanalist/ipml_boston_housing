"""Декораторы для автоматического логирования в MLflow."""

import functools
import time
from typing import Any, Callable

import mlflow
from loguru import logger


def mlflow_run(
    experiment_name: str = "boston-housing",
    run_name: str | None = None,
    tags: dict | None = None,
):
    """
    Декоратор для автоматического создания MLflow run.

    Создаёт новый MLflow run перед выполнением функции и автоматически
    логирует время выполнения.

    Args:
        experiment_name: Название эксперимента MLflow
        run_name: Имя запуска (опционально)
        tags: Теги для запуска (опционально)

    Returns:
        Декорированная функция

    Example:
        >>> @mlflow_run(experiment_name="my-exp", run_name="baseline")
        ... def train_model(params):
        ...     model = RandomForestRegressor(**params)
        ...     model.fit(X_train, y_train)
        ...     return model
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=run_name, tags=tags):
                # Логируем время начала
                start_time = time.time()
                mlflow.log_param("start_time", start_time)

                # Выполняем функцию
                result = func(*args, **kwargs)

                # Логируем время выполнения
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)

                logger.info(f"Эксперимент завершён за {duration:.2f}с")
                return result

        return wrapper

    return decorator


def log_params_decorator(func: Callable) -> Callable:
    """
    Декоратор для автоматического логирования параметров функции.

    Автоматически логирует все kwargs переданные в функцию как параметры
    MLflow. Работает только если уже есть активный MLflow run.

    Args:
        func: Декорируемая функция

    Returns:
        Декорированная функция

    Example:
        >>> @log_params_decorator
        ... def train(n_estimators=100, max_depth=10):
        ...     model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        ...     return model
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Логируем все kwargs как параметры
        if kwargs and mlflow.active_run():
            mlflow.log_params(kwargs)
            logger.debug(f"Залогированы параметры: {list(kwargs.keys())}")
        return func(*args, **kwargs)

    return wrapper


def log_metrics_decorator(metric_keys: list[str]):
    """
    Декоратор для автоматического логирования метрик из результата.

    Извлекает указанные ключи из возвращаемого словаря и логирует их
    как метрики MLflow.

    Args:
        metric_keys: Список ключей метрик для логирования

    Returns:
        Декоратор

    Example:
        >>> @log_metrics_decorator(["r2_score", "rmse"])
        ... def evaluate(model, X, y) -> dict:
        ...     y_pred = model.predict(X)
        ...     return {
        ...         "r2_score": r2_score(y, y_pred),
        ...         "rmse": np.sqrt(mean_squared_error(y, y_pred))
        ...     }
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            result = func(*args, **kwargs)

            if isinstance(result, dict) and mlflow.active_run():
                metrics_to_log = {
                    k: v
                    for k, v in result.items()
                    if k in metric_keys and isinstance(v, (int, float))
                }
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)
                    logger.debug(f"Залогированы метрики: {list(metrics_to_log.keys())}")

            return result

        return wrapper

    return decorator


def log_artifact_decorator(artifact_path: str | None = None):
    """
    Декоратор для автоматического логирования артефакта.

    Если функция возвращает путь к файлу (str или Path), этот файл
    будет залогирован как артефакт MLflow.

    Args:
        artifact_path: Путь в артефактах MLflow (опционально)

    Returns:
        Декоратор

    Example:
        >>> @log_artifact_decorator(artifact_path="plots")
        ... def create_plot(data) -> str:
        ...     fig, ax = plt.subplots()
        ...     ax.plot(data)
        ...     path = "plot.png"
        ...     fig.savefig(path)
        ...     return path
    """
    from pathlib import Path

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)

            if mlflow.active_run():
                if isinstance(result, (str, Path)) and Path(result).exists():
                    mlflow.log_artifact(str(result), artifact_path)
                    logger.debug(f"Артефакт залогирован: {result}")

            return result

        return wrapper

    return decorator


def timed_execution(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.

    Логирует время выполнения как метрику MLflow с именем
    {function_name}_duration_seconds.

    Args:
        func: Декорируемая функция

    Returns:
        Декорированная функция

    Example:
        >>> @timed_execution
        ... def expensive_computation():
        ...     # долгие вычисления
        ...     pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        if mlflow.active_run():
            metric_name = f"{func.__name__}_duration_seconds"
            mlflow.log_metric(metric_name, duration)

        logger.debug(f"{func.__name__} выполнена за {duration:.2f}с")
        return result

    return wrapper
