"""Утилиты для работы с MLflow экспериментами."""

from typing import Any

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger


def get_best_run(
    experiment_name: str,
    metric: str = "r2_score",
    ascending: bool = False,
) -> dict[str, Any]:
    """
    Получение лучшего запуска по метрике.

    Ищет в указанном эксперименте запуск с лучшим значением метрики
    и возвращает его данные.

    Args:
        experiment_name: Название эксперимента
        metric: Метрика для сортировки
        ascending: True для минимизации (RMSE), False для максимизации (R²)

    Returns:
        Словарь с информацией о лучшем запуске:
        - run_id: ID запуска
        - metrics: Все метрики запуска
        - params: Все параметры запуска
        - tags: Все теги запуска
        - artifact_uri: URI артефактов

    Example:
        >>> best = get_best_run("boston-housing", metric="r2_score")
        >>> print(f"Лучший R²: {best['metrics']['r2_score']:.4f}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Эксперимент '{experiment_name}' не найден")
        return {}

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        logger.warning(f"Нет запусков в эксперименте '{experiment_name}'")
        return {}

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "tags": best_run.data.tags,
        "artifact_uri": best_run.info.artifact_uri,
        "start_time": best_run.info.start_time,
        "end_time": best_run.info.end_time,
        "status": best_run.info.status,
    }


def load_best_model(
    experiment_name: str, metric: str = "r2_score", ascending: bool = False
):
    """
    Загрузка лучшей модели по метрике.

    Args:
        experiment_name: Название эксперимента
        metric: Метрика для выбора лучшего запуска
        ascending: True для минимизации, False для максимизации

    Returns:
        Загруженная sklearn модель

    Raises:
        ValueError: Если нет запусков в эксперименте

    Example:
        >>> model = load_best_model("boston-housing")
        >>> predictions = model.predict(X_new)
    """
    best_run = get_best_run(experiment_name, metric, ascending)
    if not best_run:
        raise ValueError(f"Нет запусков в эксперименте '{experiment_name}'")

    model_uri = f"runs:/{best_run['run_id']}/model"
    logger.info(f"Загрузка модели из {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def compare_runs(
    experiment_name: str,
    metrics: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Сравнение запусков эксперимента.

    Создаёт таблицу с метриками и параметрами для удобного сравнения
    различных запусков эксперимента.

    Args:
        experiment_name: Название эксперимента
        metrics: Список метрик для включения (по умолчанию r2_score, rmse, mae)
        top_n: Количество лучших запусков для сравнения

    Returns:
        DataFrame с колонками:
        - run_id: Сокращённый ID запуска
        - metric_*: Значения метрик
        - param_*: Значения параметров

    Example:
        >>> comparison = compare_runs("boston-housing", top_n=5)
        >>> print(comparison.to_string())
    """
    if metrics is None:
        metrics = ["r2_score", "rmse", "mae"]

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Эксперимент '{experiment_name}' не найден")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2_score DESC"],
        max_results=top_n,
    )

    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id[:8],
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "status": run.info.status,
        }
        row.update(
            {f"metric_{k}": v for k, v in run.data.metrics.items() if k in metrics}
        )
        row.update({f"param_{k}": v for k, v in run.data.params.items()})
        data.append(row)

    return pd.DataFrame(data)


def register_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "r2_score",
    ascending: bool = False,
) -> str:
    """
    Регистрация лучшей модели в Model Registry.

    Args:
        experiment_name: Название эксперимента
        model_name: Имя для регистрации модели
        metric: Метрика для выбора лучшего запуска
        ascending: True для минимизации, False для максимизации

    Returns:
        Версия зарегистрированной модели

    Raises:
        ValueError: Если нет запусков в эксперименте

    Example:
        >>> version = register_best_model("boston-housing", "boston-housing-rf")
        >>> print(f"Зарегистрирована версия: {version}")
    """
    best_run = get_best_run(experiment_name, metric, ascending)
    if not best_run:
        raise ValueError(f"Нет запусков в эксперименте '{experiment_name}'")

    model_uri = f"runs:/{best_run['run_id']}/model"
    result = mlflow.register_model(model_uri, model_name)
    logger.info(f"Модель зарегистрирована: {model_name} v{result.version}")

    return result.version


def delete_experiment_runs(
    experiment_name: str,
    keep_top_n: int = 10,
    metric: str = "r2_score",
    ascending: bool = False,
    dry_run: bool = True,
) -> list[str]:
    """
    Удаление старых запусков, кроме топ-N по метрике.

    Помогает очистить эксперимент от неудачных или устаревших запусков,
    сохраняя только лучшие результаты.

    Args:
        experiment_name: Название эксперимента
        keep_top_n: Количество лучших запусков для сохранения
        metric: Метрика для сортировки
        ascending: True для минимизации, False для максимизации
        dry_run: Если True, только показать что будет удалено

    Returns:
        Список ID удалённых запусков

    Example:
        >>> # Сначала посмотрим что будет удалено
        >>> deleted = delete_experiment_runs("boston-housing", keep_top_n=5, dry_run=True)
        >>> print(f"Будет удалено {len(deleted)} запусков")
        >>> # Теперь удалим
        >>> deleted = delete_experiment_runs("boston-housing", keep_top_n=5, dry_run=False)
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Эксперимент '{experiment_name}' не найден")
        return []

    order = "ASC" if ascending else "DESC"
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
    )

    deleted_ids = []
    for run in all_runs[keep_top_n:]:
        if dry_run:
            logger.info(f"[DRY RUN] Будет удалён run: {run.info.run_id}")
        else:
            client.delete_run(run.info.run_id)
            logger.info(f"Удалён run: {run.info.run_id}")
        deleted_ids.append(run.info.run_id)

    if deleted_ids:
        action = "Будет удалено" if dry_run else "Удалено"
        logger.info(f"{action} {len(deleted_ids)} запусков")

    return deleted_ids


def get_experiment_summary(experiment_name: str) -> dict[str, Any]:
    """
    Получение сводки по эксперименту.

    Args:
        experiment_name: Название эксперимента

    Returns:
        Словарь со сводной информацией:
        - total_runs: Общее количество запусков
        - finished_runs: Количество завершённых запусков
        - best_r2: Лучшее значение R²
        - best_rmse: Лучшее значение RMSE
        - experiment_id: ID эксперимента
        - artifact_location: Путь к артефактам

    Example:
        >>> summary = get_experiment_summary("boston-housing")
        >>> print(f"Всего запусков: {summary['total_runs']}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Эксперимент '{experiment_name}' не найден")
        return {}

    all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    finished_runs = [r for r in all_runs if r.info.status == "FINISHED"]

    r2_values = [
        r.data.metrics.get("r2_score")
        for r in finished_runs
        if r.data.metrics.get("r2_score") is not None
    ]
    rmse_values = [
        r.data.metrics.get("rmse")
        for r in finished_runs
        if r.data.metrics.get("rmse") is not None
    ]

    return {
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "artifact_location": experiment.artifact_location,
        "total_runs": len(all_runs),
        "finished_runs": len(finished_runs),
        "best_r2": max(r2_values) if r2_values else None,
        "best_rmse": min(rmse_values) if rmse_values else None,
        "worst_r2": min(r2_values) if r2_values else None,
        "worst_rmse": max(rmse_values) if rmse_values else None,
        "avg_r2": sum(r2_values) / len(r2_values) if r2_values else None,
        "avg_rmse": sum(rmse_values) / len(rmse_values) if rmse_values else None,
    }


def get_run_by_name(experiment_name: str, run_name: str) -> dict[str, Any] | None:
    """
    Получение запуска по имени.

    Args:
        experiment_name: Название эксперимента
        run_name: Имя запуска

    Returns:
        Словарь с данными запуска или None если не найден

    Example:
        >>> run = get_run_by_name("boston-housing", "rf-baseline")
        >>> if run:
        ...     print(f"R²: {run['metrics']['r2_score']}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
        "artifact_uri": run.info.artifact_uri,
        "status": run.info.status,
    }


def list_registered_models() -> pd.DataFrame:
    """
    Получение списка всех зарегистрированных моделей.

    Returns:
        DataFrame с информацией о моделях

    Example:
        >>> models = list_registered_models()
        >>> print(models[['name', 'latest_version', 'description']])
    """
    client = MlflowClient()

    models_data = []
    for rm in client.search_registered_models():
        latest_versions = rm.latest_versions
        latest_version = latest_versions[0] if latest_versions else None

        models_data.append(
            {
                "name": rm.name,
                "description": rm.description or "",
                "creation_time": rm.creation_timestamp,
                "latest_version": latest_version.version if latest_version else None,
                "latest_stage": latest_version.current_stage
                if latest_version
                else None,
                "latest_run_id": latest_version.run_id if latest_version else None,
            }
        )

    return pd.DataFrame(models_data)


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> None:
    """
    Изменение стадии модели в Model Registry.

    Args:
        model_name: Имя модели
        version: Версия модели
        stage: Новая стадия ('Staging', 'Production', 'Archived')
        archive_existing: Архивировать существующую модель на этой стадии

    Example:
        >>> transition_model_stage("boston-housing-rf", "1", "Production")
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(f"Модель {model_name} v{version} переведена в стадию {stage}")
