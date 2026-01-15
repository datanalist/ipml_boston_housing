"""MLflow трекер с поддержкой контекстного менеджера."""

from typing import Any
from pathlib import Path

import mlflow
from mlflow.models.signature import infer_signature
from loguru import logger

from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    setup_mlflow_env,
)


class MLflowExperimentTracker:
    """
    Контекстный менеджер для трекинга ML экспериментов.

    Предоставляет удобный интерфейс для работы с MLflow: автоматически
    настраивает окружение, управляет жизненным циклом run и логирует
    параметры, метрики, артефакты и модели.

    Attributes:
        experiment_name: Название эксперимента MLflow
        run: Текущий активный run (или None)

    Example:
        >>> tracker = MLflowExperimentTracker(experiment_name="boston-housing")
        >>> with tracker.start_run(run_name="gradient-boosting-v1"):
        ...     tracker.set_tags({"model_type": "GradientBoosting"})
        ...     tracker.log_params({"n_estimators": 200, "max_depth": 10})
        ...     model = GradientBoostingRegressor(n_estimators=200, max_depth=10)
        ...     model.fit(X_train, y_train)
        ...     tracker.log_metrics({"r2_score": 0.85, "rmse": 3.2})
        ...     tracker.log_model(model, "model", input_example=X_test.head(5))
        ...     print(f"Run ID: {tracker.run_id}")
    """

    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        """
        Инициализация трекера.

        Args:
            experiment_name: Название эксперимента MLflow
            tracking_uri: URI сервера MLflow Tracking
        """
        setup_mlflow_env()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.run = None
        logger.info(f"MLflow трекер: {tracking_uri}, эксперимент: {experiment_name}")

    def start_run(
        self, run_name: str | None = None, tags: dict | None = None
    ) -> "MLflowExperimentTracker":
        """
        Начало нового запуска эксперимента.

        Args:
            run_name: Имя запуска (опционально)
            tags: Начальные теги запуска (опционально)

        Returns:
            self для поддержки chaining
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Запущен run: {self.run.info.run_id}")
        return self

    def __enter__(self) -> "MLflowExperimentTracker":
        """Поддержка контекстного менеджера."""
        if self.run is None:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Завершение эксперимента."""
        mlflow.end_run()
        self.run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Логирование параметров.

        Args:
            params: Словарь параметров для логирования
        """
        mlflow.log_params(params)
        logger.debug(f"Залогированы параметры: {list(params.keys())}")

    def log_param(self, key: str, value: Any) -> None:
        """
        Логирование одного параметра.

        Args:
            key: Имя параметра
            value: Значение параметра
        """
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Логирование метрик.

        Args:
            metrics: Словарь метрик для логирования
            step: Номер шага (опционально, для временных рядов)
        """
        mlflow.log_metrics(metrics, step=step)
        for name, value in metrics.items():
            logger.info(f"Метрика {name}: {value:.4f}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """
        Логирование одной метрики.

        Args:
            key: Имя метрики
            value: Значение метрики
            step: Номер шага (опционально)
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        """
        Логирование артефакта (файла).

        Args:
            local_path: Локальный путь к файлу
            artifact_path: Путь в хранилище артефактов (опционально)
        """
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Артефакт сохранён: {local_path}")

    def log_artifacts(
        self, local_dir: str | Path, artifact_path: str | None = None
    ) -> None:
        """
        Логирование директории артефактов.

        Args:
            local_dir: Локальный путь к директории
            artifact_path: Путь в хранилище артефактов (опционально)
        """
        mlflow.log_artifacts(str(local_dir), artifact_path)
        logger.info(f"Директория артефактов сохранена: {local_dir}")

    def log_model(
        self,
        model,
        artifact_path: str,
        input_example=None,
        registered_model_name: str | None = None,
    ) -> None:
        """
        Логирование sklearn модели.

        Автоматически определяет сигнатуру модели на основе input_example.

        Args:
            model: Обученная модель sklearn
            artifact_path: Путь для сохранения модели в артефактах
            input_example: Пример входных данных для сигнатуры
            registered_model_name: Имя для регистрации модели (опционально)
        """
        signature = None
        if input_example is not None:
            predictions = model.predict(input_example)
            signature = infer_signature(input_example, predictions)

        mlflow.sklearn.log_model(
            model,
            artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Модель сохранена: {artifact_path}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Установка тегов.

        Args:
            tags: Словарь тегов
        """
        mlflow.set_tags(tags)

    def set_tag(self, key: str, value: str) -> None:
        """
        Установка одного тега.

        Args:
            key: Имя тега
            value: Значение тега
        """
        mlflow.set_tag(key, value)

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        """
        Логирование словаря как JSON/YAML артефакта.

        Args:
            dictionary: Словарь для сохранения
            artifact_file: Имя файла (с расширением .json или .yaml)
        """
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Логирование matplotlib/plotly фигуры.

        Args:
            figure: Объект фигуры matplotlib или plotly
            artifact_file: Имя файла для сохранения
        """
        mlflow.log_figure(figure, artifact_file)

    @property
    def run_id(self) -> str | None:
        """ID текущего запуска."""
        return self.run.info.run_id if self.run else None

    @property
    def artifact_uri(self) -> str | None:
        """URI хранилища артефактов текущего запуска."""
        return self.run.info.artifact_uri if self.run else None

    @property
    def experiment_id(self) -> str | None:
        """ID текущего эксперимента."""
        return self.run.info.experiment_id if self.run else None


class NestedRunTracker:
    """
    Контекстный менеджер для вложенных MLflow runs.

    Позволяет создавать иерархическую структуру экспериментов,
    например, для кросс-валидации или grid search.

    Example:
        >>> with MLflowExperimentTracker() as parent:
        ...     parent.log_params({"model": "RandomForest"})
        ...     for fold in range(5):
        ...         with NestedRunTracker(f"fold-{fold}") as child:
        ...             child.log_metrics({"accuracy": 0.85 + fold * 0.01})
    """

    def __init__(self, run_name: str | None = None, tags: dict | None = None):
        """
        Инициализация вложенного трекера.

        Args:
            run_name: Имя вложенного запуска
            tags: Теги для вложенного запуска
        """
        self.run_name = run_name
        self.tags = tags
        self.run = None

    def __enter__(self) -> "NestedRunTracker":
        """Начало вложенного запуска."""
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags, nested=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Завершение вложенного запуска."""
        mlflow.end_run()
        self.run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Логирование параметров."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Логирование метрик."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Логирование одной метрики."""
        mlflow.log_metric(key, value, step=step)

    @property
    def run_id(self) -> str | None:
        """ID текущего вложенного запуска."""
        return self.run.info.run_id if self.run else None
