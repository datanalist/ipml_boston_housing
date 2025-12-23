"""Pydantic схемы для конфигурации обучения и экспериментов."""

from typing import Any

from pydantic import Field, field_validator
from omegaconf import DictConfig, OmegaConf

from src.schemas.base import BaseConfig
from src.schemas.data_config import DataConfig
from src.schemas.model_config import ModelConfig, get_model_config_class


class TrainingConfig(BaseConfig):
    """Конфигурация процесса обучения."""

    # Параметры обучения
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Доля тестовой выборки",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Seed для воспроизводимости",
    )
    shuffle: bool = Field(
        default=True,
        description="Перемешивать данные перед разделением",
    )

    # Валидация
    cross_validation: bool = Field(
        default=False,
        description="Использовать кросс-валидацию",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Количество фолдов для CV",
    )

    # MLflow
    experiment_name: str = Field(
        default="boston-housing",
        description="Имя эксперимента в MLflow",
    )
    run_name: str | None = Field(
        default=None,
        description="Имя запуска (auto-generated если None)",
    )
    log_model: bool = Field(
        default=True,
        description="Логировать модель в MLflow",
    )
    log_plots: bool = Field(
        default=True,
        description="Логировать графики в MLflow",
    )

    # DVCLive
    use_dvclive: bool = Field(
        default=True,
        description="Использовать DVCLive для логирования",
    )
    save_dvc_exp: bool = Field(
        default=True,
        description="Сохранять DVC эксперимент",
    )

    @field_validator("cv_folds")
    @classmethod
    def validate_cv_folds(cls, v: int, info) -> int:
        """Валидация cv_folds только если cross_validation=True."""
        return v


class ExperimentConfig(BaseConfig):
    """
    Полная конфигурация эксперимента.

    Композиция всех конфигураций: модель + данные + обучение.
    """

    # Вложенные конфигурации
    model: dict[str, Any] = Field(
        default_factory=lambda: {"name": "random_forest", "n_estimators": 100},
        description="Конфигурация модели",
    )
    data: dict[str, Any] = Field(
        default_factory=lambda: {"raw_path": "data/raw/housing.csv"},
        description="Конфигурация данных",
    )
    training: dict[str, Any] = Field(
        default_factory=lambda: {"test_size": 0.2, "random_state": 42},
        description="Конфигурация обучения",
    )

    # Метаданные
    name: str = Field(
        default="default",
        description="Имя эксперимента",
    )
    description: str = Field(
        default="",
        description="Описание эксперимента",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Теги эксперимента",
    )

    def get_validated_model_config(self) -> ModelConfig:
        """Возвращает валидированную конфигурацию модели."""
        model_name = self.model.get("name", "random_forest")
        config_class = get_model_config_class(model_name)
        return config_class(**self.model)

    def get_validated_data_config(self) -> DataConfig:
        """Возвращает валидированную конфигурацию данных."""
        return DataConfig(**self.data)

    def get_validated_training_config(self) -> TrainingConfig:
        """Возвращает валидированную конфигурацию обучения."""
        return TrainingConfig(**self.training)

    def validate_all(self) -> tuple[ModelConfig, DataConfig, TrainingConfig]:
        """Валидирует все конфигурации и возвращает их."""
        return (
            self.get_validated_model_config(),
            self.get_validated_data_config(),
            self.get_validated_training_config(),
        )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "ExperimentConfig":
        """
        Создаёт ExperimentConfig из Hydra DictConfig.

        Args:
            cfg: Hydra конфигурация

        Returns:
            Валидированный ExperimentConfig
        """
        # Конвертируем OmegaConf в обычный dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls(**config_dict)

    def to_mlflow_params(self) -> dict[str, Any]:
        """Возвращает плоский словарь параметров для MLflow."""
        params = {}

        # Параметры модели с префиксом
        for key, value in self.model.items():
            params[f"model.{key}"] = value

        # Параметры данных
        for key, value in self.data.items():
            if not key.startswith("_"):  # Пропускаем приватные поля
                params[f"data.{key}"] = value

        # Параметры обучения
        for key, value in self.training.items():
            params[f"training.{key}"] = value

        # Метаданные
        params["experiment_name"] = self.name
        if self.tags:
            params["tags"] = ",".join(self.tags)

        return params
