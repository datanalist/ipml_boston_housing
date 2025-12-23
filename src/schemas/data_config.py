"""Pydantic схема для конфигурации данных."""

from pathlib import Path

from pydantic import Field, field_validator

from src.schemas.base import BaseConfig


class DataConfig(BaseConfig):
    """Конфигурация датасета Boston Housing."""

    # Пути к данным
    raw_path: str = Field(
        default="data/raw/housing.csv",
        description="Путь к сырым данным",
    )
    processed_path: str = Field(
        default="data/processed",
        description="Путь к обработанным данным",
    )

    # Параметры датасета
    target_column: str = Field(
        default="MEDV",
        description="Название целевой колонки",
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Список признаков (None = все кроме target)",
    )

    # Параметры разделения
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Доля тестовой выборки (0.05-0.5)",
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description="Seed для воспроизводимости разделения",
    )
    shuffle: bool = Field(
        default=True,
        description="Перемешивать данные перед разделением",
    )

    # Формат данных
    separator: str = Field(
        default=r"\s+",
        description="Разделитель в CSV файле",
    )
    header: int | None = Field(
        default=None,
        description="Номер строки заголовка (None = без заголовка)",
    )

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Валидация размера тестовой выборки."""
        if not 0.05 <= v <= 0.5:
            raise ValueError(f"test_size должен быть между 0.05 и 0.5, получено: {v}")
        return v

    @field_validator("raw_path")
    @classmethod
    def validate_path_format(cls, v: str) -> str:
        """Проверка формата пути."""
        if not v.endswith((".csv", ".parquet", ".json")):
            raise ValueError(
                f"Поддерживаются только .csv, .parquet, .json файлы, получено: {v}"
            )
        return v

    def get_absolute_path(self, base_path: Path) -> Path:
        """Возвращает абсолютный путь к данным."""
        path = Path(self.raw_path)
        if path.is_absolute():
            return path
        return base_path / path
