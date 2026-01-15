"""Базовые классы и утилиты для Pydantic схем."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Базовый класс конфигурации с общими настройками."""

    model_config = ConfigDict(
        # Разрешаем дополнительные поля (для совместимости с Hydra)
        extra="allow",
        # Валидация при присваивании
        validate_assignment=True,
        # Использовать имена полей (не алиасы) при сериализации
        populate_by_name=True,
        # Строгая проверка типов
        strict=False,
    )

    def to_dict(self) -> dict:
        """Преобразует конфигурацию в словарь для передачи в модель."""
        return self.model_dump(exclude_none=True)
