# Руководство по управлению конфигурациями

## Оглавление

1. [Введение](#введение)
2. [Архитектура системы конфигураций](#архитектура-системы-конфигураций)
3. [Настройка инструмента управления конфигурациями](#настройка-инструмента-управления-конфигурациями)
4. [Создание конфигураций для разных алгоритмов](#создание-конфигураций-для-разных-алгоритмов)
5. [Валидация конфигураций](#валидация-конфигураций)
6. [Система композиции конфигураций](#система-композиции-конфигураций)
7. [Примеры использования](#примеры-использования)
8. [Лучшие практики](#лучшие-практики)

---

## Введение

В данном проекте реализована продвинутая система управления конфигурациями на основе **Hydra** и **Pydantic**, обеспечивающая:

- **Модульность**: Разделение конфигураций модели, данных и обучения
- **Валидацию**: Автоматическая проверка типов и значений параметров
- **Композицию**: Гибкое комбинирование конфигураций для создания экспериментов
- **Переопределение**: Изменение параметров через командную строку
- **Воспроизводимость**: Полное логирование всех настроек эксперимента

### Используемые технологии

- **Hydra 1.3.2** - фреймворк для управления конфигурациями
- **Pydantic 2.0+** - валидация данных на основе типов Python
- **OmegaConf** - структурированные конфигурационные контейнеры
- **YAML** - декларативный формат описания конфигураций

---

## Архитектура системы конфигураций

### Структура директорий

```
conf/
├── config.yaml              # Главная конфигурация
├── model/                   # Конфигурации моделей
│   ├── linear_regression.yaml
│   ├── ridge.yaml
│   ├── lasso.yaml
│   ├── random_forest.yaml
│   ├── gradient_boosting.yaml
│   └── ...
├── data/                    # Конфигурации данных
│   └── boston.yaml
├── training/                # Конфигурации обучения
│   └── default.yaml
└── experiment/              # Готовые эксперименты
    ├── baseline.yaml
    ├── tuned.yaml
    └── linear_comparison.yaml
```

### Слои конфигураций

1. **Model Layer** (`conf/model/`) - параметры ML-моделей
2. **Data Layer** (`conf/data/`) - параметры загрузки и подготовки данных
3. **Training Layer** (`conf/training/`) - параметры процесса обучения
4. **Experiment Layer** (`conf/experiment/`) - композиции готовых экспериментов

### Схемы валидации

```
src/schemas/
├── base.py              # Базовый класс конфигурации
├── model_config.py      # Схемы для всех моделей
├── data_config.py       # Схема для данных
└── training_config.py   # Схемы для обучения и экспериментов
```

---

## Настройка инструмента управления конфигурациями

### Шаг 1: Установка зависимостей

Добавьте в `pyproject.toml`:

```toml
dependencies = [
    "hydra-core>=1.3.2",      # Основной фреймворк
    "pydantic-settings>=2.0",  # Валидация конфигураций
    "omegaconf>=2.3",          # Работа с конфигурациями
]
```

Установка:

```bash
uv sync
```

### Шаг 2: Создание главной конфигурации

Файл `conf/config.yaml`:

```yaml
# Главная конфигурация Hydra для Boston Housing ML проекта
#
# Использование:
#   uv run python src/modeling/train_hydra.py                    # defaults
#   uv run python src/modeling/train_hydra.py model=ridge        # смена модели
#   uv run python src/modeling/train_hydra.py model.alpha=0.5    # override параметра

defaults:
  - model: random_forest      # Модель по умолчанию
  - data: boston              # Датасет
  - training: default         # Параметры обучения
  - _self_                    # Переопределения из этого файла применяются последними

# Метаданные эксперимента
name: boston_housing_experiment
description: "Прогнозирование цен на недвижимость в Бостоне"
tags:
  - regression
  - boston-housing
  - scikit-learn

# Hydra настройки
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}  # Папка для одиночных запусков
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S} # Папка для множественных запусков
    subdir: ${hydra.job.num}
  job:
    chdir: false  # Не менять рабочую директорию
```

**Ключевые элементы:**

- `defaults` - список конфигураций для автоматической загрузки
- `_self_` - определяет порядок применения переопределений
- `hydra.run.dir` - директория для сохранения результатов и логов
- `hydra.job.chdir: false` - сохраняет рабочую директорию проекта

### Шаг 3: Интеграция Hydra в скрипт обучения

Файл `src/modeling/train_hydra.py`:

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Основная функция обучения с Hydra.

    Args:
        cfg: Hydra конфигурация

    Returns:
        R² score для оптимизации гиперпараметров
    """
    # Логируем полную конфигурацию
    logger.info("=" * 60)
    logger.info("HYDRA CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Валидация через Pydantic (см. следующий раздел)
    exp_config = validate_config(cfg)

    # Извлечение параметров
    model_config = exp_config.get_validated_model_config()
    data_config = exp_config.get_validated_data_config()
    training_config = exp_config.get_validated_training_config()

    # ... код обучения ...

    return r2_score

if __name__ == "__main__":
    main()
```

**Особенности интеграции:**

- `@hydra.main()` - декоратор, автоматически загружающий конфигурации
- `config_path` - относительный путь к папке с конфигурациями
- `config_name` - имя главного конфигурационного файла (без .yaml)
- `OmegaConf.to_yaml()` - конвертация конфигурации в читаемый YAML

---

## Создание конфигураций для разных алгоритмов

### Структура конфигурации модели

Каждая конфигурация модели должна содержать:

1. **Комментарий с описанием** модели
2. **Директиву `@package model`** для Hydra
3. **Обязательное поле `name`** - идентификатор модели
4. **Параметры модели** с комментариями и допустимыми диапазонами
5. **`random_state`** для воспроизводимости (если применимо)

### Пример 1: Random Forest

Файл `conf/model/random_forest.yaml`:

```yaml
# Random Forest Regressor
# @package model

name: random_forest

# Параметры ансамбля
n_estimators: 100        # Количество деревьев (10-1000)
max_depth: 10            # Максимальная глубина (1-100, null=без ограничений)

# Параметры разбиения
min_samples_split: 2     # Минимум образцов для разбиения (2-100)
min_samples_leaf: 1      # Минимум образцов в листе (1-100)
max_features: sqrt       # Признаки для разбиения: sqrt, log2, float (0-1)

# Производительность
n_jobs: -1               # Параллельные потоки (-1 = все ядра)
bootstrap: true          # Использовать bootstrap выборки

# Воспроизводимость
random_state: 42
```

**Комментарии к параметрам:**
- Указывают допустимые диапазоны значений
- Поясняют специальные значения (null, -1, "sqrt")
- Группируются по функциональному назначению

### Пример 2: Gradient Boosting

Файл `conf/model/gradient_boosting.yaml`:

```yaml
# Gradient Boosting Regressor
# @package model

name: gradient_boosting

# Параметры бустинга
n_estimators: 100        # Количество этапов (10-1000)
learning_rate: 0.1       # Скорость обучения (0-1)
max_depth: 3             # Глубина деревьев (1-50)

# Параметры разбиения
min_samples_split: 2     # Минимум образцов для разбиения
min_samples_leaf: 1      # Минимум образцов в листе

# Стохастичность
subsample: 1.0           # Доля образцов для каждого дерева (0-1)

# Функция потерь
loss: squared_error      # squared_error, absolute_error, huber, quantile

# Воспроизводимость
random_state: 42
```

### Пример 3: Ridge Regression

Файл `conf/model/ridge.yaml`:

```yaml
# Ridge Regression (L2 регуляризация)
# @package model

name: ridge

# Регуляризация
alpha: 1.0               # Коэффициент L2 регуляризации (0-1000)

# Параметры модели
fit_intercept: true      # Вычислять свободный член
solver: auto             # auto, svd, cholesky, lsqr, sparse_cg, sag, saga

# Воспроизводимость
random_state: 42
```

### Пример 4: Линейная регрессия

Файл `conf/model/linear_regression.yaml`:

```yaml
# Linear Regression (Ordinary Least Squares)
# @package model

name: linear_regression

# Параметры модели
fit_intercept: true      # Вычислять свободный член

# Примечание: LinearRegression не имеет random_state
random_state: null
```

### Полный список моделей

В проекте реализованы конфигурации для 14 моделей:

#### Линейные модели
- `linear_regression` - Обычная линейная регрессия (МНК)
- `ridge` - Ridge регрессия (L2-регуляризация)
- `lasso` - Lasso регрессия (L1-регуляризация)
- `elastic_net` - ElasticNet (комбинация L1 + L2)
- `huber` - Huber регрессор (робастная регрессия)
- `sgd` - SGD регрессор (стохастический градиентный спуск)

#### Древовидные модели
- `decision_tree` - Дерево решений
- `random_forest` - Случайный лес
- `extra_trees` - Extremely Randomized Trees
- `gradient_boosting` - Градиентный бустинг
- `adaboost` - AdaBoost
- `bagging` - Bagging регрессор

#### Другие модели
- `svr` - Support Vector Regression
- `knn` - K-ближайших соседей

### Конфигурация данных

Файл `conf/data/boston.yaml`:

```yaml
# Конфигурация датасета Boston Housing
# @package data

# Пути к данным
raw_path: data/raw/housing.csv
processed_path: data/processed

# Параметры датасета
target_column: MEDV
feature_columns: null  # null = все колонки кроме target

# Параметры разделения
test_size: 0.2
random_state: 42
shuffle: true

# Формат данных
separator: "\\s+"
header: null  # Без заголовка
```

### Конфигурация обучения

Файл `conf/training/default.yaml`:

```yaml
# Конфигурация обучения по умолчанию
# @package training

# Параметры разделения данных
test_size: 0.2
random_state: 42
shuffle: true

# Кросс-валидация
cross_validation: false
cv_folds: 5

# MLflow интеграция
experiment_name: boston-housing
run_name: null  # Автоматическая генерация
log_model: true
log_plots: true

# DVCLive интеграция
use_dvclive: true
save_dvc_exp: true
```

---

## Валидация конфигураций

### Архитектура валидации

Система валидации построена на **Pydantic** и состоит из трёх уровней:

1. **Базовый класс** (`BaseConfig`) - общие настройки валидации
2. **Специализированные классы** для каждой модели/компонента
3. **Композитный класс** (`ExperimentConfig`) для полной валидации

### Уровень 1: Базовый класс конфигурации

Файл `src/schemas/base.py`:

```python
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
```

**Ключевые настройки:**
- `extra="allow"` - не отклоняет дополнительные поля от Hydra
- `validate_assignment=True` - проверяет типы при изменении полей
- `strict=False` - разрешает неявные преобразования типов

### Уровень 2: Валидация моделей

Файл `src/schemas/model_config.py` (примеры):

```python
"""Pydantic схемы для конфигураций моделей ML."""

from typing import Any, Literal
from pydantic import Field, field_validator, model_validator
from src.schemas.base import BaseConfig


class ModelConfig(BaseConfig):
    """Базовая конфигурация модели."""

    name: str = Field(
        ...,
        description="Имя модели из реестра",
    )
    random_state: int | None = Field(
        default=42,
        ge=0,
        description="Seed для воспроизводимости",
    )

    def get_params(self) -> dict[str, Any]:
        """Возвращает параметры для создания модели (без name)."""
        params = self.model_dump(exclude={"name"}, exclude_none=True)
        return params
```

#### Пример: Валидация Ridge регрессии

```python
class RidgeConfig(ModelConfig):
    """Конфигурация Ridge регрессии (L2-регуляризация)."""

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(
        default=1.0,
        gt=0,
        le=1000,
        description="Коэффициент регуляризации (0 < alpha <= 1000)",
    )
    fit_intercept: bool = Field(
        default=True,
        description="Вычислять свободный член",
    )
    solver: Literal["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"] = Field(
        default="auto",
        description="Алгоритм решения",
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"alpha должен быть положительным, получено: {v}")
        return v
```

**Типы валидаторов:**

1. **Field constraints** - встроенные ограничения:
   - `gt`, `ge`, `lt`, `le` - числовые сравнения
   - `min_length`, `max_length` - длина строк/списков
   - `regex` - регулярные выражения

2. **Literal** - ограничение значений enum'ом:
   ```python
   solver: Literal["auto", "svd", "cholesky"]
   ```

3. **field_validator** - кастомная валидация:
   ```python
   @field_validator("learning_rate")
   @classmethod
   def validate_learning_rate(cls, v: float) -> float:
       if not 0 < v <= 1:
           raise ValueError(f"learning_rate должен быть (0, 1], получено: {v}")
       return v
   ```

#### Пример: Валидация Random Forest

```python
class RandomForestConfig(ModelConfig):
    """Конфигурация Random Forest."""

    name: Literal["random_forest"] = "random_forest"
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Количество деревьев в лесу",
    )
    max_depth: int | None = Field(
        default=10,
        ge=1,
        le=100,
        description="Максимальная глубина деревьев",
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Минимум образцов для разбиения узла",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Минимум образцов в листе",
    )
    max_features: Literal["sqrt", "log2"] | float | None = Field(
        default="sqrt",
        description="Число признаков для поиска лучшего разбиения",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Число параллельных потоков (-1 = все ядра)",
    )
    bootstrap: bool = Field(
        default=True,
        description="Использовать bootstrap выборки",
    )

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        if v < 10:
            raise ValueError(f"n_estimators должен быть >= 10, получено: {v}")
        return v
```

### Реестр конфигураций моделей

```python
MODEL_CONFIG_REGISTRY: dict[str, type[ModelConfig]] = {
    "linear_regression": LinearRegressionConfig,
    "ridge": RidgeConfig,
    "lasso": LassoConfig,
    "elastic_net": ElasticNetConfig,
    "huber": HuberConfig,
    "sgd": SGDConfig,
    "decision_tree": DecisionTreeConfig,
    "random_forest": RandomForestConfig,
    "extra_trees": ExtraTreesConfig,
    "gradient_boosting": GradientBoostingConfig,
    "adaboost": AdaBoostConfig,
    "bagging": BaggingConfig,
    "svr": SVRConfig,
    "knn": KNNConfig,
}


def get_model_config_class(model_name: str) -> type[ModelConfig]:
    """Возвращает класс конфигурации для указанной модели."""
    if model_name not in MODEL_CONFIG_REGISTRY:
        available = ", ".join(MODEL_CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Модель '{model_name}' не найдена. Доступные: {available}"
        )
    return MODEL_CONFIG_REGISTRY[model_name]
```

### Валидация данных

Файл `src/schemas/data_config.py`:

```python
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

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Валидация размера тестовой выборки."""
        if not 0.05 <= v <= 0.5:
            raise ValueError(
                f"test_size должен быть между 0.05 и 0.5, получено: {v}"
            )
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
```

### Валидация обучения

Файл `src/schemas/training_config.py`:

```python
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
```

### Уровень 3: Композитная валидация

Файл `src/schemas/training_config.py`:

```python
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
```

### Интеграция валидации в скрипт обучения

```python
def validate_config(cfg: DictConfig) -> ExperimentConfig:
    """Валидация конфигурации через Pydantic."""
    try:
        # Конвертируем OmegaConf в dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Создаём ExperimentConfig для валидации
        exp_config = ExperimentConfig(
            model=config_dict.get("model", {}),
            data=config_dict.get("data", {}),
            training=config_dict.get("training", {}),
            name=config_dict.get("name", "default"),
            description=config_dict.get("description", ""),
            tags=config_dict.get("tags", []),
        )

        # Валидируем вложенные конфигурации
        model_config, data_config, training_config = exp_config.validate_all()

        logger.success(f"✓ Конфигурация валидна: model={model_config.name}")
        return exp_config

    except Exception as e:
        logger.error(f"✗ Ошибка валидации конфигурации: {e}")
        raise


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    # Валидация конфигурации
    exp_config = validate_config(cfg)

    # Получение валидированных конфигураций
    model_config = exp_config.get_validated_model_config()
    data_config = exp_config.get_validated_data_config()
    training_config = exp_config.get_validated_training_config()

    # Извлечение параметров
    model_params = model_config.get_params()  # Без поля "name"

    # ... код обучения ...
```

### Примеры работы валидации

#### Успешная валидация

```bash
$ uv run python src/modeling/train_hydra.py model=ridge model.alpha=1.0

✓ Конфигурация валидна: model=ridge
```

#### Ошибка типа данных

```bash
$ uv run python src/modeling/train_hydra.py model=ridge model.alpha=abc

✗ Ошибка валидации конфигурации:
  Input should be a valid number, unable to parse string as a number
```

#### Ошибка диапазона значений

```bash
$ uv run python src/modeling/train_hydra.py model=ridge model.alpha=-1.0

✗ Ошибка валидации конфигурации:
  alpha должен быть положительным, получено: -1.0
```

#### Ошибка недопустимого значения enum

```bash
$ uv run python src/modeling/train_hydra.py model=ridge model.solver=invalid

✗ Ошибка валидации конфигурации:
  Input should be 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag' or 'saga'
```

---

## Система композиции конфигураций

### Принципы композиции

Hydra позволяет комбинировать конфигурации несколькими способами:

1. **Defaults** - автоматическая загрузка конфигураций
2. **Override** - замена конфигурации из defaults
3. **CLI overrides** - переопределение через командную строку
4. **Config groups** - группировка связанных конфигураций
5. **Experiments** - готовые композиции параметров

### Способ 1: Defaults списки

Главный конфигурационный файл определяет значения по умолчанию:

```yaml
# conf/config.yaml
defaults:
  - model: random_forest      # Загружает conf/model/random_forest.yaml
  - data: boston              # Загружает conf/data/boston.yaml
  - training: default         # Загружает conf/training/default.yaml
  - _self_                    # Локальные переопределения применяются последними
```

**Порядок слияния:**
1. Загружаются конфигурации из `defaults` сверху вниз
2. Применяются переопределения из самого файла (благодаря `_self_`)
3. Применяются CLI-аргументы (наивысший приоритет)

### Способ 2: Эксперименты как композиции

Создание готовых экспериментов с предустановленными параметрами.

#### Пример 1: Базовый эксперимент

Файл `conf/experiment/baseline.yaml`:

```yaml
# @package _global_
#
# Базовый эксперимент с настройками по умолчанию
# Использование: uv run python src/modeling/train_hydra.py +experiment=baseline

defaults:
  - override /model: random_forest
  - override /training: default

name: baseline_experiment
description: "Базовый эксперимент с Random Forest и параметрами по умолчанию"
tags:
  - baseline
  - random_forest

# Переопределения параметров модели
model:
  n_estimators: 100
  max_depth: 10

# Переопределения параметров обучения
training:
  experiment_name: boston-housing-baseline
```

**Использование:**

```bash
uv run python src/modeling/train_hydra.py +experiment=baseline
```

**Как это работает:**
- `@package _global_` - применяет конфигурацию к корню
- `override /model: random_forest` - заменяет модель из defaults
- Локальные `model:` и `training:` переопределяют конкретные параметры

#### Пример 2: Оптимизированный эксперимент

Файл `conf/experiment/tuned.yaml`:

```yaml
# @package _global_
#
# Настроенный эксперимент с оптимизированными параметрами
# Использование: uv run python src/modeling/train_hydra.py +experiment=tuned

defaults:
  - override /model: random_forest
  - override /training: default

name: tuned_experiment
description: "Эксперимент с оптимизированными гиперпараметрами Random Forest"
tags:
  - tuned
  - optimized
  - random_forest

# Оптимизированные параметры модели (на основе grid search)
model:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: sqrt

# Параметры обучения
training:
  experiment_name: boston-housing-tuned
  cross_validation: false
  log_model: true
  log_plots: true
```

#### Пример 3: Сравнение линейных моделей

Файл `conf/experiment/linear_comparison.yaml`:

```yaml
# @package _global_
#
# Эксперимент для сравнения линейных моделей
# Использование: uv run python src/modeling/train_hydra.py +experiment=linear_comparison

defaults:
  - override /model: ridge
  - override /training: default

name: linear_comparison
description: "Сравнение линейных моделей регрессии"
tags:
  - linear
  - comparison
  - regularization

# Параметры Ridge по умолчанию
model:
  alpha: 1.0

# Параметры обучения
training:
  experiment_name: boston-housing-linear
  cross_validation: true
  cv_folds: 5
```

### Способ 3: CLI переопределения

#### Базовое переопределение

```bash
# Изменить одно поле
uv run python src/modeling/train_hydra.py model.n_estimators=200

# Изменить несколько полей
uv run python src/modeling/train_hydra.py \
    model.n_estimators=200 \
    model.max_depth=15 \
    training.cross_validation=true
```

#### Смена группы конфигурации

```bash
# Сменить модель
uv run python src/modeling/train_hydra.py model=gradient_boosting

# Сменить модель и изменить параметры
uv run python src/modeling/train_hydra.py \
    model=gradient_boosting \
    model.learning_rate=0.05 \
    model.n_estimators=300
```

#### Комбинация эксперимента и переопределений

```bash
# Загрузить эксперимент и изменить параметры
uv run python src/modeling/train_hydra.py \
    +experiment=tuned \
    model.n_estimators=300

# Загрузить эксперимент и сменить модель
uv run python src/modeling/train_hydra.py \
    +experiment=linear_comparison \
    model=lasso \
    model.alpha=0.1
```

### Способ 4: Multirun (множественные запуски)

Hydra поддерживает запуск экспериментов с несколькими комбинациями параметров:

#### Grid search по параметрам

```bash
# Перебор параметров одной модели
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=random_forest \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,15
```

Создаст 9 запусков (3 × 3 комбинации).

#### Сравнение нескольких моделей

```bash
# Запуск разных моделей с одинаковыми данными
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=linear_regression,ridge,lasso,random_forest
```

Создаст 4 запуска (по одному для каждой модели).

#### Комплексный multirun

```bash
# Сравнение моделей с разными параметрами
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=ridge,lasso \
    model.alpha=0.1,1.0,10.0 \
    training.cross_validation=true,false
```

Создаст 12 запусков (2 модели × 3 alpha × 2 CV).

### Способ 5: Структурированные конфигурации

Для сложных экспериментов можно создавать иерархические структуры.

#### Пример: Разные датасеты для одной модели

Структура:

```
conf/
├── config.yaml
├── model/
│   └── random_forest.yaml
├── data/
│   ├── boston.yaml
│   ├── boston_normalized.yaml
│   └── boston_pca.yaml
└── experiment/
    ├── rf_baseline.yaml
    ├── rf_normalized.yaml
    └── rf_pca.yaml
```

Файл `conf/experiment/rf_normalized.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: random_forest
  - override /data: boston_normalized
  - override /training: default

name: rf_normalized_experiment
description: "Random Forest на нормализованных данных"
tags:
  - normalized
  - preprocessing
```

### Способ 6: Вложенные defaults

Конфигурации могут иметь собственные defaults списки.

Пример `conf/model/ensemble_comparison.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: random_forest  # Будет заменён в CLI

name: ensemble_comparison
description: "Сравнение ансамблевых методов"

# Параметры, общие для всех ансамблей
model:
  n_estimators: 100
  random_state: 42
```

Использование:

```bash
# Заменяем модель, но сохраняем общие параметры
uv run python src/modeling/train_hydra.py \
    +experiment=ensemble_comparison \
    model=gradient_boosting
```

---

## Примеры использования

### Пример 1: Быстрый старт

```bash
# Запуск с настройками по умолчанию (Random Forest)
uv run python src/modeling/train_hydra.py
```

**Что происходит:**
1. Загружается `conf/config.yaml`
2. Из defaults подключаются:
   - `conf/model/random_forest.yaml`
   - `conf/data/boston.yaml`
   - `conf/training/default.yaml`
3. Конфигурация валидируется через Pydantic
4. Запускается обучение модели

### Пример 2: Смена модели

```bash
# Запуск с Gradient Boosting вместо Random Forest
uv run python src/modeling/train_hydra.py model=gradient_boosting
```

### Пример 3: Настройка параметров модели

```bash
# Random Forest с увеличенными параметрами
uv run python src/modeling/train_hydra.py \
    model=random_forest \
    model.n_estimators=300 \
    model.max_depth=20 \
    model.min_samples_split=5
```

### Пример 4: Линейная регрессия с регуляризацией

```bash
# Ridge регрессия с кастомной alpha
uv run python src/modeling/train_hydra.py \
    model=ridge \
    model.alpha=10.0 \
    model.solver=cholesky
```

### Пример 5: Включение кросс-валидации

```bash
# Обучение с 10-fold cross-validation
uv run python src/modeling/train_hydra.py \
    model=random_forest \
    training.cross_validation=true \
    training.cv_folds=10
```

### Пример 6: Использование готового эксперимента

```bash
# Запуск оптимизированного эксперимента
uv run python src/modeling/train_hydra.py +experiment=tuned

# Запуск с модификацией
uv run python src/modeling/train_hydra.py \
    +experiment=tuned \
    model.n_estimators=500
```

### Пример 7: Сравнение линейных моделей

```bash
# Multirun для сравнения регуляризованных моделей
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=linear_regression,ridge,lasso,elastic_net
```

**Результат:**
- 4 отдельных запуска
- Каждый с логами в отдельной папке
- Результаты доступны в MLflow для сравнения

### Пример 8: Grid search по гиперпараметрам

```bash
# Перебор параметров Random Forest
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=random_forest \
    model.n_estimators=50,100,200,500 \
    model.max_depth=5,10,15,20 \
    model.min_samples_split=2,5,10
```

**Результат:**
- 48 запусков (4 × 4 × 3)
- Автоматическое логирование в MLflow
- Возможность выбора лучшей комбинации

### Пример 9: Сравнение с/без кросс-валидации

```bash
# Сравнение влияния CV на метрики
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=random_forest \
    training.cross_validation=true,false
```

### Пример 10: Комплексное сравнение

```bash
# Сравнение моделей с разными параметрами регуляризации
uv run python src/modeling/train_hydra.py \
    --multirun \
    +experiment=linear_comparison \
    model=ridge,lasso,elastic_net \
    model.alpha=0.01,0.1,1.0,10.0,100.0
```

**Результат:**
- 15 запусков (3 модели × 5 значений alpha)
- Каждый с кросс-валидацией (из эксперимента)
- Полное сравнение в MLflow UI

---

## Лучшие практики

### 1. Именование конфигураций

✅ **Хорошо:**
```yaml
# conf/model/gradient_boosting.yaml
name: gradient_boosting
```

❌ **Плохо:**
```yaml
# conf/model/gb.yaml
name: GB
```

**Правила:**
- Используйте snake_case для имён файлов
- Имя файла = значение поля `name`
- Имена должны быть понятными без сокращений

### 2. Комментарии в конфигурациях

✅ **Хорошо:**
```yaml
n_estimators: 100        # Количество деревьев (10-1000)
max_depth: 10            # Максимальная глубина (1-100, null=без ограничений)
```

❌ **Плохо:**
```yaml
n_estimators: 100
max_depth: 10
```

**Правила:**
- Указывайте допустимые диапазоны значений
- Поясняйте специальные значения (null, -1, "auto")
- Группируйте параметры по назначению

### 3. Значения по умолчанию

✅ **Хорошо:**
```python
class RandomForestConfig(ModelConfig):
    n_estimators: int = Field(
        default=100,       # Разумное значение по умолчанию
        ge=10,
        le=1000,
    )
```

❌ **Плохо:**
```python
class RandomForestConfig(ModelConfig):
    n_estimators: int = Field(...)  # Нет значения по умолчанию
```

**Правила:**
- Всегда предоставляйте разумные defaults
- Defaults должны давать приемлемое качество модели
- Избегайте обязательных полей (кроме `name`)

### 4. Валидация параметров

✅ **Хорошо:**
```python
learning_rate: float = Field(
    default=0.1,
    gt=0,          # Строго больше 0
    le=1.0,        # Меньше или равно 1
    description="Скорость обучения (0 < lr <= 1)",
)

@field_validator("learning_rate")
@classmethod
def validate_learning_rate(cls, v: float) -> float:
    if not 0 < v <= 1:
        raise ValueError(f"learning_rate должен быть (0, 1], получено: {v}")
    return v
```

❌ **Плохо:**
```python
learning_rate: float = 0.1  # Нет валидации
```

**Правила:**
- Используйте Field constraints (gt, ge, lt, le)
- Добавляйте field_validator для сложных проверок
- Предоставляйте информативные сообщения об ошибках

### 5. Структура экспериментов

✅ **Хорошо:**
```yaml
# @package _global_
defaults:
  - override /model: random_forest
  - override /training: default

name: tuned_experiment
description: "Детальное описание эксперимента и его целей"
tags:
  - optimized
  - production-ready

model:
  # Только изменённые параметры
  n_estimators: 200
  max_depth: 15
```

❌ **Плохо:**
```yaml
# Нет @package _global_
# Нет defaults
# Нет description

model:
  name: random_forest
  n_estimators: 200
  # ... полное дублирование всех параметров
```

**Правила:**
- Используйте `@package _global_` для экспериментов
- Указывайте `defaults` для явности
- Переопределяйте только изменённые параметры
- Добавляйте описание и теги

### 6. Директива @package

```yaml
# Для конфигураций модели/данных/обучения
# @package model

# Для экспериментов
# @package _global_
```

**Правила:**
- Конфигурации в группах (model/, data/, training/) используют `@package <group_name>`
- Эксперименты используют `@package _global_`
- Не забывайте эту директиву — без неё композиция не работает

### 7. Организация multirun

✅ **Хорошо:**
```bash
# Осмысленный grid search
uv run python src/modeling/train_hydra.py \
    --multirun \
    model=random_forest \
    model.n_estimators=50,100,200 \
    model.max_depth=10,20
```

❌ **Плохо:**
```bash
# Слишком много комбинаций (100+ запусков)
uv run python src/modeling/train_hydra.py \
    --multirun \
    model.n_estimators=10,50,100,200,500,1000 \
    model.max_depth=5,10,15,20,25 \
    model.min_samples_split=2,5,10,20 \
    model.min_samples_leaf=1,2,5,10
```

**Правила:**
- Ограничивайте количество комбинаций (< 50 запусков)
- Используйте логарифмическую шкалу для параметров (0.1, 1, 10, 100)
- Сначала делайте грубый поиск, затем — уточняющий

### 8. Логирование конфигураций

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    # Всегда логируйте полную конфигурацию в начале
    logger.info("=" * 60)
    logger.info("HYDRA CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # ... валидация и обучение ...
```

**Правила:**
- Логируйте полную конфигурацию в начале запуска
- Используйте `OmegaConf.to_yaml()` для читаемого вывода
- Сохраняйте конфигурацию в MLflow для воспроизводимости

### 9. Версионирование конфигураций

```yaml
# conf/config.yaml
defaults:
  - model: random_forest
  - data: boston
  - training: default
  - _self_

# Версия схемы конфигураций
config_version: "1.0"

# Hydra настройки
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**Правила:**
- Добавляйте поле `config_version` для отслеживания изменений
- При breaking changes повышайте major версию
- Сохраняйте старые конфигурации для совместимости

### 10. Тестирование конфигураций

Создайте тесты для валидации:

```python
# tests/test_configs.py
import pytest
from omegaconf import OmegaConf
from src.schemas.training_config import ExperimentConfig


def test_default_config_is_valid():
    """Проверка, что config.yaml валиден."""
    cfg = OmegaConf.load("conf/config.yaml")

    # Должна пройти валидация
    exp_config = ExperimentConfig.from_hydra(cfg)
    model_config, data_config, training_config = exp_config.validate_all()

    assert model_config.name == "random_forest"
    assert data_config.test_size == 0.2


def test_all_model_configs_are_valid():
    """Проверка, что все конфигурации моделей валидны."""
    import glob

    for config_file in glob.glob("conf/model/*.yaml"):
        cfg = OmegaConf.load(config_file)
        model_name = cfg.get("name")

        # Должна пройти валидация
        from src.schemas.model_config import get_model_config_class
        config_class = get_model_config_class(model_name)
        model_config = config_class(**OmegaConf.to_container(cfg))

        assert model_config.name == model_name


def test_invalid_alpha_raises_error():
    """Проверка, что невалидные параметры отклоняются."""
    from src.schemas.model_config import RidgeConfig

    with pytest.raises(ValueError, match="alpha должен быть положительным"):
        RidgeConfig(name="ridge", alpha=-1.0)
```

**Правила:**
- Тестируйте все дефолтные конфигурации
- Проверяйте валидацию границ параметров
- Автоматизируйте проверку всех файлов в conf/

---

## Заключение

Реализованная система управления конфигурациями обеспечивает:

✅ **Гибкость** - легко менять модели и параметры через CLI  
✅ **Безопасность** - автоматическая валидация всех параметров  
✅ **Масштабируемость** - простое добавление новых моделей и экспериментов  
✅ **Воспроизводимость** - полное логирование конфигураций  
✅ **Удобство** - готовые эксперименты и multirun для поиска гиперпараметров  

### Следующие шаги

1. **Изучите существующие конфигурации** в `conf/`
2. **Запустите примеры** из раздела "Примеры использования"
3. **Создайте свой эксперимент** в `conf/experiment/`
4. **Добавьте новую модель** (config + schema + tests)
5. **Настройте автоматизацию** через DVC pipelines или Airflow

### Полезные ссылки

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)

---

**Автор:** Команда разработки IPML Boston Housing  
**Дата:** Декабрь 2025  
**Версия:** 1.0
