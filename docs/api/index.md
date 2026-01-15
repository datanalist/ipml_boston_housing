# API Reference

Добро пожаловать в справочник API проекта Boston Housing Price Prediction.

---

## 📚 Обзор модулей

Проект организован в следующие основные модули:

| Модуль | Описание | Путь |
|--------|----------|------|
| **ml_models** | Загрузчик моделей (14 алгоритмов регрессии) | `src/ml_models/` |
| **tracking** | MLflow трекинг (декораторы, менеджеры) | `src/tracking/` |
| **config** | Конфигурации и настройки | `conf/` |
| **dataset** | Работа с данными | `src/data/` |
| **schemas** | Pydantic схемы валидации | `src/schemas/` |

---

## 🔍 Быстрый поиск

### Модели

```python
from src.ml_models.model_loader import create_model, save_model, load_model

# Создание модели
model = create_model("random_forest", custom_params={"n_estimators": 200})

# Сохранение модели
save_model(model, "my_rf_model")

# Загрузка модели
model = load_model("random_forest")
```

Подробнее см. модуль `src/ml_models/model_loader.py`

### Трекинг

```python
from src.tracking.decorators import mlflow_run, log_metrics

@mlflow_run(experiment_name="my_experiment")
@log_metrics()
def train_model():
    # Ваш код обучения
    return {"rmse": 3.5, "r2": 0.85}
```

Подробнее см. модуль `src/tracking/`

### Конфигурации

```python
from hydra import compose, initialize
from src.schemas import ExperimentConfig

# Инициализация Hydra
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")

# Валидация через Pydantic
config = ExperimentConfig(**cfg)
```

Подробнее см. [Руководство по конфигурациям](../guides/CONFIGURATION_MANAGEMENT.md)

---

## 📖 Детальная документация

### Модели (ml_models)

Модуль `src/ml_models/model_loader.py`:

- `create_model()` — создание модели по имени
- `save_model()` — сохранение модели
- `load_model()` — загрузка модели
- `get_available_models()` — список доступных моделей
- `get_model_info()` — информация о модели

### Трекинг (tracking)

Модуль `src/tracking/`:

- `@mlflow_run` — декоратор для автоматического MLflow run
- `@log_params` — декоратор для логирования параметров
- `@log_metrics` — декоратор для логирования метрик
- `MLflowExperimentTracker` — класс для управления экспериментами
- `get_best_run()` — получение лучшего запуска
- `load_best_model()` — загрузка лучшей модели

### Конфигурации (config)

Модуль `src/schemas/` (см. [Руководство по конфигурациям](../guides/CONFIGURATION_MANAGEMENT.md)):

- `BaseConfig` — базовый класс конфигурации
- `ModelConfig` — конфигурация модели
- `DataConfig` — конфигурация данных
- `TrainingConfig` — конфигурация обучения
- `ExperimentConfig` — полная конфигурация эксперимента
- Специализированные конфигурации для каждой модели

### Датасет (dataset)

Модуль `src/data/`:

- `load_boston_housing()` — загрузка датасета
- `validate_data()` — валидация данных
- `split_data()` — разделение на train/test
- `get_data_info()` — информация о датасете

---

## 🎯 Примеры использования

### Полный пример обучения

```python
from src.ml_models.model_loader import create_model
from src.tracking.decorators import mlflow_run, log_metrics, log_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

@mlflow_run(experiment_name="boston_housing")
@log_params(params={"model_type": "random_forest"})
@log_metrics()
def train_and_evaluate():
    # Загрузка данных
    data = pd.read_csv("data/raw/housing.csv")
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создание и обучение модели
    model = create_model("random_forest", custom_params={
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42
    })
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "r2": r2}

# Запуск
metrics = train_and_evaluate()
print(f"RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
```

### Работа с Hydra конфигурациями

```python
import hydra
from omegaconf import DictConfig
from src.ml_models.model_loader import create_model

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Создание модели из конфигурации
    model = create_model(
        cfg.model.name,
        custom_params=cfg.model.get("params", {})
    )

    # Обучение
    # ...

if __name__ == "__main__":
    main()
```

---

## 🔗 Дополнительные ресурсы

- [Примеры использования](../examples/index.md) — больше практических примеров
- [Руководства](../guides/index.md) — подробные инструкции
- [GitHub Repository](https://github.com/yourusername/ipml_boston_housing) — исходный код

---

## 📝 Конвенции кода

Проект следует следующим конвенциям:

### Типизация

Все функции имеют type hints:

```python
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest"
) -> tuple[Any, dict[str, float]]:
    """Train model and return it with metrics."""
    ...
```

### Docstrings

Используется Google style:

```python
def create_model(model_name: str, custom_params: dict | None = None) -> Any:
    """Create a scikit-learn model by name.

    Args:
        model_name: Name of the model to create.
        custom_params: Custom parameters to override defaults.

    Returns:
        Instantiated scikit-learn model.

    Raises:
        ValueError: If model_name is not recognized.

    Example:
        >>> model = create_model("random_forest", {"n_estimators": 200})
        >>> model.fit(X_train, y_train)
    """
    ...
```

### Именование

- **Функции**: `snake_case`
- **Классы**: `PascalCase`
- **Константы**: `UPPER_SNAKE_CASE`
- **Приватные**: `_leading_underscore`

---

## 🤝 Вклад в проект

При добавлении нового кода:

1. ✅ Добавьте type hints
2. ✅ Напишите docstrings (Google style)
3. ✅ Добавьте unit tests
4. ✅ Запустите `make lint` и `make format`
5. ✅ Обновите документацию

См. [Contributing Guide](../about.md) для деталей.
