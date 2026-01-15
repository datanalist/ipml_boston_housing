# Примеры использования

Этот раздел содержит практические примеры использования различных компонентов проекта.

---

## 📚 Обзор

| Пример | Сложность | Описание |
|--------|-----------|----------|
| **Базовое обучение** | ⭐ Начальный | Простое обучение одной модели |
| **Hydra конфигурации** | ⭐⭐ Средний | Управление конфигурациями |
| **Airflow DAG** | ⭐⭐ Средний | Оркестрация ML пайплайна |
| **MLflow трекинг** | ⭐⭐⭐ Продвинутый | Трекинг экспериментов |

---

## 🚀 Быстрый старт

### 1. Простейший пример

```python
from src.ml_models.model_loader import create_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создание и обучение модели
model = create_model("random_forest")
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### 2. С использованием Hydra

```python
import hydra
from omegaconf import DictConfig
from src.ml_models.model_loader import create_model

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Модель создается из конфигурации
    model = create_model(cfg.model.name, custom_params=cfg.model)
    # ... обучение ...

if __name__ == "__main__":
    main()
```

Запуск:

```bash
python script.py model=gradient_boosting
```

### 3. С MLflow трекингом

```python
from src.tracking.decorators import mlflow_run, log_metrics

@mlflow_run(experiment_name="boston_housing")
@log_metrics()
def train():
    # ... обучение ...
    return {"rmse": 3.5, "r2": 0.85}

metrics = train()
```

---

## 📖 Подробные примеры

### Базовое обучение

Узнайте, как:

- Загрузить данные
- Создать модель
- Обучить и оценить
- Сохранить результаты

См. [Руководство по экспериментам](../guides/EXPERIMENTS.md)

### Hydra конфигурации

Узнайте, как:

- Создавать конфигурации
- Переопределять параметры
- Использовать композицию
- Запускать multirun

См. [Управление конфигурациями](../guides/CONFIGURATION_MANAGEMENT.md)

### Airflow DAG

Узнайте, как:

- Создать DAG
- Определить зависимости
- Запустить пайплайн
- Мониторить выполнение

См. [Airflow ML Pipeline](../guides/airflow_ml_pipeline.md)

### MLflow трекинг

Узнайте, как:

- Настроить эксперимент
- Логировать параметры и метрики
- Сохранять артефакты
- Сравнивать модели

См. [MLflow + DVC + MinIO](../guides/MLFLOW+DVC+MINIO.md)

---

## 💡 Полезные паттерны

### Паттерн 1: Декоратор для трекинга

```python
from functools import wraps
import mlflow

def track_experiment(experiment_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                # Логирование параметров
                mlflow.log_params(kwargs)

                # Выполнение функции
                result = func(*args, **kwargs)

                # Логирование метрик
                if isinstance(result, dict):
                    mlflow.log_metrics(result)

                return result
        return wrapper
    return decorator

@track_experiment("my_experiment")
def train_model(n_estimators=100, max_depth=10):
    # ... обучение ...
    return {"rmse": 3.5, "r2": 0.85}
```

### Паттерн 2: Контекстный менеджер для ресурсов

```python
from contextlib import contextmanager
import mlflow

@contextmanager
def mlflow_context(experiment_name: str, run_name: str | None = None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        try:
            yield run
        finally:
            mlflow.end_run()

# Использование
with mlflow_context("boston_housing", "rf_tuning"):
    model = create_model("random_forest")
    model.fit(X_train, y_train)
    mlflow.log_param("n_estimators", 200)
```

### Паттерн 3: Фабрика моделей

```python
class ModelFactory:
    _models = {
        "rf": lambda: create_model("random_forest"),
        "gb": lambda: create_model("gradient_boosting"),
        "ridge": lambda: create_model("ridge"),
    }

    @classmethod
    def create(cls, model_type: str, **kwargs):
        if model_type not in cls._models:
            raise ValueError(f"Unknown model: {model_type}")
        model = cls._models[model_type]()
        if kwargs:
            model.set_params(**kwargs)
        return model

# Использование
model = ModelFactory.create("rf", n_estimators=200, max_depth=15)
```

---

## 🎯 Сценарии использования

### Сценарий 1: Быстрое прототипирование

```bash
# Одна команда для запуска эксперимента
uv run python src/modeling/train_hydra.py \
    model=random_forest \
    model.n_estimators=200
```

### Сценарий 2: Grid Search

```bash
# Перебор параметров
uv run python src/modeling/train_hydra.py --multirun \
    model=random_forest \
    model.n_estimators=100,200,300 \
    model.max_depth=10,15,20
```

### Сценарий 3: Сравнение моделей

```bash
# Обучение нескольких моделей
uv run python src/modeling/train_hydra.py --multirun \
    model=random_forest,gradient_boosting,ridge,lasso

# Результаты можно сравнить в MLflow UI
```

### Сценарий 4: Production пайплайн

1. Запустите Docker инфраструктуру: `make docker-up`
2. Откройте Airflow UI: http://localhost:8080
3. Запустите DAG: `boston_housing_cached`
4. Мониторьте через Airflow UI
5. Проверьте результаты в MLflow: http://localhost:5000

---

## 📝 Шаблоны кода

### Шаблон обучения

```python
"""Template for model training."""
import hydra
from omegaconf import DictConfig
from src.ml_models.model_loader import create_model
from src.tracking.decorators import mlflow_run, log_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Load data
    data = pd.read_csv(cfg.data.path)
    X = data.drop(cfg.data.target, axis=1)
    y = data[cfg.data.target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )

    # Create model
    model = create_model(cfg.model.name, custom_params=cfg.model)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

if __name__ == "__main__":
    main()
```

### Шаблон DAG

```python
"""Template for Airflow DAG."""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def task_function():
    """Your task logic."""
    pass

with DAG(
    'my_ml_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id='load_data',
        python_callable=task_function,
    )

    task2 = PythonOperator(
        task_id='train_model',
        python_callable=task_function,
    )

    task1 >> task2
```

---

## 🔗 Дополнительные ресурсы

- [API Reference](../api/index.md) — полная документация API
- [Руководства](../guides/index.md) — подробные инструкции
- [GitHub Repository](https://github.com/yourusername/ipml_boston_housing) — исходный код

---

## ❓ Нужна помощь?

Если пример не работает:

1. Убедитесь, что установлены все зависимости: `make requirements`
2. Проверьте версию Python: `python --version` (требуется 3.13+)
3. Проверьте [Troubleshooting](../reproducibility/troubleshooting.md)
4. Создайте [Issue на GitHub](https://github.com/yourusername/ipml_boston_housing/issues)
