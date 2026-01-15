# 📊 Отчёт: Трекинг ML-экспериментов

**Проект:** Boston Housing Price Prediction  
**Дата:** 23 декабря 2025 (обновлено)  
**Автор:** Макаров Михаил Владимирович (студенческий проект по IPML)  

---

## 📋 Содержание

1. [Настройка выбранного инструмента трекинга](#1-настройка-выбранного-инструмента)
2. [Проведение экспериментов](#2-проведение-экспериментов)
3. [Интеграция с кодом](#3-интеграция-с-кодом)
4. [Инструкции по запуску](#4-инструкции-по-запуску)
5. [Настройка инструмента оркестрации](#5-настройка-инструмента-оркестрации)
6. [Настройка инструмента конфигураций](#6-настройка-инструмента-конфигураций)
7. [Интеграция и тестирование систем](#7-интеграция-и-тестирование-систем)

---

## 1. Настройка выбранного инструмента трекинга

### 1.1 Установка и настройка выбранного инструмента (MLflow)

**Выбранный стек технологий:**

| Инструмент | Версия | Назначение |
|------------|--------|------------|
| **MLflow** | 3.7.0+ | Трекинг экспериментов, Model Registry |
| **MinIO** | latest | S3-совместимое хранилище артефактов |
| **DVC** | 3.64.2+ | Версионирование данных |
| **Apache Airflow** | 2.8.1 | Оркестрация ML пайплайнов |
| **PostgreSQL** | 15 | База метаданных Airflow |
| **Redis** | 7-alpine | Брокер сообщений Celery |
| **Nginx** | alpine | Reverse proxy с Basic Auth |
| **Docker** | — | Контейнеризация инфраструктуры |

**Установка зависимостей:**

```bash
# Через uv (пакетный менеджер)
uv add "mlflow[auth]>=3.7.0" "boto3>=1.41.5" "dvc[s3]>=3.64.2"

# Обновление pyproject.toml
```

**Файл `pyproject.toml`:**

```toml
[project]
name = "ipml-boston-housing"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "cookiecutter-data-science>=2.3.0",
    "dvc[s3]>=3.64.2",
    "dvc-gdrive>=3.0.1",
    "pre-commit>=4.5.0",
    "ruff>=0.14.6",
    "numpy>=2.3.5",
    "pandas>=2.3.3",
    "dvclive>=3.49.0",
    "loguru>=0.7.3",
    "scikit-learn>=1.7.2",
    "click>=8.3.1",
    "dvc-s3>=3.2.2",
    "mlflow[auth]>=3.7.0",
    "boto3>=1.41.5",
]

[dependency-groups]
dev = [
    "detect-secrets>=1.5.0",
]
```

### 1.2 Настройка базы данных/облачного хранилища (MinIO)

**Архитектура решения:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Experiment Lifecycle                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Код/Скрипты │───▶│   MLflow     │───▶│      MinIO           │   │
│  │  обучения    │    │   Tracking   │    │  (Artifact Store)    │   │
│  └──────────────┘    │   Server     │    │                      │   │
│                      └──────────────┘    │  ┌────────────────┐  │   │
│                             │            │  │ mlflow-artifacts│  │   │
│  ┌──────────────┐           │            │  │  └─ models/     │  │   │
│  │     DVC      │───────────┼───────────▶│  │  └─ metrics/    │  │   │
│  │  (Версии     │           │            │  └────────────────┘  │   │
│  │   данных)    │           │            │                      │   │
│  └──────────────┘           │            │  ┌────────────────┐  │   │
│        │                    │            │  │boston-housing- │  │   │
│        ▼                    │            │  │     data       │  │   │
│  ┌──────────────┐           │            │  └────────────────┘  │   │
│  │     Git      │◀──────────┘            └──────────────────────┘   │
│  │  (.dvc файлы)│                                                   │
│  └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Docker-конфигурация MinIO (`docker/Dockerfile.minio`):**

```dockerfile
FROM minio/minio:latest

LABEL maintainer="Boston Housing ML Project"
LABEL description="MinIO storage для хранения ML артефактов"

ENV MINIO_ROOT_USER=minioadmin0
ENV MINIO_ROOT_PASSWORD=minioadmin1230
ENV MINIO_CONSOLE_ADDRESS=":9001"

EXPOSE 9000 9001
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=20s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9000/minio/health/live || exit 1

CMD ["server", "/data", "--console-address", ":9001"]
```

**Бакеты в MinIO:**

| Бакет | Назначение |
|-------|------------|
| `boston-housing-data` | Данные и модели (DVC) |
| `mlflow-artifacts` | Артефакты MLflow (модели, метрики, графики) |

**Скриншот MinIO Object Browser:**

![UI MinIO](figures/6.png)

*Рис. 1: Веб-консоль MinIO с бакетами для хранения данных DVC и артефактов MLflow*

### 1.3 Создание проекта и экспериментов

**Запуск инфраструктуры:**

```bash
# Запуск MinIO и MLflow
docker-compose up -d minio mlflow nginx

# Создание бакетов
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data
mc mb local/mlflow-artifacts
```

**Создание эксперимента в MLflow:**

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("boston-housing")
```

### 1.4 Настройка аутентификации и доступа

Реализована **двухуровневая аутентификация**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Архитектура аутентификации                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                    │
│  │   Браузер    │─────┐                                              │
│  │   (UI)       │     │                                              │
│  └──────────────┘     │                                              │
│                       ▼                                              │
│  ┌──────────────┐   ┌──────────────────────────┐                     │
│  │ Python SDK   │   │     Nginx (порт 5000)    │                     │
│  │ (mlflow.*)   │──▶│   Basic Auth (htpasswd)  │                     │
│  └──────────────┘   └──────────────────────────┘                     │
│        │                        │                                    │
│        │                        ▼                                    │
│        │            ┌──────────────────────────┐                     │
│        └───────────▶│  MLflow Server (внутр.)  │                     │
│                     │   Basic Auth (built-in)  │                     │
│                     └──────────────────────────┘                     │
│                                 │                                    │
│                                 ▼                                    │
│                     ┌──────────────────────────┐                     │
│                     │    MinIO (S3 артефакты)  │                     │
│                     │    Access Key + Secret   │                     │
│                     └──────────────────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Уровни защиты:**

| Уровень | Компонент | Метод защиты |
|---------|-----------|--------------|
| 1 | MinIO (S3) | Access Key + Secret Key |
| 2 | Nginx (UI/браузер) | Basic Auth (htpasswd) |
| 3 | MLflow (API/SDK) | Basic Auth (встроенный) |

**Конфигурация `.env`:**

```bash
# MinIO
MINIO_ROOT_USER=minioadmin0
MINIO_ROOT_PASSWORD=minioadmin1230

# MLflow Admin
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=secure_password_123

# MLflow Tracking (для Python SDK)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=secure_password_123

# S3/MinIO для артефактов
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin0
AWS_SECRET_ACCESS_KEY=minioadmin1230
```

**Доступ к сервисам:**

| Сервис | URL | Логин/Пароль | Описание |
|--------|-----|--------------|----------|
| **Airflow UI** | http://localhost:8080 | admin / admin | Управление DAGs и мониторинг |
| **MLflow UI** | http://localhost:5000 | admin / secure_password_123 | Веб-интерфейс MLflow |
| **MinIO Console** | http://localhost:9001 | minioadmin0 / minioadmin1230 | Управление хранилищем |
| **MinIO S3 API** | http://localhost:9000 | — | S3 API для артефактов |

---

## 2. Проведение экспериментов

### 2.1 Проведение 19 экспериментов с разными алгоритмами

Реализован комплексный набор экспериментов через Airflow DAG `boston_housing_experiments` с параллельным обучением 19 моделей машинного обучения:

**Линейные модели (7 экспериментов):**
| # | Алгоритм | Параметры | Описание |
|---|----------|-----------|----------|
| 1 | Linear Regression | — | Baseline линейная регрессия |
| 2 | Ridge | alpha=0.1 | Ridge α=0.1 |
| 3 | Ridge | alpha=1.0 | Ridge α=1.0 |
| 4 | Ridge | alpha=10.0 | Ridge α=10.0 |
| 5 | Lasso | alpha=0.1 | Lasso α=0.1 |
| 6 | ElasticNet | alpha=0.5, l1_ratio=0.5 | ElasticNet |
| 7 | Huber Regressor | epsilon=1.35 | Huber Regressor |

**Древовидные модели и ансамбли (9 экспериментов):**
| # | Алгоритм | Параметры | Описание |
|---|----------|-----------|----------|
| 8 | Decision Tree | max_depth=5 | Decision Tree d=5 |
| 9 | Decision Tree | max_depth=10 | Decision Tree d=10 |
| 10 | Random Forest | n_estimators=100, max_depth=10 | RF n=100 |
| 11 | Random Forest | n_estimators=200, max_depth=15 | RF n=200 |
| 12 | Extra Trees | n_estimators=100, max_depth=10 | ExtraTrees |
| 13 | Gradient Boosting | n_estimators=100, learning_rate=0.1 | GBM lr=0.1 |
| 14 | Gradient Boosting | n_estimators=200, learning_rate=0.05 | GBM lr=0.05 |
| 15 | AdaBoost | n_estimators=50, learning_rate=1.0 | AdaBoost |
| 16 | Bagging | n_estimators=20 | Bagging |

**Другие модели (3 эксперимента):**
| # | Алгоритм | Параметры | Описание |
|---|----------|-----------|----------|
| 17 | SVR | kernel=rbf, C=1.0 | SVR RBF |
| 18 | KNN | n_neighbors=5, weights=uniform | KNN k=5 |
| 19 | KNN | n_neighbors=10, weights=distance | KNN k=10 |

**Скриншот проведенных экспериментов:**

![UI MLFlow с экспериментами](figures/5.png)

*Рис. 2: MLflow UI с проведенными и запущенными экспериментами*

### 2.2 Настройка логирования метрик, параметров и артефактов

**Логируемые данные:**

| Тип | Примеры |
|-----|---------|
| **Параметры** | `n_estimators`, `max_depth`, `learning_rate`, `test_size` |
| **Метрики** | `r2_score`, `rmse`, `mae`, `mape` |
| **Артефакты** | Модели (`.pkl`), графики важности признаков, CSV с результатами |
| **Теги** | `model_type`, `framework`, `dataset` |

**Пример логирования:**

```python
import mlflow

with mlflow.start_run(run_name="rf-baseline"):
    # Параметры
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
    })

    # Метрики
    mlflow.log_metrics({
        "r2_score": 0.8665,
        "rmse": 3.13,
        "mae": 2.09,
    })

    # Артефакты
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_importance.csv")

    # Теги
    mlflow.set_tags({
        "model_type": "RandomForest",
        "framework": "sklearn",
    })
```


### 2.3 Система сравнения экспериментов

MLflow предоставляет мощные инструменты для сравнения:

**Через UI:**
- Таблица экспериментов с сортировкой по метрикам
- Графики сравнения (Parallel Coordinates, Scatter Plot)
- Визуализация изменения метрик по времени

**Через API:**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Получение топ-10 моделей по R² Score
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.r2_score DESC"],
    max_results=10,
)

for run in runs:
    r2 = run.data.metrics.get("r2_score", 0)
    params = run.data.params
    print(f"Run {run.info.run_id[:8]}: R²={r2:.4f}")
```

**Через CLI:**

```bash
# Список экспериментов
mlflow experiments search

# Поиск запусков с фильтрацией
mlflow runs list --experiment-id 1

# Сравнение двух запусков
mlflow runs compare <run_id_1> <run_id_2>
```

### 2.4 Фильтрация и поиск экспериментов

**Возможности фильтрации в MLflow:**

```python
# Поиск по метрикам
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.r2_score > 0.85",
)

# Поиск по параметрам
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="params.n_estimators = '200'",
)

# Поиск по тегам
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.model_type = 'RandomForest'",
)

# Комбинированный поиск
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="""
        metrics.r2_score > 0.80
        AND params.max_depth != 'None'
        AND tags.framework = 'sklearn'
    """,
    order_by=["metrics.rmse ASC"],
)
```

**Скриншот фильтрации экспериментов (изображение недоступно):**

---

## 3. Интеграция с кодом

### 3.1 Интеграция MLflow в Python код

**Структура модуля `src/tracking/`:**

```
src/tracking/
├── __init__.py
├── decorators.py      # 5 декораторов для автоматического логирования
├── mlflow_tracker.py  # 2 контекстных менеджера
└── utils.py           # 9 утилит для работы с экспериментами
```

**Модуль конфигурации `src/config/mlflow_config.py`:**

```python
"""Конфигурация MLflow для проекта."""

import os

# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "boston-housing")

# MinIO/S3 для артефактов
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin0")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin1230")


def setup_mlflow_env():
    """Настройка переменных окружения для MLflow + S3."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
```

### 3.2 Декораторы для автоматического логирования

**Модуль `src/tracking/decorators.py`:**

Реализовано **5 декораторов** для автоматического логирования:

| Декоратор | Назначение |
|-----------|------------|
| `@mlflow_run` | Автоматическое создание MLflow run с логированием времени выполнения |
| `@log_params_decorator` | Автоматическое логирование kwargs функции как параметров |
| `@log_metrics_decorator` | Извлечение и логирование метрик из возвращаемого словаря |
| `@log_artifact_decorator` | Автоматическое логирование файла-результата как артефакта |
| `@timed_execution` | Измерение и логирование времени выполнения функции |

```python
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
                start_time = time.time()
                mlflow.log_param("start_time", start_time)
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)
                logger.info(f"Эксперимент завершён за {duration:.2f}с")
                return result

        return wrapper
    return decorator


def log_params_decorator(func: Callable) -> Callable:
    """
    Декоратор для автоматического логирования параметров функции.
    Автоматически логирует все kwargs переданные в функцию как параметры MLflow.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if kwargs and mlflow.active_run():
            mlflow.log_params(kwargs)
            logger.debug(f"Залогированы параметры: {list(kwargs.keys())}")
        return func(*args, **kwargs)
    return wrapper


def log_metrics_decorator(metric_keys: list[str]):
    """
    Декоратор для автоматического логирования метрик из результата.
    Извлекает указанные ключи из возвращаемого словаря и логирует их как метрики.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            result = func(*args, **kwargs)
            if isinstance(result, dict) and mlflow.active_run():
                metrics_to_log = {
                    k: v for k, v in result.items()
                    if k in metric_keys and isinstance(v, (int, float))
                }
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)
            return result
        return wrapper
    return decorator


def log_artifact_decorator(artifact_path: str | None = None):
    """
    Декоратор для автоматического логирования артефакта.
    Если функция возвращает путь к файлу, этот файл будет залогирован.
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
    Логирует время выполнения как метрику {function_name}_duration_seconds.
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
```

**Пример использования декораторов:**

```python
from src.tracking.decorators import mlflow_run, log_params_decorator, log_metrics_decorator

@mlflow_run(experiment_name="boston-housing", run_name="rf-experiment")
@log_params_decorator
def train_model(n_estimators=100, max_depth=10, **params):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

@log_metrics_decorator(["r2_score", "rmse", "mae"])
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

# Использование
model = train_model(n_estimators=200, max_depth=15)
metrics = evaluate_model(model, X_test, y_test)
```

### 3.3 Контекстные менеджеры

**Модуль `src/tracking/mlflow_tracker.py`:**

Реализовано **2 класса** для управления экспериментами:

| Класс | Назначение |
|-------|------------|
| `MLflowExperimentTracker` | Основной контекстный менеджер для трекинга экспериментов |
| `NestedRunTracker` | Контекстный менеджер для вложенных runs (кросс-валидация, grid search) |

```python
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
        setup_mlflow_env()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run = None
        logger.info(f"MLflow трекер: {tracking_uri}, эксперимент: {experiment_name}")

    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Начало нового запуска эксперимента."""
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Запущен run: {self.run.info.run_id}")
        return self

    def __enter__(self):
        if self.run is None:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        self.run = None

    def log_params(self, params: dict[str, Any]):
        """Логирование параметров."""
        mlflow.log_params(params)
        logger.debug(f"Залогированы параметры: {list(params.keys())}")

    def log_param(self, key: str, value: Any):
        """Логирование одного параметра."""
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Логирование метрик."""
        mlflow.log_metrics(metrics, step=step)
        for name, value in metrics.items():
            logger.info(f"Метрика {name}: {value:.4f}")

    def log_metric(self, key: str, value: float, step: int | None = None):
        """Логирование одной метрики."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None):
        """Логирование артефакта (файла)."""
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Артефакт сохранён: {local_path}")

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None):
        """Логирование директории артефактов."""
        mlflow.log_artifacts(str(local_dir), artifact_path)
        logger.info(f"Директория артефактов сохранена: {local_dir}")

    def log_model(self, model, artifact_path: str, input_example=None,
                  registered_model_name: str | None = None):
        """Логирование sklearn модели с автоматическим определением сигнатуры."""
        signature = None
        if input_example is not None:
            predictions = model.predict(input_example)
            signature = infer_signature(input_example, predictions)
        mlflow.sklearn.log_model(
            model, artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Модель сохранена: {artifact_path}")

    def set_tags(self, tags: dict[str, str]):
        """Установка тегов."""
        mlflow.set_tags(tags)

    def set_tag(self, key: str, value: str):
        """Установка одного тега."""
        mlflow.set_tag(key, value)

    def log_dict(self, dictionary: dict, artifact_file: str):
        """Логирование словаря как JSON/YAML артефакта."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure, artifact_file: str):
        """Логирование matplotlib/plotly фигуры."""
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
        self.run_name = run_name
        self.tags = tags
        self.run = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags, nested=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        self.run = None

    def log_params(self, params: dict[str, Any]):
        """Логирование параметров."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Логирование метрик."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None):
        """Логирование одной метрики."""
        mlflow.log_metric(key, value, step=step)

    @property
    def run_id(self) -> str | None:
        """ID текущего вложенного запуска."""
        return self.run.info.run_id if self.run else None
```

**Пример использования контекстного менеджера:**

```python
from src.tracking.mlflow_tracker import MLflowExperimentTracker

tracker = MLflowExperimentTracker(experiment_name="boston-housing")

with tracker.start_run(run_name="gradient-boosting-v1"):
    tracker.set_tags({"model_type": "GradientBoosting", "framework": "sklearn"})
    tracker.log_params({"n_estimators": 200, "max_depth": 10})

    # Обучение модели
    model = GradientBoostingRegressor(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Оценка
    metrics = evaluate_model(model, X_test, y_test)
    tracker.log_metrics(metrics)

    # Сохранение модели
    tracker.log_model(model, "model", input_example=X_test.head(5))

    print(f"Run ID: {tracker.run_id}")
    print(f"Artifacts: {tracker.artifact_uri}")
```

### 3.4 Утилиты для работы с экспериментами

**Модуль `src/tracking/utils.py`:**

Реализовано **9 утилит** для работы с MLflow экспериментами:

| Функция | Назначение |
|---------|------------|
| `get_best_run` | Получение лучшего запуска по метрике |
| `load_best_model` | Загрузка лучшей модели из эксперимента |
| `compare_runs` | Сравнение запусков в виде DataFrame |
| `register_best_model` | Регистрация лучшей модели в Model Registry |
| `delete_experiment_runs` | Удаление старых запусков (с dry-run режимом) |
| `get_experiment_summary` | Получение сводки по эксперименту |
| `get_run_by_name` | Поиск запуска по имени |
| `list_registered_models` | Список зарегистрированных моделей |
| `transition_model_stage` | Изменение стадии модели в Registry |

```python
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

    Args:
        experiment_name: Название эксперимента
        metric: Метрика для сортировки
        ascending: True для минимизации (RMSE), False для максимизации (R²)

    Returns:
        Словарь с информацией о лучшем запуске
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
        return {}

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "tags": best_run.data.tags,
        "artifact_uri": best_run.info.artifact_uri,
        "start_time": best_run.info.start_time,
        "status": best_run.info.status,
    }


def load_best_model(experiment_name: str, metric: str = "r2_score"):
    """Загрузка лучшей модели по метрике."""
    best_run = get_best_run(experiment_name, metric)
    if not best_run:
        raise ValueError(f"Нет запусков в эксперименте '{experiment_name}'")

    model_uri = f"runs:/{best_run['run_id']}/model"
    logger.info(f"Загрузка модели из {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def compare_runs(
    experiment_name: str,
    metrics: list[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Сравнение запусков эксперимента."""
    if metrics is None:
        metrics = ["r2_score", "rmse", "mae"]

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

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
        row.update({f"metric_{k}": v for k, v in run.data.metrics.items() if k in metrics})
        row.update({f"param_{k}": v for k, v in run.data.params.items()})
        data.append(row)

    return pd.DataFrame(data)


def get_experiment_summary(experiment_name: str) -> dict[str, Any]:
    """Получение сводки по эксперименту."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return {}

    all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    finished_runs = [r for r in all_runs if r.info.status == "FINISHED"]

    r2_values = [r.data.metrics.get("r2_score") for r in finished_runs
                 if r.data.metrics.get("r2_score") is not None]

    return {
        "experiment_name": experiment_name,
        "total_runs": len(all_runs),
        "finished_runs": len(finished_runs),
        "best_r2": max(r2_values) if r2_values else None,
        "avg_r2": sum(r2_values) / len(r2_values) if r2_values else None,
    }


def delete_experiment_runs(
    experiment_name: str,
    keep_top_n: int = 10,
    metric: str = "r2_score",
    dry_run: bool = True,
) -> list[str]:
    """Удаление старых запусков, кроме топ-N по метрике."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )

    deleted_ids = []
    for run in all_runs[keep_top_n:]:
        if dry_run:
            logger.info(f"[DRY RUN] Будет удалён run: {run.info.run_id}")
        else:
            client.delete_run(run.info.run_id)
            logger.info(f"Удалён run: {run.info.run_id}")
        deleted_ids.append(run.info.run_id)

    return deleted_ids
```

**Пример использования утилит:**

```python
from src.tracking.utils import (
    get_best_run,
    load_best_model,
    compare_runs,
    register_best_model,
)

# Получение лучшего запуска
best = get_best_run("boston-housing", metric="r2_score")
print(f"Лучший R²: {best['metrics']['r2_score']:.4f}")

# Загрузка лучшей модели
model = load_best_model("boston-housing")
predictions = model.predict(X_new)

# Сравнение экспериментов
comparison = compare_runs("boston-housing", top_n=5)
print(comparison)

# Регистрация лучшей модели
version = register_best_model("boston-housing", "boston-housing-rf")
print(f"Зарегистрирована версия: {version}")
```

---

## 4. Инструкции по запуску

### 4.1 Быстрый старт

```bash
# 1. Клонирование репозитория
git clone <repo-url>
cd ipml_boston_housing

# 2. Установка зависимостей
uv sync

# 3. Создание файла .env
cat > .env << 'EOF'
# MinIO
MINIO_ROOT_USER=minioadmin0
MINIO_ROOT_PASSWORD=minioadmin1230

# MLflow
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=secure_password_123
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=secure_password_123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_FLASK_SERVER_SECRET_KEY=mlflow-secret-key-change-me

# S3/MinIO для артефактов
AWS_ACCESS_KEY_ID=minioadmin0
AWS_SECRET_ACCESS_KEY=minioadmin1230

# Airflow
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_UID=50000
EOF

# 4. Запуск инфраструктуры
# Запуск полного стека (включая Airflow)
docker-compose up -d

# ИЛИ запуск только ML инфраструктуры (без Airflow)
# docker-compose up -d minio mlflow nginx

# 5. Создание бакетов
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data
mc mb local/mlflow-artifacts

# 6. Загрузка данных
dvc pull

# 7. Запуск экспериментов
# Локально
python src/modeling/train.py -n 200 -d 15

# Через Airflow (полный набор из 19 экспериментов)
# ВАЖНО: Дождитесь инициализации Airflow (2-3 минуты)
# Проверьте готовность: docker-compose logs airflow-init
# Откройте http://localhost:8080 (admin/admin)
# Запустите DAG: boston_housing_experiments

# 8. Просмотр результатов
# MLflow UI: http://localhost:5000 (admin / secure_password_123)
# Airflow UI: http://localhost:8080 (admin / admin)
# MinIO Console: http://localhost:9001 (minioadmin0 / minioadmin1230)
```

### 4.2 Полезные команды

```bash
# MLflow
mlflow experiments search                # Список экспериментов
mlflow runs list --experiment-id 1       # Запуски эксперимента
mlflow models list                       # Зарегистрированные модели

# Airflow
# Веб-интерфейс: http://localhost:8080 (admin/admin)
docker-compose logs -f airflow-webserver   # Логи веб-сервера
docker-compose logs -f airflow-scheduler   # Логи планировщика
docker-compose logs -f airflow-worker      # Логи воркера

# Docker (полная инфраструктура)
docker-compose ps                        # Статус всех контейнеров
docker-compose logs -f mlflow            # Логи MLflow
docker-compose restart airflow-webserver # Перезапуск Airflow UI
docker-compose down && docker-compose up -d  # Полный перезапуск

# DVC
dvc status                               # Статус данных
dvc pull                                 # Загрузка данных
dvc push                                 # Отправка данных
```

---

## 5. Настройка инструмента оркестрации

**Выполненные задания:**
- ✅ Установлен и настроен Apache Airflow 2.8.1 с CeleryExecutor
- ✅ Создан workflow для ML пайплайна (3 DAG с различными паттернами)
- ✅ Настроены зависимости между этапами (линейные, параллельные, условные)
- ✅ Реализовано кэширование в MinIO и параллельное выполнение до 8 задач

### 5.1 Установка и настройка выбранного инструмента (Apache Airflow)

**Выбранный инструмент оркестрации:** Apache Airflow 2.8.1

**Архитектура развертывания:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Airflow Stack                             │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Webserver   │  Scheduler   │   Worker     │   airflow-init    │
│  (порт 8080) │              │  (Celery)    │                   │
└──────┬───────┴──────┬───────┴──────┬───────┴───────────────────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Инфраструктура                              │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│ PostgreSQL  │   Redis     │   MLflow    │   MinIO     │ Nginx  │
│ (Metadata)  │ (Broker)    │ (Tracking)  │ (Storage)   │ (Auth) │
└─────────────┴─────────────┴─────────────┴─────────────┴────────┘
```

**Реализованная архитектура через Docker Compose:**

Полный стек развёрнут в `docker-compose.yml` с 10 сервисами:

| Сервис | Порт | Назначение |
|--------|------|------------|
| **airflow-webserver** | 8080 | Веб-интерфейс Airflow |
| **airflow-scheduler** | — | Планировщик задач |
| **airflow-worker** | — | Исполнитель задач (Celery) |
| **airflow-init** | — | Инициализация БД и создание пользователя |
| **postgres** | — | База метаданных Airflow |
| **redis** | — | Брокер сообщений Celery |
| **mlflow** | 5000 (через nginx) | Tracking server с аутентификацией |
| **minio** | 9000, 9001 | S3-совместимое хранилище |
| **nginx** | 5000 | Reverse proxy с Basic Auth для MLflow |
| **train** | — | Контейнер для обучения моделей (профиль train) |

```dockerfile
# docker/Dockerfile.airflow (фрагмент)
FROM apache/airflow:2.8.1-python3.11

# Установка Python зависимостей проекта
RUN pip install --no-cache-dir \
    numpy>=2.0.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.5.0 \
    mlflow>=2.10.0 \
    boto3>=1.34.0 \
    loguru>=0.7.0 \
    python-dotenv>=1.0.0 \
    matplotlib>=3.8.0 \
    apache-airflow-providers-celery>=3.5.0 \
    apache-airflow-providers-redis>=3.5.0 \
    apache-airflow-providers-postgres>=5.8.0 \
    apache-airflow-providers-amazon>=8.0.0

# Настройка Python paths для доступа к src/
ENV PYTHONPATH="/opt/airflow:/opt/airflow/src:${PYTHONPATH}"
```

**Docker Compose конфигурация:**

```yaml
x-airflow-common: &airflow-common
  build:
    context: .
    dockerfile: docker/Dockerfile.airflow
  environment: &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow  # pragma: allowlist secret
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow  # pragma: allowlist secret
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    # Параллелизм
    AIRFLOW__CORE__PARALLELISM: 16
    AIRFLOW__CORE__DAG_CONCURRENCY: 8
    AIRFLOW__CELERY__WORKER_CONCURRENCY: 4
    # Интеграция с ML инфраструктурой
    MLFLOW_TRACKING_URI: http://mlflow:5000
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
  volumes:
    - ./airflow/dags:/opt/airflow/dags
    - ./airflow/logs:/opt/airflow/logs
    - ./airflow/plugins:/opt/airflow/plugins
    - ./data:/opt/airflow/data
    - ./src:/opt/airflow/src
```

**Веб-интерфейс Apache Airflow:**

![Доступные DAGs](figures/2.png)

*Рис. 3: Доступные DAGs экспериментов в Airflow UI*

![DAG с экспериментом](figures/1.png)

*Рис. 4: DAG Airflow с экспериментом boston_housing_experiments*

![Docker-compose консоль](figures/3.png)

*Рис. 5: Docker-compose консоль с запущенными сервисами*

![Gantt диаграмма](figures/4.png)

*Рис. 6: Gantt диаграмма с запущенными задачами в Airflow*

### 5.2 Создание workflow для ML пайплайна

**Структура `airflow/dags/`:**

```
airflow/dags/
├── boston_housing_simple.py       # Простой последовательный пайплайн
├── boston_housing_experiments.py  # Параллельное обучение 19 моделей
└── boston_housing_cached.py       # Пайплайн с кэшированием в MinIO
```

Реализовано **3 DAG** с различными паттернами оркестрации:

#### DAG 1: boston_housing_simple

**Назначение:** Простой последовательный пайплайн для быстрого прототипирования с одной моделью Random Forest.

**Архитектура:**
```
download_data → validate_data → train_model → evaluate_model → save_artifacts
```

**Особенности:**
- Обучение одной модели Random Forest
- Сохранение результатов в локальные файлы
- Простая последовательная архитектура без параллелизма

**Конфигурация:**
```python
@dag(
    dag_id="boston_housing_simple",
    schedule=None,  # Ручной запуск
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "boston_housing", "random_forest"],
)
```

#### DAG 2: boston_housing_experiments

**Назначение:** Параллельное обучение 19 ML моделей с агрегацией результатов и автоматическим логированием в MLflow.

**Архитектура:**
```
                    download_data
                         │
                    validate_data
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Linear Models   Tree Models    Other Models
      (7 шт.)        (9 шт.)        (3 шт.)
         │               │               │
         └───────────────┼───────────────┘
                         │
                  aggregate_results
                         │
                   generate_report
                         │
              ┌──────────┼──────────┐
              │          │          │
         Save to     Log to     Upload to
         Local      MLflow      MinIO
```

**Особенности:**
- Максимум 8 параллельных задач (`max_active_tasks=8`)
- Автоматическое логирование в MLflow с тегами
- Сохранение артефактов в MinIO (отчёты, результаты, лучшая модель)
- Генерация подробного Markdown отчёта

**Параллельное выполнение через expand():**
```python
# Линейные модели (7 экспериментов)
linear_results = train_single_model.expand(
    model_config=LINEAR_MODELS,
    data_info=[data_info] * len(LINEAR_MODELS),
)

# Древовидные модели (9 экспериментов)
tree_results = train_single_model.expand(
    model_config=TREE_MODELS,
    data_info=[data_info] * len(TREE_MODELS),
)

# Другие модели (3 эксперимента)
other_results = train_single_model.expand(
    model_config=OTHER_MODELS,
    data_info=[data_info] * len(OTHER_MODELS),
)
```

#### DAG 3: boston_housing_cached

**Назначение:** ML пайплайн с интеллектуальным кэшированием в MinIO.

**Особенности:**
- `ShortCircuitOperator` для пропуска задач если модель в кэше
- Проверка хэша входных данных для инвалидации кэша
- Автоматическое сохранение в кэш после обучения

**Логика кэширования:**
```python
def check_cache_exists(data_path: str, **kwargs) -> bool:
    """
    True: продолжить выполнение (нет кэша, нужно обучать)
    False: пропустить downstream задачи (кэш найден)
    """
    cache = MinIOCache(bucket_name="airflow-cache")
    prefix = "models/random_forest_cached"
    exists, cache_key = cache.check_cache(prefix, MODEL_PARAMS, data_path)
    return not exists  # Инвертируем для ShortCircuitOperator
```

### 5.3 Настройка зависимостей между этапами

**Типы зависимостей:**

| Тип | Описание | Пример |
|------|----------|---------|
| **Линейные** | Последовательное выполнение | `data >> validate >> train >> evaluate` |
| **Параллельные** | Независимое выполнение групп задач | `expand()` для множественного обучения |
| **Условные** | Выполнение зависит от результата предыдущих задач | `ShortCircuitOperator` для кэширования |
| **Trigger Rules** | Гибкая логика запуска | `none_failed_min_one_success` |

**Примеры настройки зависимостей:**

```python
# Линейные зависимости
data_path >> cache_check >> train_result >> save_result >> summary

# Условные зависимости с trigger rules
@task(trigger_rule="none_failed_min_one_success")
def generate_summary(cache_result=None, train_save_result=None):
    # Выполняется если хотя бы один upstream task успешен
    pass

# Параллельные зависимости
aggregated = aggregate_results(
    linear_results=linear_results,    # Ждёт завершения всех линейных моделей
    tree_results=tree_results,        # Ждёт завершения всех древовидных моделей
    other_results=other_results,      # Ждёт завершения всех других моделей
)
```

### 5.4 Реализация кэширования и параллельного выполнения

**Структура модуля `airflow/plugins/`:**

```
airflow/plugins/
├── __init__.py
└── minio_cache.py  # Система кэширования артефактов в MinIO
```

#### Система кэширования на базе MinIO

**Модуль `airflow/plugins/minio_cache.py`:**

Реализован класс `MinIOCache` и 3 вспомогательные функции для кэширования:

| Компонент | Назначение |
|-----------|------------|
| `MinIOCache` | Класс для работы с кэшем в MinIO (проверка, загрузка, скачивание) |
| `check_model_cache` | Функция для ShortCircuitOperator (проверка наличия модели в кэше) |
| `get_cached_model` | Получение кэшированной модели из MinIO |
| `save_model_to_cache` | Сохранение обученной модели в кэш с метаданными |

```python
class MinIOCache:
    """
    Класс для кэширования артефактов в MinIO.

    Позволяет:
    - Проверять существование файлов по хэш-ключам
    - Вычислять хэши для проверки изменений данных
    - Загружать/скачивать артефакты
    """

    def __init__(self, bucket_name: str = "airflow-cache"):
        self.endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
        self.bucket_name = bucket_name
        self.client = boto3.client("s3", endpoint_url=self.endpoint_url, ...)
        self._ensure_bucket_exists()

    def compute_file_hash(self, file_path: str) -> str:
        """Вычисляет MD5 хэш файла."""
        ...

    def compute_params_hash(self, params: dict) -> str:
        """Вычисляет хэш параметров."""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def get_cache_key(self, prefix: str, params: dict, data_hash: str = None) -> str:
        """
        Генерирует ключ кэша на основе параметров и хэша данных.
        Формат: prefix_params_hash_data_hash
        """
        params_hash = self.compute_params_hash(params)
        if data_hash:
            return f"{prefix}_{params_hash}_{data_hash}"
        return f"{prefix}_{params_hash}"

    def check_cache(self, prefix: str, params: dict, data_path: str = None) -> tuple[bool, str]:
        """Проверяет наличие кэшированного результата."""
        ...

    def upload(self, local_path: str, key: str) -> str:
        """Загружает файл в MinIO."""
        ...

    def download(self, key: str, local_path: str) -> str:
        """Скачивает файл из MinIO."""
        ...

    def put_json(self, key: str, data: dict) -> str:
        """Сохраняет JSON в MinIO."""
        ...

    def get_json(self, key: str) -> dict:
        """Читает JSON из MinIO."""
        ...
```

**Интеграция с Airflow:**

```python
# Проверка кэша
cache_check = ShortCircuitOperator(
    task_id="check_cache",
    python_callable=check_cache_exists,
    op_kwargs={"data_path": data_path},
    provide_context=True,
)

# Сохранение в кэш после обучения
save_result = save_to_cache(train_result)
```

#### Параллельное выполнение

**Настройки параллелизма в Docker Compose:**

```yaml
environment:
  AIRFLOW__CORE__PARALLELISM: 16          # Макс. задач во всём Airflow
  AIRFLOW__CORE__DAG_CONCURRENCY: 8       # Макс. задач на один DAG
  AIRFLOW__CELERY__WORKER_CONCURRENCY: 4  # Макс. задач на воркер
```

**Реализация через expand() operator:**

```python
@dag(max_active_tasks=8)  # Ограничение на уровне DAG
def boston_housing_experiments_dag():

    # Параллельное обучение 19 моделей
    linear_results = train_single_model.expand(
        model_config=LINEAR_MODELS,      # 7 конфигураций
        data_info=[data_info] * 7,
    )

    tree_results = train_single_model.expand(
        model_config=TREE_MODELS,        # 9 конфигураций  
        data_info=[data_info] * 9,
    )

    other_results = train_single_model.expand(
        model_config=OTHER_MODELS,       # 3 конфигурации
        data_info=[data_info] * 3,
    )
```

**Мониторинг параллельного выполнения:**

- **Airflow UI:** Граф зависимостей с реальным временем выполнения
- **Логи:** Структурированное логирование через `loguru`
- **Метрики:** Время выполнения каждой задачи в MLflow

---

## 6. Настройка инструмента конфигураций

**Выполненные задания:**
- ✅ Настроен Hydra + Pydantic для управления конфигурациями
- ✅ Созданы конфигурации для 14 различных алгоритмов ML
- ✅ Настроена валидация конфигураций через Pydantic схемы
- ✅ Создана система композиции конфигураций с CLI переопределением

### 6.1 Настройка выбранного инструмента для управления конфигурациями

**Выбранный стек:** Hydra + Pydantic — комбинация для ML проектов, где Hydra обеспечивает композицию YAML-конфигураций и CLI переопределение, а Pydantic — строгую валидацию схем.

**Установка зависимостей:**

```bash
uv add "hydra-core>=1.3.2" "pydantic-settings>=2.0"
```

**Архитектура системы конфигураций:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hydra + Pydantic Configuration System                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     YAML Configurations (conf/)                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │config.yaml  │  │ model/*.yaml│  │ experiment/*.yaml       │  │    │
│  │  │  (defaults) │  │ (14 моделей)│  │ (композиции)            │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │    │
│  │         │                │                      │                │    │
│  │         └────────────────┴──────────────────────┘                │    │
│  │                          │                                       │    │
│  │                    ┌─────▼─────┐                                 │    │
│  │                    │   Hydra   │  CLI: model=ridge model.alpha=0.5│   │
│  │                    │  Compose  │                                 │    │
│  │                    └─────┬─────┘                                 │    │
│  └──────────────────────────┼───────────────────────────────────────┘    │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐    │
│  │                 Pydantic Validation (src/schemas/)               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ModelConfig  │  │ DataConfig  │  │ TrainingConfig          │  │    │
│  │  │ (14 схем)   │  │ (валидация) │  │ (эксперименты)          │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │    │
│  │         │                │                      │                │    │
│  │         └────────────────┴──────────────────────┘                │    │
│  │                          │                                       │    │
│  │                    ┌─────▼─────┐                                 │    │
│  │                    │Experiment │                                 │    │
│  │                    │  Config   │                                 │    │
│  │                    └─────┬─────┘                                 │    │
│  └──────────────────────────┼───────────────────────────────────────┘    │
│                             │                                            │
│                       ┌─────▼─────┐                                      │
│                       │train_hydra│  DVCLive + MLflow интеграция        │
│                       │   .py     │                                      │
│                       └───────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Структура файлов конфигурации:**

```
conf/                              # Hydra конфигурации
├── config.yaml                    # Главная конфигурация с defaults
├── model/                         # 14 конфигураций моделей
│   ├── random_forest.yaml
│   ├── gradient_boosting.yaml
│   ├── ridge.yaml
│   ├── lasso.yaml
│   ├── elastic_net.yaml
│   ├── linear_regression.yaml
│   ├── huber.yaml
│   ├── sgd.yaml
│   ├── decision_tree.yaml
│   ├── extra_trees.yaml
│   ├── adaboost.yaml
│   ├── bagging.yaml
│   ├── svr.yaml
│   └── knn.yaml
├── data/
│   └── boston.yaml                # Конфигурация датасета
├── training/
│   └── default.yaml               # Параметры обучения
└── experiment/                    # Композиции для экспериментов
    ├── baseline.yaml
    ├── tuned.yaml
    └── linear_comparison.yaml

src/schemas/                       # Pydantic схемы валидации
├── __init__.py
├── base.py                        # Базовые классы
├── model_config.py                # 14 схем параметров моделей
├── data_config.py                 # Схема данных
└── training_config.py             # Схема обучения + ExperimentConfig
```

### 6.2 Создание конфигураций для разных алгоритмов

**Главная конфигурация `conf/config.yaml`:**

```yaml
# Главная конфигурация Hydra для Boston Housing ML проекта
defaults:
  - model: random_forest      # Модель по умолчанию
  - data: boston              # Датасет
  - training: default         # Параметры обучения
  - _self_                    # Переопределения применяются последними

name: boston_housing_experiment
description: "Прогнозирование цен на недвижимость в Бостоне"
tags:
  - regression
  - boston-housing
  - scikit-learn

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false  # Не менять рабочую директорию
```

**Пример конфигурации модели `conf/model/random_forest.yaml`:**

```yaml
# @package model
name: random_forest

# Параметры ансамбля
n_estimators: 100        # Количество деревьев (10-1000)
max_depth: 10            # Максимальная глубина (1-100)
min_samples_split: 2     # Минимум образцов для разбиения
min_samples_leaf: 1      # Минимум образцов в листе
max_features: sqrt       # Признаки для разбиения

# Производительность
n_jobs: -1               # Параллельные потоки (-1 = все ядра)
bootstrap: true          # Использовать bootstrap выборки

# Воспроизводимость
random_state: 42
```

**Пример конфигурации эксперимента `conf/experiment/tuned.yaml`:**

```yaml
# @package _global_
# Настроенный эксперимент с оптимизированными параметрами

defaults:
  - override /model: random_forest
  - override /training: default

name: tuned_experiment
description: "Эксперимент с оптимизированными гиперпараметрами"
tags:
  - tuned
  - optimized

model:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
  min_samples_leaf: 2

training:
  experiment_name: boston-housing-tuned
  cross_validation: false
```

**Все 14 поддерживаемых моделей:**

| Категория | Модели | Конфигурационный файл |
|-----------|--------|----------------------|
| **Линейные** | Linear Regression, Ridge, Lasso, ElasticNet, Huber, SGD | `conf/model/{linear_regression,ridge,lasso,elastic_net,huber,sgd}.yaml` |
| **Ансамбли** | Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging | `conf/model/{random_forest,extra_trees,gradient_boosting,adaboost,bagging}.yaml` |
| **Другие** | Decision Tree, SVR, KNN | `conf/model/{decision_tree,svr,knn}.yaml` |

### 6.3 Настройка валидации конфигураций

**Pydantic схемы для строгой валидации типов (`src/schemas/model_config.py`):**

```python
from pydantic import Field, field_validator
from src.schemas.base import BaseConfig

class RandomForestConfig(BaseConfig):
    """Конфигурация Random Forest с валидацией."""

    name: Literal["random_forest"] = "random_forest"

    n_estimators: int = Field(
        default=100,
        ge=10,          # >= 10
        le=1000,        # <= 1000
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
    random_state: int = Field(default=42, ge=0)

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        if v < 10:
            raise ValueError(f"n_estimators должен быть >= 10, получено: {v}")
        return v

    def get_params(self) -> dict:
        """Возвращает параметры для создания модели."""
        return self.model_dump(exclude={"name"}, exclude_none=True)
```

**Валидация конфигурации эксперимента (`src/schemas/training_config.py`):**

```python
class ExperimentConfig(BaseConfig):
    """Полная конфигурация эксперимента с композицией."""

    model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    name: str = "default"
    tags: list[str] = []

    def get_validated_model_config(self) -> ModelConfig:
        """Возвращает валидированную конфигурацию модели."""
        model_name = self.model.get("name", "random_forest")
        config_class = get_model_config_class(model_name)
        return config_class(**self.model)

    def validate_all(self) -> tuple[ModelConfig, DataConfig, TrainingConfig]:
        """Валидирует все конфигурации и возвращает их."""
        return (
            self.get_validated_model_config(),
            self.get_validated_data_config(),
            self.get_validated_training_config(),
        )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "ExperimentConfig":
        """Создаёт ExperimentConfig из Hydra DictConfig."""
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls(**config_dict)
```

**Валидация в скрипте обучения (`src/modeling/train_hydra.py`):**

```python
def validate_config(cfg: DictConfig) -> ExperimentConfig:
    """Валидация конфигурации через Pydantic."""
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        exp_config = ExperimentConfig(
            model=config_dict.get("model", {}),
            data=config_dict.get("data", {}),
            training=config_dict.get("training", {}),
        )

        # Валидируем все вложенные конфигурации
        model_config, data_config, training_config = exp_config.validate_all()
        logger.success(f"Конфигурация валидна: model={model_config.name}")

        return exp_config
    except ValidationError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise
```

### 6.4 Создание системы композиции конфигураций

**Hydra defaults для композиции:**

```yaml
# conf/config.yaml
defaults:
  - model: random_forest      # Подключает conf/model/random_forest.yaml
  - data: boston              # Подключает conf/data/boston.yaml
  - training: default         # Подключает conf/training/default.yaml
  - _self_                    # Переопределения из этого файла
```

**Эксперименты как композиции (`conf/experiment/baseline.yaml`):**

```yaml
# @package _global_
defaults:
  - override /model: random_forest
  - override /training: default

name: baseline_experiment
model:
  n_estimators: 100
  max_depth: 10
```

**Примеры использования CLI:**

```bash
# Базовый запуск (Random Forest по умолчанию)
uv run python src/modeling/train_hydra.py

# Смена модели
uv run python src/modeling/train_hydra.py model=gradient_boosting

# Переопределение параметров
uv run python src/modeling/train_hydra.py model=random_forest \
    model.n_estimators=500 \
    model.max_depth=20

# Готовый эксперимент
uv run python src/modeling/train_hydra.py +experiment=tuned

# Multirun (несколько моделей параллельно)
uv run python src/modeling/train_hydra.py --multirun \
    model=ridge,lasso,elastic_net

# Просмотр конфигурации без запуска
uv run python src/modeling/train_hydra.py --cfg job
```

**Результаты тестирования:**

| Эксперимент | Модель | R² Score | RMSE | Команда |
|-------------|--------|----------|------|---------|
| Default | Random Forest | 0.8482 | 3.34 | `model=random_forest` |
| Tuned | Random Forest (оптим.) | 0.8482 | 3.34 | `+experiment=tuned` |
| Ridge | Ridge Regression | 0.6662 | 4.95 | `model=ridge` |

**Интеграция с MLflow и DVCLive:**

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    # Валидация через Pydantic
    exp_config = validate_config(cfg)
    model_config = exp_config.get_validated_model_config()

    # DVCLive для логирования
    with Live(save_dvc_exp=True) as live:
        live.log_param("model_name", model_config.name)
        for key, value in model_config.get_params().items():
            live.log_param(f"model.{key}", value)

        # ... обучение модели ...

        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

    return metrics["r2_score"]
```

---

## 7. Интеграция и тестирование систем

**Выполненные задания:**
- ✅ Интегрированы все инструменты (Airflow + MLflow + MinIO + Hydra)
- ✅ Создана система мониторинга выполнения (loguru + Airflow UI + MLflow UI)
- ✅ Настроены уведомления о результатах (callbacks + сводные отчеты)
- ✅ Протестирована воспроизводимость результатов (random seed + кэширование)

### 7.1 Интеграция выбранных инструментов

**Архитектура интегрированной системы:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Интегрированная ML Платформа                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Apache Airflow │    │      MLflow      │    │      MinIO       │   │
│  │  (Оркестрация)   │◀──▶│   (Трекинг)      │◀──▶│   (Хранилище)    │   │
│  │                  │    │                  │    │                  │   │
│  │ • 3 DAG          │    │ • Эксперименты   │    │ • Артефакты      │   │
│  │ • Параллелизм    │    │ • Метрики        │    │ • Кэширование    │   │
│  │ • Кэширование    │    │ • Модели         │    │ • S3 API         │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘   │
│           │                        │                        │           │
│           ▼                        ▼                        ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        Общая интеграция                             │ │
│  ├─────────────────────────────────────────────────────────────────────┤ │
│  │ • Docker Compose оркестрация всех сервисов                          │ │
│  │ • Общие переменные окружения (.env)                                 │ │
│  │ • Единая аутентификация (Nginx + Basic Auth)                       │ │
│  │ • Мониторинг через структурированные логи (loguru)                  │ │
│  │ • Автоматические уведомления в Airflow UI + MLflow UI              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ключевые точки интеграции:**

### **1. Airflow → MLflow интеграция**

```python
# В Airflow DAG
def generate_report(aggregated: dict) -> dict:
    """Автоматическое логирование результатов в MLflow из Airflow."""

    import mlflow

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("boston_housing_experiments")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"experiments_summary_{timestamp}"):
        # Логирование агрегированных метрик
        mlflow.log_metric("total_experiments", aggregated["total_experiments"])
        mlflow.log_metric("best_r2", aggregated["summary"]["best_r2"])
        mlflow.log_metric("mean_r2", aggregated["summary"]["mean_r2"])

        # Логирование параметров лучшей модели
        mlflow.log_param("best_model", aggregated["best_model"]["run_id"])

        # Логирование артефактов
        mlflow.log_artifact(report_path, "reports")
        mlflow.log_artifact(aggregated["results_path"], "results")

        # Теги для идентификации источника
        mlflow.set_tag("source", "airflow")
        mlflow.set_tag("dag", "boston_housing_experiments")
```

### **2. Airflow → MinIO интеграция**

```python
# Кэширование артефактов в MinIO из Airflow
class MinIOCache:
    def __init__(self):
        # Используем те же переменные окружения что и в Airflow
        self.endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

# Использование в Airflow task
@task
def save_to_cache(train_result: dict) -> dict:
    """Сохранение модели в MinIO с автоматической генерацией ключей."""

    from minio_cache import save_model_to_cache

    model_uri = save_model_to_cache(
        model_path=train_result["model_path"],
        model_name="random_forest_cached",
        params=train_result["params"],
        data_path=train_result["data_path"],
        metrics=train_result["metrics"],
        bucket_name="airflow-cache",
    )

    return {"status": "saved", "cache_uri": model_uri}
```

### **3. MLflow → MinIO интеграция**

```python
# MLflow автоматически использует MinIO для артефактов
# Настройка в переменных окружения:

MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin  
AWS_SECRET_ACCESS_KEY=minioadmin

# Все артефакты MLflow автоматически сохраняются в MinIO
mlflow.sklearn.log_model(model, "model")  # → s3://mlflow-artifacts/...
mlflow.log_artifact("report.html")        # → s3://mlflow-artifacts/...
```

### 7.2 Создание системы мониторинга выполнения

**Многоуровневая система мониторинга:**

### **Уровень 1: Структурированные логи (loguru)**

```python
from loguru import logger

# Конфигурация логгера в src/config.py
logger.remove(0)  # Удаляем стандартный хэндлер
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO"
)

# Использование в ML пайплайнах
@task
def train_single_model(model_config: dict, data_info: dict) -> dict:
    """Обучение с подробным логированием."""

    model_name = model_config["name"]
    params = model_config["params"]

    logger.info(f"🚀 Начало обучения: {model_name}")
    logger.debug(f"   Параметры: {params}")
    logger.debug(f"   Размер данных: train={data_info['n_train']}, test={data_info['n_test']}")

    start_time = time.time()

    try:
        # Обучение модели
        model = create_model(model_name, params)
        model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logger.info(f"✅ Обучение завершено за {train_time:.2f}с")

        # Вычисление метрик
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

        logger.success(f"📊 Метрики: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}")

        return {
            "run_id": f"{model_name}_{params_str}",
            "metrics": metrics,
            "train_time": train_time,
        }

    except Exception as e:
        logger.error(f"❌ Ошибка обучения {model_name}: {e}")
        raise
```

### **Уровень 2: Мониторинг Airflow**

```python
# Настройки мониторинга в default_args DAG
default_args = {
    "owner": "boston_housing",
    "depends_on_past": False,
    "email_on_failure": False,    # Отключаем email, используем UI
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": log_failure_callback,
    "on_success_callback": log_success_callback,
}

def log_failure_callback(context):
    """Callback при неудачном выполнении задачи."""
    task_instance = context.get('task_instance')
    dag_run = context.get('dag_run')

    logger.error(f"🔥 Task failed: {task_instance.dag_id}.{task_instance.task_id}")
    logger.error(f"   DAG run: {dag_run.run_id}")
    logger.error(f"   Execution date: {context.get('execution_date')}")

    # Можно добавить отправку в внешние системы мониторинга
    # send_alert_to_slack(task_instance, dag_run)

def log_success_callback(context):
    """Callback при успешном выполнении задачи."""
    task_instance = context.get('task_instance')
    duration = task_instance.end_date - task_instance.start_date

    logger.success(f"✅ Task completed: {task_instance.dag_id}.{task_instance.task_id}")
    logger.info(f"   Duration: {duration.total_seconds():.2f}s")
```

### **Уровень 3: Мониторинг через MLflow UI**

```python
# Автоматическое логирование системных метрик
@mlflow_run(experiment_name="boston-housing")
def train_with_monitoring(model_name: str, params: dict):
    """Обучение с мониторингом производительности."""

    import psutil
    import time

    # Мониторинг ресурсов
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu_percent = process.cpu_percent()

    # Логирование системной информации
    mlflow.log_param("system_memory_gb", psutil.virtual_memory().total / 1024**3)
    mlflow.log_param("system_cpu_count", psutil.cpu_count())
    mlflow.log_param("start_memory_mb", start_memory)

    start_time = time.time()

    # Обучение модели
    model = train_model(model_name, params)

    # Мониторинг после обучения
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = end_memory - start_memory

    # Логирование метрик производительности
    mlflow.log_metric("train_time_seconds", end_time - start_time)
    mlflow.log_metric("memory_usage_mb", end_memory)
    mlflow.log_metric("memory_delta_mb", memory_delta)
    mlflow.log_metric("cpu_percent", process.cpu_percent())

    return model
```

### 7.3 Настройка уведомлений о результатах

**Система уведомлений через интерфейсы:**

### **1. Airflow UI Dashboard**

```python
# Кастомные операторы для создания rich-уведомлений
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

@task
def send_completion_notification(results: dict) -> dict:
    """Создание уведомления о завершении эксперимента."""

    total_experiments = results.get("total_experiments", 0)
    best_r2 = results.get("summary", {}).get("best_r2", 0)
    best_model = results.get("best_model", {}).get("run_id", "unknown")

    # Создание подробного статуса для Airflow UI
    notification = {
        "status": "SUCCESS" if best_r2 > 0.8 else "WARNING",
        "summary": f"Завершено {total_experiments} экспериментов",
        "best_result": f"Лучшая модель: {best_model} (R²={best_r2:.4f})",
        "timestamp": datetime.now().isoformat(),
    }

    # Логирование для отображения в Airflow UI
    logger.info("=" * 60)
    logger.info("🎉 ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    logger.info("=" * 60)
    logger.info(f"📊 Всего экспериментов: {total_experiments}")
    logger.info(f"🏆 Лучший результат: R² = {best_r2:.4f}")
    logger.info(f"🤖 Лучшая модель: {best_model}")
    logger.info("=" * 60)

    return notification
```

### **2. MLflow UI интеграция**

```python
# Автоматическое создание сводных run'ов в MLflow
@task
def create_mlflow_summary(aggregated_results: dict) -> dict:
    """Создание сводного run'а в MLflow с результатами всех экспериментов."""

    import mlflow

    mlflow.set_experiment("boston_housing_summary")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"batch_summary_{timestamp}"):

        # Логирование агрегированных метрик
        summary = aggregated_results["summary"]
        mlflow.log_metric("experiments_count", aggregated_results["total_experiments"])
        mlflow.log_metric("best_r2_score", summary["best_r2"])
        mlflow.log_metric("mean_r2_score", summary["mean_r2"])
        mlflow.log_metric("std_r2_score", summary["std_r2"])
        mlflow.log_metric("worst_r2_score", summary["worst_r2"])

        # Создание визуализации результатов
        results_df = pd.read_csv(aggregated_results["results_path"])

        # График сравнения моделей
        fig, ax = plt.subplots(figsize=(12, 8))
        results_df.plot(x='run_id', y='r2_score', kind='bar', ax=ax)
        ax.set_title('Сравнение R² Score по моделям')
        ax.set_xlabel('Модель')
        ax.set_ylabel('R² Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = "/tmp/models_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path, "plots")

        # Теги для навигации
        mlflow.set_tag("experiment_type", "batch_comparison")
        mlflow.set_tag("source", "airflow_batch")
        mlflow.set_tag("algorithms_count", len(results_df))

        logger.success("✅ Сводный отчёт создан в MLflow")

        return {
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "summary_created": True
        }
```

### **3. Интеграция с внешними системами**

```python
# Настройка webhook'ов для внешних уведомлений
@task
def send_webhook_notification(results: dict) -> dict:
    """Отправка webhook уведомления во внешние системы."""

    import requests

    webhook_url = os.environ.get("WEBHOOK_URL")
    if not webhook_url:
        logger.warning("WEBHOOK_URL не настроен, пропускаем внешние уведомления")
        return {"status": "skipped"}

    payload = {
        "text": f"🤖 ML Pipeline завершён",
        "attachments": [{
            "color": "good" if results["summary"]["best_r2"] > 0.8 else "warning",
            "fields": [
                {
                    "title": "Экспериментов",
                    "value": str(results["total_experiments"]),
                    "short": True
                },
                {
                    "title": "Лучший R²",
                    "value": f"{results['summary']['best_r2']:.4f}",
                    "short": True
                },
                {
                    "title": "Лучшая модель",
                    "value": results["best_model"]["run_id"],
                    "short": False
                }
            ]
        }]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.success("✅ Webhook уведомление отправлено")
        return {"status": "sent", "response_code": response.status_code}
    except requests.RequestException as e:
        logger.warning(f"⚠️ Ошибка отправки webhook: {e}")
        return {"status": "error", "error": str(e)}
```

### 7.4 Тестирование воспроизводимости

**Система обеспечения воспроизводимости результатов:**

### **1. Детерминированность через фиксацию seed'ов**

```python
# Глобальная настройка random seed во всех экспериментах
def set_reproducible_environment(random_state: int = 42):
    """Настройка окружения для воспроизводимых результатов."""

    import random
    import numpy as np
    import os

    # Python random
    random.seed(random_state)

    # NumPy random  
    np.random.seed(random_state)

    # Scikit-learn по умолчанию
    os.environ['PYTHONHASHSEED'] = str(random_state)

    logger.info(f"🔒 Установлен random_state={random_state} для воспроизводимости")

# Применение во всех моделях
BASE_EXPERIMENT_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
}

# Все модели наследуют детерминированность
def create_model(model_name: str, params: dict):
    """Создание модели с принудительной установкой random_state."""

    # Принудительно добавляем random_state если модель его поддерживает
    if hasattr(model_class, '_check_n_features') and 'random_state' not in params:
        params['random_state'] = BASE_EXPERIMENT_CONFIG['random_state']

    return model_class(**params)
```

### **2. Кэширование для проверки идентичности**

```python
class ReproducibilityTester:
    """Класс для тестирования воспроизводимости результатов."""

    def __init__(self, cache: MinIOCache):
        self.cache = cache

    def test_model_reproducibility(
        self,
        model_name: str,
        params: dict,
        data_path: str,
        tolerance: float = 1e-6
    ) -> dict:
        """
        Тестирует воспроизводимость модели путём двойного обучения.

        Returns:
            dict с результатами теста воспроизводимости
        """
        logger.info(f"🧪 Тест воспроизводимости: {model_name}")

        results = []

        # Обучаем модель дважды с одинаковыми параметрами
        for run_num in [1, 2]:
            logger.info(f"   Запуск #{run_num}")

            # Обучение
            model = create_model(model_name, params)
            X_train, X_test, y_train, y_test = load_and_split_data(data_path)

            set_reproducible_environment(params.get('random_state', 42))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "predictions_hash": hashlib.md5(y_pred.tobytes()).hexdigest()
            }

            results.append(metrics)

        # Сравнение результатов
        r2_diff = abs(results[0]["r2_score"] - results[1]["r2_score"])
        rmse_diff = abs(results[0]["rmse"] - results[1]["rmse"])
        predictions_identical = results[0]["predictions_hash"] == results[1]["predictions_hash"]

        is_reproducible = (
            r2_diff < tolerance and
            rmse_diff < tolerance and
            predictions_identical
        )

        test_result = {
            "model_name": model_name,
            "is_reproducible": is_reproducible,
            "r2_difference": r2_diff,
            "rmse_difference": rmse_diff,
            "predictions_identical": predictions_identical,
            "tolerance": tolerance,
            "run1_results": results[0],
            "run2_results": results[1],
        }

        if is_reproducible:
            logger.success(f"✅ Модель {model_name} воспроизводима")
        else:
            logger.warning(f"⚠️ Модель {model_name} НЕ воспроизводима")
            logger.warning(f"   R² разность: {r2_diff}")
            logger.warning(f"   RMSE разность: {rmse_diff}")

        return test_result

# Интеграция в Airflow DAG
@task
def test_reproducibility() -> dict:
    """Task для тестирования воспроизводимости ключевых моделей."""

    cache = MinIOCache(bucket_name="reproducibility-tests")
    tester = ReproducibilityTester(cache)

    # Тестируем топ-3 модели
    test_models = [
        {"name": "random_forest", "params": {"n_estimators": 100, "random_state": 42}},
        {"name": "gradient_boosting", "params": {"n_estimators": 100, "random_state": 42}},
        {"name": "ridge", "params": {"alpha": 1.0, "random_state": 42}},
    ]

    test_results = []
    for model_config in test_models:
        result = tester.test_model_reproducibility(
            model_config["name"],
            model_config["params"],
            "/opt/airflow/data/raw/housing.csv"
        )
        test_results.append(result)

    # Сводная статистика
    reproducible_count = sum(1 for r in test_results if r["is_reproducible"])
    total_count = len(test_results)

    summary = {
        "total_tests": total_count,
        "reproducible_count": reproducible_count,
        "reproducibility_rate": reproducible_count / total_count,
        "test_results": test_results,
    }

    logger.info(f"🧪 Тесты воспроизводимости: {reproducible_count}/{total_count}")

    return summary
```

### **3. Версионирование данных через DVC**

```python
@task
def validate_data_version() -> dict:
    """Проверка версии данных для обеспечения воспроизводимости."""

    import subprocess

    try:
        # Проверяем статус DVC
        result = subprocess.run(
            ["dvc", "status"],
            cwd="/opt/airflow",
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            if not result.stdout.strip():
                logger.success("✅ DVC: данные актуальны")
                data_status = "up_to_date"
            else:
                logger.warning("⚠️ DVC: обнаружены изменения в данных")
                logger.info(f"DVC статус:\n{result.stdout}")
                data_status = "modified"
        else:
            logger.error(f"❌ DVC ошибка: {result.stderr}")
            data_status = "error"

        # Получаем хэш текущих данных
        data_path = "/opt/airflow/data/raw/housing.csv"
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        return {
            "data_status": data_status,
            "data_hash": data_hash,
            "dvc_output": result.stdout,
        }

    except subprocess.TimeoutExpired:
        logger.error("❌ DVC команда превысила таймаут")
        return {"data_status": "timeout"}
    except FileNotFoundError:
        logger.warning("⚠️ DVC не установлен")
        return {"data_status": "dvc_not_found"}
```

---

## 📚 Документация проекта

Подробные руководства находятся в директории `docs/guides/`:

| Файл | Описание |
|------|----------|
| [`MLFLOW+DVC+MINIO.md`](../docs/guides/MLFLOW+DVC+MINIO.md) | Полное руководство по MLflow + DVC + MinIO |
| [`MINIO+DVC.md`](../docs/guides/MINIO+DVC.md) | Настройка MinIO и DVC |
| [`PRE-COMMIT.md`](../docs/guides/PRE-COMMIT.md) | Настройка pre-commit хуков |
| [`DOCKER.md`](../docs/guides/DOCKER.md) | Руководство по Docker и контейнеризации |
| [`ENV.md`](../docs/guides/ENV.md) | Настройка переменных окружения |
| [`EXPERIMENTS.md`](../docs/guides/EXPERIMENTS.md) | Руководство по проведению экспериментов |
| [`EXPERIMENTS-ADVANCED.md`](../docs/guides/EXPERIMENTS-ADVANCED.md) | Продвинутые техники экспериментов |
| [`TRACKING-INTEGRATION.md`](../docs/guides/TRACKING-INTEGRATION.md) | Интеграция системы трекинга |
| [`airflow_ml_pipeline.md`](../docs/guides/airflow_ml_pipeline.md) | Airflow ML пайплайны |

---

## 📝 Заключение

В рамках лабораторной работы выполнены все поставленные задачи:

### 1. Настройка выбранного инструмента трекинга

**Основные достижения:**
- Установлен MLflow 3.7.0+ с поддержкой аутентификации
- Развёрнуто облачное хранилище MinIO для артефактов
- Настроена двухуровневая аутентификация (Nginx + MLflow + MinIO)
- Проведено 19 экспериментов с различными алгоритмами
- Интегрирован MLflow в Python код (5 декораторов, 2 контекстных менеджера, 9 утилит)

### 2. Настройка выбранного инструмента оркестрации

**✅ Установлен и настроен Apache Airflow 2.8.1**
- Развёрнут через Docker Compose с CeleryExecutor
- Настроена интеграция с PostgreSQL (метаданные) и Redis (брокер)
- Интегрирован с MLflow и MinIO

**✅ Создан workflow для ML пайплайна**
- Реализовано 3 DAG с различными паттернами:
  - `boston_housing_simple` — последовательный пайплайн для прототипирования
  - `boston_housing_experiments` — параллельное обучение 19 моделей
  - `boston_housing_cached` — пайплайн с интеллектуальным кэшированием

**✅ Настроены зависимости между этапами**
- Линейные зависимости (последовательное выполнение)
- Параллельные зависимости (через `expand()` operator)
- Условные зависимости (через `ShortCircuitOperator`)
- Trigger rules для гибкой логики запуска

**✅ Реализовано кэширование и параллельное выполнение**
- Система кэширования на базе MinIO с проверкой MD5 хэшей
- Параллелизм до 8 задач одновременно
- Автоматическая инвалидация кэша при изменении данных или параметров

### 3. Настройка выбранного инструмента конфигураций

**✅ Настроен инструмент для управления конфигурациями**
- Реализована система на базе **Hydra + Pydantic**
- Hydra обеспечивает композицию YAML-конфигураций и CLI переопределение
- Pydantic гарантирует строгую валидацию типов и параметров

**✅ Созданы конфигурации для разных алгоритмов**
- 14 YAML-конфигураций в `conf/model/`:
  - 6 линейных моделей (Linear Regression, Ridge, Lasso, ElasticNet, Huber, SGD)
  - 5 ансамблевых моделей (Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging)
  - 3 других алгоритма (Decision Tree, SVR, KNN)
- Каждая конфигурация содержит документированные параметры с значениями по умолчанию

**✅ Настроена валидация конфигураций**
- Реализованы Pydantic схемы в `src/schemas/`:
  - Строгая валидация типов для всех параметров
  - Ограничения диапазонов (`n_estimators >= 10`, `0 < alpha <= 1000`)
  - Кастомные валидаторы через `@field_validator`
  - Автоматическая документация через `Field(description=...)`

**✅ Создана система композиции конфигураций**
- Hydra композиция через `defaults` в `config.yaml`
- Группы экспериментов в `conf/experiment/`
- CLI переопределение: `model=ridge model.alpha=0.5`
- Multirun режим: `--multirun model=ridge,lasso,elastic_net`

### 4. Интеграция и тестирование

**✅ Интегрированы выбранные инструменты**
- Полная интеграция Airflow + MLflow + MinIO + Hydra
- Единые переменные окружения через `.env`
- Docker Compose оркестрация 10 сервисов
- Общая аутентификация через Nginx

**✅ Создана система мониторинга выполнения**
- Уровень 1: Структурированные логи через loguru
- Уровень 2: Мониторинг Airflow с callbacks (on_failure, on_success)
- Уровень 3: Мониторинг производительности в MLflow (CPU, память, время)

**✅ Настроены уведомления о результатах**
- Airflow UI Dashboard с rich-уведомлениями
- MLflow UI с автоматическими сводными отчётами
- Поддержка webhook'ов для внешних систем (Slack, Discord)

**✅ Протестирована воспроизводимость**
- Детерминированность через фиксацию random seeds
- Класс `ReproducibilityTester` для проверки идентичности результатов
- Версионирование данных через DVC
- Кэширование для проверки консистентности

### 5. Отчёт о проделанной работе

**✅ Создан подробный отчёт**
- Отчёт в формате Markdown с описанием всех компонентов
- Добавлены схемы архитектуры и примеры кода
- Отчёт обновлён на основе актуальной структуры проекта
- Сохранён в Git репозитории

**Используемый стек технологий:**

| Категория | Технология | Версия | Назначение |
|-----------|------------|--------|------------|
| **Язык и пакеты** | Python + uv | 3.13 | Основной язык разработки, пакетный менеджер |
| **Оркестрация** | Apache Airflow | 2.8.1 | Оркестрация ML пайплайнов, планировщик задач |
| **Трекинг** | MLflow | 3.7.0+ | Трекинг экспериментов, Model Registry |
| **Хранилище** | MinIO | latest | S3-совместимое хранилище артефактов и кэширование |
| **Версионирование данных** | DVC + DVCLive | 3.64.2+ | Версионирование данных и моделей, метрики |
| **База данных** | PostgreSQL | 15 | Метаданные Airflow |
| **Брокер сообщений** | Redis | 7-alpine | Celery broker для Airflow |
| **Контейнеризация** | Docker + Docker Compose | — | Развертывание инфраструктуры |
| **Обратный прокси** | Nginx | alpine | Аутентификация и маршрутизация |
| **ML библиотека** | scikit-learn | 1.7.2+ | 13 алгоритмов машинного обучения |
| **Конфигурации** | Hydra + Pydantic | 1.3.2+ / 2.0+ | Композиция YAML-конфигураций, валидация схем, CLI переопределение |
| **Логирование** | loguru | 0.7.3+ | Структурированное логирование |
| **Качество кода** | Ruff + pre-commit | 0.14.6+ | Линтинг и форматирование кода |
| **Безопасность** | detect-secrets | 1.5.0+ | Обнаружение секретов в коде |
| **Утилиты** | click, pandas, numpy | 2.3.3+ | CLI, обработка данных, вычисления |

---
