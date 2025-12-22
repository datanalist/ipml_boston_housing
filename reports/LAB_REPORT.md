# 📊 Отчёт: Трекинг ML-экспериментов

**Проект:** Boston Housing Price Prediction  
**Дата:** 20 декабря 2025  
**Автор:** Студенческий проект по IPML  

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
| **MLflow** | 2.18.0+ | Трекинг экспериментов, Model Registry |
| **MinIO** | latest | S3-совместимое хранилище артефактов |
| **DVC** | 3.64.2+ | Версионирование данных |
| **Nginx** | alpine | Reverse proxy с Basic Auth |
| **Docker** | — | Контейнеризация инфраструктуры |

**Установка зависимостей:**

```bash
# Через uv (пакетный менеджер)
uv add mlflow boto3

# Обновление pyproject.toml
```

**Файл `pyproject.toml`:**

```toml
[project]
dependencies = [
    "mlflow>=2.18.0",
    "boto3>=1.35.0",
    "dvc[s3]>=3.64.2",
    # ... другие зависимости
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

![MinIO Object Browser](figures/minio.png)

*Рис. 1: Веб-консоль MinIO с бакетами для хранения данных DVC и артефактов MLflow*

### 1.3 Создание проекта и экспериментов

**Запуск инфраструктуры:**s

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

| Сервис | URL | Описание |
|--------|-----|----------|
| MLflow UI | http://localhost:5000 | Веб-интерфейс MLflow |
| MinIO Console | http://localhost:9001 | Управление хранилищем |
| MinIO S3 API | http://localhost:9000 | S3 API для артефактов |

---

## 2. Проведение экспериментов

### 2.1 Проведение 15+ экспериментов с разными алгоритмами

Проведены эксперименты с различными алгоритмами машинного обучения:

| # | Алгоритм | Параметры | R² Score | RMSE |
|---|----------|-----------|----------|------|
| 1 | Random Forest | n=100, d=10 | 0.8512 | 3.30 |
| 2 | Random Forest | n=200, d=15 | 0.8665 | 3.13 |
| 3 | Random Forest | n=50, d=5 | 0.8234 | 3.60 |
| 4 | Random Forest | n=300, d=20 | 0.8701 | 3.08 |
| 5 | Gradient Boosting | n=100, d=5 | 0.8543 | 3.27 |
| 6 | Gradient Boosting | n=200, d=10 | 0.8712 | 3.07 |
| 7 | Ridge Regression | alpha=1.0 | 0.7234 | 4.50 |
| 8 | Lasso Regression | alpha=0.1 | 0.7156 | 4.57 |
| 9 | ElasticNet | alpha=0.5, l1_ratio=0.5 | 0.7089 | 4.62 |
| 10 | SVR | C=1.0, kernel=rbf | 0.7823 | 3.99 |
| 11 | KNN | n_neighbors=5 | 0.6512 | 5.05 |
| 12 | Decision Tree | d=10 | 0.7456 | 4.32 |
| 13 | AdaBoost | n=100 | 0.8123 | 3.71 |
| 14 | Bagging | n=50 | 0.8345 | 3.48 |
| 15 | Extra Trees | n=200, d=15 | 0.8623 | 3.18 |
| 16 | Huber Regressor | epsilon=1.35 | 0.7012 | 4.68 |

**Скриншот группировки экспериментов:**

![Groups](figures/groups.png)

*Рис. 2: Группировка экспериментов в MLflow UI*

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

**Скриншот метрики R² Score:**

![R2 Score](figures/r2_score.png)

*Рис. 3: Сравнение метрики R² Score для различных экспериментов*

**Скриншот времени обучения:**

![Train Time](figures/train_time_seconds.png)

*Рис. 4: Время обучения моделей в секундах*

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

**Скриншот фильтрации экспериментов:**

![Filters](figures/filters.png)

*Рис. 5: Интерфейс фильтрации и поиска экспериментов в MLflow UI*

---

## 3. Интеграция с кодом

### 3.1 Интеграция MLflow в Python код

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

    Пример использования:
        @mlflow_run(experiment_name="my-exp", run_name="baseline")
        def train_model(params):
            ...
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

    Пример:
        @log_params_decorator
        def train(n_estimators=100, max_depth=10):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Логируем все kwargs как параметры
        if kwargs:
            mlflow.log_params(kwargs)
        return func(*args, **kwargs)
    return wrapper


def log_metrics_decorator(metric_keys: list[str]):
    """
    Декоратор для автоматического логирования метрик из результата.

    Пример:
        @log_metrics_decorator(["r2_score", "rmse"])
        def evaluate(model, X, y) -> dict:
            return {"r2_score": 0.85, "rmse": 3.2}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            result = func(*args, **kwargs)

            if isinstance(result, dict):
                metrics_to_log = {
                    k: v for k, v in result.items()
                    if k in metric_keys and isinstance(v, (int, float))
                }
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)

            return result
        return wrapper
    return decorator
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
    """Контекстный менеджер для трекинга ML экспериментов."""

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
        """Поддержка контекстного менеджера."""
        if self.run is None:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение эксперимента."""
        mlflow.end_run()
        self.run = None

    def log_params(self, params: dict[str, Any]):
        """Логирование параметров."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """Логирование метрик."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None):
        """Логирование артефакта."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_model(self, model, artifact_path: str, input_example=None,
                  registered_model_name: str | None = None):
        """Логирование sklearn модели."""
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

    def set_tags(self, tags: dict[str, str]):
        """Установка тегов."""
        mlflow.set_tags(tags)

    @property
    def run_id(self) -> str | None:
        return self.run.info.run_id if self.run else None

    @property
    def artifact_uri(self) -> str | None:
        return self.run.info.artifact_uri if self.run else None
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

```python
"""Утилиты для работы с MLflow экспериментами."""

from typing import Any
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


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
        ascending: True для минимизации, False для максимизации

    Returns:
        Словарь с информацией о лучшем запуске
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

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
    }


def load_best_model(experiment_name: str, metric: str = "r2_score"):
    """Загрузка лучшей модели по метрике."""
    best_run = get_best_run(experiment_name, metric)
    if not best_run:
        raise ValueError(f"Нет запусков в эксперименте {experiment_name}")

    model_uri = f"runs:/{best_run['run_id']}/model"
    return mlflow.sklearn.load_model(model_uri)


def compare_runs(
    experiment_name: str,
    metrics: list[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Сравнение запусков эксперимента.

    Returns:
        DataFrame с метриками и параметрами
    """
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
        row = {"run_id": run.info.run_id[:8]}
        row.update({f"metric_{k}": v for k, v in run.data.metrics.items() if k in metrics})
        row.update({f"param_{k}": v for k, v in run.data.params.items()})
        data.append(row)

    return pd.DataFrame(data)


def register_best_model(
    experiment_name: str,
    model_name: str,
    metric: str = "r2_score",
) -> str:
    """
    Регистрация лучшей модели в Model Registry.

    Returns:
        Версия зарегистрированной модели
    """
    best_run = get_best_run(experiment_name, metric)
    if not best_run:
        raise ValueError(f"Нет запусков в эксперименте {experiment_name}")

    model_uri = f"runs:/{best_run['run_id']}/model"
    result = mlflow.register_model(model_uri, model_name)

    return result.version


def cleanup_old_runs(
    experiment_name: str,
    keep_top_n: int = 10,
    metric: str = "r2_score",
):
    """
    Удаление старых запусков, кроме топ-N по метрике.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Получаем все запуски отсортированные по метрике
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )

    # Удаляем все кроме топ-N
    for run in all_runs[keep_top_n:]:
        client.delete_run(run.info.run_id)
        print(f"Удалён run: {run.info.run_id}")
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
MINIO_ROOT_USER=minioadmin0
MINIO_ROOT_PASSWORD=minioadmin1230
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=secure_password_123
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=secure_password_123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin0
AWS_SECRET_ACCESS_KEY=minioadmin1230
EOF

# 4. Запуск инфраструктуры
docker-compose up -d minio mlflow nginx

# 5. Создание бакетов
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data
mc mb local/mlflow-artifacts

# 6. Загрузка данных
dvc pull

# 7. Запуск эксперимента
python src/modeling/train_mlflow.py --run-name "my-experiment"

# 8. Просмотр результатов
# Откройте http://localhost:5000
# Логин: admin / secure_password_123
```

### 4.2 Полезные команды

```bash
# MLflow
mlflow experiments search                # Список экспериментов
mlflow runs list --experiment-id 1       # Запуски эксперимента
mlflow models list                       # Зарегистрированные модели

# Docker
docker-compose ps                        # Статус контейнеров
docker-compose logs -f mlflow            # Логи MLflow
docker-compose restart mlflow nginx      # Перезапуск

# DVC
dvc status                               # Статус данных
dvc pull                                 # Загрузка данных
dvc push                                 # Отправка данных
```

---

## 5. Настройка инструмента оркестрации

### 5.1 Установка и настройка выбранного инструмента (Apache Airflow)

**Выбранный инструмент оркестрации:** Apache Airflow 2.8.0+

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

**Установка и настройка через Docker Compose:**

```dockerfile
# docker/Dockerfile.airflow
FROM apache/airflow:2.8.0-python3.11

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Настройка Python paths
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"
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

### 5.2 Создание workflow для ML пайплайна

Реализовано **3 DAG** с различными паттернами оркестрации:

#### DAG 1: boston_housing_simple

**Назначение:** Простой последовательный пайплайн для быстрого прототипирования.

**Архитектура:**
```
download_data → validate_data → train_model → evaluate_model → save_artifacts
```

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

**Назначение:** Параллельное обучение 19 ML моделей с агрегацией результатов.

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
```

**Параллельное выполнение:**
```python
# Линейные модели
linear_results = train_single_model.expand(
    model_config=LINEAR_MODELS,
    data_info=[data_info] * len(LINEAR_MODELS),
)

# Древовидные модели  
tree_results = train_single_model.expand(
    model_config=TREE_MODELS,
    data_info=[data_info] * len(TREE_MODELS),
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

|| Тип | Описание | Пример |
||------|----------|---------|
|| **Линейные** | Последовательное выполнение | `data >> validate >> train >> evaluate` |
|| **Параллельные** | Независимое выполнение групп задач | `expand()` для множественного обучения |
|| **Условные** | Выполнение зависит от результата предыдущих задач | `ShortCircuitOperator` для кэширования |
|| **Trigger Rules** | Гибкая логика запуска | `none_failed_min_one_success` |

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

#### Система кэширования на базе MinIO

**Модуль `airflow/plugins/minio_cache.py`:**

```python
class MinIOCache:
    """
    Класс для кэширования артефактов в MinIO.

    Позволяет:
    - Проверять существование файлов по хэш-ключам
    - Вычислять хэши для проверки изменений данных
    - Загружать/скачивать артефакты
    """

    def get_cache_key(self, prefix: str, params: dict, data_hash: str = None) -> str:
        """
        Генерирует ключ кэша на основе параметров и хэша данных.

        Формат: prefix_params_hash_data_hash
        Пример: "models/random_forest_a1b2c3d4_e5f6g7h8"
        """
        params_hash = self.compute_params_hash(params)
        if data_hash:
            return f"{prefix}_{params_hash}_{data_hash}"
        return f"{prefix}_{params_hash}"
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

### 6.1 Настройка выбранного инструмента для управления конфигурациями

**Выбранный подход:** Модульная система конфигураций на основе переменных окружения и Python модулей.

**Архитектура системы конфигураций:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Система конфигураций                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │    .env     │───▶│ os.environ  │───▶│  Python модули      │  │
│  │ (локально)  │    │ (runtime)   │    │ config/*.py         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        │                   │                        │           │
│        ▼                   ▼                        ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │docker-compose│    │  Airflow    │    │   ML Scripts       │  │
│  │    .yml     │    │   DAGs      │    │  (experiments)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Основные модули конфигурации:**

|| Модуль | Назначение | Содержание |
||--------|------------|------------|
|| `src/config.py` | Базовые пути проекта | Директории данных, путь к проекту, настройки логгера |
|| `src/config/mlflow_config.py` | MLflow и MinIO | URI трекинга, эндпоинты S3, ключи доступа |
|| `.env` | Секреты и переменные окружения | Пароли, ключи API, настройки БД |
|| `docker-compose.yml` | Инфраструктурные настройки | Порты, volumes, переменные для контейнеров |

### 6.2 Создание конфигураций для разных алгоритмов

**Централизованные конфигурации алгоритмов в DAG:**

```python
# Линейные модели (7 конфигураций)
LINEAR_MODELS = [
    {
        "name": "linear_regression",
        "params": {},
        "description": "Baseline линейная регрессия",
    },
    {"name": "ridge", "params": {"alpha": 0.1}, "description": "Ridge α=0.1"},
    {"name": "ridge", "params": {"alpha": 1.0}, "description": "Ridge α=1.0"},
    {"name": "ridge", "params": {"alpha": 10.0}, "description": "Ridge α=10.0"},
    {"name": "lasso", "params": {"alpha": 0.1}, "description": "Lasso α=0.1"},
    {
        "name": "elastic_net",
        "params": {"alpha": 0.5, "l1_ratio": 0.5},
        "description": "ElasticNet",
    },
    {"name": "huber", "params": {"epsilon": 1.35}, "description": "Huber Regressor"},
]

# Древовидные модели (9 конфигураций)
TREE_MODELS = [
    {
        "name": "decision_tree",
        "params": {"max_depth": 5},
        "description": "Decision Tree d=5",
    },
    {
        "name": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10},
        "description": "RF n=100",
    },
    {
        "name": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.1},
        "description": "GBM lr=0.1",
    },
    # ... ещё 6 конфигураций
]
```

**Фабрика моделей с валидацией:**

```python
def create_model(model_name: str, params: dict):
    """Создаёт модель по имени и параметрам с валидацией."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    models = {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        # ... остальные модели
    }

    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Неизвестная модель: {model_name}")

    # Валидация параметров перед созданием
    try:
        return model_class(**params)
    except TypeError as e:
        raise ValueError(f"Некорректные параметры для {model_name}: {e}")
```

### 6.3 Настройка валидации конфигураций

**Валидация на уровне Airflow tasks:**

```python
@task
def validate_data(data_path: str) -> dict:
    """Валидация данных и подготовка train/test split."""
    import pandas as pd
    from loguru import logger

    logger.info(f"🔍 Валидация данных: {data_path}")

    df = pd.read_csv(data_path, sep=r"\s+", header=None)

    # Проверки схемы данных
    expected_columns = 14  # 13 признаков + 1 целевая переменная
    assert len(df.columns) == expected_columns, (
        f"Ожидалось {expected_columns} колонок, получено {len(df.columns)}"
    )

    # Проверка пропущенных значений
    missing_values = df.isnull().sum().sum()
    assert missing_values == 0, f"Обнаружено {missing_values} пропущенных значений"

    # Валидация диапазонов
    target = df.iloc[:, -1]  # Последняя колонка - целевая
    assert target.min() > 0, "Целевая переменная должна быть положительной"
    assert target.max() < 100, "Целевая переменная превышает разумный максимум"

    logger.success(f"✅ Валидация пройдена: {len(df)} записей")

    return {
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }
```

**Валидация переменных окружения:**

```python
# src/config/mlflow_config.py
def setup_mlflow_env():
    """Настройка переменных окружения для MLflow + S3 с валидацией."""

    # Валидация обязательных переменных
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise EnvironmentError(
            f"Отсутствуют обязательные переменные окружения: {missing_vars}"
        )

    # Валидация форматов
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri.startswith(("http://", "https://")):
        raise ValueError(f"MLFLOW_TRACKING_URI должен начинаться с http:// или https://")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
```

### 6.4 Создание системы композиции конфигураций

**Наследование и переопределение конфигураций:**

```python
# Базовые настройки для всех экспериментов
BASE_EXPERIMENT_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": False,
}

# Специфичные настройки для семейств алгоритмов
TREE_MODEL_DEFAULTS = {
    **BASE_EXPERIMENT_CONFIG,
    "max_features": "sqrt",
    "bootstrap": True,
}

ENSEMBLE_MODEL_DEFAULTS = {
    **TREE_MODEL_DEFAULTS,
    "n_estimators": 100,
    "warm_start": False,
}

# Композиция финальных конфигураций
def compose_model_config(base_config: dict, model_specific: dict) -> dict:
    """Композиция конфигураций с приоритетом специфичных параметров."""

    # Глубокое слияние конфигураций
    result = base_config.copy()
    result.update(model_specific)

    # Логирование применённых настроек
    logger.info(f"Финальная конфигурация: {result}")

    return result

# Использование
random_forest_config = compose_model_config(
    ENSEMBLE_MODEL_DEFAULTS,
    {"max_depth": 10, "min_samples_split": 5}
)
```

**Переменные окружения с иерархией:**

```bash
# .env - базовые настройки
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=boston-housing

# Переопределение для разных окружений  
# .env.development
MLFLOW_TRACKING_URI=http://localhost:5000
DEBUG=true

# .env.production  
MLFLOW_TRACKING_URI=https://mlflow.production.com
DEBUG=false
BATCH_SIZE=128
```

**Конфигурационные классы с валидацией:**

```python
from pathlib import Path
from typing import Optional
import os

class ProjectConfig:
    """Центральная конфигурация проекта с валидацией."""

    def __init__(self):
        self.proj_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.proj_root / "data"
        self.models_dir = self.data_dir / "models"

        # Валидация путей
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

    @property
    def mlflow_tracking_uri(self) -> str:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        if not uri.startswith(("http://", "https://")):
            raise ValueError(f"Некорректный MLFLOW_TRACKING_URI: {uri}")
        return uri

    @property
    def experiment_name(self) -> str:
        return os.getenv("MLFLOW_EXPERIMENT_NAME", "boston-housing")

    def validate(self) -> bool:
        """Валидация всей конфигурации."""
        try:
            _ = self.mlflow_tracking_uri
            _ = self.experiment_name
            assert self.data_dir.exists()
            return True
        except (ValueError, AssertionError) as e:
            logger.error(f"Ошибка валидации конфигурации: {e}")
            return False

# Глобальный экземпляр
config = ProjectConfig()
```

---

## 7. Интеграция и тестирование систем

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

---

## 📝 Заключение

В рамках лабораторной работы выполнены все поставленные задачи:

### Настройка выбранного инструмента трекинга

1. **Установка и настройка MLflow**: Установлен MLflow 2.18.0+, развёрнуто облачное хранилище MinIO, создан проект и эксперименты, настроена двухуровневая аутентификация (Nginx Basic Auth + MLflow Auth + MinIO Access Keys)

2. **Проведение экспериментов**: Проведено 16 экспериментов с различными алгоритмами (Random Forest, Gradient Boosting, Ridge, Lasso, ElasticNet, SVR, KNN, Decision Tree, AdaBoost, Bagging, Extra Trees, Huber Regressor), настроено логирование метрик, параметров и артефактов, создана система сравнения экспериментов, настроена фильтрация и поиск

3. **Интеграция с кодом**: Интегрирован MLflow в Python код, созданы декораторы для автоматического логирования (`@mlflow_run`, `@log_params_decorator`, `@log_metrics_decorator`), настроены контекстные менеджеры (`MLflowExperimentTracker`), созданы утилиты для работы с экспериментами

### Настройка выбранного инструмента оркестрации

4. **Установка и настройка Apache Airflow**: Развернут Airflow 2.8.0+ через Docker Compose с CeleryExecutor, PostgreSQL, Redis, интегрирован с MLflow и MinIO

5. **Создание workflow для ML пайплайна**: Реализовано 3 DAG:
   - `boston_housing_simple`: простой последовательный пайплайн
   - `boston_housing_experiments`: параллельное обучение 19 моделей с агрегацией
   - `boston_housing_cached`: пайплайн с интеллектуальным кэшированием в MinIO

6. **Настройка зависимостей между этапами**: Реализованы линейные, параллельные и условные зависимости, trigger rules для гибкой логики запуска

7. **Реализация кэширования и параллельного выполнения**: Создана система кэширования на базе MinIO с проверкой хэшей данных и параметров, настроен параллелизм через expand() operator (до 8 задач одновременно)

### Настройка выбранного инструмента конфигураций

8. **Настройка инструмента управления конфигурациями**: Реализована модульная система конфигураций на основе переменных окружения (.env), Python модулей (config.py, mlflow_config.py) и Docker Compose

9. **Создание конфигураций для разных алгоритмов**: Созданы централизованные конфигурации для 19 алгоритмов (7 линейных моделей, 9 древовидных моделей, 3 других алгоритма) с фабрикой моделей

10. **Настройка валидации конфигураций**: Реализована валидация на уровне данных, параметров моделей, переменных окружения с подробным логированием ошибок

11. **Создание системы композиции конфигураций**: Настроено наследование и переопределение конфигураций, иерархия переменных окружения, конфигурационные классы с валидацией

### Интеграция и тестирование

12. **Интеграция выбранных инструментов**: Полная интеграция Airflow + MLflow + MinIO через единые переменные окружения, Docker Compose оркестрацию, общую аутентификацию

13. **Создание системы мониторинга выполнения**: Реализован многоуровневый мониторинг: структурированные логи (loguru), мониторинг Airflow с callbacks, мониторинг производительности в MLflow

14. **Настройка уведомлений о результатах**: Система уведомлений через Airflow UI, MLflow UI с автоматическим созданием сводных отчётов, интеграция с внешними системами через webhook'и

15. **Тестирование воспроизводимости**: Система обеспечения детерминированности через фиксацию random seeds, кэширование для проверки идентичности результатов, версионирование данных через DVC

### Отчёт о проделанной работе

16. **Создание отчёта**: Создан данный подробный отчёт в формате Markdown с описанием всех компонентов системы, добавлены схемы архитектуры и примеры кода, отчёт сохранён в Git репозитории

**Используемый стек технологий:**

|| Категория | Технология | Версия | Назначение |
||-----------|------------|--------|------------|
|| **Язык и пакеты** | Python + uv | 3.13 | Основной язык разработки, пакетный менеджер |
|| **Оркестрация** | Apache Airflow | 2.8.0+ | Оркестрация ML пайплайнов, планировщик задач |
|| **Трекинг** | MLflow | 2.18.0+ | Трекинг экспериментов, Model Registry |
|| **Хранилище** | MinIO | latest | S3-совместимое хранилище артефактов и кэширование |
|| **Конфигурации** | python-dotenv + модули | — | Управление конфигурациями через .env и Python |
|| **Версионирование данных** | DVC | 3.64.2+ | Версионирование данных и моделей |
|| **База данных** | PostgreSQL | 13 | Метаданные Airflow |
|| **Брокер сообщений** | Redis | 7 | Celery broker для Airflow |
|| **Контейнеризация** | Docker + Docker Compose | — | Развертывание инфраструктуры |
|| **Обратный прокси** | Nginx | alpine | Аутентификация и маршрутизация |
|| **ML библиотека** | scikit-learn | 1.3+ | Алгоритмы машинного обучения |
|| **Логирование** | loguru | — | Структурированное логирование |
|| **Качество кода** | Ruff + pre-commit | — | Линтинг и форматирование кода |

---
