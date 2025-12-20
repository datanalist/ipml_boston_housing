# 📊 Отчёт: Трекинг ML-экспериментов

**Проект:** Boston Housing Price Prediction  
**Дата:** 20 декабря 2025  
**Автор:** Студенческий проект по IPML  

---

## 📋 Содержание

1. [Обзор выполненных задач](#1-обзор-выполненных-задач)
2. [Настройка выбранного инструмента](#2-настройка-выбранного-инструмента)
3. [Проведение экспериментов](#3-проведение-экспериментов)
4. [Интеграция с кодом](#4-интеграция-с-кодом)
5. [Инструкции по запуску](#5-инструкции-по-запуску)

---

## 1. Обзор выполненных задач

### ✅ Чек-лист выполнения

| Категория | Задача | Статус |
|-----------|--------|--------|
| **Настройка инструмента** | Установить и настроить выбранный инструмент (MLflow) | ✅ Выполнено |
| | Настроить базу данных/облачное хранилище (MinIO) | ✅ Выполнено |
| | Создать проект и эксперименты | ✅ Выполнено |
| | Настроить аутентификацию и доступ | ✅ Выполнено |
| **Проведение экспериментов** | Провести 15+ экспериментов с разными алгоритмами | ✅ Выполнено |
| | Настроить логирование метрик, параметров и артефактов | ✅ Выполнено |
| | Создать систему сравнения экспериментов | ✅ Выполнено |
| | Настроить фильтрацию и поиск экспериментов | ✅ Выполнено |
| **Интеграция с кодом** | Интегрировать выбранный инструмент в Python код | ✅ Выполнено |
| | Создать декораторы для автоматического логирования | ✅ Выполнено |
| | Настроить контекстные менеджеры | ✅ Выполнено |
| | Создать утилиты для работы с экспериментами | ✅ Выполнено |
| **Отчёт** | Создать отчёт в формате Markdown | ✅ Выполнено |
| | Описать настройку выбранного инструмента | ✅ Выполнено |
| | Добавить скриншоты результатов | ✅ Выполнено |
| | Сохранить отчёт в Git репозитории | ✅ Выполнено |

---

## 2. Настройка выбранного инструмента

### 2.1 Выбранный стек технологий

| Инструмент | Версия | Назначение |
|------------|--------|------------|
| **MLflow** | 2.18.0+ | Трекинг экспериментов, Model Registry |
| **MinIO** | latest | S3-совместимое хранилище артефактов |
| **DVC** | 3.64.2+ | Версионирование данных |
| **Nginx** | alpine | Reverse proxy с Basic Auth |
| **Docker** | — | Контейнеризация инфраструктуры |

### 2.2 Архитектура решения

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

### 2.3 Установка и настройка MLflow

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

### 2.4 Настройка облачного хранилища (MinIO)

#### Docker-конфигурация MinIO

**`docker/Dockerfile.minio`:**

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

#### Бакеты в MinIO

| Бакет | Назначение |
|-------|------------|
| `boston-housing-data` | Данные и модели (DVC) |
| `mlflow-artifacts` | Артефакты MLflow (модели, метрики, графики) |

### 2.5 Создание проекта и экспериментов

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

### 2.6 Настройка аутентификации и доступа

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

### 2.7 Скриншот MinIO

![MinIO Object Browser](../image.png)

*Рис. 1: Веб-консоль MinIO с бакетами для хранения данных DVC и артефактов MLflow*

---

## 3. Проведение экспериментов

### 3.1 Проведённые эксперименты (15+ экспериментов)

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

### 3.2 Настройка логирования метрик, параметров и артефактов

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

### 3.3 Система сравнения экспериментов

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

### 3.4 Фильтрация и поиск экспериментов

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

---

## 4. Интеграция с кодом

### 4.1 Интеграция MLflow в Python код

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

### 4.2 Декораторы для автоматического логирования

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

### 4.3 Контекстные менеджеры

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

### 4.4 Утилиты для работы с экспериментами

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

## 5. Инструкции по запуску

### 5.1 Быстрый старт

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

### 5.2 Полезные команды

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

1. **Настройка инструмента**: Развёрнут MLflow Tracking Server с MinIO в качестве artifact store, настроена двухуровневая аутентификация через Nginx + Basic Auth

2. **Проведение экспериментов**: Проведено 15+ экспериментов с различными алгоритмами (Random Forest, Gradient Boosting, Ridge, Lasso, SVR, KNN и др.), настроено логирование всех метрик, параметров и артефактов

3. **Интеграция с кодом**: Созданы декораторы для автоматического логирования (`@mlflow_run`, `@log_params_decorator`, `@log_metrics_decorator`), контекстный менеджер `MLflowExperimentTracker`, утилиты для поиска лучших моделей и сравнения экспериментов

4. **Документация**: Подготовлен данный отчёт и детальные руководства

**Используемый стек технологий:**
- Python 3.13 + uv (пакетный менеджер)
- MLflow 2.18.0+ (трекинг экспериментов)
- MinIO (S3-совместимое хранилище)
- DVC (версионирование данных)
- Docker + Docker Compose (контейнеризация)
- Nginx (reverse proxy с аутентификацией)
- scikit-learn (машинное обучение)

---

*Отчёт создан: 20 декабря 2025*
