# 🔬 MLflow + DVC + MinIO: Полное руководство

Это руководство описывает интеграцию **MLflow** для трекинга экспериментов с **DVC** для версионирования данных и **MinIO** для хранения артефактов.

## 📋 Содержание

1. [Архитектура решения](#архитектура-решения)
2. [Установка и настройка](#установка-и-настройка)
3. [Настройка MinIO для MLflow](#настройка-minio-для-mlflow)
4. [Запуск MLflow Tracking Server](#запуск-mlflow-tracking-server)
5. [Интеграция MLflow с кодом](#интеграция-mlflow-с-кодом)
6. [Связка MLflow и DVC](#связка-mlflow-и-dvc)
7. [Workflow: полный цикл эксперимента](#workflow-полный-цикл-эксперимента)
8. [Примеры использования](#примеры-использования)
9. [Сравнение MLflow и DVCLive](#сравнение-mlflow-и-dvclive)
10. [Устранение неполадок](#устранение-неполадок)

---

## Архитектура решения

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
│                                          │  │ mlflow-artifacts│  │   │
│  ┌──────────────┐                        │  │  └─ models/     │  │   │
│  │     DVC      │───────────────────────▶│  │  └─ metrics/    │  │   │
│  │  (Версии     │                        │  └────────────────┘  │   │
│  │   данных)    │                        │                      │   │
│  └──────────────┘                        │  ┌────────────────┐  │   │
│        │                                 │  │boston-housing- │  │   │
│        │                                 │  │     data       │  │   │
│        ▼                                 │  │  └─ raw/       │  │   │
│  ┌──────────────┐                        │  │  └─ models/    │  │   │
│  │     Git      │                        │  └────────────────┘  │   │
│  │  (.dvc файлы,│                        └──────────────────────┘   │
│  │   метаданные)│                                                   │
│  └──────────────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Разделение обязанностей

| Компонент | Назначение |
|-----------|------------|
| **MLflow** | Трекинг экспериментов, метрик, параметров, UI для сравнения |
| **DVC** | Версионирование больших файлов данных и моделей |
| **MinIO** | S3-совместимое хранилище для артефактов MLflow и данных DVC |
| **Git** | Версионирование кода, `.dvc` файлов, конфигураций |

---

## Установка и настройка

### Шаг 1: Установка зависимостей

Добавьте MLflow и boto3 в `pyproject.toml`:

```bash
# Через uv
uv add mlflow boto3

# Или через pip
pip install mlflow boto3
```

Обновлённый `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... существующие зависимости ...
    "mlflow>=2.18.0",
    "boto3>=1.35.0",
]
```

### Шаг 2: Обновление docker-compose.yml

Добавьте сервис MLflow в `docker-compose.yml`:

```yaml
services:
  # ... существующие сервисы (minio) ...

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.18.0
    container_name: boston_housing_mlflow
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
      - AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://mlflow-artifacts/
    volumes:
      - mlflow_data:/mlflow
    depends_on:
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - boston_housing_network

volumes:
  mlflow_data:

networks:
  boston_housing_network:
    driver: bridge
```

### Шаг 3: Создание файла переменных окружения

Создайте/обновите файл `.env` в корне проекта:

```bash
# MinIO
MINIO_ROOT_USER=minioadmin0
MINIO_ROOT_PASSWORD=minioadmin1230

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin0
AWS_SECRET_ACCESS_KEY=minioadmin1230
```

---

## Настройка MinIO для MLflow

### Шаг 1: Запуск MinIO

```bash
docker-compose up -d minio
```

### Шаг 2: Создание бакета для артефактов MLflow

#### Через веб-консоль (http://localhost:9001):

1. Войдите с учётными данными: `minioadmin0` / `minioadmin1230`
2. Перейдите в **Buckets** → **Create Bucket**
3. Создайте бакет: `mlflow-artifacts`

#### Через MinIO Client:

```bash
# Настройка алиаса
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230

# Создание бакета для MLflow
mc mb local/mlflow-artifacts

# Проверка
mc ls local
# Ожидаемый вывод:
# [2024-XX-XX XX:XX:XX]     0B boston-housing-data/
# [2024-XX-XX XX:XX:XX]     0B mlflow-artifacts/
```

### Шаг 3: Настройка политик доступа (опционально)

Для production-окружения рекомендуется создать отдельного пользователя:

```bash
# Создание пользователя для MLflow
mc admin user add local mlflow_user mlflow_secret_password

# Создание политики доступа
cat > /tmp/mlflow-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::mlflow-artifacts",
        "arn:aws:s3:::mlflow-artifacts/*"
      ]
    }
  ]
}
EOF

# Применение политики
mc admin policy create local mlflow-policy /tmp/mlflow-policy.json
mc admin policy attach local mlflow-policy --user mlflow_user
```

---

## Запуск MLflow Tracking Server

### Вариант 1: Через Docker Compose (рекомендуется)

```bash
# Запуск MinIO и MLflow
docker-compose up -d minio mlflow

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f mlflow
```

### Вариант 2: Локальный запуск (для разработки)

```bash
# Экспорт переменных окружения
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin0
export AWS_SECRET_ACCESS_KEY=minioadmin1230

# Запуск MLflow сервера
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://mlflow-artifacts/
```

### Проверка запуска

После запуска MLflow UI доступен по адресу: **http://localhost:5000**

```bash
# Проверка здоровья сервера
curl http://localhost:5000/health
# Ожидаемый ответ: OK

# Проверка API
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

---

## Интеграция MLflow с кодом

### Шаг 1: Создание конфигурации MLflow

Создайте файл `src/config/mlflow_config.py`:

```python
"""Конфигурация MLflow для проекта."""

import os
from pathlib import Path


# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "boston-housing")

# MinIO/S3 для артефактов
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin0")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin1230")

# Artifact paths
ARTIFACT_BUCKET = "mlflow-artifacts"


def setup_mlflow_env():
    """Настройка переменных окружения для MLflow + S3."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
```

### Шаг 2: Создание обёртки для MLflow

Создайте файл `src/tracking/mlflow_tracker.py`:

```python
"""MLflow трекер для экспериментов."""

import pickle
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models.signature import infer_signature
from loguru import logger

from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    setup_mlflow_env,
)


class MLflowExperimentTracker:
    """Класс для трекинга ML экспериментов через MLflow."""
    
    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        """
        Инициализация трекера.
        
        Args:
            experiment_name: Название эксперимента в MLflow
            tracking_uri: URI MLflow Tracking Server
        """
        # Настройка окружения для S3
        setup_mlflow_env()
        
        # Подключение к MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Создание/получение эксперимента
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.run = None
        
        logger.info(f"MLflow трекер инициализирован: {tracking_uri}")
        logger.info(f"Эксперимент: {experiment_name}")
    
    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Начало нового запуска эксперимента."""
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Запущен эксперимент: {self.run.info.run_id}")
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
        """Логирование параметров эксперимента."""
        mlflow.log_params(params)
        logger.debug(f"Залогированы параметры: {list(params.keys())}")
    
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
    
    def log_model(
        self,
        model,
        artifact_path: str,
        input_example=None,
        registered_model_name: str | None = None,
    ):
        """
        Логирование модели sklearn.
        
        Args:
            model: Обученная модель
            artifact_path: Путь в хранилище артефактов
            input_example: Пример входных данных для сигнатуры
            registered_model_name: Имя для регистрации в Model Registry
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
        logger.info(f"Модель залогирована: {artifact_path}")
        
        if registered_model_name:
            logger.info(f"Модель зарегистрирована: {registered_model_name}")
    
    def log_figure(self, figure, artifact_file: str):
        """Логирование matplotlib/plotly фигуры."""
        mlflow.log_figure(figure, artifact_file)
    
    def set_tags(self, tags: dict[str, str]):
        """Установка тегов для запуска."""
        mlflow.set_tags(tags)
    
    @property
    def run_id(self) -> str | None:
        """ID текущего запуска."""
        return self.run.info.run_id if self.run else None
    
    @property
    def artifact_uri(self) -> str | None:
        """URI артефактов текущего запуска."""
        return self.run.info.artifact_uri if self.run else None
```

### Шаг 3: Обновление скрипта обучения

Создайте `src/modeling/train_mlflow.py`:

```python
"""
Обучение модели Random Forest с трекингом через MLflow.
"""

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR, RAW_DATA_DIR, HOUSING_DATA_FILE
from src.tracking.mlflow_tracker import MLflowExperimentTracker


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Загрузка данных Boston Housing."""
    logger.info(f"Загрузка данных из {data_path}")
    
    df = pd.read_csv(data_path, sep=r"\s+", header=None)
    
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
    ]
    df.columns = column_names
    
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    
    logger.info(f"Загружено {len(df)} записей, {len(X.columns)} признаков")
    return X, y


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Оценка модели и расчёт метрик."""
    y_pred = model.predict(X_test)
    
    return {
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
    }


@click.command()
@click.option("--n-estimators", "-n", default=100, type=int)
@click.option("--max-depth", "-d", default=10, type=int)
@click.option("--min-samples-split", "-s", default=5, type=int)
@click.option("--min-samples-leaf", "-l", default=2, type=int)
@click.option("--test-size", "-t", default=0.2, type=float)
@click.option("--random-state", "-r", default=42, type=int)
@click.option("--run-name", default=None, type=str, help="Имя запуска в MLflow")
@click.option("--register-model", is_flag=True, help="Зарегистрировать модель в Model Registry")
def main(
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    test_size: float,
    random_state: int,
    run_name: str | None,
    register_model: bool,
):
    """Обучение модели Random Forest с MLflow трекингом."""
    
    actual_max_depth = None if max_depth == 0 else max_depth
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": actual_max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "test_size": test_size,
    }
    
    data_file = RAW_DATA_DIR / HOUSING_DATA_FILE
    
    if not data_file.exists():
        logger.error(f"Файл данных не найден: {data_file}")
        logger.info("Выполните 'dvc pull' для загрузки данных из MinIO")
        raise click.Abort()
    
    # Инициализация MLflow трекера
    tracker = MLflowExperimentTracker()
    
    with tracker.start_run(run_name=run_name):
        # Теги для идентификации
        tracker.set_tags({
            "model_type": "RandomForest",
            "framework": "sklearn",
            "dataset": "boston_housing",
        })
        
        # Логирование параметров
        tracker.log_params(params)
        
        # Загрузка данных
        X, y = load_data(data_file)
        tracker.log_params({
            "n_samples": len(X),
            "n_features": len(X.columns),
        })
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        tracker.log_params({
            "train_size": len(X_train),
            "test_size_actual": len(X_test),
        })
        
        # Обучение модели
        logger.info("Обучение модели Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=actual_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logger.success("Модель обучена!")
        
        # Оценка модели
        metrics = evaluate_model(model, X_test, y_test)
        tracker.log_metrics(metrics)
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Сохраняем важность признаков как артефакт
        importance_path = Path("feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        tracker.log_artifact(importance_path)
        importance_path.unlink()  # Удаляем временный файл
        
        # Логирование модели в MLflow
        model_name = "boston-housing-rf" if register_model else None
        tracker.log_model(
            model,
            artifact_path="model",
            input_example=X_test.head(5),
            registered_model_name=model_name,
        )
        
        # Также сохраняем локально для DVC
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "random_forest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.success(f"Модель сохранена локально: {model_path}")
        
        # Итоговый вывод
        logger.info("\n" + "=" * 50)
        logger.info("📈 ИТОГОВЫЕ МЕТРИКИ:")
        logger.info(f"  R² Score:  {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE:      {metrics['rmse']:.4f}")
        logger.info(f"  MAE:       {metrics['mae']:.4f}")
        logger.info(f"  MAPE:      {metrics['mape']:.2f}%")
        logger.info("=" * 50)
        logger.info(f"\n🔗 MLflow Run ID: {tracker.run_id}")
        logger.info(f"📁 Artifacts: {tracker.artifact_uri}")


if __name__ == "__main__":
    main()
```

---

## Связка MLflow и DVC

### Философия интеграции

| Что храним | Где храним | Почему |
|------------|------------|--------|
| **Данные** | DVC → MinIO (`boston-housing-data`) | Версионирование больших файлов, связь с Git |
| **Метрики/параметры** | MLflow Tracking Server | Быстрый поиск, сравнение, UI |
| **Артефакты моделей** | MLflow → MinIO (`mlflow-artifacts`) | Автоматическое сохранение, Model Registry |
| **Версии моделей (production)** | DVC → MinIO | Явное версионирование, воспроизводимость |

### Рекомендуемый workflow

```bash
# 1. Загрузка данных через DVC
dvc pull

# 2. Обучение с трекингом в MLflow
python src/modeling/train_mlflow.py -n 200 -d 15 --run-name "baseline-v1"

# 3. Анализ результатов в MLflow UI
# http://localhost:5000

# 4. Если модель хорошая - сохраняем через DVC
dvc add data/models/random_forest.pkl
git add data/models/random_forest.pkl.dvc
git commit -m "model: RF n=200 d=15, R²=0.89"
dvc push

# 5. (Опционально) Регистрируем в MLflow Model Registry
python src/modeling/train_mlflow.py --register-model
```

### Автоматизация связки (скрипт)

Создайте `scripts/run_experiment.sh`:

```bash
#!/bin/bash
set -e

# Параметры эксперимента
N_ESTIMATORS=${1:-100}
MAX_DEPTH=${2:-10}
RUN_NAME=${3:-"experiment"}

echo "🚀 Запуск эксперимента: $RUN_NAME"
echo "   n_estimators=$N_ESTIMATORS, max_depth=$MAX_DEPTH"

# 1. Убедиться что данные актуальны
echo "📥 Проверка данных DVC..."
dvc pull

# 2. Запустить обучение с MLflow
echo "🔬 Обучение модели..."
python src/modeling/train_mlflow.py \
    -n $N_ESTIMATORS \
    -d $MAX_DEPTH \
    --run-name "$RUN_NAME"

# 3. Спросить пользователя о сохранении
read -p "💾 Сохранить модель в DVC? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    dvc add data/models/random_forest.pkl
    git add data/models/random_forest.pkl.dvc
    git commit -m "model: $RUN_NAME (n=$N_ESTIMATORS, d=$MAX_DEPTH)"
    dvc push
    echo "✅ Модель сохранена в DVC"
fi

echo "🎉 Эксперимент завершён!"
```

---

## Workflow: полный цикл эксперимента

### Шаг 1: Подготовка инфраструктуры

```bash
# Запуск MinIO и MLflow
docker-compose up -d minio mlflow

# Проверка сервисов
docker-compose ps

# Создание бакетов (если ещё не созданы)
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data --ignore-existing
mc mb local/mlflow-artifacts --ignore-existing
```

### Шаг 2: Загрузка данных

```bash
# Загрузка данных из DVC
dvc pull

# Проверка
ls -la data/raw/
```

### Шаг 3: Запуск эксперимента

```bash
# Базовый эксперимент
python src/modeling/train_mlflow.py --run-name "baseline"

# Эксперимент с другими параметрами
python src/modeling/train_mlflow.py \
    -n 200 -d 15 -s 10 \
    --run-name "deep-forest"
```

### Шаг 4: Анализ в MLflow UI

1. Откройте http://localhost:5000
2. Выберите эксперимент `boston-housing`
3. Сравните метрики разных запусков
4. Выберите лучшую модель

### Шаг 5: Сохранение лучшей модели

```bash
# Добавление модели в DVC
dvc add data/models/random_forest.pkl

# Коммит метаданных
git add data/models/random_forest.pkl.dvc
git commit -m "model: best RF (R²=0.89, n=200, d=15)"

# Отправка в MinIO
dvc push
```

### Шаг 6: Регистрация в Model Registry (опционально)

```bash
# Повторный запуск с регистрацией
python src/modeling/train_mlflow.py \
    -n 200 -d 15 \
    --run-name "production-candidate" \
    --register-model
```

В MLflow UI появится зарегистрированная модель в разделе **Models**.

---

## Примеры использования

### Пример 1: Быстрый эксперимент

```bash
# Минимальная модель для проверки пайплайна
python src/modeling/train_mlflow.py -n 10 -d 5 --run-name "quick-test"
```

### Пример 2: Grid Search с MLflow

```python
"""Grid search с логированием в MLflow."""

import itertools
from src.tracking.mlflow_tracker import MLflowExperimentTracker

# Параметры для поиска
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

# Генерация комбинаций
combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

tracker = MLflowExperimentTracker(experiment_name="grid-search")

for i, combo in enumerate(combinations):
    params = dict(zip(param_names, combo))
    
    with tracker.start_run(run_name=f"grid-{i:03d}"):
        tracker.log_params(params)
        
        # Обучение и оценка модели
        # ... код обучения ...
        
        tracker.log_metrics(metrics)
```

### Пример 3: Загрузка модели из MLflow

```python
import mlflow

# Загрузка по Run ID
run_id = "abc123..."
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Загрузка из Model Registry
model = mlflow.sklearn.load_model("models:/boston-housing-rf/Production")

# Предсказание
predictions = model.predict(X_new)
```

### Пример 4: Сравнение экспериментов через API

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Получение всех запусков эксперимента
experiment = client.get_experiment_by_name("boston-housing")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2_score DESC"],
    max_results=10,
)

# Вывод топ-10 моделей
print("🏆 Топ-10 моделей по R² Score:")
for run in runs:
    r2 = run.data.metrics.get("r2_score", 0)
    n_est = run.data.params.get("n_estimators", "?")
    print(f"  {run.info.run_id[:8]}... R²={r2:.4f}, n_estimators={n_est}")
```

---

## Сравнение MLflow и DVCLive

| Аспект | MLflow | DVCLive |
|--------|--------|---------|
| **UI** | Полнофункциональный веб-интерфейс | Статические HTML-отчёты |
| **Сравнение** | Встроенное сравнение экспериментов | Через `dvc exp show` |
| **Model Registry** | ✅ Полноценный реестр моделей | ❌ Нет (используйте DVC) |
| **Интеграция с Git** | Отдельная система | Тесная интеграция |
| **Масштабируемость** | Серверная архитектура | Файловое хранение |
| **Сложность** | Требует сервер | Работает локально |
| **Артефакты** | S3/GCS/Azure/local | Через DVC remote |

### Когда использовать что

**Используйте MLflow если:**
- Нужен удобный UI для сравнения экспериментов
- Работаете в команде и нужен централизованный сервер
- Нужен Model Registry для управления версиями моделей
- Планируете интеграцию с deployment системами

**Используйте DVCLive если:**
- Простой проект с небольшим числом экспериментов
- Нужна тесная интеграция с Git
- Не хотите поднимать дополнительные сервисы
- Фокус на воспроизводимости через Git

**Используйте оба:**
- MLflow для трекинга экспериментов и метрик
- DVC для версионирования данных и финальных моделей

---

## Устранение неполадок

### MLflow не может подключиться к MinIO

**Симптом:**
```
botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL
```

**Решения:**

```bash
# 1. Проверьте, запущен ли MinIO
docker ps | grep minio

# 2. Проверьте переменные окружения
echo $MLFLOW_S3_ENDPOINT_URL
echo $AWS_ACCESS_KEY_ID

# 3. Проверьте сетевое подключение
curl http://localhost:9000/minio/health/live

# 4. Если MLflow в Docker — используйте имя сервиса
# В docker-compose: http://minio:9000 (не localhost!)
```

### Бакет не найден

**Симптом:**
```
botocore.exceptions.ClientError: Bucket does not exist
```

**Решение:**

```bash
# Создайте бакет
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/mlflow-artifacts
```

### MLflow UI не открывается

**Симптом:** http://localhost:5000 недоступен

**Решения:**

```bash
# 1. Проверьте статус контейнера
docker-compose ps mlflow

# 2. Просмотрите логи
docker-compose logs mlflow

# 3. Проверьте порт
netstat -tlnp | grep 5000

# 4. Перезапустите сервис
docker-compose restart mlflow
```

### Ошибка при логировании модели

**Симптом:**
```
mlflow.exceptions.MlflowException: Model registry features are not supported
```

**Решение:**
Model Registry требует backend store на базе БД (не файловой системы):

```bash
# Используйте SQLite или PostgreSQL
mlflow server --backend-store-uri sqlite:///mlflow.db ...
```

### Конфликт портов

**Симптом:** Порт 5000 или 9000 уже занят

**Решение:**

```yaml
# docker-compose.yml - измените порты
services:
  mlflow:
    ports:
      - "5001:5000"  # MLflow на порту 5001
  minio:
    ports:
      - "9002:9000"  # MinIO API на порту 9002
```

Обновите `.env`:
```bash
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_S3_ENDPOINT_URL=http://localhost:9002
```

---

## 📚 Полезные ссылки

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow with S3](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)
- [DVC Documentation](https://dvc.org/doc)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## ⚡ Быстрый старт (TL;DR)

```bash
# 1. Установка зависимостей
uv add mlflow boto3

# 2. Запуск инфраструктуры
docker-compose up -d minio mlflow

# 3. Создание бакета для MLflow
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/mlflow-artifacts

# 4. Экспорт переменных (для локального запуска)
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin0
export AWS_SECRET_ACCESS_KEY=minioadmin1230

# 5. Запуск эксперимента
python src/modeling/train_mlflow.py --run-name "my-experiment"

# 6. Просмотр результатов
# Откройте http://localhost:5000

# 7. Сохранение модели в DVC
dvc add data/models/random_forest.pkl
git add data/models/random_forest.pkl.dvc
git commit -m "model: добавлена модель из эксперимента"
dvc push

# Готово! 🎉
```

---

## 🔧 Финальная структура проекта

```
ipml_boston_housing/
├── .dvc/
│   ├── config              # DVC remote config (MinIO)
│   └── config.local        # Credentials (не в git)
├── data/
│   ├── raw/                # Данные (под DVC)
│   ├── models/             # Модели (под DVC)
│   ├── raw.dvc             # DVC метаданные
│   └── models.dvc
├── docker/
│   └── Dockerfile.minio
├── src/
│   ├── config/
│   │   └── mlflow_config.py    # Конфиг MLflow
│   ├── modeling/
│   │   ├── train.py            # Обучение с DVCLive
│   │   └── train_mlflow.py     # Обучение с MLflow
│   └── tracking/
│       └── mlflow_tracker.py   # MLflow обёртка
├── docker-compose.yml      # MinIO + MLflow
├── .env                    # Переменные окружения
└── pyproject.toml          # Зависимости
```

