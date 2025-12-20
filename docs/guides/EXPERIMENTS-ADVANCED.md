# 🔬 Продвинутое руководство по экспериментам

Это руководство описывает проведение масштабных экспериментов с различными алгоритмами машинного обучения, настройку логирования и систему сравнения результатов.

## 📋 Содержание

1. [Обзор](#обзор)
2. [Доступные алгоритмы](#доступные-алгоритмы)
3. [Проведение 19 экспериментов](#проведение-19-экспериментов)
4. [Настройка логирования](#настройка-логирования)
5. [Система сравнения экспериментов](#система-сравнения-экспериментов)
6. [Фильтрация и поиск экспериментов](#фильтрация-и-поиск-экспериментов)
7. [Автоматизация экспериментов](#автоматизация-экспериментов)
8. [Best Practices](#best-practices)

---

## Обзор

В проекте доступно **14 алгоритмов регрессии** из scikit-learn, которые можно комбинировать с различными гиперпараметрами для создания **19 экспериментов**. Для трекинга используется связка:

| Инструмент | Назначение |
|------------|------------|
| **MLflow** | Централизованный трекинг экспериментов, UI для сравнения |
| **DVCLive** | Локальный трекинг с интеграцией в Git/DVC |
| **MinIO** | Хранение артефактов (модели, графики) |

---

## Доступные алгоритмы

### Линейные модели

| Алгоритм | Ключ | Описание | Ключевые параметры |
|----------|------|----------|-------------------|
| Linear Regression | `linear_regression` | Обычная линейная регрессия (МНК) | — |
| Ridge | `ridge` | L2-регуляризация | `alpha` |
| Lasso | `lasso` | L1-регуляризация | `alpha` |
| Elastic Net | `elastic_net` | L1+L2 регуляризация | `alpha`, `l1_ratio` |
| Huber Regressor | `huber` | Робастная регрессия | `epsilon` |
| SGD Regressor | `sgd` | Стохастический градиентный спуск | `learning_rate`, `max_iter` |

### Древовидные модели и ансамбли

| Алгоритм | Ключ | Описание | Ключевые параметры |
|----------|------|----------|-------------------|
| Decision Tree | `decision_tree` | Дерево решений | `max_depth` |
| Random Forest | `random_forest` | Случайный лес | `n_estimators`, `max_depth` |
| Extra Trees | `extra_trees` | Экстремально рандомизированные деревья | `n_estimators`, `max_depth` |
| Gradient Boosting | `gradient_boosting` | Градиентный бустинг | `n_estimators`, `learning_rate` |
| AdaBoost | `adaboost` | Адаптивный бустинг | `n_estimators`, `learning_rate` |
| Bagging | `bagging` | Бэггинг регрессор | `n_estimators` |

### Другие модели

| Алгоритм | Ключ | Описание | Ключевые параметры |
|----------|------|----------|-------------------|
| SVR | `svr` | Опорные вектора | `kernel`, `C`, `epsilon` |
| KNN | `knn` | K ближайших соседей | `n_neighbors`, `weights` |

---

## Проведение 19 экспериментов

### Скрипт для массовых экспериментов

Скрипт `scripts/run_experiments.py` содержит полную реализацию:

```python
"""
Скрипт для запуска множественных экспериментов с разными алгоритмами ML.

Логирует метрики, параметры и артефакты в MLflow.
"""

import os
import sys
import time
import pickle
import tempfile
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml_models.model_loader import create_model, MODEL_REGISTRY
from src.config import RAW_DATA_DIR, HOUSING_DATA_FILE, MODELS_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ 19 ЭКСПЕРИМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS_CONFIG = [
    # ─────────────────────────────────────────────────────────────────────────
    # Линейные модели (7 экспериментов)
    # ─────────────────────────────────────────────────────────────────────────
    {"name": "linear_regression", "params": {}, "description": "Baseline линейная регрессия"},
    {"name": "ridge", "params": {"alpha": 0.1}, "description": "Ridge со слабой регуляризацией"},
    {"name": "ridge", "params": {"alpha": 1.0}, "description": "Ridge стандартный"},
    {"name": "ridge", "params": {"alpha": 10.0}, "description": "Ridge с сильной регуляризацией"},
    {"name": "lasso", "params": {"alpha": 0.1}, "description": "Lasso для отбора признаков"},
    {"name": "elastic_net", "params": {"alpha": 0.5, "l1_ratio": 0.5}, "description": "Elastic Net комбинированный"},
    {"name": "huber", "params": {"epsilon": 1.35}, "description": "Huber робастная регрессия"},

    # ─────────────────────────────────────────────────────────────────────────
    # Древовидные модели и ансамбли (9 экспериментов)
    # ─────────────────────────────────────────────────────────────────────────
    {"name": "decision_tree", "params": {"max_depth": 5}, "description": "Дерево решений (shallow)"},
    {"name": "decision_tree", "params": {"max_depth": 10}, "description": "Дерево решений (deep)"},
    {"name": "random_forest", "params": {"n_estimators": 100, "max_depth": 10}, "description": "Random Forest стандартный"},
    {"name": "random_forest", "params": {"n_estimators": 200, "max_depth": 15}, "description": "Random Forest большой"},
    {"name": "extra_trees", "params": {"n_estimators": 100, "max_depth": 10}, "description": "Extra Trees"},
    {"name": "gradient_boosting", "params": {"n_estimators": 100, "learning_rate": 0.1}, "description": "Gradient Boosting стандартный"},
    {"name": "gradient_boosting", "params": {"n_estimators": 200, "learning_rate": 0.05}, "description": "Gradient Boosting медленный"},
    {"name": "adaboost", "params": {"n_estimators": 50, "learning_rate": 1.0}, "description": "AdaBoost"},
    {"name": "bagging", "params": {"n_estimators": 20}, "description": "Bagging регрессор"},

    # ─────────────────────────────────────────────────────────────────────────
    # Другие модели (3 эксперимента)
    # ─────────────────────────────────────────────────────────────────────────
    {"name": "svr", "params": {"kernel": "rbf", "C": 1.0}, "description": "SVR с RBF ядром"},
    {"name": "knn", "params": {"n_neighbors": 5, "weights": "uniform"}, "description": "KNN k=5 uniform"},
    {"name": "knn", "params": {"n_neighbors": 10, "weights": "distance"}, "description": "KNN k=10 distance"},
]


def load_data():
    """Загрузка и подготовка данных Boston Housing."""
    data_file = RAW_DATA_DIR / HOUSING_DATA_FILE

    if not data_file.exists():
        logger.error(f"Файл данных не найден: {data_file}")
        logger.info("Выполните 'dvc pull' для загрузки данных")
        sys.exit(1)

    df = pd.read_csv(data_file, sep=r"\s+", header=None)

    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    df.columns = column_names

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_algorithm_family(model_name: str) -> str:
    """Определение семейства алгоритма для тегирования."""
    linear_models = ["linear_regression", "ridge", "lasso", "elastic_net", "huber", "sgd"]
    tree_models = ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "adaboost", "bagging"]

    if model_name in linear_models:
        return "linear"
    elif model_name in tree_models:
        return "tree_ensemble"
    else:
        return "other"


def evaluate_model(model, X_test, y_test):
    """Оценка модели и расчёт метрик."""
    y_pred = model.predict(X_test)

    return {
        "r2_score": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }, y_pred


def run_single_experiment(config, X_train, X_test, y_train, y_test, experiment_idx, total_experiments):
    """Запуск одного эксперимента с полным логированием в MLflow."""

    model_name = config["name"]
    custom_params = config["params"]
    description = config.get("description", "")

    # Генерация уникального имени run
    param_str = "_".join([f"{k}={v}" for k, v in custom_params.items()])
    run_name = f"{model_name}_{param_str}" if param_str else model_name

    with mlflow.start_run(run_name=run_name):
        # Логирование параметров
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("model_description", MODEL_REGISTRY[model_name]["description"])
        mlflow.log_param("experiment_description", description)

        for param_name, param_value in custom_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("random_state", 42)

        # Обучение с замером времени
        model = create_model(model_name, custom_params)

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        metrics, y_pred = evaluate_model(model, X_test, y_test)

        # Логирование метрик
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_metric("train_time_seconds", train_time)

        # Логирование модели и артефактов
        mlflow.sklearn.log_model(model, "sklearn_model")

        # Теги
        mlflow.set_tag("algorithm_family", get_algorithm_family(model_name))
        mlflow.set_tag("experiment_type", "model_comparison")
        mlflow.set_tag("dataset", "boston_housing")

        logger.info(f"✅ {run_name}: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}")

        return {
            "run_name": run_name,
            "model_type": model_name,
            **metrics,
            "train_time": train_time
        }


def main():
    """Основная функция запуска экспериментов."""

    # Настройка MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("boston_housing_model_comparison")

    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data()
    logger.info(f"Данные загружены: train={len(X_train)}, test={len(X_test)}")

    # Запуск экспериментов
    results = []
    total = len(EXPERIMENTS_CONFIG)

    for i, config in enumerate(EXPERIMENTS_CONFIG, 1):
        logger.info(f"\n[{i}/{total}] Запуск: {config['name']}")
        result = run_single_experiment(config, X_train, X_test, y_train, y_test, i, total)
        results.append(result)

    # Сводка результатов
    results_df = pd.DataFrame(results).sort_values("r2_score", ascending=False)

    logger.info("\n" + "=" * 60)
    logger.info("📊 ТОП-5 МОДЕЛЕЙ ПО R² SCORE:")
    for _, row in results_df.head(5).iterrows():
        logger.info(f"  {row['run_name']}: R²={row['r2_score']:.4f}")

    # Сохранение результатов
    results_path = Path(__file__).parent.parent / "data" / "experiments" / "results_summary.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    logger.info(f"\n✅ Завершено {len(EXPERIMENTS_CONFIG)} экспериментов")
    logger.info(f"💾 Результаты сохранены: {results_path}")
    logger.info(f"🌐 Откройте MLflow UI: {mlflow_uri}")


if __name__ == "__main__":
    main()
```

### Запуск экспериментов

```bash
# 1. Убедитесь, что MLflow сервер запущен
docker-compose up -d mlflow

# 2. Настройте переменные окружения
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

# 3. Запустите эксперименты
python scripts/run_experiments.py
```

### План 19 экспериментов

| # | Алгоритм | Параметры | Цель эксперимента |
|---|----------|-----------|-------------------|
| 1 | Linear Regression | по умолчанию | Baseline линейная регрессия |
| 2 | Ridge | α=0.1 | Ridge со слабой регуляризацией |
| 3 | Ridge | α=1.0 | Ridge стандартный |
| 4 | Ridge | α=10.0 | Ridge с сильной регуляризацией |
| 5 | Lasso | α=0.1 | Lasso для отбора признаков |
| 6 | Elastic Net | α=0.5, l1_ratio=0.5 | Elastic Net комбинированный |
| 7 | Huber | ε=1.35 | Huber робастная регрессия |
| 8 | Decision Tree | depth=5 | Дерево решений (shallow) |
| 9 | Decision Tree | depth=10 | Дерево решений (deep) |
| 10 | Random Forest | n=100, depth=10 | Random Forest стандартный |
| 11 | Random Forest | n=200, depth=15 | Random Forest большой |
| 12 | Extra Trees | n=100, depth=10 | Extra Trees |
| 13 | Gradient Boosting | n=100, lr=0.1 | Gradient Boosting стандартный |
| 14 | Gradient Boosting | n=200, lr=0.05 | Gradient Boosting медленный |
| 15 | AdaBoost | n=50, lr=1.0 | AdaBoost |
| 16 | Bagging | n=20 | Bagging регрессор |
| 17 | SVR (RBF) | kernel=rbf, C=1.0 | SVR с RBF ядром |
| 18 | KNN | k=5, uniform | KNN k=5 uniform |
| 19 | KNN | k=10, distance | KNN k=10 distance |

---

## Настройка логирования

### Структура логирования

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Что логировать?                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ПАРАМЕТРЫ (log_param)              МЕТРИКИ (log_metric)              │
│  ├── model_type                     ├── r2_score                      │
│  ├── model_description              ├── rmse                          │
│  ├── experiment_description         ├── mae                           │
│  ├── n_estimators, max_depth...     ├── mape                          │
│  ├── train_size                     ├── train_time_seconds            │
│  ├── test_size                      ├── inference_time_seconds        │
│  ├── n_features                     └── predictions_per_second        │
│  ├── random_state                                                     │
│  └── test_split_ratio                                                 │
│                                                                       │
│  АРТЕФАКТЫ (log_artifact)           ТЕГИ (set_tag)                    │
│  ├── sklearn_model/                 ├── algorithm_family              │
│  ├── model_artifacts/*.pkl          ├── experiment_type               │
│  ├── plots/predictions_scatter.png  ├── dataset                       │
│  ├── plots/residuals.png            ├── author                        │
│  ├── plots/feature_importance.png   ├── environment                   │
│  ├── predictions/predictions.csv    └── mlflow.note.content           │
│  └── config/experiment_config.json                                    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Пример полного логирования

```python
import mlflow
import mlflow.sklearn
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import time

def run_full_experiment(model, model_name, X_train, X_test, y_train, y_test, custom_params):
    """Полный эксперимент с расширенным логированием."""

    mlflow.set_experiment("boston_housing_model_comparison")

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # ═══════════════════════════════════════════════════════════
        # 1. ЛОГИРОВАНИЕ ПАРАМЕТРОВ
        # ═══════════════════════════════════════════════════════════

        # Параметры модели
        mlflow.log_param("model_type", model_name)
        for param_name, param_value in custom_params.items():
            mlflow.log_param(param_name, param_value)

        # Параметры данных
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", list(X_train.columns))

        # ═══════════════════════════════════════════════════════════
        # 2. ОБУЧЕНИЕ С ЗАМЕРОМ ВРЕМЕНИ
        # ═══════════════════════════════════════════════════════════

        import time

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_inference = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_inference

        # ═══════════════════════════════════════════════════════════
        # 3. ЛОГИРОВАНИЕ МЕТРИК
        # ═══════════════════════════════════════════════════════════

        # Основные метрики качества
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mape", np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        # Метрики производительности
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("inference_time_seconds", inference_time)
        mlflow.log_metric("predictions_per_second", len(X_test) / inference_time)

        # ═══════════════════════════════════════════════════════════
        # 4. ЛОГИРОВАНИЕ АРТЕФАКТОВ
        # ═══════════════════════════════════════════════════════════

        # 4.1 Модель (автоматический MLflow формат)
        mlflow.sklearn.log_model(model, "sklearn_model")

        # 4.2 Модель в pickle формате
        with open("/tmp/model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("/tmp/model.pkl", "model_artifacts")

        # 4.3 График важности признаков (для tree-based моделей)
        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=True)

            ax.barh(importances["feature"], importances["importance"])
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")

            fig.tight_layout()
            fig.savefig("/tmp/feature_importance.png", dpi=150)
            mlflow.log_artifact("/tmp/feature_importance.png", "plots")
            plt.close(fig)

        # 4.4 График предсказаний vs реальных значений
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predictions vs Actual")

        fig.tight_layout()
        fig.savefig("/tmp/predictions_scatter.png", dpi=150)
        mlflow.log_artifact("/tmp/predictions_scatter.png", "plots")
        plt.close(fig)

        # 4.5 CSV с предсказаниями
        predictions_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": y_pred,
            "error": y_test.values - y_pred,
            "abs_error": np.abs(y_test.values - y_pred)
        })
        predictions_df.to_csv("/tmp/predictions.csv", index=False)
        mlflow.log_artifact("/tmp/predictions.csv", "predictions")

        # ═══════════════════════════════════════════════════════════
        # 5. ЛОГИРОВАНИЕ ТЕГОВ
        # ═══════════════════════════════════════════════════════════

        mlflow.set_tag("algorithm_family", get_algorithm_family(model_name))
        mlflow.set_tag("experiment_type", "model_comparison")
        mlflow.set_tag("dataset", "boston_housing")
        mlflow.set_tag("author", "data_scientist")
        mlflow.set_tag("environment", "development")
        mlflow.set_tag("mlflow.note.content", f"Эксперимент с {model_name}")

        # ═══════════════════════════════════════════════════════════
        # 6. ЛОГИРОВАНИЕ КОНФИГУРАЦИИ
        # ═══════════════════════════════════════════════════════════

        config = {
            "model_type": model_name,
            "params": custom_params,
            "data": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X_train.columns)
            },
            "timestamp": datetime.now().isoformat()
        }

        with open("/tmp/config.json", "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact("/tmp/config.json", "config")

        return mlflow.active_run().info.run_id
```

### Логирование в DVCLive (альтернатива)

```python
from dvclive import Live
import json

def run_dvclive_experiment(model, model_name, X_train, X_test, y_train, y_test, params):
    """Эксперимент с логированием через DVCLive."""

    with Live(save_dvc_exp=True, dir=f"dvclive/{model_name}") as live:
        # Параметры
        live.log_param("model_type", model_name)
        for param_name, param_value in params.items():
            live.log_param(param_name, param_value)

        # Обучение
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Метрики
        live.log_metric("r2_score", r2_score(y_test, y_pred))
        live.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
        live.log_metric("mae", mean_absolute_error(y_test, y_pred))

        # Артефакт модели
        model_path = f"data/models/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        live.log_artifact(model_path, type="model", name=model_name)

        # График (DVCLive поддерживает matplotlib)
        if hasattr(model, "feature_importances_"):
            live.log_sklearn_plot("feature_importances", model, X_train.columns)
```

---

## Система сравнения экспериментов

### MLflow UI — Сравнение через веб-интерфейс

```bash
# Откройте MLflow UI
open http://localhost:5000
```

**Функции сравнения в UI:**

1. **Compare** — выбор нескольких экспериментов для сравнения
2. **Charts** — визуализация метрик на графиках
3. **Parallel Coordinates** — параллельные координаты для анализа зависимостей
4. **Scatter Plot** — scatter-графики для любых двух метрик/параметров

### Сравнение через Python API

```python
import mlflow
from mlflow.tracking import MlflowClient

def compare_experiments(experiment_name: str, metric: str = "r2_score", top_n: int = 10):
    """Сравнение экспериментов программно."""

    client = MlflowClient()

    # Получение эксперимента
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Эксперимент '{experiment_name}' не найден")

    # Получение всех запусков
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n
    )

    # Формирование таблицы сравнения
    comparison = []
    for run in runs:
        comparison.append({
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name,
            "model_type": run.data.params.get("model_type", "unknown"),
            "r2_score": run.data.metrics.get("r2_score"),
            "rmse": run.data.metrics.get("rmse"),
            "mae": run.data.metrics.get("mae"),
            "train_time": run.data.metrics.get("train_time_seconds"),
        })

    df = pd.DataFrame(comparison)
    print(f"\n📊 ТОП-{top_n} экспериментов по {metric}:\n")
    print(df.to_string(index=False))

    return df


def compare_runs_detailed(run_ids: list[str]):
    """Детальное сравнение конкретных запусков."""

    client = MlflowClient()

    print("\n" + "=" * 80)
    print("📊 ДЕТАЛЬНОЕ СРАВНЕНИЕ ЗАПУСКОВ")
    print("=" * 80)

    for run_id in run_ids:
        run = client.get_run(run_id)

        print(f"\n🔹 Run: {run.info.run_name} ({run_id[:8]})")
        print("-" * 40)

        print("  Параметры:")
        for param, value in sorted(run.data.params.items()):
            print(f"    {param}: {value}")

        print("  Метрики:")
        for metric, value in sorted(run.data.metrics.items()):
            print(f"    {metric}: {value:.4f}" if isinstance(value, float) else f"    {metric}: {value}")

        print("  Теги:")
        for tag, value in sorted(run.data.tags.items()):
            if not tag.startswith("mlflow."):
                print(f"    {tag}: {value}")


# Использование
compare_experiments("boston_housing_model_comparison", metric="r2_score", top_n=5)
compare_runs_detailed(["run_id_1", "run_id_2", "run_id_3"])
```

### Визуализация сравнения

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_comparison(experiment_name: str):
    """Визуализация сравнения метрик."""

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=50
    )

    # Сбор данных
    data = []
    for run in runs:
        data.append({
            "model": run.data.params.get("model_type", "unknown"),
            "r2_score": run.data.metrics.get("r2_score", 0),
            "rmse": run.data.metrics.get("rmse", 0),
            "mae": run.data.metrics.get("mae", 0),
            "family": run.data.tags.get("algorithm_family", "unknown")
        })

    df = pd.DataFrame(data)

    # График 1: R² Score по моделям
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar chart: R² Score
    ax1 = axes[0, 0]
    df_sorted = df.sort_values("r2_score", ascending=True)
    colors = df_sorted["family"].map({
        "linear": "#3498db",
        "tree_ensemble": "#2ecc71",
        "other": "#e74c3c"
    })
    ax1.barh(df_sorted["model"], df_sorted["r2_score"], color=colors)
    ax1.set_xlabel("R² Score")
    ax1.set_title("R² Score по моделям")
    ax1.axvline(x=0.8, color='r', linestyle='--', label='Threshold (0.8)')

    # Bar chart: RMSE
    ax2 = axes[0, 1]
    df_sorted = df.sort_values("rmse", ascending=False)
    ax2.barh(df_sorted["model"], df_sorted["rmse"], color="#9b59b6")
    ax2.set_xlabel("RMSE")
    ax2.set_title("RMSE по моделям (меньше = лучше)")

    # Scatter: R² vs RMSE
    ax3 = axes[1, 0]
    for family in df["family"].unique():
        family_df = df[df["family"] == family]
        ax3.scatter(family_df["r2_score"], family_df["rmse"], label=family, s=100, alpha=0.7)
    ax3.set_xlabel("R² Score")
    ax3.set_ylabel("RMSE")
    ax3.set_title("R² Score vs RMSE")
    ax3.legend()

    # Box plot по семействам
    ax4 = axes[1, 1]
    df.boxplot(column="r2_score", by="family", ax=ax4)
    ax4.set_title("Распределение R² по семействам алгоритмов")
    ax4.set_xlabel("Algorithm Family")
    ax4.set_ylabel("R² Score")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=150)
    plt.show()
```

### DVC для сравнения экспериментов

```bash
# Просмотр всех экспериментов
dvc exp show

# Сравнение конкретных экспериментов
dvc exp diff exp-abc123 exp-def456

# Табличный вывод с сортировкой
dvc exp show --sort-by r2_score --sort-order desc

# Экспорт в CSV
dvc exp show --csv > experiments.csv
```

---

## Фильтрация и поиск экспериментов

### MLflow Search API

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# ═══════════════════════════════════════════════════════════════════
# ФИЛЬТРАЦИЯ ПО МЕТРИКАМ
# ═══════════════════════════════════════════════════════════════════

# Найти все запуски с R² > 0.85
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.r2_score > 0.85",
    order_by=["metrics.r2_score DESC"]
)

# Найти запуски с RMSE < 3.0
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.rmse < 3.0"
)

# Комбинированный фильтр: R² > 0.8 AND RMSE < 3.5
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.r2_score > 0.8 AND metrics.rmse < 3.5"
)

# ═══════════════════════════════════════════════════════════════════
# ФИЛЬТРАЦИЯ ПО ПАРАМЕТРАМ
# ═══════════════════════════════════════════════════════════════════

# Найти все Random Forest модели
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="params.model_type = 'random_forest'"
)

# Найти модели с n_estimators >= 100
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="params.n_estimators >= '100'"  # строковое сравнение!
)

# ═══════════════════════════════════════════════════════════════════
# ФИЛЬТРАЦИЯ ПО ТЕГАМ
# ═══════════════════════════════════════════════════════════════════

# Найти все tree-based модели
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.algorithm_family = 'tree_ensemble'"
)

# Найти эксперименты конкретного автора
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="tags.author = 'data_scientist'"
)

# ═══════════════════════════════════════════════════════════════════
# ФИЛЬТРАЦИЯ ПО СТАТУСУ И ВРЕМЕНИ
# ═══════════════════════════════════════════════════════════════════

# Только успешные запуски
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="attributes.status = 'FINISHED'"
)

# Запуски за последние 7 дней
from datetime import datetime, timedelta
week_ago = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

runs = client.search_runs(
    experiment_ids=["1"],
    filter_string=f"attributes.start_time > {week_ago}"
)

# ═══════════════════════════════════════════════════════════════════
# КОМПЛЕКСНЫЕ ЗАПРОСЫ
# ═══════════════════════════════════════════════════════════════════

# Лучшие tree-based модели с R² > 0.85
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="""
        tags.algorithm_family = 'tree_ensemble'
        AND metrics.r2_score > 0.85
        AND attributes.status = 'FINISHED'
    """,
    order_by=["metrics.r2_score DESC"],
    max_results=10
)

for run in runs:
    print(f"{run.info.run_name}: R²={run.data.metrics['r2_score']:.4f}")
```

### Утилита для поиска

```python
def search_experiments(
    experiment_name: str,
    metric_filters: dict = None,
    param_filters: dict = None,
    tag_filters: dict = None,
    top_n: int = 10,
    sort_by: str = "r2_score",
    ascending: bool = False
):
    """
    Универсальная функция поиска экспериментов.

    Args:
        experiment_name: Имя эксперимента
        metric_filters: {"r2_score": "> 0.8", "rmse": "< 4.0"}
        param_filters: {"model_type": "random_forest"}
        tag_filters: {"algorithm_family": "tree_ensemble"}
        top_n: Количество результатов
        sort_by: Метрика для сортировки
        ascending: Порядок сортировки

    Returns:
        DataFrame с результатами
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Построение filter_string
    conditions = []

    if metric_filters:
        for metric, condition in metric_filters.items():
            conditions.append(f"metrics.{metric} {condition}")

    if param_filters:
        for param, value in param_filters.items():
            conditions.append(f"params.{param} = '{value}'")

    if tag_filters:
        for tag, value in tag_filters.items():
            conditions.append(f"tags.{tag} = '{value}'")

    filter_string = " AND ".join(conditions) if conditions else ""

    # Выполнение поиска
    order_direction = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=[f"metrics.{sort_by} {order_direction}"],
        max_results=top_n
    )

    # Формирование результатов
    results = []
    for run in runs:
        results.append({
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name,
            "model_type": run.data.params.get("model_type"),
            **{f"metric_{k}": v for k, v in run.data.metrics.items()}
        })

    return pd.DataFrame(results)


# Примеры использования
# ─────────────────────────────────────────────────────────────────

# Найти лучшие tree-based модели
df = search_experiments(
    "boston_housing_model_comparison",
    metric_filters={"r2_score": "> 0.85"},
    tag_filters={"algorithm_family": "tree_ensemble"},
    top_n=5
)

# Найти все Random Forest с разными параметрами
df = search_experiments(
    "boston_housing_model_comparison",
    param_filters={"model_type": "random_forest"},
    sort_by="r2_score"
)

# Найти модели с низкой ошибкой
df = search_experiments(
    "boston_housing_model_comparison",
    metric_filters={"rmse": "< 3.0", "mae": "< 2.5"},
    top_n=10
)
```

### Поиск через MLflow UI

В веб-интерфейсе MLflow доступен поиск через строку фильтра:

```
# Примеры фильтров для UI
metrics.r2_score > 0.85
params.model_type = "random_forest"
tags.algorithm_family = "tree_ensemble"
metrics.r2_score > 0.8 AND metrics.rmse < 4.0
```

### Поиск через DVC

```bash
# Поиск по метрикам
dvc exp show --drop-param --keep-metric r2_score --sort-by r2_score

# Фильтрация по значению
dvc exp show | grep "random_forest"

# Экспорт для дальнейшего анализа
dvc exp show --json > experiments.json
```

---

## Автоматизация экспериментов

### Bash скрипт для запуска всех экспериментов

```bash
#!/bin/bash
# Запуск всех 19 экспериментов

set -e

echo "🚀 Запуск всех экспериментов..."

# Настройка окружения
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

# Запуск всех экспериментов через единый скрипт
python scripts/run_experiments.py

echo "🎉 Все эксперименты завершены!"
echo "Откройте MLflow UI: http://localhost:5000"
echo "Результаты сохранены в data/experiments/results_summary.csv"
```

### Grid Search с логированием

```python
from sklearn.model_selection import GridSearchCV
import mlflow

def run_grid_search_experiment(
    model_name: str,
    param_grid: dict,
    X_train, X_test, y_train, y_test,
    cv: int = 5
):
    """Grid Search с автоматическим логированием в MLflow."""

    mlflow.set_experiment("boston_housing_grid_search")

    with mlflow.start_run(run_name=f"gridsearch_{model_name}"):
        # Создание базовой модели
        base_model = create_model(model_name)

        # Grid Search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Логирование лучших параметров
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", cv)
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)

        # Логирование результатов CV
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Оценка на тестовой выборке
        y_pred = grid_search.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_metric("test_r2_score", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)

        # Логирование лучшей модели
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

        # Логирование всех результатов CV
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv("/tmp/cv_results.csv", index=False)
        mlflow.log_artifact("/tmp/cv_results.csv", "cv_results")

        return grid_search.best_estimator_, grid_search.best_params_


# Пример использования
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

best_model, best_params = run_grid_search_experiment(
    "random_forest",
    param_grid,
    X_train, X_test, y_train, y_test
)
```

---

## Best Practices

### 1. Именование экспериментов

```python
# ✅ Хорошо: информативные имена
mlflow.set_experiment("boston_housing_regression_v2")
run_name = f"rf_n{n_estimators}_d{max_depth}_{datetime.now().strftime('%Y%m%d')}"

# ❌ Плохо: неинформативные имена
mlflow.set_experiment("test")
run_name = "experiment_1"
```

### 2. Структура тегов

```python
# Рекомендуемые теги
mlflow.set_tag("algorithm_family", "tree_ensemble")  # Семейство алгоритма
mlflow.set_tag("experiment_type", "model_comparison") # Тип эксперимента
mlflow.set_tag("dataset", "boston_housing")          # Набор данных
mlflow.set_tag("data_version", "v1.2")               # Версия данных
mlflow.set_tag("author", "your_name")                # Автор
mlflow.set_tag("environment", "development")         # Окружение
```

### 3. Воспроизводимость

```python
# Фиксация seed
random_state = 42
mlflow.log_param("random_state", random_state)

# Версии библиотек
mlflow.log_param("sklearn_version", sklearn.__version__)
mlflow.log_param("python_version", sys.version)

# Хэш данных
data_hash = hashlib.md5(X_train.values.tobytes()).hexdigest()[:8]
mlflow.log_param("data_hash", data_hash)
```

### 4. Документирование экспериментов

```python
# Добавление заметок к запуску
mlflow.set_tag("mlflow.note.content", """
## Цель эксперимента
Проверка влияния глубины дерева на качество модели.

## Гипотеза
Увеличение max_depth улучшит R² до определённого предела.

## Результаты
- max_depth=10: R²=0.85
- max_depth=15: R²=0.87
- max_depth=20: R²=0.86 (переобучение)
""")
```

### 5. Организация артефактов

```
artifacts/
├── sklearn_model/
│   ├── model.pkl
│   ├── MLmodel
│   ├── conda.yaml
│   └── requirements.txt
├── model_artifacts/
│   └── {model_name}.pkl
├── plots/
│   ├── predictions_scatter.png
│   ├── residuals.png
│   └── feature_importance.png  # для tree-based моделей
├── predictions/
│   └── predictions.csv
└── config/
    └── experiment_config.json
```

---

## 🚀 Быстрый старт

```bash
# 1. Запустите инфраструктуру
docker-compose up -d mlflow minio

# 2. Настройте окружение
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

# 3. Запустите все 19 экспериментов
python scripts/run_experiments.py

# 4. Откройте UI для анализа
open http://localhost:5000

# 5. Просмотрите сводку результатов
cat data/experiments/results_summary.csv
```

---

## 📚 Полезные ссылки

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Search Syntax](https://mlflow.org/docs/latest/search-syntax.html)
- [DVCLive Documentation](https://dvc.org/doc/dvclive)
- [scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
