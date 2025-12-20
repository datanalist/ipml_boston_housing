# Интеграция MLflow Tracking в проект

Данный документ описывает модули для автоматического логирования экспериментов машинного обучения с помощью MLflow.

## Содержание

1. [Обзор модулей](#обзор-модулей)
2. [Конфигурация MLflow](#конфигурация-mlflow)
3. [Декораторы для логирования](#декораторы-для-логирования)
4. [Контекстные менеджеры](#контекстные-менеджеры)
5. [Утилиты для работы с экспериментами](#утилиты-для-работы-с-экспериментами)
6. [Примеры использования](#примеры-использования)

---

## Обзор модулей

Созданы следующие модули для интеграции MLflow в проект:

```
src/
├── config/
│   ├── __init__.py
│   └── mlflow_config.py      # Конфигурация MLflow и S3/MinIO
└── tracking/
    ├── __init__.py
    ├── decorators.py         # Декораторы для автоматического логирования
    ├── mlflow_tracker.py     # Контекстные менеджеры
    └── utils.py              # Утилиты для работы с экспериментами
```

---

## Конфигурация MLflow

**Модуль:** `src/config/mlflow_config.py`

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URL сервера MLflow Tracking |
| `MLFLOW_EXPERIMENT_NAME` | `boston-housing` | Название эксперимента по умолчанию |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | URL MinIO/S3 для артефактов |
| `AWS_ACCESS_KEY_ID` | `minioadmin0` | Ключ доступа MinIO |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin1230` | Секретный ключ MinIO |

### Использование

```python
from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    setup_mlflow_env,
)

# Настройка переменных окружения для работы с S3
setup_mlflow_env()
```

---

## Декораторы для логирования

**Модуль:** `src/tracking/decorators.py`

### `@mlflow_run`

Декоратор для автоматического создания MLflow run. Оборачивает функцию в контекст эксперимента и автоматически логирует время выполнения.

```python
from src.tracking.decorators import mlflow_run

@mlflow_run(experiment_name="boston-housing", run_name="rf-baseline")
def train_model(n_estimators=100, max_depth=10):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

# Автоматически создаётся run и логируется время выполнения
model = train_model()
```

**Параметры:**
- `experiment_name` (str): Название эксперимента
- `run_name` (str, optional): Имя запуска
- `tags` (dict, optional): Теги для запуска

### `@log_params_decorator`

Автоматически логирует все kwargs функции как параметры MLflow.

```python
from src.tracking.decorators import log_params_decorator

@log_params_decorator
def train(n_estimators=100, max_depth=10, learning_rate=0.1):
    # n_estimators, max_depth, learning_rate будут залогированы как параметры
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    return model.fit(X_train, y_train)
```

### `@log_metrics_decorator`

Извлекает указанные ключи из возвращаемого словаря и логирует их как метрики.

```python
from src.tracking.decorators import log_metrics_decorator

@log_metrics_decorator(["r2_score", "rmse", "mae"])
def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    return {
        "r2_score": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
    }

# r2_score, rmse, mae будут автоматически залогированы
metrics = evaluate(model, X_test, y_test)
```

### `@log_artifact_decorator`

Если функция возвращает путь к файлу, этот файл будет залогирован как артефакт.

```python
from src.tracking.decorators import log_artifact_decorator

@log_artifact_decorator(artifact_path="plots")
def create_plot(data) -> str:
    fig, ax = plt.subplots()
    ax.plot(data)
    path = "residuals.png"
    fig.savefig(path)
    return path  # Файл будет залогирован в артефакты
```

### `@timed_execution`

Измеряет время выполнения функции и логирует его как метрику.

```python
from src.tracking.decorators import timed_execution

@timed_execution
def expensive_computation():
    # Долгие вычисления
    pass

# Метрика expensive_computation_duration_seconds будет залогирована
```

### Комбинирование декораторов

```python
from src.tracking.decorators import mlflow_run, log_params_decorator, log_metrics_decorator

@mlflow_run(experiment_name="boston-housing", run_name="rf-tuned")
@log_params_decorator
def train_and_evaluate(n_estimators=100, max_depth=10):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metrics({
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    })

    return model

# Один вызов — полный эксперимент
model = train_and_evaluate(n_estimators=200, max_depth=15)
```

---

## Контекстные менеджеры

**Модуль:** `src/tracking/mlflow_tracker.py`

### `MLflowExperimentTracker`

Полнофункциональный контекстный менеджер для трекинга экспериментов.

```python
from src.tracking.mlflow_tracker import MLflowExperimentTracker

tracker = MLflowExperimentTracker(experiment_name="boston-housing")

with tracker.start_run(run_name="gradient-boosting-v1"):
    # Установка тегов
    tracker.set_tags({
        "model_type": "GradientBoosting",
        "framework": "sklearn",
        "author": "data-science-team"
    })

    # Логирование параметров
    tracker.log_params({
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.1
    })

    # Обучение модели
    model = GradientBoostingRegressor(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Оценка и логирование метрик
    y_pred = model.predict(X_test)
    tracker.log_metrics({
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred)
    })

    # Логирование артефактов
    tracker.log_artifact("reports/figures/residuals.png", artifact_path="plots")

    # Сохранение модели
    tracker.log_model(
        model,
        artifact_path="model",
        input_example=X_test.head(5),
        registered_model_name="boston-housing-gb"
    )

    # Получение информации о run
    print(f"Run ID: {tracker.run_id}")
    print(f"Artifacts: {tracker.artifact_uri}")
```

**Методы MLflowExperimentTracker:**

| Метод | Описание |
|-------|----------|
| `start_run(run_name, tags)` | Начало нового запуска |
| `log_params(params)` | Логирование словаря параметров |
| `log_param(key, value)` | Логирование одного параметра |
| `log_metrics(metrics, step)` | Логирование словаря метрик |
| `log_metric(key, value, step)` | Логирование одной метрики |
| `log_artifact(local_path, artifact_path)` | Логирование файла |
| `log_artifacts(local_dir, artifact_path)` | Логирование директории |
| `log_model(model, artifact_path, ...)` | Логирование sklearn модели |
| `set_tags(tags)` | Установка тегов |
| `set_tag(key, value)` | Установка одного тега |
| `log_dict(dictionary, artifact_file)` | Логирование JSON/YAML |
| `log_figure(figure, artifact_file)` | Логирование matplotlib/plotly фигуры |

**Свойства:**

| Свойство | Описание |
|----------|----------|
| `run_id` | ID текущего запуска |
| `artifact_uri` | URI хранилища артефактов |
| `experiment_id` | ID эксперимента |

### `NestedRunTracker`

Контекстный менеджер для вложенных запусков (кросс-валидация, grid search).

```python
from src.tracking.mlflow_tracker import MLflowExperimentTracker, NestedRunTracker

with MLflowExperimentTracker(experiment_name="cv-experiment") as parent:
    parent.log_params({"model": "RandomForest", "cv_folds": 5})

    scores = []
    for fold in range(5):
        with NestedRunTracker(run_name=f"fold-{fold}") as child:
            # Обучение на fold
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)

            child.log_metrics({"fold_score": score})
            scores.append(score)

    # Агрегированные метрики в родительском run
    parent.log_metrics({
        "mean_cv_score": np.mean(scores),
        "std_cv_score": np.std(scores)
    })
```

---

## Утилиты для работы с экспериментами

**Модуль:** `src/tracking/utils.py`

### `get_best_run`

Получение лучшего запуска по метрике.

```python
from src.tracking.utils import get_best_run

# Лучший по R² (максимизация)
best = get_best_run("boston-housing", metric="r2_score")
print(f"Run ID: {best['run_id']}")
print(f"R²: {best['metrics']['r2_score']:.4f}")
print(f"Параметры: {best['params']}")

# Лучший по RMSE (минимизация)
best = get_best_run("boston-housing", metric="rmse", ascending=True)
```

### `load_best_model`

Загрузка лучшей модели для инференса.

```python
from src.tracking.utils import load_best_model

# Загрузка лучшей модели по R²
model = load_best_model("boston-housing", metric="r2_score")
predictions = model.predict(X_new)

# Загрузка лучшей модели по RMSE
model = load_best_model("boston-housing", metric="rmse", ascending=True)
```

### `compare_runs`

Сравнение запусков в виде DataFrame.

```python
from src.tracking.utils import compare_runs

# Топ-10 запусков
comparison = compare_runs("boston-housing", top_n=10)
print(comparison[["run_id", "run_name", "metric_r2_score", "metric_rmse"]])

# Сравнение по конкретным метрикам
comparison = compare_runs(
    "boston-housing",
    metrics=["r2_score", "rmse", "mae", "train_time"],
    top_n=5
)
```

### `register_best_model`

Регистрация лучшей модели в Model Registry.

```python
from src.tracking.utils import register_best_model

version = register_best_model(
    experiment_name="boston-housing",
    model_name="boston-housing-production",
    metric="r2_score"
)
print(f"Зарегистрирована версия: {version}")
```

### `delete_experiment_runs`

Очистка старых/неудачных запусков.

```python
from src.tracking.utils import delete_experiment_runs

# Предварительный просмотр (dry run)
to_delete = delete_experiment_runs(
    "boston-housing",
    keep_top_n=10,
    dry_run=True  # Только показать, не удалять
)
print(f"Будет удалено: {len(to_delete)} запусков")

# Фактическое удаление
deleted = delete_experiment_runs(
    "boston-housing",
    keep_top_n=10,
    dry_run=False
)
```

### `get_experiment_summary`

Сводка по эксперименту.

```python
from src.tracking.utils import get_experiment_summary

summary = get_experiment_summary("boston-housing")
print(f"Всего запусков: {summary['total_runs']}")
print(f"Завершённых: {summary['finished_runs']}")
print(f"Лучший R²: {summary['best_r2']:.4f}")
print(f"Лучший RMSE: {summary['best_rmse']:.4f}")
print(f"Средний R²: {summary['avg_r2']:.4f}")
```

### `get_run_by_name`

Поиск запуска по имени.

```python
from src.tracking.utils import get_run_by_name

run = get_run_by_name("boston-housing", "rf-baseline-v2")
if run:
    print(f"R²: {run['metrics']['r2_score']}")
    print(f"Параметры: {run['params']}")
```

### `list_registered_models`

Список всех зарегистрированных моделей.

```python
from src.tracking.utils import list_registered_models

models = list_registered_models()
print(models[["name", "latest_version", "latest_stage"]])
```

### `transition_model_stage`

Перевод модели между стадиями (Staging → Production).

```python
from src.tracking.utils import transition_model_stage

# Перевод в Production
transition_model_stage(
    model_name="boston-housing-production",
    version="3",
    stage="Production",
    archive_existing=True  # Архивировать текущую Production версию
)
```

---

## Примеры использования

### Полный пайплайн эксперимента

```python
from src.tracking import (
    MLflowExperimentTracker,
    mlflow_run,
    log_params_decorator,
    get_best_run,
    compare_runs,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Вариант 1: Через контекстный менеджер
tracker = MLflowExperimentTracker(experiment_name="boston-housing")

with tracker.start_run(run_name="rf-optimized"):
    tracker.set_tags({"model_type": "RandomForest", "stage": "optimization"})

    params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5}
    tracker.log_params(params)

    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    tracker.log_metrics({
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    })

    tracker.log_model(model, "model", input_example=X_test.head())


# Вариант 2: Через декораторы
@mlflow_run(experiment_name="boston-housing", run_name="gb-experiment")
@log_params_decorator
def run_experiment(n_estimators=100, max_depth=5, learning_rate=0.1):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metrics({
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    })

    return model

model = run_experiment(n_estimators=150, max_depth=8, learning_rate=0.05)


# Анализ результатов
best = get_best_run("boston-housing")
print(f"Лучший результат: R² = {best['metrics']['r2_score']:.4f}")

comparison = compare_runs("boston-housing", top_n=5)
print(comparison)
```

### Grid Search с вложенными runs

```python
from src.tracking import MLflowExperimentTracker, NestedRunTracker
from itertools import product

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
}

with MLflowExperimentTracker(experiment_name="boston-grid-search") as parent:
    parent.log_params({"search_type": "grid", "model": "RandomForest"})
    parent.log_dict(param_grid, "param_grid.json")

    best_score = 0
    best_params = None

    for n_est, depth in product(param_grid["n_estimators"], param_grid["max_depth"]):
        with NestedRunTracker(run_name=f"n{n_est}_d{depth}") as child:
            child.log_params({"n_estimators": n_est, "max_depth": depth})

            model = RandomForestRegressor(n_estimators=n_est, max_depth=depth)
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            child.log_metrics({"r2_score": score})

            if score > best_score:
                best_score = score
                best_params = {"n_estimators": n_est, "max_depth": depth}

    parent.log_params({"best_" + k: v for k, v in best_params.items()})
    parent.log_metrics({"best_r2_score": best_score})
```

---

## Заключение

Созданные модули обеспечивают:

- **Простоту использования**: Декораторы и контекстные менеджеры минимизируют boilerplate код
- **Гибкость**: Можно выбрать подходящий стиль (декораторы или контекстные менеджеры)
- **Полноту**: Поддержка всех основных операций MLflow (параметры, метрики, артефакты, модели)
- **Удобство анализа**: Утилиты для сравнения экспериментов и работы с Model Registry

Все модули интегрированы с существующей инфраструктурой проекта (MinIO для артефактов, централизованная конфигурация).
