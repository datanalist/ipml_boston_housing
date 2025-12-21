# Apache Airflow для ML пайплайнов Boston Housing

## Обзор

Данный гайд описывает интеграцию Apache Airflow в проект Boston Housing для оркестрации ML пайплайнов. Реализация включает:

- Установку Airflow через Docker Compose
- Три DAG для различных сценариев использования
- Параллельное обучение моделей
- Кэширование артефактов в MinIO
- Интеграцию с MLflow для трекинга экспериментов

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Airflow Stack                             │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Webserver   │  Scheduler   │   Worker     │   airflow-init    │
│  (порт 8080) │              │  (Celery)    │                   │
└──────┬───────┴──────┬───────┴──────┬───────┴───────────────────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
       ┌──────────────┴──────────────┐
       │         PostgreSQL          │
       │    (метаданные Airflow)     │
       └──────────────┬──────────────┘
                      │
       ┌──────────────┴──────────────┐
       │           Redis             │
       │    (брокер Celery)          │
       └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   ML Infrastructure                              │
├──────────────────┬──────────────────┬───────────────────────────┤
│      MinIO       │     MLflow       │        Nginx              │
│  (S3 хранилище)  │   (трекинг)      │   (reverse proxy)         │
│  порты 9000/9001 │   порт 5000      │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
```

## Структура файлов

```
ipml_boston_housing/
├── docker-compose.yml          # Конфигурация всех сервисов
├── docker/
│   └── Dockerfile.airflow      # Docker образ Airflow с зависимостями
├── airflow/
│   ├── dags/
│   │   ├── boston_housing_simple.py       # Простой пайплайн
│   │   ├── boston_housing_experiments.py  # Параллельные эксперименты
│   │   └── boston_housing_cached.py       # Пайплайн с кэшированием
│   ├── logs/                   # Логи выполнения
│   └── plugins/
│       └── minio_cache.py      # Утилиты кэширования
└── scripts/
    └── start_airflow.sh        # Скрипт быстрого запуска
```

## Быстрый старт

### 1. Запуск инфраструктуры

```bash
# Из корня проекта
cd /home/user/ipml_boston_housing

# Вариант 1: через скрипт
./scripts/start_airflow.sh

# Вариант 2: вручную
export AIRFLOW_UID=$(id -u)
docker-compose up -d
```

### 2. Проверка статуса

```bash
docker-compose ps
```

Ожидаемый результат:
```
NAME                              STATUS
boston_housing_airflow_scheduler  Up (healthy)
boston_housing_airflow_webserver  Up (healthy)
boston_housing_airflow_worker     Up (healthy)
boston_housing_minio              Up (healthy)
boston_housing_mlflow             Up (healthy)
boston_housing_postgres           Up (healthy)
boston_housing_redis              Up (healthy)
```

### 3. Доступ к веб-интерфейсам

| Сервис | URL | Логин/Пароль |
|--------|-----|--------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

## DAGs

### 1. boston_housing_simple

**Назначение:** Простой последовательный пайплайн для обучения одной модели Random Forest.

**Этапы:**
```
download_data → validate_data → train_model → evaluate_model → save_artifacts
```

**Задачи:**

| Задача | Описание |
|--------|----------|
| `download_data` | Загрузка данных Boston Housing из интернета |
| `validate_data` | Проверка качества данных, подсчёт статистик |
| `train_model` | Обучение Random Forest с параметрами по умолчанию |
| `evaluate_model` | Расчёт метрик: R², RMSE, MAE, MAPE |
| `save_artifacts` | Сохранение модели в MinIO, логирование в MLflow |

**Запуск:**
```bash
# Через Airflow CLI
docker-compose exec airflow-webserver airflow dags trigger boston_housing_simple

# Или через UI: нажать кнопку ▶ напротив DAG
```

### 2. boston_housing_experiments

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

**Модели:**

| Группа | Модели |
|--------|--------|
| Linear | LinearRegression, Ridge (α=0.1, 1.0, 10.0), Lasso, ElasticNet, Huber |
| Tree | DecisionTree, RandomForest (2 конфига), ExtraTrees, GradientBoosting (2 конфига), AdaBoost, Bagging |
| Other | SVR, KNN (2 конфига) |

**Параллелизм:**
- Используется `expand()` для динамического создания задач
- До 8 параллельных задач (`max_active_tasks=8`)
- CeleryExecutor с 4 воркерами

**Результаты:**
- CSV с метриками всех моделей: `data/experiments/all_results.csv`
- Markdown отчёт: `data/experiments/report_<timestamp>.md`
- Артефакты в MinIO: `s3://airflow-artifacts/`

### 3. boston_housing_cached

**Назначение:** Пайплайн с кэшированием для пропуска повторных вычислений.

**Логика кэширования:**
1. Вычисляется хэш входных данных + параметров модели
2. Проверяется наличие модели в MinIO по этому хэшу
3. Если найдена — обучение пропускается (ShortCircuitOperator)
4. Если нет — обучение выполняется, результат сохраняется в кэш

**Этапы:**
```
download_data → check_cache ─┬─[кэш есть]──→ use_cached_model ──┐
                             │                                   │
                             └─[кэша нет]─→ train_model ────────┤
                                                │                │
                                          save_to_cache         │
                                                │                │
                                                └────────────────┴──→ generate_summary
```

## Кэширование в MinIO

### Класс MinIOCache

Расположение: `airflow/plugins/minio_cache.py`

**Основные методы:**

```python
from minio_cache import MinIOCache

cache = MinIOCache(bucket_name="airflow-cache")

# Проверка существования кэша
exists, cache_key = cache.check_cache(
    prefix="models/random_forest",
    params={"n_estimators": 100, "max_depth": 10},
    data_path="/path/to/data.csv"
)

# Загрузка файла в кэш
cache.upload("/local/model.pkl", "models/model_abc123.pkl")

# Скачивание из кэша
cache.download("models/model_abc123.pkl", "/local/model.pkl")

# Сохранение/чтение JSON
cache.put_json("metrics/run_001.json", {"r2": 0.85})
metrics = cache.get_json("metrics/run_001.json")
```

**Алгоритм формирования ключа кэша:**
```
cache_key = f"{prefix}_{md5(params)}_{md5(data_file)}"
```

### Функции для ShortCircuitOperator

```python
from minio_cache import check_model_cache

# Возвращает True если нужно обучать (кэша нет)
# Возвращает False если можно пропустить (кэш найден)
need_train = check_model_cache(
    model_name="random_forest",
    params={"n_estimators": 100},
    data_path="/data/housing.csv"
)
```

## Конфигурация

### Переменные окружения

Файл `.env`:
```bash
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# MLflow
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=admin

# Airflow
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_UID=50000
```

### Параметры параллелизма

В `docker-compose.yml`:
```yaml
x-airflow-common:
  environment:
    AIRFLOW__CORE__PARALLELISM: 16          # Макс. задач во всём Airflow
    AIRFLOW__CORE__DAG_CONCURRENCY: 8       # Макс. задач на один DAG
    AIRFLOW__CELERY__WORKER_CONCURRENCY: 4  # Макс. задач на воркер
```

## Мониторинг и отладка

### Просмотр логов

```bash
# Логи webserver
docker-compose logs -f airflow-webserver

# Логи scheduler
docker-compose logs -f airflow-scheduler

# Логи worker
docker-compose logs -f airflow-worker

# Все логи Airflow
docker-compose logs -f airflow-webserver airflow-scheduler airflow-worker
```

### Проверка состояния DAG

```bash
# Список DAGs
docker-compose exec airflow-webserver airflow dags list

# Состояние задач
docker-compose exec airflow-webserver airflow tasks list boston_housing_simple

# История запусков
docker-compose exec airflow-webserver airflow dags list-runs -d boston_housing_simple
```

### Ручной запуск задачи

```bash
docker-compose exec airflow-webserver airflow tasks test \
    boston_housing_simple download_data 2024-01-01
```

## Расширение

### Добавление нового DAG

1. Создайте файл в `airflow/dags/`:

```python
from datetime import datetime
from airflow.decorators import dag, task

@dag(
    dag_id="my_new_dag",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["custom"],
)
def my_new_dag():
    @task
    def my_task():
        print("Hello from my task!")
        return {"status": "success"}

    my_task()

my_new_dag()
```

2. DAG автоматически появится в UI через несколько секунд

### Добавление новой модели в эксперименты

Отредактируйте `airflow/dags/boston_housing_experiments.py`:

```python
# Добавьте в соответствующий список
OTHER_MODELS = [
    # ... существующие модели ...
    {
        "name": "my_model",
        "params": {"param1": "value1"},
        "description": "Описание модели"
    },
]

# Добавьте создание модели в функцию create_model()
def create_model(model_name: str, params: dict):
    models = {
        # ... существующие модели ...
        "my_model": MyModelClass,
    }
    # ...
```

## Устранение неполадок

### DAG не появляется в UI

1. Проверьте синтаксис Python:
```bash
docker-compose exec airflow-webserver python /opt/airflow/dags/boston_housing_simple.py
```

2. Проверьте логи scheduler:
```bash
docker-compose logs airflow-scheduler | grep -i error
```

### Задача зависает

1. Проверьте состояние воркера:
```bash
docker-compose exec airflow-worker celery -A airflow.providers.celery.executors.celery_executor.app inspect active
```

2. Перезапустите воркер:
```bash
docker-compose restart airflow-worker
```

### Ошибка подключения к MinIO

1. Проверьте доступность MinIO:
```bash
curl http://localhost:9000/minio/health/live
```

2. Проверьте переменные окружения:
```bash
docker-compose exec airflow-worker env | grep -E "(AWS|MINIO)"
```

### Ошибка подключения к MLflow

1. Проверьте доступность MLflow:
```bash
curl http://localhost:5000/health
```

2. Проверьте креденшелы:
```bash
docker-compose exec airflow-worker env | grep MLFLOW
```

## Полезные команды

```bash
# Остановка всех сервисов
docker-compose down

# Остановка с удалением данных
docker-compose down --volumes

# Пересборка образов
docker-compose build --no-cache

# Масштабирование воркеров
docker-compose up -d --scale airflow-worker=3

# Очистка всех DAG runs
docker-compose exec airflow-webserver airflow db clean --clean-before-timestamp "2099-01-01"
```

## Ссылки

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [TaskFlow API](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html)
- [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
