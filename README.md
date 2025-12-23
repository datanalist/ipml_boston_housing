# Предсказание цен на недвижимость в Бостоне

## 📋 Описание проекта

Учебный проект по машинному обучению для предсказания медианной стоимости домов в различных районах Бостона на основе социально-экономических и географических характеристик, используя лучшие инженерные практики области DS.

**Источник данных**: [Kaggle - Boston House Prices](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)

## 📚 Домашние задания

| № ДЗ | Ветка | Отчёт | Подробный отчёт | Описание |
|------|-------|-------|-----------------|----------|
| 1 | [hw1](https://github.com/datanalist/ipml_boston_housing/tree/hw1) | [📄](reports/PROJECT_SETUP_REPORT.md) | [📖](reports/LAB_REPORT.md) | Настройка проекта, структура, инструменты качества кода |
| 2 | [hw2](https://github.com/datanalist/ipml_boston_housing/tree/hw2) | [📄](reports/HW2_VERSIONING_REPORT.md) | [📖](reports/LAB_REPORT.md) | Версионирование данных и моделей: DVC, MinIO, DVCLive, Docker |
| 3 | [hw3](https://github.com/datanalist/ipml_boston_housing/tree/hw3) | [📄](reports/LAB_REPORT.md) | [📖](reports/LAB_REPORT.md) | Трекинг ML-экспериментов: MLflow, декораторы, контекстные менеджеры |
| 4 | [hw4](https://github.com/datanalist/ipml_boston_housing/tree/hw4) | [📄](reports/LAB_REPORT.md) | [📖](reports/LAB_REPORT.md) | Оркестрация ML-пайплайнов: Airflow, Hydra, Pydantic, интеграция систем |

## 🛠️ Технологии

| Категория | Инструмент | Описание |
|-----------|------------|----------|
| **Язык** | Python 3.13 | Основной язык |
| **Пакетный менеджер** | uv | Быстрый менеджер зависимостей |
| **Структура** | Cookiecutter DS | Шаблон проекта |
| **Качество кода** | ruff, pre-commit | Линтинг и форматирование |
| **Версионирование данных** | DVC | Data Version Control |
| **Хранилище** | MinIO | S3-совместимое локальное хранилище |
| **Эксперименты** | DVCLive, MLflow | Отслеживание метрик и параметров |
| **Оркестрация** | Apache Airflow | Планировщик ML-пайплайнов |
| **Конфигурации** | Hydra, Pydantic | Управление конфигурациями и валидация |
| **База данных** | PostgreSQL | Метаданные Airflow |
| **Брокер сообщений** | Redis | Celery broker для Airflow |
| **Контейнеризация** | Docker | Воспроизводимое окружение |

## 🚀 Быстрый старт

### Локальный запуск

```bash
# Клонирование репозитория
git clone git@github.com:datanalist/ipml_boston_housing.git
cd ipml_boston_housing

# Создание окружения
uv venv .venv --python=3.13
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Установка зависимостей
uv sync

# Запуск MinIO и загрузка данных
docker-compose up -d minio
dvc pull

# Обучение модели
python src/modeling/train.py -n 200 -d 15
```

### Запуск в Docker

```bash
# Сборка и запуск обучения
docker-compose build train
docker-compose run --rm train -n 200 -d 15

# Просмотр результатов
cat dvclive/metrics.json
```

## 📁 Структура проекта

```
ipml_boston_housing/
├── airflow/              # Apache Airflow конфигурация
│   ├── dags/            # DAG определения (3 пайплайна)
│   ├── plugins/         # Кастомные плагины (MinIO кэш)
│   └── logs/            # Логи Airflow
├── conf/                 # Hydra конфигурации
│   ├── model/           # Конфигурации 14 моделей ML
│   ├── data/            # Конфигурации данных
│   ├── training/        # Конфигурации обучения
│   ├── experiment/      # Готовые эксперименты
│   └── config.yaml      # Основной конфиг
├── config/               # Конфигурационные файлы
│   ├── mlflow/          # MLflow аутентификация
│   └── nginx/           # Nginx конфигурация
├── data/                 # Данные (версионируются через DVC)
│   ├── raw/             # Исходные данные
│   ├── interim/         # Промежуточные данные
│   ├── proccesed/       # Обработанные данные
│   ├── models/          # Обученные модели
│   └── *.dvc            # DVC-файлы
├── docker/               # Dockerfile'ы
│   ├── Dockerfile.airflow
│   ├── Dockerfile.app
│   ├── Dockerfile.mlflow
│   └── Dockerfile.minio
├── docs/                 # Документация
│   ├── guides/          # Руководства (10+ файлов)
│   └── docs/            # MkDocs документация
├── dvclive/              # Метрики и параметры экспериментов
├── notebooks/            # Jupyter ноутбуки
├── references/           # Референсные материалы
├── reports/              # Отчёты по домашним заданиям
├── scripts/              # Вспомогательные скрипты
│   ├── download_data.py
│   ├── run_experiments.py
│   └── start_airflow.sh
├── src/                  # Исходный код
│   ├── config.py        # Основная конфигурация
│   ├── dataset.py       # Загрузка данных
│   ├── features.py      # Инженерия признаков
│   ├── plots.py         # Визуализация
│   ├── config/           # Конфигурационные модули
│   │   └── mlflow_config.py
│   ├── integration/      # Интеграция систем
│   │   ├── health_check.py
│   │   └── utils.py
│   ├── ml_models/        # Загрузка моделей
│   │   └── model_loader.py
│   ├── modeling/         # Обучение и предсказания
│   │   ├── train.py      # Базовое обучение
│   │   ├── train_hydra.py # Обучение с Hydra
│   │   └── predict.py    # Предсказания
│   ├── monitoring/       # Мониторинг пайплайнов
│   │   ├── airflow_callbacks.py
│   │   ├── logger.py
│   │   └── pipeline_monitor.py
│   ├── notifications/    # Уведомления
│   │   ├── notifier.py
│   │   └── templates.py
│   ├── schemas/          # Pydantic схемы валидации
│   │   ├── base.py
│   │   ├── data_config.py
│   │   ├── model_config.py
│   │   └── training_config.py
│   └── tracking/         # MLflow трекинг
│       ├── decorators.py      # 5 декораторов
│       ├── mlflow_tracker.py  # 2 контекстных менеджера
│       └── utils.py           # 9 утилит
├── tests/                # Тесты
├── docker-compose.yml    # Оркестрация контейнеров
├── dvc.yaml              # Конфигурация DVC пайплайна
├── pyproject.toml        # Зависимости
└── uv.lock               # Фиксация версий
```

## 🔧 Основные команды

```bash
# Код
make lint                 # Проверить код
make format               # Отформатировать код
make test                 # Запустить тесты

# DVC
dvc pull                  # Загрузить данные из MinIO
dvc push                  # Отправить данные в MinIO
dvc exp show              # История экспериментов

# Docker
docker-compose up -d minio              # Запуск MinIO
docker-compose up -d                   # Запуск всех сервисов (Airflow, MLflow, MinIO)
docker-compose run --rm train           # Обучение в контейнере

# Airflow
docker-compose up -d airflow            # Запуск Airflow
# Доступ: http://localhost:8080 (admin/admin)

# MLflow
docker-compose up -d mlflow             # Запуск MLflow
# Доступ: http://localhost:5000 (admin/secure_password_123)
```

## 📊 Результаты модели

| Метрика | Значение |
|---------|----------|
| R² Score | 0.866 |
| RMSE | 3.13 |
| MAE | 2.09 |
| MAPE | 11.3% |

## 🌿 Структура веток

- **`main`** — продакшн версия
- **`dev`** — основная разработка
- **`hw*`** — домашние задания
- **`research`** — исследования

## 📖 Документация

- [Руководство по MinIO + DVC](docs/guides/MINIO+DVC.md)
- [Запуск экспериментов](docs/guides/EXPERIMENTS.md)
- [Работа с Docker](docs/guides/DOCKER.md)
- [Интеграция MLflow](docs/guides/TRACKING-INTEGRATION.md)
- [MLflow + DVC + MinIO](docs/guides/MLFLOW+DVC+MINIO.md)
- [Airflow ML пайплайны](docs/guides/airflow_ml_pipeline.md)

## 👨‍🎓 Автор

Студенческий учебный проект по машинному обучению

## 📄 Лицензия

Проект создан в образовательных целях.
