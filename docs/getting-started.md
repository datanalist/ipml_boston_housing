# Быстрый старт

Это руководство поможет вам быстро начать работу с проектом Boston Housing Price Prediction.

---

## 📋 Предварительные требования

Убедитесь, что у вас установлено следующее:

- **Python 3.13**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — современный пакетный менеджер
- **Docker + Docker Compose** (опционально, для полной инфраструктуры)
- **Git**

### Установка uv

=== "Linux/macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "pip"

    ```bash
    pip install uv
    ```

---

## 🚀 Установка проекта

### Вариант 1: Автоматическая установка (рекомендуется)

```bash
# 1. Клонирование репозитория
git clone https://github.com/yourusername/ipml_boston_housing
cd ipml_boston_housing

# 2. Полная автоматическая настройка
make setup
```

Команда `make setup` выполнит:
- ✅ Создание виртуального окружения
- ✅ Установку всех зависимостей
- ✅ Настройку pre-commit хуков
- ✅ Загрузку данных через DVC
- ✅ Запуск Docker-инфраструктуры

### Вариант 2: Пошаговая установка

```bash
# 1. Клонирование репозитория
git clone https://github.com/yourusername/ipml_boston_housing
cd ipml_boston_housing

# 2. Создание окружения и установка зависимостей
make create_environment
make requirements

# 3. Настройка pre-commit хуков
make pre-commit

# 4. Загрузка данных
make dvc-pull

# 5. Запуск инфраструктуры (опционально)
make docker-up
```

---

## 🎯 Первое обучение модели

### Способ 1: Hydra (рекомендуется)

Hydra предоставляет гибкое управление конфигурациями:

```bash
# Базовый запуск (Random Forest по умолчанию)
uv run python src/modeling/train_hydra.py

# Смена модели
uv run python src/modeling/train_hydra.py model=gradient_boosting

# Переопределение параметров
uv run python src/modeling/train_hydra.py \
    model=random_forest \
    model.n_estimators=500 \
    model.max_depth=20

# Готовые эксперименты
uv run python src/modeling/train_hydra.py +experiment=tuned
```

### Способ 2: Airflow DAG

```bash
# 1. Запуск Airflow
docker-compose up -d

# 2. Открыть Web UI
# URL: http://localhost:8080
# Логин: admin
# Пароль: admin

# 3. Выбрать и запустить DAG:
#    - boston_housing_simple — одна модель
#    - boston_housing_experiments — 19 моделей параллельно
#    - boston_housing_cached — с кэшированием
```

### Способ 3: Классический CLI

```bash
# Локально
python src/modeling/train.py -n 200 -d 15

# Через Docker
docker-compose run --rm train -n 200 -d 15
```

---

## 📊 Просмотр результатов

### DVCLive метрики

```bash
# Просмотр метрик
cat dvclive/metrics.json

# Пример вывода:
# {
#   "train": {
#     "rmse": 2.456,
#     "r2": 0.8912,
#     "mae": 1.789
#   },
#   "test": {
#     "rmse": 3.129,
#     "r2": 0.8665,
#     "mae": 2.090
#   }
# }
```

### Hydra конфигурации и логи

```bash
# Конфигурация последнего запуска
cat outputs/$(ls -t outputs | head -1)/$(ls -t outputs/$(ls -t outputs | head -1) | head -1)/.hydra/config.yaml

# Логи обучения
cat outputs/$(ls -t outputs | head -1)/$(ls -t outputs/$(ls -t outputs | head -1) | head -1)/train_hydra.log
```

### MLflow UI

```bash
# Открыть MLflow UI
open http://localhost:5000

# Или через браузер:
# http://localhost:5000
```

### Airflow UI

```bash
# Открыть Airflow UI
open http://localhost:8080

# Или через браузер:
# http://localhost:8080
# Логин: admin, Пароль: admin
```

---

## 🐳 Управление Docker-инфраструктурой

```bash
# Запуск всех сервисов
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f airflow-worker
docker-compose logs -f mlflow

# Остановка
docker-compose down

# Полная очистка (включая volumes)
docker-compose down -v
```

### Доступные сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| **Airflow Web UI** | http://localhost:8080 | Управление ML пайплайнами |
| **MLflow UI** | http://localhost:5000 | Трекинг экспериментов |
| **MinIO Console** | http://localhost:9001 | S3-хранилище (admin/minioadmin) |

---

## 🔧 Полезные команды

```bash
# Список всех Makefile команд
make help

# Качество кода
make lint                # Проверка
make format              # Форматирование
make test                # Тесты

# DVC
make dvc-pull            # Загрузка данных
make dvc-push            # Отправка данных
make dvc-status          # Статус

# Docker
make docker-up           # Запуск
make docker-down         # Остановка
make docker-logs         # Логи
make docker-status       # Статус
```

---

## ❓ Что дальше?

1. **[Руководства](guides/index.md)** — изучите подробные гайды по всем компонентам
2. **[Примеры использования](examples/index.md)** — практические примеры
3. **[API Reference](api/index.md)** — документация кода
4. **[Развертывание](deployment/index.md)** — продвинутые настройки развертывания

---

## 🆘 Проблемы?

Если возникли проблемы:

1. Проверьте [Troubleshooting](reproducibility/troubleshooting.md)
2. Убедитесь, что все зависимости установлены: `make requirements`
3. Проверьте статус Docker: `docker-compose ps`
4. Посмотрите логи: `docker-compose logs`

---

## 🎓 Следующие шаги

- 📖 Изучите [управление конфигурациями Hydra](guides/CONFIGURATION_MANAGEMENT.md)
- 🔀 Настройте [Airflow ML Pipeline](guides/airflow_ml_pipeline.md)
- 📊 Запустите [продвинутые эксперименты](guides/EXPERIMENTS-ADVANCED.md)
- 🔄 Настройте [версионирование с DVC](guides/MINIO+DVC.md)
