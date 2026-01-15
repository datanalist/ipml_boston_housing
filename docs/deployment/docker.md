# Docker развертывание

Этот раздел содержит подробные инструкции по развертыванию проекта с использованием Docker Compose.

---

## 📋 Системные требования

- **OS**: Linux, macOS, Windows (с WSL2)
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **RAM**: минимум 8 GB (рекомендуется 16 GB)
- **CPU**: минимум 4 ядра
- **Дисковое пространство**: минимум 10 GB

---

## 🏗️ Архитектура

Проект использует Docker Compose для оркестрации следующих сервисов:

```
┌────────────────────────────────────────────────────────────────┐
│                    boston_housing_network                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              AIRFLOW ORCHESTRATION                        │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐                 │  │
│  │  │Webserver│  │Scheduler │  │ Worker  │                 │  │
│  │  │  :8080  │  │          │  │(Celery) │                 │  │
│  │  └────┬────┘  └────┬─────┘  └────┬────┘                 │  │
│  │       └────────────┼─────────────┘                       │  │
│  │                    │                                      │  │
│  │       ┌────────────┴────────────┐                        │  │
│  │       │                         │                        │  │
│  │  ┌────▼────┐            ┌───────▼──────┐                │  │
│  │  │PostgreSQL│            │    Redis     │                │  │
│  │  │:5432     │            │    :6379     │                │  │
│  │  └──────────┘            └──────────────┘                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             ML TRACKING & STORAGE                         │  │
│  │  ┌────────┐   ┌────────┐   ┌──────────────┐             │  │
│  │  │ MinIO  │   │ Nginx  │   │    MLflow    │             │  │
│  │  │ :9000  │   │ :5000  │──▶│   Tracking   │             │  │
│  │  │ :9001  │   │(BasicAuth)│   sqlite+S3   │             │  │
│  │  └────┬───┘   └────────┘   └──────────────┘             │  │
│  │       └──────────────────────────────────────            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│                        ┌──────────────┐                         │
│                        │Train (CLI)   │                         │
│                        │Python + uv   │                         │
│                        └──────────────┘                         │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Быстрый старт

### Установка Docker

=== "Ubuntu/Debian"

    ```bash
    # Обновление пакетов
    sudo apt-get update

    # Установка Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh

    # Добавление пользователя в группу docker
    sudo usermod -aG docker $USER
    newgrp docker

    # Проверка
    docker --version
    docker-compose --version
    ```

=== "macOS"

    ```bash
    # Установка Docker Desktop
    brew install --cask docker

    # Запустите Docker Desktop из Applications
    # Проверка
    docker --version
    docker-compose --version
    ```

=== "Windows (WSL2)"

    1. Установите [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
    2. Включите WSL2 backend в настройках Docker Desktop
    3. Проверьте установку в WSL2:

    ```bash
    docker --version
    docker-compose --version
    ```

### Запуск инфраструктуры

```bash
# 1. Клонирование репозитория
git clone https://github.com/yourusername/ipml_boston_housing
cd ipml_boston_housing

# 2. Создание .env файла (если нужно)
cp .env.example .env

# 3. Запуск всех сервисов
make docker-up

# Или напрямую
docker-compose up -d
```

### Проверка статуса

```bash
# Через Makefile
make docker-status

# Или напрямую
docker-compose ps
```

Ожидаемый вывод:

```
NAME                          STATUS              PORTS
airflow-webserver             Up 30 seconds       0.0.0.0:8080->8080/tcp
airflow-scheduler             Up 30 seconds       8080/tcp
airflow-worker                Up 30 seconds       8080/tcp
postgres                      Up 31 seconds       5432/tcp
redis                         Up 31 seconds       6379/tcp
minio                         Up 31 seconds       0.0.0.0:9000-9001->9000-9001/tcp
nginx                         Up 30 seconds       0.0.0.0:5000->80/tcp
```

---

## 🌐 Доступ к сервисам

После запуска доступны следующие веб-интерфейсы:

| Сервис | URL | Логин | Пароль | Описание |
|--------|-----|-------|--------|----------|
| **Airflow UI** | http://localhost:8080 | `admin` | `admin` | Управление ML пайплайнами |
| **MLflow UI** | http://localhost:5000 | `admin` | `password` | Трекинг экспериментов |
| **MinIO Console** | http://localhost:9001 | `minioadmin` | `minioadmin` | S3-хранилище |

### Airflow Web UI

```bash
# Откройте в браузере
open http://localhost:8080

# Или через curl для проверки
curl http://localhost:8080/health
```

**Что можно делать:**
- Просмотр и запуск DAG
- Мониторинг выполнения задач
- Просмотр логов
- Управление подключениями и переменными

### MLflow UI

```bash
# Откройте в браузере
open http://localhost:5000

# Basic Auth: admin / password
```

**Что можно делать:**
- Просмотр экспериментов
- Сравнение моделей
- Визуализация метрик
- Скачивание артефактов

### MinIO Console

```bash
# Откройте в браузере
open http://localhost:9001
```

**Что можно делать:**
- Управление buckets
- Загрузка/скачивание файлов
- Управление доступом
- Просмотр статистики

---

## 🎯 Запуск экспериментов

### Через Airflow DAG

1. Откройте Airflow UI: http://localhost:8080
2. Найдите нужный DAG:
   - `boston_housing_simple` — базовый пайплайн
   - `boston_housing_experiments` — 19 моделей параллельно
   - `boston_housing_cached` — с кэшированием
3. Нажмите кнопку "Trigger DAG"
4. Следите за выполнением в Graph View

### Через CLI контейнер

```bash
# Запуск обучения в контейнере
docker-compose run --rm train python src/modeling/train_hydra.py

# С параметрами
docker-compose run --rm train python src/modeling/train_hydra.py \
    model=gradient_boosting \
    model.n_estimators=300

# Multirun
docker-compose run --rm train python src/modeling/train_hydra.py \
    --multirun model=ridge,lasso,elastic_net
```

---

## 📊 Мониторинг и логи

### Просмотр логов

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-worker
docker-compose logs -f mlflow
docker-compose logs -f minio

# Последние N строк
docker-compose logs --tail=100 airflow-worker
```

### Выполнение команд в контейнерах

```bash
# Airflow CLI
docker-compose exec airflow-webserver airflow dags list
docker-compose exec airflow-webserver airflow tasks list boston_housing_experiments

# Bash в контейнере
docker-compose exec airflow-webserver bash
docker-compose exec airflow-worker bash

# Python в контейнере train
docker-compose run --rm train python
```

### Проверка здоровья сервисов

```bash
# Health check всех сервисов
docker-compose ps

# Health check через API
curl http://localhost:8080/health  # Airflow
curl http://localhost:9000/minio/health/live  # MinIO
```

---

## 🔧 Управление Docker

### Остановка и запуск

```bash
# Остановка всех сервисов
make docker-down
# или
docker-compose down

# Остановка с удалением volumes (ОСТОРОЖНО: удалит данные!)
docker-compose down -v

# Остановка конкретного сервиса
docker-compose stop airflow-worker

# Запуск остановленных сервисов
docker-compose start
```

### Перезапуск

```bash
# Перезапуск всех сервисов
make docker-restart
# или
docker-compose restart

# Перезапуск конкретного сервиса
docker-compose restart airflow-scheduler
docker-compose restart airflow-worker
```

### Пересборка образов

```bash
# Пересборка всех образов
make docker-build
# или
docker-compose build

# Пересборка конкретного сервиса
docker-compose build airflow-webserver

# Пересборка без кэша
docker-compose build --no-cache
```

### Очистка

```bash
# Удаление неиспользуемых образов
docker image prune

# Удаление всех остановленных контейнеров
docker container prune

# Полная очистка Docker (ОСТОРОЖНО!)
docker system prune -a --volumes
```

---

## ⚙️ Конфигурация

### Переменные окружения

Создайте файл `.env` в корне проекта:

```bash
# Airflow
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
# pragma: allowlist secret
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
# pragma: allowlist secret
AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow
AIRFLOW__CELERY__BROKER_URL=redis://:@redis:6379/0
AIRFLOW__CORE__FERNET_KEY=
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth

# MLflow
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
# pragma: allowlist secret
AWS_ACCESS_KEY_ID=minioadmin
# pragma: allowlist secret
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_TRACKING_URI=http://nginx:80

# MinIO
# pragma: allowlist secret
MINIO_ROOT_USER=minioadmin
# pragma: allowlist secret
MINIO_ROOT_PASSWORD=minioadmin

# Python
PYTHONUNBUFFERED=1
```

### Настройка ресурсов

Отредактируйте `docker-compose.yml` для изменения лимитов:

```yaml
services:
  airflow-worker:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Volumes

Данные сохраняются в именованных volumes:

```bash
# Список volumes
docker volume ls | grep boston

# Инспекция volume
docker volume inspect boston_housing_postgres-db-volume

# Резервное копирование volume
docker run --rm \
  -v boston_housing_postgres-db-volume:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/postgres-backup.tar.gz -C /data .

# Восстановление volume
docker run --rm \
  -v boston_housing_postgres-db-volume:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/postgres-backup.tar.gz -C /data
```

---

## 🐛 Troubleshooting

### Проблема: Сервисы не запускаются

**Решение:**

```bash
# Проверьте логи
docker-compose logs

# Проверьте порты
sudo lsof -i :8080
sudo lsof -i :5000

# Пересоздайте контейнеры
docker-compose down -v
docker-compose up -d
```

### Проблема: Airflow worker падает

**Решение:**

```bash
# Увеличьте память для Docker
# Docker Desktop → Settings → Resources → Memory: 8GB+

# Проверьте логи worker
docker-compose logs airflow-worker

# Перезапустите worker
docker-compose restart airflow-worker
```

### Проблема: MinIO недоступен

**Решение:**

```bash
# Проверьте статус
docker-compose ps minio

# Проверьте health
curl http://localhost:9000/minio/health/live

# Пересоздайте MinIO
docker-compose stop minio
docker-compose rm minio
docker-compose up -d minio
```

### Проблема: MLflow не может подключиться к MinIO

**Решение:**

```bash
# Проверьте переменные окружения
docker-compose exec mlflow env | grep MINIO

# Проверьте доступность MinIO изнутри контейнера
docker-compose exec mlflow curl http://minio:9000/minio/health/live

# Пересоздайте network
make docker-recreate
```

---

## 📚 Следующие шаги

После успешного развертывания:

1. 🔀 Изучите [Airflow ML Pipeline](../guides/airflow_ml_pipeline.md)
2. 📊 Настройте [MLflow трекинг](../guides/MLFLOW+DVC+MINIO.md)
3. 🚀 Запустите [продвинутые эксперименты](../guides/EXPERIMENTS-ADVANCED.md)
4. 📈 Создавайте [отчеты об экспериментах](../reports/index.md)

---

## 🆘 Нужна помощь?

- [Troubleshooting](../reproducibility/troubleshooting.md) — решение распространенных проблем
- [Docker Guide](../guides/DOCKER.md) — подробное руководство по Docker
- [GitHub Issues](https://github.com/yourusername/ipml_boston_housing/issues) — сообщите о проблеме
