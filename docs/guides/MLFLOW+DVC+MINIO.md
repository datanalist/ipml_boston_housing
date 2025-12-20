# 🔬 MLflow + DVC + MinIO: Полное руководство

Это руководство описывает интеграцию **MLflow** для трекинга экспериментов с **DVC** для версионирования данных и **MinIO** для хранения артефактов.

## 📋 Содержание

1. [Архитектура решения](#архитектура-решения)
2. [Установка и настройка](#установка-и-настройка)
3. [Настройка MinIO для MLflow](#настройка-minio-для-mlflow)
4. [Запуск MLflow Tracking Server](#запуск-mlflow-tracking-server)
5. [Аутентификация и контроль доступа](#аутентификация-и-контроль-доступа)
6. [Интеграция MLflow с кодом](#интеграция-mlflow-с-кодом)
7. [Связка MLflow и DVC](#связка-mlflow-и-dvc)
8. [Workflow: полный цикл эксперимента](#workflow-полный-цикл-эксперимента)
9. [Примеры использования](#примеры-использования)
10. [Сравнение MLflow и DVCLive](#сравнение-mlflow-и-dvclive)
11. [Устранение неполадок](#устранение-неполадок)

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

### Шаг 3: Настройка политик доступа в MinIO (опционально)

Для production-окружения рекомендуется создать отдельного пользователя:

```bash
# Создание пользователя для MLflow
mc admin user add local mlflow_user mlflow_secret_password

# Создание файла политики доступа
# Примечание: если mc установлен через snap, используйте ~/mlflow-policy.json
# вместо /tmp/, т.к. snap не имеет доступа к /tmp
cat > ~/mlflow-policy.json << 'EOF'
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

# Добавление политики в MinIO (синтаксис для mc >= 2023)
mc admin policy add local mlflow-policy ~/mlflow-policy.json

# Назначение политики пользователю
mc admin policy set local mlflow-policy user=mlflow_user

# Проверка назначенных политик
mc admin user info local mlflow_user
```

> **Примечание**: Синтаксис команд `mc admin policy` зависит от версии MinIO Client.
> - Старые версии: `mc admin policy create/attach`
> - Новые версии (2023+): `mc admin policy add/set`

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

## Аутентификация и контроль доступа

По умолчанию MLflow Tracking Server не требует аутентификации. Для production-окружения необходимо настроить защиту на нескольких уровнях.

### Уровни защиты

| Уровень | Компонент | Метод защиты |
|---------|-----------|--------------|
| 1 | MinIO (S3) | Access Key + Secret Key |
| 2 | MLflow UI/API | Basic Auth / OAuth / Reverse Proxy |
| 3 | Сеть | Firewall, VPN, приватная сеть |

---

### Вариант 1: MLflow с Basic Auth в Docker (рекомендуется)

MLflow поддерживает встроенную аутентификацию начиная с версии 2.5+.

> **Важно**: Образ `ubuntu/mlflow:2.1.1` слишком старый для auth. Нужно использовать кастомный образ или официальный `ghcr.io/mlflow/mlflow:v2.18.0`.

---

#### Шаг 1: Создание Dockerfile для MLflow с auth

Создайте файл `docker/Dockerfile.mlflow`:

```dockerfile
FROM python:3.11-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка MLflow с auth и boto3 для S3
RUN pip install --no-cache-dir \
    mlflow[auth]==2.18.0 \
    boto3 \
    psycopg2-binary

# Создание директории для данных
RUN mkdir -p /mlflow/data

WORKDIR /mlflow

# Порт MLflow
EXPOSE 5000

# Точка входа
ENTRYPOINT ["mlflow", "server"]
```

#### Шаг 2: Создание конфигурации basic_auth.ini

Создайте директорию и файл `config/mlflow/basic_auth.ini`:

```bash
mkdir -p config/mlflow
```

```ini
[mlflow]
# Права по умолчанию для новых пользователей: READ, EDIT, MANAGE, NO_PERMISSIONS
default_permission = READ

# База данных для хранения пользователей (внутри контейнера)
database_uri = sqlite:////mlflow/data/auth.db

# Учётные данные администратора (ОБЯЗАТЕЛЬНО СМЕНИТЕ!)
admin_username = admin
admin_password = mlflow_admin_secure_password_123

# Функция авторизации
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
```

#### Шаг 3: Обновление docker-compose.yml

Замените секцию `mlflow` в `docker-compose.yml`:

```yaml
services:
  # ... minio service ...

  mlflow:
    build:
      context: ./docker
      dockerfile: Dockerfile.mlflow
    container_name: boston_housing_mlflow
    ports:
      - "5000:5000"
    environment:
      # Подключение к MinIO для артефактов
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
      - AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
      # Путь к конфигу аутентификации
      - MLFLOW_AUTH_CONFIG_PATH=/mlflow/config/basic_auth.ini
    command: >
      --app-name basic-auth
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/data/mlflow.db
      --default-artifact-root s3://mlflow-artifacts/
    volumes:
      # Конфиг аутентификации
      - ./config/mlflow/basic_auth.ini:/mlflow/config/basic_auth.ini:ro
      # Персистентное хранилище для БД
      - mlflow_data:/mlflow/data
    depends_on:
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    networks:
      - boston_housing_network
```

#### Шаг 4: Пересборка и запуск

```bash
# Пересобрать образ MLflow
docker-compose build mlflow

# Запустить сервисы
docker-compose up -d minio mlflow

# Проверить логи
docker-compose logs -f mlflow
```

При первом запуске создаётся admin-пользователь с учётными данными из `basic_auth.ini`:
- **Username**: `admin`
- **Password**: `mlflow_admin_secure_password_123`

#### Шаг 5: Проверка аутентификации

```bash
# Без авторизации — получим 401 Unauthorized
curl http://localhost:5000/api/2.0/mlflow/experiments/search
# {"error_code": "UNAUTHENTICATED", ...}

# С авторизацией — успех
curl -u admin:mlflow_admin_secure_password_123 \
    http://localhost:5000/api/2.0/mlflow/experiments/search
# {"experiments": [...]}
```

#### Шаг 6: Управление пользователями через API

```bash
# Создание нового пользователя
curl -X POST http://localhost:5000/api/2.0/mlflow/users/create \
    -H "Content-Type: application/json" \
    -u admin:mlflow_admin_secure_password_123 \
    -d '{"username": "data_scientist", "password": "ds_secure_pwd_456"}'

# Получение списка пользователей
curl -u admin:mlflow_admin_secure_password_123 \
    http://localhost:5000/api/2.0/mlflow/users/list

# Смена пароля пользователя
curl -X PATCH http://localhost:5000/api/2.0/mlflow/users/update-password \
    -H "Content-Type: application/json" \
    -u admin:mlflow_admin_secure_password_123 \
    -d '{"username": "data_scientist", "password": "new_password_789"}'

# Удаление пользователя
curl -X DELETE http://localhost:5000/api/2.0/mlflow/users/delete \
    -H "Content-Type: application/json" \
    -u admin:mlflow_admin_secure_password_123 \
    -d '{"username": "data_scientist"}'
```

#### Шаг 7: Подключение из Python-кода

```python
import os
import mlflow

# Способ 1: Через переменные окружения (рекомендуется)
os.environ["MLFLOW_TRACKING_USERNAME"] = "data_scientist"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ds_secure_pwd_456"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("boston-housing")

with mlflow.start_run(run_name="my-experiment"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("r2_score", 0.89)
```

Или через `.env` файл:

```bash
# .env (добавьте к существующим переменным)
MLFLOW_TRACKING_USERNAME=data_scientist
MLFLOW_TRACKING_PASSWORD=ds_secure_pwd_456
```

```python
from dotenv import load_dotenv
load_dotenv()

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
# Credentials подхватятся автоматически из окружения
```

#### Шаг 8: Обновление src/config/mlflow_config.py

```python
"""Конфигурация MLflow для проекта."""

import os


# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "boston-housing")

# Аутентификация MLflow
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# MinIO/S3 для артефактов
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin0")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin1230")


def setup_mlflow_env():
    """Настройка переменных окружения для MLflow + S3 + Auth."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    # Аутентификация (если заданы)
    if MLFLOW_TRACKING_USERNAME:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    if MLFLOW_TRACKING_PASSWORD:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
```

---

### Быстрый старт: MLflow с Auth в Docker

```bash
# 1. Создать структуру
mkdir -p config/mlflow docker

# 2. Создать Dockerfile.mlflow (см. выше)

# 3. Создать basic_auth.ini
cat > config/mlflow/basic_auth.ini << 'EOF'
[mlflow]
default_permission = READ
database_uri = sqlite:////mlflow/data/auth.db
admin_username = admin
admin_password = mlflow_admin_secure_password_123
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

# 4. Обновить docker-compose.yml (см. выше)

# 5. Собрать и запустить
docker-compose build mlflow
docker-compose up -d minio mlflow

# 6. Проверить
curl -u admin:mlflow_admin_secure_password_123 http://localhost:5000/api/2.0/mlflow/experiments/search

# 7. Добавить credentials в .env
echo 'MLFLOW_TRACKING_USERNAME=admin' >> .env
echo 'MLFLOW_TRACKING_PASSWORD=mlflow_admin_secure_password_123' >> .env

# Готово! 🎉
```

---

### Вариант 2: Nginx Reverse Proxy с Basic Auth

Для более гибкой настройки используйте Nginx.

#### Шаг 1: Создание файла паролей

```bash
# Установка apache2-utils (для htpasswd)
sudo apt install apache2-utils

# Создание файла паролей
htpasswd -c ./config/htpasswd admin
htpasswd ./config/htpasswd data_scientist
htpasswd ./config/htpasswd ml_engineer
```

#### Шаг 2: Конфигурация Nginx

Создайте файл `config/nginx.conf`:

```nginx
upstream mlflow {
    server mlflow:5000;
}

server {
    listen 80;
    server_name mlflow.localhost;

    # Basic Auth
    auth_basic "MLflow Tracking Server";
    auth_basic_user_file /etc/nginx/htpasswd;

    location / {
        proxy_pass http://mlflow;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (для live updates)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check endpoint (без авторизации)
    location /health {
        auth_basic off;
        proxy_pass http://mlflow/health;
    }
}
```

#### Шаг 3: Добавление Nginx в docker-compose.yml

```yaml
services:
  nginx:
    image: nginx:alpine
    container_name: boston_housing_nginx
    ports:
      - "8080:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./config/htpasswd:/etc/nginx/htpasswd:ro
    depends_on:
      - mlflow
    networks:
      - boston_housing_network

  mlflow:
    # ... (без публикации порта наружу)
    expose:
      - "5000"
    # ports: убрать!
```

#### Шаг 4: Подключение через Nginx

```bash
# Доступ через браузер с авторизацией
# http://localhost:8080

# Из кода
export MLFLOW_TRACKING_URI=http://localhost:8080
export MLFLOW_TRACKING_USERNAME=data_scientist
export MLFLOW_TRACKING_PASSWORD=your_password
```

---

### Вариант 3: OAuth 2.0 / OIDC (для корпоративных сред)

Для интеграции с корпоративными identity providers (Keycloak, Okta, Azure AD).

#### Шаг 1: Установка oauth-proxy

```yaml
services:
  oauth2-proxy:
    image: quay.io/oauth2-proxy/oauth2-proxy:v7.5.1
    container_name: boston_housing_oauth_proxy
    ports:
      - "4180:4180"
    environment:
      - OAUTH2_PROXY_PROVIDER=oidc
      - OAUTH2_PROXY_OIDC_ISSUER_URL=https://your-idp.example.com/realms/ml
      - OAUTH2_PROXY_CLIENT_ID=mlflow
      - OAUTH2_PROXY_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - OAUTH2_PROXY_COOKIE_SECRET=${COOKIE_SECRET}
      - OAUTH2_PROXY_UPSTREAMS=http://mlflow:5000
      - OAUTH2_PROXY_EMAIL_DOMAINS=*
      - OAUTH2_PROXY_HTTP_ADDRESS=0.0.0.0:4180
    depends_on:
      - mlflow
    networks:
      - boston_housing_network
```

---

### Настройка прав доступа (RBAC) в MLflow

MLflow поддерживает role-based access control начиная с версии 2.5+.

#### Доступные роли

| Роль | Права |
|------|-------|
| `READER` | Просмотр экспериментов и runs |
| `EDITOR` | Создание/редактирование runs, логирование метрик |
| `ADMIN` | Управление экспериментами, удаление |

#### Назначение прав на эксперимент

```python
from mlflow.server.auth import set_experiment_permission

# Дать права на эксперимент
set_experiment_permission(
    experiment_id="1",
    username="data_scientist",
    permission="EDITOR"
)

set_experiment_permission(
    experiment_id="1",
    username="ml_engineer",
    permission="READER"
)
```

Через REST API:

```bash
# Назначение прав
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/permissions/create \
    -H "Content-Type: application/json" \
    -u admin:password \
    -d '{
        "experiment_id": "1",
        "username": "data_scientist",
        "permission": "EDIT"
    }'
```

---

### Настройка учётных данных MinIO для команды

#### Создание отдельных пользователей MinIO

```bash
# Настройка alias
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230

# Создание пользователей для команды
mc admin user add local alice alice_secret_key
mc admin user add local bob bob_secret_key

# Создание групп
mc admin group add local data-scientists alice
mc admin group add local ml-engineers bob

# Создание политики только на чтение
cat > ~/readonly-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::*"]
    }
  ]
}
EOF

mc admin policy add local readonly ~/readonly-policy.json
mc admin policy set local readonly group=ml-engineers

# Полный доступ для data scientists
cat > ~/readwrite-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": ["arn:aws:s3:::mlflow-artifacts", "arn:aws:s3:::mlflow-artifacts/*"]
    }
  ]
}
EOF

mc admin policy add local readwrite ~/readwrite-policy.json
mc admin policy set local readwrite group=data-scientists
```

#### Файл .env для разных пользователей

```bash
# .env.alice
AWS_ACCESS_KEY_ID=alice
AWS_SECRET_ACCESS_KEY=alice_secret_key
MLFLOW_TRACKING_USERNAME=alice
MLFLOW_TRACKING_PASSWORD=alice_mlflow_password

# .env.bob
AWS_ACCESS_KEY_ID=bob
AWS_SECRET_ACCESS_KEY=bob_secret_key
MLFLOW_TRACKING_USERNAME=bob
MLFLOW_TRACKING_PASSWORD=bob_mlflow_password
```

---

### Рекомендации по безопасности

1. **Никогда не коммитьте credentials в Git**
   ```gitignore
   # .gitignore
   .env
   .env.*
   config/htpasswd
   *.ini
   ```

2. **Используйте переменные окружения или секреты**
   ```bash
   # Для CI/CD используйте GitHub Secrets, GitLab CI Variables и т.д.
   export MLFLOW_TRACKING_PASSWORD=$MLFLOW_SECRET
   ```

3. **Регулярно ротируйте ключи**
   ```bash
   # Смена пароля MinIO пользователя
   mc admin user update local alice new_secret_key
   ```

4. **Ограничьте сетевой доступ**
   - MLflow и MinIO должны быть доступны только из внутренней сети
   - Используйте VPN для удалённого доступа
   - Настройте firewall rules

5. **Включите TLS/HTTPS**
   ```yaml
   # Для production обязательно используйте HTTPS
   nginx:
     volumes:
       - ./certs:/etc/nginx/certs:ro
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
