# 🔐 Настройка файла .env

## 📋 Описание

Файл `.env` — это файл конфигурации переменных окружения, который хранит секретные данные и настройки для локальной разработки. Он **не должен** попадать в Git-репозиторий, так как содержит чувствительную информацию (пароли, ключи доступа).

### Роль .env в проекте

| Аспект | Описание |
|--------|----------|
| **Безопасность** | Хранение секретов вне кода (пароли, API-ключи) |
| **Гибкость** | Разные настройки для dev/staging/production окружений |
| **Docker** | Автоматическая загрузка переменных в контейнеры через `env_file` |
| **Python** | Загрузка через `python-dotenv` в `src/config.py` |

---

## 🚀 Быстрый старт

### 1. Создание файла

```bash
# Скопируйте шаблон (если есть) или создайте новый
cp .env.example .env

# Или создайте вручную
touch .env
```

### 2. Заполнение переменных

Откройте `.env` в редакторе и заполните по шаблону ниже.

---

## 📝 Шаблон .env

```env
# ============================================
# MinIO - S3-совместимое хранилище
# ============================================
# Учётные данные администратора MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# ============================================
# MLflow - Tracking Server
# ============================================
# Учётные данные администратора MLflow (для Basic Auth)
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=adminpassword123

# ============================================
# MLflow Client (для Python-скриптов)
# ============================================
# URL MLflow Tracking Server
MLFLOW_TRACKING_URI=http://localhost:5000

# Название эксперимента по умолчанию
MLFLOW_EXPERIMENT_NAME=boston-housing

# URL MinIO S3 API (для артефактов)
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# ============================================
# AWS/S3 Credentials (для MinIO)
# ============================================
# Используются клиентами для подключения к MinIO
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
```

---

## 📖 Описание переменных

### MinIO (S3-хранилище)

| Переменная | Описание | Пример |
|------------|----------|--------|
| `MINIO_ROOT_USER` | Логин администратора MinIO | `minioadmin` |
| `MINIO_ROOT_PASSWORD` | Пароль администратора MinIO (мин. 8 символов) | `minioadmin123` |

> ⚠️ **Важно**: `MINIO_ROOT_PASSWORD` должен быть не менее 8 символов!

### MLflow Server

| Переменная | Описание | Пример |
|------------|----------|--------|
| `MLFLOW_ADMIN_USERNAME` | Логин для входа в MLflow UI | `admin` |
| `MLFLOW_ADMIN_PASSWORD` | Пароль для входа в MLflow UI | `adminpassword123` |

### MLflow Client (Python)

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MLFLOW_TRACKING_URI` | URL MLflow сервера | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Название эксперимента | `boston-housing` |
| `MLFLOW_S3_ENDPOINT_URL` | URL MinIO S3 API | `http://localhost:9000` |

### AWS/S3 Credentials

| Переменная | Описание | Связь с MinIO |
|------------|----------|---------------|
| `AWS_ACCESS_KEY_ID` | Ключ доступа к S3 | = `MINIO_ROOT_USER` |
| `AWS_SECRET_ACCESS_KEY` | Секретный ключ S3 | = `MINIO_ROOT_PASSWORD` |

> 💡 **Совет**: `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` должны совпадать с учётными данными MinIO для корректной работы MLflow с артефактами.

---

## 🔧 Примеры конфигураций

### Локальная разработка (Development)

```env
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# MLflow
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=admin123

# Client
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=boston-housing-dev
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# S3
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
```

### Docker-окружение

```env
# MinIO
MINIO_ROOT_USER=minio_user
MINIO_ROOT_PASSWORD=minio_secure_pass_2024

# MLflow
MLFLOW_ADMIN_USERNAME=mlflow_admin
MLFLOW_ADMIN_PASSWORD=mlflow_secure_pass_2024

# Client (внутри Docker-сети используем имя сервиса)
MLFLOW_TRACKING_URI=http://nginx:80
MLFLOW_EXPERIMENT_NAME=boston-housing
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# S3
AWS_ACCESS_KEY_ID=minio_user
AWS_SECRET_ACCESS_KEY=minio_secure_pass_2024
```

### Production (рекомендации)

```env
# Используйте сложные пароли!
MINIO_ROOT_USER=prod_minio_admin
MINIO_ROOT_PASSWORD=Pr0d_M1n10_S3cur3_P@ssw0rd!

MLFLOW_ADMIN_USERNAME=prod_mlflow_admin
MLFLOW_ADMIN_PASSWORD=Pr0d_MLfl0w_S3cur3_P@ssw0rd!

# Production URLs
MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com
MLFLOW_S3_ENDPOINT_URL=https://s3.yourcompany.com

AWS_ACCESS_KEY_ID=prod_minio_admin
AWS_SECRET_ACCESS_KEY=Pr0d_M1n10_S3cur3_P@ssw0rd!
```

---

## 🔒 Безопасность

### ✅ Что нужно делать

1. **Добавьте `.env` в `.gitignore`**:
   ```gitignore
   # Secrets
   .env
   .env.local
   .env.*.local
   ```

2. **Создайте `.env.example`** — шаблон без реальных значений:
   ```env
   MINIO_ROOT_USER=your_minio_user
   MINIO_ROOT_PASSWORD=your_minio_password
   # ... и т.д.
   ```

3. **Используйте сложные пароли** в production:
   - Минимум 12 символов
   - Буквы, цифры, спецсимволы
   - Уникальные для каждого сервиса

### ❌ Чего избегать

- **НЕ** коммитьте `.env` в репозиторий
- **НЕ** используйте одинаковые пароли для разных окружений
- **НЕ** храните production-секреты в dev-окружении
- **НЕ** передавайте `.env` через мессенджеры/email

---

## 🐛 Решение проблем

### Переменные не загружаются

1. Проверьте, что файл называется именно `.env` (с точкой)
2. Убедитесь, что файл находится в корне проекта
3. Проверьте формат: `KEY=value` (без пробелов вокруг `=`)

### Docker не видит переменные

```yaml
# docker-compose.yml должен содержать:
services:
  your-service:
    env_file:
      - .env
```

### Python не загружает переменные

```python
# В начале скрипта:
from dotenv import load_dotenv
load_dotenv()  # Загружает .env автоматически
```

### MinIO отказывает в доступе

- Убедитесь, что `AWS_ACCESS_KEY_ID` = `MINIO_ROOT_USER`
- Убедитесь, что `AWS_SECRET_ACCESS_KEY` = `MINIO_ROOT_PASSWORD`
- Пароль MinIO должен быть >= 8 символов

---

## 📚 Связанные гайды

- [🐳 Docker](DOCKER.md) — запуск инфраструктуры
- [📦 MinIO + DVC](MINIO+DVC.md) — версионирование данных
- [📈 MLflow + DVC + MinIO](MLFLOW+DVC+MINIO.md) — трекинг экспериментов

---

## 📎 Полезные ссылки

- [python-dotenv](https://github.com/theskumar/python-dotenv) — библиотека для загрузки .env
- [Docker env_file](https://docs.docker.com/compose/environment-variables/) — документация Docker
- [12-Factor App: Config](https://12factor.net/config) — best practices хранения конфигов
