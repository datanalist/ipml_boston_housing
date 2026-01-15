# Troubleshooting

Решения распространенных проблем при работе с проектом.

---

## 🐍 Проблемы с Python

### Проблема: Python 3.13 не найден

**Симптомы:**
```bash
python: command not found
# или
Python 3.12 instead of 3.13
```

**Решение:**

=== "Ubuntu/Debian"

    ```bash
    sudo apt update
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.13 python3.13-venv python3.13-dev
    ```

=== "macOS"

    ```bash
    brew install python@3.13
    echo 'export PATH="/opt/homebrew/opt/python@3.13/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```

=== "pyenv"

    ```bash
    curl https://pyenv.run | bash
    pyenv install 3.13
    pyenv global 3.13
    ```

---

### Проблема: uv не найден

**Симптомы:**
```bash
uv: command not found
```

**Решение:**

```bash
# Установка uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Добавить в PATH
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Проверка
uv --version
```

---

### Проблема: Виртуальное окружение не активируется

**Симптомы:**
```bash
which python  # показывает системный Python
```

**Решение:**

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (WSL2)
source .venv/bin/activate

# Если не помогло, пересоздайте окружение
rm -rf .venv
uv venv
source .venv/bin/activate
```

---

## 📦 Проблемы с зависимостями

### Проблема: Ошибки при установке пакетов

**Симптомы:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Решение:**

```bash
# Очистите кэш
uv cache clean

# Пересоздайте окружение
rm -rf .venv
uv venv
source .venv/bin/activate

# Обновите uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установите зависимости заново
uv sync
```

---

### Проблема: Конфликт зависимостей

**Симптомы:**
```
ERROR: Cannot install package-a and package-b because these package versions have conflicting dependencies
```

**Решение:**

```bash
# Проверьте uv.lock
cat uv.lock | grep conflicting-package

# Обновите конкретный пакет
uv add package-name --upgrade

# Или пересоздайте lock
rm uv.lock
uv sync
```

---

## 🐳 Проблемы с Docker

### Проблема: Docker не запускается

**Симптомы:**
```bash
Cannot connect to the Docker daemon
```

**Решение:**

```bash
# Запустите Docker daemon
sudo systemctl start docker

# Проверьте статус
sudo systemctl status docker

# Добавьте пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker

# Проверка
docker ps
```

---

### Проблема: Порты заняты

**Симптомы:**
```
ERROR: for airflow-webserver  Cannot start service: Ports are not available: listen tcp 0.0.0.0:8080: bind: address already in use
```

**Решение:**

```bash
# Найдите процесс, занимающий порт
sudo lsof -i :8080

# Убейте процесс
sudo kill -9 PID

# Или измените порт в docker-compose.yml
ports:
  - "8081:8080"  # Вместо 8080:8080
```

---

### Проблема: Контейнеры падают

**Симптомы:**
```bash
docker-compose ps
# Показывает Exit 1 или Restarting
```

**Решение:**

```bash
# Посмотрите логи
docker-compose logs airflow-worker

# Увеличьте память для Docker Desktop
# Settings → Resources → Memory: 8GB+

# Перезапустите контейнеры
docker-compose down
docker-compose up -d

# Если не помогает, пересоздайте
docker-compose down -v
docker-compose up -d
```

---

### Проблема: Network ошибки

**Симптомы:**
```
ERROR: Network boston_housing_network declared as external, but could not be found
```

**Решение:**

```bash
# Создайте network вручную
docker network create boston_housing_network

# Или используйте Makefile
make docker-recreate

# Проверьте networks
docker network ls
```

---

## 📊 Проблемы с данными

### Проблема: Данные не найдены

**Симптомы:**
```python
FileNotFoundError: data/raw/housing.csv
```

**Решение:**

```bash
# Попробуйте DVC
make dvc-pull

# Если DVC не настроен, загрузите напрямую
make download-data

# Проверьте наличие файла
ls -lh data/raw/housing.csv
```

---

### Проблема: DVC ошибки

**Симптомы:**
```
ERROR: failed to pull data from the cloud - connection error
```

**Решение:**

```bash
# Проверьте конфигурацию DVC
dvc remote list
dvc remote --local list

# Проверьте доступность MinIO
curl http://localhost:9000/minio/health/live

# Если MinIO недоступен, используйте прямую загрузку
make download-data-force
```

---

## ⚙️ Проблемы с конфигурацией

### Проблема: Hydra не находит конфигурацию

**Симптомы:**
```python
hydra.errors.MissingConfigException: Cannot find primary config 'config'
```

**Решение:**

```bash
# Проверьте структуру
ls conf/config.yaml

# Убедитесь, что запускаете из корня проекта
cd /path/to/ipml_boston_housing
python src/modeling/train_hydra.py

# Или укажите путь явно
python src/modeling/train_hydra.py --config-path=../conf
```

---

### Проблема: Pydantic валидация не проходит

**Симптомы:**
```python
pydantic.error_wrappers.ValidationError: 1 validation error for RandomForestConfig
```

**Решение:**

```bash
# Проверьте значения параметров
cat conf/model/random_forest.yaml

# Убедитесь, что значения в допустимых диапазонах
# Например, n_estimators: 10-1000, max_depth: 1-50

# Исправьте конфигурацию
vim conf/model/random_forest.yaml
```

---

## 🔄 Проблемы с Airflow

### Проблема: DAG не появляется в UI

**Симптомы:**
DAG файл создан, но не отображается в Airflow UI

**Решение:**

```bash
# Проверьте синтаксис DAG
python airflow/dags/your_dag.py

# Проверьте логи scheduler
docker-compose logs airflow-scheduler

# Перезапустите scheduler
docker-compose restart airflow-scheduler

# Проверьте в Airflow UI: Browse → DAG Errors
```

---

### Проблема: Task падает с ошибкой

**Симптомы:**
Task показывает статус "Failed" в UI

**Решение:**

```bash
# Посмотрите логи task в UI
# Graph View → Кликните на task → Log

# Или через CLI
docker-compose exec airflow-webserver airflow tasks logs boston_housing_simple train_model

# Увеличьте timeout если task долго выполняется
# В DAG:
task = PythonOperator(
    task_id='train',
    execution_timeout=timedelta(minutes=30),  # Увеличьте
)
```

---

## 📈 Проблемы с MLflow

### Проблема: MLflow UI недоступен

**Симптомы:**
```
curl: (7) Failed to connect to localhost port 5000: Connection refused
```

**Решение:**

```bash
# Проверьте статус контейнеров
docker-compose ps mlflow nginx

# Проверьте логи
docker-compose logs mlflow
docker-compose logs nginx

# Перезапустите сервисы
docker-compose restart mlflow nginx

# Проверьте basic auth
curl -u admin:password http://localhost:5000
```

---

### Проблема: Эксперименты не логируются

**Симптомы:**
Код выполняется, но в MLflow UI ничего нет

**Решение:**

```python
# Проверьте переменные окружения
import os
print(os.getenv("MLFLOW_TRACKING_URI"))

# Должно быть: http://localhost:5000 (или nginx:80 внутри Docker)

# Установите явно
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# Проверьте эксперимент
mlflow.set_experiment("boston_housing")
```

---

## 🔧 Проблемы с pre-commit

### Проблема: Pre-commit хуки не запускаются

**Симптомы:**
Коммит проходит без проверок

**Решение:**

```bash
# Установите хуки
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# Проверьте установку
ls .git/hooks/pre-commit

# Запустите вручную
pre-commit run --all-files
```

---

### Проблема: Ruff ошибки

**Симптомы:**
```
ruff: error: Found 10 errors
```

**Решение:**

```bash
# Автоматическое исправление
ruff check --fix .

# Форматирование
ruff format .

# Проверка
ruff check .
```

---

## 💾 Проблемы с памятью

### Проблема: Out of Memory

**Симптомы:**
```
MemoryError: Unable to allocate array
# или
Killed
```

**Решение:**

```bash
# Увеличьте память для Docker
# Docker Desktop → Settings → Resources → Memory: 8GB+

# Уменьшите параллелизм Airflow
# В .env:
AIRFLOW__CORE__PARALLELISM=4  # Вместо 16
AIRFLOW__CELERY__WORKER_CONCURRENCY=2  # Вместо 4

# Обучайте модели последовательно
python src/modeling/train_hydra.py model=rf
# Вместо multirun
```

---

## 🆘 Общие рекомендации

### Если ничего не помогает

1. **Полная переустановка:**

```bash
# Удалите все
rm -rf .venv
rm -rf outputs
docker-compose down -v

# Установите заново
make setup
```

2. **Проверьте окружение:**

```bash
python scripts/check_environment.py
```

3. **Проверьте версии:**

```bash
python --version
uv --version
docker --version
docker-compose --version
```

4. **Посмотрите логи:**

```bash
# Python
cat outputs/*/*/train_hydra.log

# Docker
docker-compose logs

# Airflow
docker-compose logs airflow-scheduler airflow-worker
```

5. **Создайте Issue:**

Если проблема не решается, создайте [Issue на GitHub](https://github.com/yourusername/ipml_boston_housing/issues) с:
- Описанием проблемы
- Командами, которые выполняли
- Полным выводом ошибки
- Версиями Python, Docker, ОС

---

## 📚 Дополнительные ресурсы

- [Пошаговая инструкция](step-by-step.md) — детальное руководство
- [Проверка зависимостей](dependencies.md) — проверка установки
- [Воспроизводимость](index.md) — полная инструкция
- [Быстрый старт](../getting-started.md) — минимальная установка
