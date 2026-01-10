# Локальная установка

Этот раздел содержит инструкции по локальной установке проекта без использования Docker.

---

## 📋 Системные требования

- **OS**: Linux, macOS, Windows (WSL2)
- **Python**: 3.13+
- **RAM**: минимум 4 GB
- **Дисковое пространство**: минимум 2 GB

---

## 🚀 Пошаговая установка

### Шаг 1: Установка uv

[uv](https://docs.astral.sh/uv/) — современный пакетный менеджер для Python.

=== "Linux/macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (WSL2)"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "pip"

    ```bash
    pip install uv
    ```

Проверьте установку:

```bash
uv --version
```

### Шаг 2: Клонирование репозитория

```bash
git clone https://github.com/yourusername/ipml_boston_housing
cd ipml_boston_housing
```

### Шаг 3: Создание виртуального окружения

```bash
# Автоматически с помощью Makefile
make create_environment

# Или вручную
uv venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate  # Windows
```

### Шаг 4: Установка зависимостей

```bash
# Через Makefile (рекомендуется)
make requirements

# Или вручную
uv sync
uv sync --group dev
uv sync --group docs
```

### Шаг 5: Настройка pre-commit хуков

```bash
make pre-commit

# Или вручную
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

### Шаг 6: Загрузка данных

```bash
# Через DVC
make dvc-pull

# Или напрямую (если DVC не настроен)
make download-data
```

---

## ✅ Проверка установки

### Автоматическая проверка

```bash
python scripts/check_environment.py
```

Ожидаемый вывод:

```
✓ Python version: 3.13.x
✓ uv installed
✓ Virtual environment: active
✓ Dependencies installed: 28/28
✓ Data files present
✓ Configuration files valid
✓ Pre-commit hooks installed

All checks passed! ✓
```

### Ручная проверка

```bash
# Проверка Python
python --version

# Проверка установленных пакетов
uv pip list

# Проверка данных
ls data/raw/housing.csv

# Проверка конфигураций
ls conf/config.yaml
```

---

## 🎯 Первый запуск

### Базовое обучение модели

```bash
uv run python src/modeling/train_hydra.py
```

### Просмотр результатов

```bash
# Метрики DVCLive
cat dvclive/metrics.json

# Логи Hydra
ls -ltr outputs/
cat outputs/*/*/train_hydra.log
```

---

## ⚙️ Настройка (опционально)

### Настройка DVC с MinIO (локально)

Если вы хотите использовать локальное хранилище MinIO без Docker:

1. Установите MinIO:

```bash
# Linux
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/

# macOS
brew install minio/stable/minio

# Запуск
minio server ~/minio-data --console-address ":9001"
```

2. Настройте DVC:

```bash
# Создайте bucket через MinIO Console (http://localhost:9001)
# Затем настройте DVC remote
dvc remote modify myremote endpointurl http://localhost:9000
dvc remote modify myremote access_key_id minioadmin
dvc remote modify myremote secret_access_key minioadmin
```

### Настройка MLflow (локально)

```bash
# Запуск MLflow сервера
uv run mlflow ui --host 0.0.0.0 --port 5000 &

# Или с backend storage
uv run mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
```

Откройте http://localhost:5000 для просмотра экспериментов.

---

## 🔧 Управление зависимостями

### Добавление новых пакетов

```bash
# Добавить зависимость
uv add package-name

# Добавить dev-зависимость
uv add --group dev package-name

# Добавить docs-зависимость
uv add --group docs package-name

# Обновить зависимости
uv sync
```

### Обновление зависимостей

```bash
# Обновить все пакеты
uv sync --upgrade

# Обновить конкретный пакет
uv add --upgrade package-name

# Заморозить зависимости
uv lock
```

---

## 🐛 Возможные проблемы

### Проблема: Python 3.13 не найден

**Решение:**

```bash
# Установка через pyenv
curl https://pyenv.run | bash
pyenv install 3.13
pyenv global 3.13

# Проверка
python --version
```

### Проблема: uv не найден после установки

**Решение:**

```bash
# Добавьте в PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Для постоянного эффекта добавьте в ~/.bashrc или ~/.zshrc
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Проблема: Ошибка при установке зависимостей

**Решение:**

```bash
# Очистите кэш
uv cache clean

# Пересоздайте окружение
rm -rf .venv
uv venv
source .venv/bin/activate
uv sync
```

### Проблема: Данные не загружаются через DVC

**Решение:**

```bash
# Загрузите данные напрямую
make download-data-force

# Или вручную скачайте с Kaggle
# https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
# Поместите housing.csv в data/raw/
```

---

## 📚 Следующие шаги

После успешной установки:

1. 📖 Изучите [примеры использования](../examples/index.md)
2. ⚙️ Настройте [управление конфигурациями](../guides/CONFIGURATION_MANAGEMENT.md)
3. 🚀 Запустите [эксперименты](../guides/EXPERIMENTS.md)
4. 📊 Изучите [трекинг с MLflow](../guides/MLFLOW+DVC+MINIO.md)

---

## 🆘 Нужна помощь?

- [Troubleshooting](../reproducibility/troubleshooting.md) — решение распространенных проблем
- [Проверка зависимостей](../reproducibility/dependencies.md) — детальная проверка
- [GitHub Issues](https://github.com/yourusername/ipml_boston_housing/issues) — сообщите о проблеме
