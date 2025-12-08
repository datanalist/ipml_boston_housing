# 📊 Отчёт ДЗ №2: Версионирование данных и моделей

**Проект:** Boston Housing Price Prediction  
**Дата:** Декабрь 2025  

---

## ✅ Чек-лист выполнения

### Настройка инструмента для данных 

| Задача | Статус | Реализация |
|--------|--------|------------|
| Установить и настроить DVC | ✅ | `uv add dvc[s3] dvc-s3` |
| Настроить remote storage (S3/Local) | ✅ | MinIO через Docker |
| Создать систему версионирования данных | ✅ | `data/raw.dvc`, `data/models.dvc` |
| Настроить автоматическое создание версий | ✅ | DVCLive с `save_dvc_exp=True` |

### Настройка инструмента для моделей 

| Задача | Статус | Реализация |
|--------|--------|------------|
| Настроить инструмент для моделей | ✅ | DVCLive в `train.py` |
| Создать систему версионирования моделей | ✅ | `data/models/random_forest.pkl.dvc` |
| Настроить метаданные для моделей | ✅ | `dvclive/params.yaml`, `metrics.json` |
| Создать систему сравнения версий | ✅ | `dvc exp show`, `dvc exp diff` |

### Воспроизводимость (2 балла)

| Задача | Статус | Реализация |
|--------|--------|------------|
| Создать инструкции по воспроизведению | ✅ | `docs/guides/*.md` |
| Настроить фиксацию версий зависимостей | ✅ | `pyproject.toml` + `uv.lock` |
| Протестировать воспроизводимость | ✅ | `random_state=42` |
| Создать Docker контейнер | ✅ | `Dockerfile.app`, `Dockerfile.minio` |

### Отчёт 

| Задача | Статус |
|--------|--------|
| Создать отчёт в формате Markdown | ✅ |
| Описать настройку инструментов | ✅ |
| Добавить скриншоты результатов | ✅ |
| Сохранить отчёт в Git | ✅ |



---

## 1. Версионирование данных (DVC + MinIO)

### 1.1 Установка

```bash
uv add dvc[s3] dvc-s3 dvclive
dvc init
```

### 1.2 Настройка MinIO

**docker-compose.yml:**
```yaml
services:
  minio:
    build:
      context: ./docker
      dockerfile: Dockerfile.minio
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # Web Console
    volumes:
      - ./minio_data:/data
```

**Запуск:**
```bash
docker-compose up -d minio
```

**Доступ:** http://localhost:9001 (minioadmin0 / minioadmin1230)

### 1.3 Подключение DVC к MinIO

```bash
dvc remote add -d minio s3://boston-housing-data
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id minioadmin0
dvc remote modify --local minio secret_access_key minioadmin1230
dvc remote modify minio use_ssl false
```

**Результат (.dvc/config):**
```ini
[core]
    remote = minio
['remote "minio"']
    url = s3://boston-housing-data
    endpointurl = http://localhost:9000
    use_ssl = false
```

### 1.4 Версионирование данных

```bash
# Добавление данных
dvc add data/raw
dvc add data/models/random_forest.pkl

# Отправка в MinIO
dvc push

# Получение данных
dvc pull
```

**Структура:**
```
data/
├── raw/                      # Исходные данные
│   └── housing.csv
├── raw.dvc                   # MD5: 040008edfc98ff4a18d0e870096bb2ef
├── models/
│   ├── random_forest.pkl
│   └── random_forest.pkl.dvc
└── models.dvc
```

### 1.5 Скриншот MinIO

![MinIO](../image.png)

*Веб-консоль MinIO с бакетом boston-housing-data*

---

## 2. Версионирование моделей (DVCLive)

### 2.1 Интеграция в код

**src/modeling/train.py:**
```python
from dvclive import Live

with Live(save_dvc_exp=True) as live:
    # Логирование параметров
    live.log_param("n_estimators", 200)
    live.log_param("max_depth", 15)
    
    # Обучение модели
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Логирование метрик
    live.log_metric("r2_score", 0.866)
    live.log_metric("rmse", 3.13)
    
    # Логирование артефакта
    live.log_artifact("data/models/random_forest.pkl", type="model")
```

### 2.2 Конфигурация пайплайна

**dvc.yaml:**
```yaml
params:
- dvclive/params.yaml
metrics:
- dvclive/metrics.json
plots:
- dvclive/plots/metrics:
    x: step
artifacts:
  random_forest:
    path: data/models/random_forest.pkl
    type: model
```

### 2.3 Метаданные модели

**dvclive/params.yaml:**
```yaml
n_estimators: 200
max_depth: 15
min_samples_split: 5
min_samples_leaf: 2
random_state: 42
test_size: 0.2
n_samples: 506
n_features: 13
train_size: 404
test_size_actual: 102
```

**dvclive/metrics.json:**
```json
{
    "r2_score": 0.8664603178027023,
    "rmse": 3.1293721570875954,
    "mae": 2.0902853281690428,
    "mape": 11.337048938411472
}
```

### 2.4 Сравнение версий

```bash
# История экспериментов
dvc exp show

# Сравнение экспериментов
dvc exp diff <exp1> <exp2>

# Текущие метрики
dvc metrics show
```

---

## 3. Воспроизводимость (Docker)

### 3.1 Архитектура

```
┌─────────────────────────────────────────────┐
│           Docker Network                     │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐  │
│  │    MinIO     │    │      Train       │  │
│  │  :9000/:9001 │    │  Python 3.13     │  │
│  │  S3 Storage  │    │  uv + sklearn    │  │
│  └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────┘
```

### 3.2 ML-контейнер

**docker/Dockerfile.app:**
```dockerfile
FROM python:3.13-slim

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
RUN mkdir -p data/raw data/models dvclive

ENTRYPOINT ["uv", "run", "python", "src/modeling/train.py"]
CMD ["--n-estimators", "100", "--max-depth", "10"]
```

### 3.3 Фиксация зависимостей

**pyproject.toml:**
```toml
[project]
requires-python = ">=3.13"
dependencies = [
    "dvc[s3]>=3.64.2",
    "dvclive>=3.49.0",
    "scikit-learn>=1.7.2",
    # ...
]
```

**uv.lock** — 2600+ строк с точными версиями и SHA256-хешами.

### 3.4 Запуск

```bash
# Сборка
docker-compose build train

# Запуск с параметрами
docker-compose run --rm train -n 200 -d 15 -t 0.25

# Результаты сохраняются в ./dvclive/ и ./data/models/
```

---

## 4. Инструкции по воспроизведению

### Полный цикл

```bash
# 1. Клонирование
git clone git@github.com:datanalist/ipml_boston_housing.git
cd ipml_boston_housing

# 2. Окружение
uv sync

# 3. Инфраструктура
docker-compose up -d minio

# 4. Данные
dvc pull

# 5. Обучение
docker-compose run --rm train -n 200 -d 15

# 6. Результаты
cat dvclive/metrics.json

# 7. Сохранение
dvc add data/models/random_forest.pkl
git add . && git commit -m "exp: RF n=200 d=15"
dvc push && git push
```

---

## 5. Результаты

| Метрика | Значение | Качество |
|---------|----------|----------|
| R² Score | 0.866 | ✅ Хорошо (>0.80) |
| RMSE | 3.13 | ✅ Приемлемо |
| MAE | 2.09 | ✅ Хорошо (<2.5) |
| MAPE | 11.3% | ✅ Хорошо (10-15%) |

---

## 📚 Документация

- [MinIO + DVC](../docs/guides/MINIO+DVC.md)
- [Эксперименты](../docs/guides/EXPERIMENTS.md)  
- [Docker](../docs/guides/DOCKER.md)

---

*Отчёт создан: Декабрь 2025*

