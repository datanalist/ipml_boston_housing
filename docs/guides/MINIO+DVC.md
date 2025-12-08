# 🗄️ Руководство по MinIO + DVC

Это руководство описывает настройку локального S3-совместимого хранилища MinIO и подключение DVC для версионирования данных и моделей.

## 📋 Содержание

1. [Что такое MinIO и DVC](#что-такое-minio-и-dvc)
2. [Запуск MinIO](#запуск-minio)
3. [Настройка MinIO](#настройка-minio)
4. [Подключение DVC к MinIO](#подключение-dvc-к-minio)
5. [Основные команды DVC](#основные-команды-dvc)
6. [Типичные сценарии использования](#типичные-сценарии-использования)
7. [Устранение неполадок](#устранение-неполадок)

---

## Что такое MinIO и DVC

### MinIO
**MinIO** — это высокопроизводительное объектное хранилище, совместимое с Amazon S3 API. Позволяет локально развернуть S3-подобное хранилище для:
- Хранения больших датасетов
- Сохранения обученных моделей
- Хранения артефактов экспериментов

### DVC (Data Version Control)
**DVC** — это система версионирования данных для ML-проектов. Позволяет:
- Версионировать большие файлы данных и модели
- Отслеживать эксперименты
- Воспроизводить ML-пайплайны
- Делиться данными между членами команды

---

## Запуск MinIO

### Предварительные требования

- Установленный [Docker](https://www.docker.com/products/docker-desktop/)
- Docker Compose

### Способ 1: Через Docker Compose (рекомендуется)

```bash
# Из корневой директории проекта
docker-compose up -d minio
```

### Способ 2: Напрямую через Docker

```powershell
# Windows PowerShell
docker run -d `
  --name boston_housing_minio `
  -p 9000:9000 `
  -p 9001:9001 `
  -v ${PWD}/minio_data:/data `
  -e MINIO_ROOT_USER=minioadmin0 `
  -e MINIO_ROOT_PASSWORD=minioadmin1230 `
  minio/minio server /data --console-address ":9001"
```

```bash
# Linux/macOS
docker run -d \
  --name boston_housing_minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v ./minio_data:/data \
  -e MINIO_ROOT_USER=minioadmin0 \
  -e MINIO_ROOT_PASSWORD=minioadmin1230 \
  minio/minio server /data --console-address ":9001"
```

### Проверка запуска

```bash
# Проверка статуса контейнера
docker ps | grep minio

# Проверка логов
docker logs boston_housing_minio
```

После запуска доступны:
- **S3 API**: http://localhost:9000
- **Веб-консоль**: http://localhost:9001

---

## Настройка MinIO

### Доступ к веб-консоли

1. Откройте в браузере: http://localhost:9001
2. Введите учётные данные:
   - **Username**: `minioadmin0`
   - **Password**: `minioadmin1230`

### Создание бакета для DVC

#### Через веб-консоль:

1. В меню слева выберите **Buckets**
2. Нажмите **Create Bucket**
3. Введите имя: `boston-housing-data`
4. Нажмите **Create Bucket**

#### Через командную строку (mc — MinIO Client):

```bash
# Установка MinIO Client
# Windows (через chocolatey):
choco install minio-client

# Или скачайте с https://min.io/download#/windows

# Настройка алиаса для подключения
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230

# Создание бакета
mc mb local/boston-housing-data

# Проверка списка бакетов
mc ls local
```

### Структура локальных данных

```
minio_data/
├── raw/              # Исходные данные
├── processed/        # Обработанные данные
├── models/           # Обученные модели
└── experiments/      # Артефакты экспериментов
```

---

## Подключение DVC к MinIO

### Шаг 1: Инициализация DVC

```bash
# Если DVC ещё не инициализирован в проекте
dvc init

# Проверка инициализации
git status
# Должны появиться: .dvc/, .dvcignore
```

### Шаг 2: Настройка удалённого хранилища

```bash
# Добавление MinIO как remote storage
dvc remote add -d minio s3://boston-housing-data

# Настройка endpoint URL для MinIO
dvc remote modify minio endpointurl http://localhost:9000

# Настройка учётных данных
dvc remote modify minio access_key_id minioadmin0
dvc remote modify minio secret_access_key minioadmin1230

# Отключение проверки SSL (для локального использования)
dvc remote modify minio use_ssl false
```

### Шаг 3: Проверка конфигурации

```bash
# Просмотр настроек DVC
dvc remote list
cat .dvc/config
```

Файл `.dvc/config` должен выглядеть так:

```ini
[core]
    remote = minio
['remote "minio"']
    url = s3://boston-housing-data
    endpointurl = http://localhost:9000
    access_key_id = minioadmin0
    secret_access_key = minioadmin1230
    use_ssl = false
```

### Шаг 4: Коммит конфигурации

```bash
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "feat: настройка DVC с MinIO хранилищем"
```

---

## Основные команды DVC

### Добавление файлов под версионирование

```bash
# Добавить файл данных
dvc add minio_data/raw/housing.csv

# Добавить всю директорию
dvc add minio_data/processed

# Добавить модель
dvc add minio_data/models/random_forest.pkl
```

После добавления появятся файлы `.dvc`:
- `minio_data/raw/housing.csv.dvc` — метаданные для DVC
- Оригинальный файл добавится в локальный `.gitignore`

### Отправка данных в хранилище

```bash
# Отправить все отслеживаемые данные
dvc push

# Отправить конкретный файл
dvc push minio_data/raw/housing.csv.dvc
```

### Получение данных из хранилища

```bash
# Скачать все данные
dvc pull

# Скачать конкретный файл
dvc pull minio_data/raw/housing.csv.dvc
```

### Проверка статуса

```bash
# Статус локальных изменений
dvc status

# Сравнение с remote
dvc status --remote
```

### Работа с версиями

```bash
# Переключение на версию данных из определённого коммита
git checkout <commit-hash>
dvc checkout

# Вернуться к последней версии
git checkout main
dvc checkout
```

---

## Типичные сценарии использования

### Сценарий 1: Первоначальная загрузка данных

```bash
# 1. Скачайте датасет и поместите в minio_data/raw/
# 2. Добавьте под контроль DVC
dvc add minio_data/raw/housing.csv

# 3. Закоммитьте .dvc файл
git add minio_data/raw/housing.csv.dvc minio_data/raw/.gitignore
git commit -m "data: добавлен исходный датасет Boston Housing"

# 4. Отправьте данные в MinIO
dvc push
```

### Сценарий 2: Обновление данных

```bash
# 1. Обновите файл данных
# 2. Пересчитайте хеш DVC
dvc add minio_data/raw/housing.csv

# 3. Закоммитьте изменения
git add minio_data/raw/housing.csv.dvc
git commit -m "data: обновлён датасет"

# 4. Отправьте новую версию
dvc push
```

### Сценарий 3: Сохранение обученной модели

```bash
# 1. После обучения модели сохраните в minio_data/models/
dvc add minio_data/models/best_model.pkl

# 2. Закоммитьте
git add minio_data/models/best_model.pkl.dvc minio_data/models/.gitignore
git commit -m "model: добавлена лучшая модель RandomForest (R²=0.87)"

# 3. Отправьте в хранилище
dvc push
```

### Сценарий 4: Клонирование проекта новым участником

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd ipml_boston_housing

# 2. Установить зависимости
uv sync

# 3. Запустить MinIO (или подключиться к общему)
docker-compose up -d minio

# 4. Скачать все данные
dvc pull
```

### Сценарий 5: Откат к предыдущей версии данных

```bash
# Найти нужный коммит
git log --oneline minio_data/raw/housing.csv.dvc

# Откатиться к версии
git checkout <commit-hash> -- minio_data/raw/housing.csv.dvc
dvc checkout minio_data/raw/housing.csv.dvc

# Или полный откат всего проекта
git checkout <commit-hash>
dvc checkout
```

---

## Устранение неполадок

### Ошибка подключения к MinIO

**Симптом**: `ERROR: Unable to connect to the remote storage`

**Решения**:
```bash
# 1. Проверьте, запущен ли контейнер
docker ps | grep minio

# 2. Проверьте доступность endpoint
curl http://localhost:9000/minio/health/live

# 3. Перезапустите MinIO
docker-compose restart minio
```

### Ошибка аутентификации

**Симптом**: `Access Denied` или `Invalid credentials`

**Решения**:
```bash
# Проверьте учётные данные в конфигурации
cat .dvc/config

# Обновите credentials
dvc remote modify minio access_key_id minioadmin0
dvc remote modify minio secret_access_key minioadmin1230
```

### Бакет не найден

**Симптом**: `Bucket does not exist`

**Решения**:
```bash
# Создайте бакет через mc
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data

# Или через веб-консоль http://localhost:9001
```

### DVC push завис

**Симптом**: Команда `dvc push` долго выполняется

**Решения**:
```bash
# Проверьте размер файлов
dvc status

# Используйте verbose режим для диагностики
dvc push -v

# Проверьте сетевое подключение к MinIO
curl -I http://localhost:9000
```

### Конфликт версий файлов

**Симптом**: `Error: file is already tracked by DVC`

**Решения**:
```bash
# Удалите из отслеживания и добавьте заново
dvc remove minio_data/raw/housing.csv.dvc
dvc add minio_data/raw/housing.csv
```

### Файл .dvc игнорируется git

**Симптом**: `ERROR: bad DVC file name '...' is git-ignored`

**Решения**:
Убедитесь, что в `.gitignore` есть исключения для `.dvc` файлов:
```gitignore
# Data files
minio_data/

# Но НЕ игнорируем .dvc файлы
!**/*.dvc
!**/.gitignore
```

---

## 📚 Полезные ссылки

- [Документация DVC](https://dvc.org/doc)
- [DVC с S3-совместимыми хранилищами](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
- [MinIO Documentation](https://min.io/docs/minio/linux/index.html)
- [MinIO Client (mc) Reference](https://min.io/docs/minio/linux/reference/minio-mc.html)

---

## ⚡ Быстрый старт (TL;DR)

```bash
# 1. Запуск MinIO
docker-compose up -d minio

# 2. Создание бакета (через браузер http://localhost:9001 или mc)
mc alias set local http://localhost:9000 minioadmin0 minioadmin1230
mc mb local/boston-housing-data

# 3. Настройка DVC
dvc init
dvc remote add -d minio s3://boston-housing-data
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin0
dvc remote modify minio secret_access_key minioadmin1230
dvc remote modify minio use_ssl false

# 4. Добавление данных
dvc add minio_data/raw/housing.csv
git add minio_data/raw/housing.csv.dvc .dvc/config
git commit -m "feat: настройка DVC + MinIO, добавлены данные"
dvc push

# Готово! 🎉
```
