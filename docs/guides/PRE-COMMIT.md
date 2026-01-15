# 🔒 Руководство по Pre-commit Hooks

Это руководство описывает настройку и использование pre-commit hooks для автоматической проверки кода и данных перед коммитом.

## 📋 Содержание

1. [Что такое Pre-commit](#что-такое-pre-commit)
2. [Установка](#установка)
3. [Конфигурация проекта](#конфигурация-проекта)
4. [Использование](#использование)
5. [Описание hooks](#описание-hooks)
6. [Устранение неполадок](#устранение-неполадок)

---

## Что такое Pre-commit

**Pre-commit** — это фреймворк для управления git hooks, который автоматически запускает проверки перед каждым коммитом.

### Зачем нужен Pre-commit?

- 🔍 **Автоматическая проверка кода** — линтинг и форматирование
- 📊 **Синхронизация данных** — проверка DVC файлов
- 🚫 **Предотвращение ошибок** — не даёт закоммитить "плохой" код
- 👥 **Единый стандарт** — одинаковые правила для всей команды

### Hooks в проекте

#### 🔧 Общие проверки (pre-commit-hooks v5.0.0)

| Hook | Назначение |
|------|------------|
| `trailing-whitespace` | Удаление пробелов в конце строк |
| `end-of-file-fixer` | Добавление newline в конец файла |
| `check-yaml` | Проверка синтаксиса YAML |
| `check-json` | Проверка синтаксиса JSON |
| `check-toml` | Проверка синтаксиса TOML |
| `check-added-large-files` | Блокировка файлов >500KB |
| `check-merge-conflict` | Поиск нерешённых merge конфликтов |
| `check-executables-have-shebangs` | Проверка shebang в исполняемых файлах |
| `check-symlinks` | Проверка символических ссылок |
| `detect-private-key` | Детекция приватных ключей |
| `check-case-conflict` | Проверка конфликтов регистра в именах файлов |
| `check-docstring-first` | Проверка docstring в начале Python модулей |

#### 🐍 Python (Ruff v0.14.6)

| Hook | Назначение |
|------|------------|
| `ruff-check` | Линтинг Python кода с автоисправлением |
| `ruff-format` | Форматирование кода |

#### 📄 YAML (yamllint v1.37.0)

| Hook | Назначение |
|------|------------|
| `yamllint` | Линтинг YAML файлов (конфиг: `.yamllint.yaml`) |

#### 🐚 Shell (shellcheck-py v0.10.0.1)

| Hook | Назначение |
|------|------------|
| `shellcheck` | Проверка синтаксиса shell скриптов |

#### 🔐 Безопасность (detect-secrets v1.5.0)

| Hook | Назначение |
|------|------------|
| `detect-secrets` | Поиск случайно закоммиченных секретов |

#### 📝 Git Commits (commitizen v4.8.3)

| Hook | Stage | Назначение |
|------|-------|------------|
| `commitizen` | commit-msg | Проверка формата Conventional Commits |

#### 🚫 Отключенные hooks

| Hook | Причина |
|------|---------|
| `hadolint-docker` | Требует рабочий Docker daemon |
| `dvc-pre-commit`, `dvc-pre-push` | DVC hooks отключены (опционально) |

---

## Установка

### Шаг 1: Установка pre-commit

Pre-commit уже добавлен в зависимости проекта:

```bash
# Синхронизация зависимостей
uv sync
```

Или установка вручную:

```bash
uv add pre-commit
```

### Шаг 2: Установка hooks

```bash
# Установка всех типов hooks
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Или по отдельности:

```bash
# Только pre-commit (проверка перед коммитом)
uv run pre-commit install

# Commit-msg (проверка формата сообщения коммита)
uv run pre-commit install --hook-type commit-msg
```

### Шаг 3: Проверка установки

```bash
# Должны появиться файлы в .git/hooks/
ls .git/hooks/

# Ожидаемый вывод:
# pre-commit
# commit-msg
```

---

## Конфигурация проекта

### Файл `.pre-commit-config.yaml`

```yaml
repos:
# Общие проверки для всех файлов
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v5.0.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-json
    - id: check-toml
    - id: check-added-large-files
      args: ['--maxkb=500']
    - id: check-merge-conflict
    - id: check-executables-have-shebangs
    - id: check-symlinks
    - id: detect-private-key
    - id: check-case-conflict
    - id: check-docstring-first

# Python - Ruff (linter + formatter)
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.14.6
  hooks:
    - id: ruff-check
      types_or: [python, pyi]
      args: [--fix]
    - id: ruff-format
      types_or: [python, pyi]

# YAML - линтинг
- repo: https://github.com/adrienverge/yamllint
  rev: v1.37.0
  hooks:
    - id: yamllint
      args: [-c, .yamllint.yaml]

# Shell scripts - проверка синтаксиса
- repo: https://github.com/shellcheck-py/shellcheck-py
  rev: v0.10.0.1
  hooks:
    - id: shellcheck
      args: [--severity=warning]

# Secrets detection
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.5.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']

# Git commit messages - Conventional Commits
- repo: https://github.com/commitizen-tools/commitizen
  rev: v4.8.3
  hooks:
    - id: commitizen
      stages: [commit-msg]
```

### Обновление версий hooks

```bash
# Обновить все hooks до последних версий
uv run pre-commit autoupdate

# Проверить обновления
git diff .pre-commit-config.yaml
```

---

## Использование

### Автоматический запуск

После установки hooks запускаются автоматически:

```bash
# При коммите — запускаются ruff и dvc-pre-commit
git add .
git commit -m "feat: добавлена новая функция"

# При push — запускается dvc-pre-push
git push origin main

# При checkout — запускается dvc-post-checkout
git checkout feature-branch
```

### Ручной запуск

```bash
# Проверить все файлы
uv run pre-commit run --all-files

# Проверить конкретный hook
uv run pre-commit run ruff-check --all-files
uv run pre-commit run ruff-format --all-files
uv run pre-commit run dvc-pre-commit --all-files

# Проверить только staged файлы
uv run pre-commit run
```

### Пропуск hooks (не рекомендуется)

```bash
# Коммит без проверок (в экстренных случаях)
git commit -m "hotfix" --no-verify

# Push без проверок
git push --no-verify
```

⚠️ **Внимание:** Используйте `--no-verify` только в исключительных случаях!

---

## Описание hooks

### Общие проверки (pre-commit-hooks)

**trailing-whitespace** — удаляет лишние пробелы в конце строк
**end-of-file-fixer** — добавляет newline в конец файлов
**check-yaml/json/toml** — проверяет синтаксис конфигурационных файлов
**check-added-large-files** — блокирует файлы больше 500KB
**check-merge-conflict** — находит нерешённые merge конфликты
**detect-private-key** — находит случайно добавленные приватные ключи

### Ruff Check

**Что делает:** Проверяет Python код на ошибки и стилистические проблемы.

**Проверки:**
- Неиспользуемые импорты
- Неиспользуемые переменные
- Синтаксические ошибки
- Нарушения PEP 8
- Потенциальные баги

**Пример ошибки:**
```
src/modeling/train.py:15:1: F401 `os` imported but unused
```

**Исправление:**
- Автоматически с `--fix`
- Или вручную: удалить неиспользуемый импорт

### Ruff Format

**Что делает:** Автоматически форматирует код в едином стиле.

**Форматирование:**
- Отступы
- Кавычки
- Переносы строк
- Пробелы

**Пример:** До и после форматирования:

```python
# До
def foo(x,y):return x+y

# После
def foo(x, y):
    return x + y
```

### Yamllint

**Что делает:** Проверяет YAML файлы на соответствие стандартам.

**Конфигурация:** `.yamllint.yaml`

**Проверки:**
- Правильные отступы
- Длина строк
- Кавычки
- Пустые строки

### Shellcheck

**Что делает:** Анализирует shell скрипты на ошибки и потенциальные проблемы.

**Пример ошибки:**
```
docker/entrypoint.sh:5: warning: Quote this to prevent word splitting [SC2086]
```

### Detect-secrets

**Что делает:** Сканирует файлы на наличие случайно закоммиченных секретов (пароли, API ключи, токены).

**Конфигурация:** `.secrets.baseline` — файл с известными "безопасными" секретами (false positives)

**Пример ошибки:**
```
Potential secret detected in config/settings.py:42
Type: High Entropy String
```

**Обновление baseline:**
```bash
# Если секрет безопасен (false positive)
detect-secrets scan --baseline .secrets.baseline
```

### Commitizen

**Что делает:** Проверяет, что сообщения коммитов соответствуют формату [Conventional Commits](https://www.conventionalcommits.org/).

**Stage:** `commit-msg`

**Формат сообщений:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Типы коммитов:**
- `feat:` — новая функциональность
- `fix:` — исправление бага
- `docs:` — изменения в документации
- `style:` — форматирование, стиль кода
- `refactor:` — рефакторинг
- `test:` — добавление тестов
- `chore:` — обслуживание, зависимости

**Пример:**
```bash
# ✅ Правильно
git commit -m "feat(api): добавлен эндпоинт предсказаний"
git commit -m "fix: исправлена ошибка загрузки модели"
git commit -m "docs: обновлено README"

# ❌ Неправильно
git commit -m "добавил фичу"
git commit -m "fix bug"
```

---

## Типичные сценарии

### Сценарий 1: Обычный рабочий процесс

```bash
# 1. Внесите изменения в код
vim src/modeling/train.py

# 2. Добавьте файлы
git add .

# 3. Коммит (hooks запустятся автоматически)
git commit -m "feat: улучшена модель"
# ✓ trailing-whitespace: Passed
# ✓ check-yaml: Passed
# ✓ ruff-check: Passed
# ✓ ruff-format: Passed
# ✓ detect-secrets: Passed
# ✓ commitizen: Passed

# 4. Push
git push
```

### Сценарий 2: Ошибка в формате коммита

```bash
# Неправильный формат сообщения
git commit -m "добавил фичу"
# ✗ commitizen: Failed
# commit validation: failed!
# please enter a commit message in the commitizen format.

# Правильный формат
git commit -m "feat: добавлена новая функция предсказания"
# ✓ commitizen: Passed
```

### Сценарий 3: Найден потенциальный секрет

```bash
# При коммите найден секрет
git commit -m "feat: добавлен конфиг"
# ✗ detect-secrets: Failed
# Potential secret detected

# Если это false positive — обновите baseline
detect-secrets scan --baseline .secrets.baseline
git add .secrets.baseline
git commit -m "chore: обновлён baseline секретов"

# Если это реальный секрет — удалите его из кода!
```

### Сценарий 4: Исправление ошибок линтера

```bash
# Коммит не проходит
git commit -m "feat: new feature"
# ✗ ruff-check: Failed
# src/train.py:10: F401 unused import

# Запустить автоисправление
uv run pre-commit run ruff-check --all-files

# Или исправить вручную и повторить
git add .
git commit -m "feat: new feature"
```

### Сценарий 5: Проверка YAML файлов

```bash
# Ошибка в YAML
git commit -m "chore: обновлён docker-compose"
# ✗ yamllint: Failed
# docker-compose.yml:15: wrong indentation

# Исправьте файл и повторите
vim docker-compose.yml
git add docker-compose.yml
git commit -m "chore: обновлён docker-compose"
```

---

## Устранение неполадок

### Hook не запускается

**Причина:** Hooks не установлены

**Решение:**
```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### "No files to check"

**Причина:** Нет staged файлов нужного типа

**Решение:**
```bash
# Добавьте файлы
git add src/

# Или проверьте все файлы
uv run pre-commit run --all-files
```

### Ruff конфликтует с существующим кодом

**Решение:** Отформатируйте весь код один раз:
```bash
uv run ruff format .
uv run ruff check --fix .
git add .
git commit -m "style: форматирование кода"
```

### Commitizen отклоняет сообщение коммита

**Причина:** Сообщение не соответствует Conventional Commits

**Решение:**
```bash
# Используйте правильный формат
git commit -m "feat: описание новой функции"
git commit -m "fix: описание исправления"
git commit -m "docs: описание изменений в документации"

# Или используйте интерактивный режим commitizen
uv run cz commit
```

### Detect-secrets находит false positive

**Решение:** Обновите baseline файл:
```bash
# Сканирование и обновление baseline
detect-secrets scan --baseline .secrets.baseline

# Добавьте обновлённый baseline
git add .secrets.baseline
```

### Yamllint жалуется на docker-compose

**Причина:** Использование специальных конструкций (anchors, templates)

**Решение:** В конфигурации уже включён `--unsafe` для `check-yaml`. Если yamllint всё ещё жалуется, проверьте `.yamllint.yaml`.

### Слишком медленные проверки

**Решение:** Запускайте только на изменённых файлах:
```bash
# По умолчанию проверяются только staged файлы
git add src/modeling/train.py
uv run pre-commit run

# Вместо
uv run pre-commit run --all-files
```

### Нужно пропустить один раз

```bash
# В экстренном случае
git commit -m "hotfix: критическое исправление" --no-verify

# Потом обязательно исправьте и закоммитьте нормально
uv run pre-commit run --all-files
git add .
git commit -m "fix: исправление после hotfix"
```

---

## 🚀 Быстрый старт (TL;DR)

```bash
# 1. Установка hooks
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# 2. Проверка всех файлов
uv run pre-commit run --all-files

# 3. Теперь hooks работают автоматически при коммитах
# Используйте Conventional Commits для сообщений:
git commit -m "feat: описание функции"

# Обновление hooks
uv run pre-commit autoupdate
```

---

## 📚 Полезные ссылки

- [Pre-commit Documentation](https://pre-commit.com/)
- [Pre-commit Hooks Collection](https://github.com/pre-commit/pre-commit-hooks)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Yamllint Documentation](https://yamllint.readthedocs.io/)
- [Shellcheck Documentation](https://www.shellcheck.net/)
- [Detect-secrets Documentation](https://github.com/Yelp/detect-secrets)
- [Commitizen Documentation](https://commitizen-tools.github.io/commitizen/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
