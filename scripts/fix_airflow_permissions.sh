#!/bin/bash
# Скрипт для исправления прав доступа на файлы Airflow
# Использует sudo для изменения владельца файлов

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

echo "Исправление прав доступа на файлы Airflow..."
echo "Текущий пользователь: UID=$CURRENT_UID, GID=$CURRENT_GID"
echo ""

# Исправление прав на файлы
if [ -d "$PROJECT_ROOT/airflow/dags" ]; then
    echo "Исправление прав на файлы в airflow/dags..."
    sudo chown -R "$CURRENT_UID:$CURRENT_GID" "$PROJECT_ROOT/airflow/dags"/*.py 2>/dev/null || true
    sudo chown -R "$CURRENT_UID:$CURRENT_GID" "$PROJECT_ROOT/airflow/dags" 2>/dev/null || true
fi

if [ -d "$PROJECT_ROOT/airflow/plugins" ]; then
    echo "Исправление прав на файлы в airflow/plugins..."
    sudo chown -R "$CURRENT_UID:$CURRENT_GID" "$PROJECT_ROOT/airflow/plugins"/*.py 2>/dev/null || true
    sudo chown -R "$CURRENT_UID:$CURRENT_GID" "$PROJECT_ROOT/airflow/plugins" 2>/dev/null || true
fi

echo "Готово!"
