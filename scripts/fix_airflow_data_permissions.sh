#!/bin/bash
# Скрипт для исправления прав доступа к директориям данных в контейнерах Airflow

set -e

echo "🔧 Исправление прав доступа к директориям данных в контейнерах Airflow..."

# Список контейнеров Airflow
CONTAINERS=(
    "boston_housing_airflow_worker"
    "boston_housing_airflow_scheduler"
    "boston_housing_airflow_webserver"
)

# Директории для исправления прав
DIRECTORIES=(
    "/opt/airflow/data/models"
    "/opt/airflow/data/experiments"
    "/opt/airflow/data/raw"
)

for container in "${CONTAINERS[@]}"; do
    echo "📦 Обработка контейнера: $container"

    # Проверяем, запущен ли контейнер
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        for dir in "${DIRECTORIES[@]}"; do
            echo "  📁 Исправление прав для: $dir"
            docker exec -u root "$container" chown -R airflow:root "$dir" 2>/dev/null || true
            docker exec -u root "$container" chmod -R 775 "$dir" 2>/dev/null || true
        done
        echo "  ✅ Права доступа исправлены в $container"
    else
        echo "  ⚠️  Контейнер $container не запущен, пропускаем"
    fi
done

echo ""
echo "✅ Исправление прав доступа завершено!"
echo ""
echo "Проверка прав доступа:"
docker exec boston_housing_airflow_worker ls -la /opt/airflow/data/models/ | head -5
