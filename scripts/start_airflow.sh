#!/bin/bash
# =============================================================================
# Скрипт для запуска Airflow в Docker Compose
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🚀 Запуск Boston Housing ML Infrastructure с Airflow"
echo "=================================================="

# Создание необходимых директорий
echo "📁 Создание директорий Airflow..."
mkdir -p airflow/dags airflow/logs airflow/plugins

# Установка AIRFLOW_UID для корректных прав доступа
AIRFLOW_UID=$(id -u)
export AIRFLOW_UID
echo "   AIRFLOW_UID=$AIRFLOW_UID"

# Проверка наличия .env файла
if [ ! -f ".env" ]; then
    echo "⚠️  Файл .env не найден. Создаём шаблон..."
    cat > .env << 'EOF'
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# MLflow
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=admin
MLFLOW_FLASK_SERVER_SECRET_KEY=mlflow-secret-key-change-me

# Airflow
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_UID=50000
EOF
    echo "✅ Создан файл .env"
fi

# Остановка старых контейнеров (если есть)
echo ""
echo "🛑 Остановка старых контейнеров..."
docker-compose down --remove-orphans 2>/dev/null || true

# Сборка образов
echo ""
echo "🔨 Сборка Docker образов..."
docker-compose build

# Запуск инфраструктуры
echo ""
echo "🚀 Запуск сервисов..."
docker-compose up -d

# Ожидание готовности
echo ""
echo "⏳ Ожидание готовности сервисов..."
sleep 10

# Проверка статуса
echo ""
echo "📊 Статус сервисов:"
docker-compose ps

echo ""
echo "=================================================="
echo "✅ Инфраструктура запущена!"
echo ""
echo "🌐 Веб-интерфейсы:"
echo "   Airflow UI:  http://localhost:8080 (admin/admin)"
echo "   MLflow UI:   http://localhost:5000"
echo "   MinIO UI:    http://localhost:9001"
echo ""
echo "📝 DAGs доступны:"
echo "   - boston_housing_simple      (простой пайплайн)"
echo "   - boston_housing_experiments (параллельные эксперименты)"
echo "   - boston_housing_cached      (с кэшированием)"
echo ""
echo "🛠️  Команды:"
echo "   docker-compose logs -f airflow-webserver  # Логи Airflow"
echo "   docker-compose down                       # Остановка"
echo "=================================================="
