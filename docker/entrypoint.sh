#!/bin/bash
set -e

# Подстановка переменных окружения в конфиг аутентификации
if [ -f /mlflow/config/basic_auth.ini.template ]; then
    envsubst < /mlflow/config/basic_auth.ini.template > /mlflow/config/basic_auth.ini
    echo "✅ Конфиг аутентификации сгенерирован из шаблона"
fi

# Запуск MLflow server с переданными аргументами
exec mlflow server "$@"
