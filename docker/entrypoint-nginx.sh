#!/bin/sh
set -e

# Установка htpasswd (apache2-utils)
apk add --no-cache apache2-utils > /dev/null 2>&1

HTPASSWD_FILE="/etc/nginx/htpasswd"

# Создаём htpasswd с admin пользователем из env
if [ -n "$MLFLOW_ADMIN_USERNAME" ] && [ -n "$MLFLOW_ADMIN_PASSWORD" ]; then
    echo "✅ Генерация htpasswd для пользователя: $MLFLOW_ADMIN_USERNAME"
    # Генерируем htpasswd файл
    htpasswd -cb "$HTPASSWD_FILE" "$MLFLOW_ADMIN_USERNAME" "$MLFLOW_ADMIN_PASSWORD"
    echo "✅ htpasswd файл создан"
else
    echo "❌ MLFLOW_ADMIN_USERNAME или MLFLOW_ADMIN_PASSWORD не заданы!"
    exit 1
fi

# Запуск nginx
exec nginx -g "daemon off;"

