"""
DAG: Boston Housing Simple Pipeline
===================================
Простой последовательный пайплайн для обучения модели Random Forest.

Этапы:
1. download_data - Загрузка данных
2. validate_data - Валидация данных
3. train_model - Обучение модели
4. evaluate_model - Оценка модели
5. save_to_minio - Сохранение артефактов в MinIO
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

# Добавляем пути проекта
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/src")

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ DAG
# ═══════════════════════════════════════════════════════════════════════════════

default_args = {
    "owner": "boston_housing",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Пути к данным
DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
HOUSING_DATA_FILE = "housing.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════


@dag(
    dag_id="boston_housing_simple",
    default_args=default_args,
    description="Простой ML пайплайн: загрузка → обучение → оценка",
    schedule=None,  # Ручной запуск
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "boston_housing", "random_forest"],
)
def boston_housing_simple_dag():
    """
    Простой последовательный пайплайн для обучения модели Random Forest
    на датасете Boston Housing.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1: Загрузка данных
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def download_data() -> str:
        """
        Загружает данные Boston Housing из интернета.
        Возвращает путь к файлу данных.
        """
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        from loguru import logger

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RAW_DATA_DIR / HOUSING_DATA_FILE

        # Проверяем, существует ли файл (кэширование)
        if output_path.exists():
            logger.info(f"✅ Данные уже существуют: {output_path}")
            return str(output_path)

        # URL источника данных
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

        logger.info(f"📥 Загрузка данных из {url}")

        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                content = response.read().decode("utf-8")

            # Конвертируем CSV в формат с пробельным разделителем
            lines = content.strip().split("\n")
            if lines and "," in lines[0]:
                data_lines = []
                for line in lines[1:]:  # Пропускаем заголовок
                    parts = line.split(",")
                    data_lines.append(" ".join(parts))
                content = "\n".join(data_lines)

            # Сохраняем данные
            with open(output_path, "w") as f:
                f.write(content)

            logger.success(f"✅ Данные сохранены: {output_path}")
            return str(output_path)

        except (URLError, HTTPError) as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2: Валидация данных
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def validate_data(data_path: str) -> dict:
        """
        Проверяет качество загруженных данных.
        Возвращает статистику по данным.
        """
        import pandas as pd
        from loguru import logger

        logger.info(f"🔍 Валидация данных: {data_path}")

        # Загружаем данные
        df = pd.read_csv(data_path, sep=r"\s+", header=None)

        # Названия колонок
        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]
        df.columns = column_names

        # Проверки
        stats = {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "missing_values": int(df.isnull().sum().sum()),
            "target_mean": float(df["MEDV"].mean()),
            "target_std": float(df["MEDV"].std()),
        }

        # Валидация
        assert stats["n_samples"] > 0, "Датасет пуст!"
        assert stats["missing_values"] == 0, "Есть пропущенные значения!"
        assert stats["n_features"] == 13, (
            f"Ожидалось 13 признаков, получено {stats['n_features']}"
        )

        logger.success(f"✅ Валидация пройдена: {stats['n_samples']} записей")
        logger.info(
            f"   Среднее MEDV: {stats['target_mean']:.2f} ± {stats['target_std']:.2f}"
        )

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3: Обучение модели
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def train_model(data_path: str, data_stats: dict) -> dict:
        """
        Обучает модель Random Forest.
        Возвращает путь к модели и параметры.
        """
        import pickle

        import pandas as pd
        from loguru import logger
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        logger.info("🚀 Начало обучения модели Random Forest")

        # Загружаем данные
        df = pd.read_csv(data_path, sep=r"\s+", header=None)
        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]
        df.columns = column_names

        X = df.drop("MEDV", axis=1)
        y = df["MEDV"]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Параметры модели
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }

        # Обучение
        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Сохранение модели
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "random_forest_airflow.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"✅ Модель сохранена: {model_path}")

        # Важность признаков
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]

        logger.info("📊 Топ-5 признаков:")
        for feat, imp in top_features:
            logger.info(f"   {feat}: {imp:.4f}")

        return {
            "model_path": str(model_path),
            "params": params,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_columns": list(X.columns),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 4: Оценка модели
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def evaluate_model(data_path: str, train_result: dict) -> dict:
        """
        Оценивает обученную модель на тестовых данных.
        Возвращает метрики качества.
        """
        import pickle

        import numpy as np
        import pandas as pd
        from loguru import logger
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        logger.info("📈 Оценка модели")

        # Загружаем модель
        with open(train_result["model_path"], "rb") as f:
            model = pickle.load(f)

        # Загружаем данные
        df = pd.read_csv(data_path, sep=r"\s+", header=None)
        column_names = [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ]
        df.columns = column_names

        X = df.drop("MEDV", axis=1)
        y = df["MEDV"]

        # Воспроизводим то же разбиение
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Предсказания
        y_pred = model.predict(X_test)

        # Метрики
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        }

        logger.success("✅ Метрики модели:")
        logger.info(f"   R² Score:  {metrics['r2_score']:.4f}")
        logger.info(f"   RMSE:      {metrics['rmse']:.4f}")
        logger.info(f"   MAE:       {metrics['mae']:.4f}")
        logger.info(f"   MAPE:      {metrics['mape']:.2f}%")

        return {
            "model_path": train_result["model_path"],
            "metrics": metrics,
            "params": train_result["params"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 5: Сохранение в MinIO и логирование в MLflow
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def save_artifacts(evaluation_result: dict) -> dict:
        """
        Сохраняет артефакты модели в MinIO и логирует в MLflow.
        """
        import json
        from datetime import datetime

        import boto3
        from botocore.client import Config
        from loguru import logger

        logger.info("💾 Сохранение артефактов")

        # Конфигурация MinIO
        minio_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

        # Создаём клиент S3
        s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            config=Config(signature_version="s3v4"),
        )

        # Имя бакета
        bucket_name = "airflow-artifacts"

        # Создаём бакет если не существует
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except Exception:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"📦 Создан бакет: {bucket_name}")

        # Загружаем модель в MinIO
        model_path = evaluation_result["model_path"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"models/random_forest_{timestamp}.pkl"

        s3_client.upload_file(model_path, bucket_name, s3_key)
        logger.success(f"✅ Модель загружена в MinIO: s3://{bucket_name}/{s3_key}")

        # Сохраняем метрики как JSON
        metrics_key = f"metrics/random_forest_{timestamp}.json"
        metrics_json = json.dumps(
            {
                "metrics": evaluation_result["metrics"],
                "params": evaluation_result["params"],
                "timestamp": timestamp,
            },
            indent=2,
        )

        s3_client.put_object(
            Bucket=bucket_name,
            Key=metrics_key,
            Body=metrics_json.encode("utf-8"),
            ContentType="application/json",
        )
        logger.success(f"✅ Метрики сохранены: s3://{bucket_name}/{metrics_key}")

        # Логирование в MLflow (если доступен)
        try:
            import mlflow

            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("boston_housing_airflow")

            with mlflow.start_run(run_name=f"airflow_run_{timestamp}"):
                # Логируем параметры
                for param_name, param_value in evaluation_result["params"].items():
                    mlflow.log_param(param_name, param_value)

                # Логируем метрики
                for metric_name, metric_value in evaluation_result["metrics"].items():
                    mlflow.log_metric(metric_name, metric_value)

                # Логируем артефакт модели
                mlflow.log_artifact(model_path, "model")

                mlflow.set_tag("source", "airflow")
                mlflow.set_tag("dag", "boston_housing_simple")

            logger.success("✅ Результаты залогированы в MLflow")

        except Exception as e:
            logger.warning(f"⚠️ Не удалось залогировать в MLflow: {e}")

        return {
            "status": "success",
            "minio_model_path": f"s3://{bucket_name}/{s3_key}",
            "minio_metrics_path": f"s3://{bucket_name}/{metrics_key}",
            "metrics": evaluation_result["metrics"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ
    # ─────────────────────────────────────────────────────────────────────────

    # Загрузка данных
    data_path = download_data()

    # Валидация данных
    data_stats = validate_data(data_path)

    # Обучение модели (зависит от валидации)
    train_result = train_model(data_path, data_stats)

    # Оценка модели
    eval_result = evaluate_model(data_path, train_result)

    # Сохранение артефактов
    save_artifacts(eval_result)


# Создание DAG
boston_housing_simple_dag()
