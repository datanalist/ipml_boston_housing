"""
DAG: Boston Housing Cached Pipeline
====================================
Пайплайн с кэшированием через MinIO.

Особенности:
- ShortCircuitOperator для пропуска задач если модель в кэше
- Проверка хэша входных данных
- Автоматическое сохранение в кэш после обучения
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator

# Добавляем пути проекта
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow/plugins")

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

default_args = {
    "owner": "boston_housing",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
HOUSING_DATA_FILE = "housing.csv"

# Параметры модели для кэширования
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════


@dag(
    dag_id="boston_housing_cached",
    default_args=default_args,
    description="ML пайплайн с кэшированием в MinIO",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "boston_housing", "caching"],
)
def boston_housing_cached_dag():
    """
    DAG с кэшированием моделей в MinIO.

    Если модель с такими параметрами уже обучена на тех же данных -
    пропускает обучение и использует кэшированную модель.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Загрузка данных
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def download_data() -> str:
        """Загружает данные Boston Housing."""
        from urllib.request import Request, urlopen

        from loguru import logger

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RAW_DATA_DIR / HOUSING_DATA_FILE

        if output_path.exists():
            logger.info(f"✅ Данные уже существуют: {output_path}")
            return str(output_path)

        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        logger.info(f"📥 Загрузка данных из {url}")

        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8")

        lines = content.strip().split("\n")
        if lines and "," in lines[0]:
            data_lines = [" ".join(line.split(",")) for line in lines[1:]]
            content = "\n".join(data_lines)

        with open(output_path, "w") as f:
            f.write(content)

        logger.success(f"✅ Данные сохранены: {output_path}")
        return str(output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Проверка кэша
    # ─────────────────────────────────────────────────────────────────────────
    def check_cache_exists(data_path: str, **kwargs) -> bool:
        """
        Проверяет наличие модели в кэше.

        Используется с ShortCircuitOperator:
        - True: продолжить выполнение (нет кэша, нужно обучать)
        - False: пропустить downstream задачи (кэш найден)
        """
        from minio_cache import MinIOCache
        from loguru import logger

        logger.info("🔍 Проверка кэша в MinIO...")

        try:
            cache = MinIOCache(bucket_name="airflow-cache")
            prefix = "models/random_forest_cached"
            exists, cache_key = cache.check_cache(prefix, MODEL_PARAMS, data_path)

            if exists:
                logger.info(f"✅ Модель найдена в кэше: {cache_key}")
                # Сохраняем ключ кэша для использования позже
                kwargs["ti"].xcom_push(key="cache_key", value=cache_key)
                return False  # Пропустить обучение
            else:
                logger.info("❌ Кэш не найден, требуется обучение")
                kwargs["ti"].xcom_push(key="cache_key", value=cache_key)
                return True  # Продолжить обучение

        except Exception as e:
            logger.warning(f"⚠️ Ошибка проверки кэша: {e}")
            return True  # При ошибке - обучаем

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Обучение модели
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def train_model(data_path: str) -> dict:
        """Обучает модель Random Forest."""
        import pickle

        import numpy as np
        import pandas as pd
        from loguru import logger
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        logger.info("🚀 Обучение модели Random Forest")

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Обучение
        model = RandomForestRegressor(**MODEL_PARAMS, n_jobs=-1)
        model.fit(X_train, y_train)

        # Метрики
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

        # Сохранение
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "random_forest_cached.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"✅ Модель обучена: R²={metrics['r2_score']:.4f}")

        return {
            "model_path": str(model_path),
            "data_path": data_path,
            "params": MODEL_PARAMS,
            "metrics": metrics,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Сохранение в кэш
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def save_to_cache(train_result: dict) -> dict:
        """Сохраняет обученную модель в кэш MinIO."""
        from minio_cache import save_model_to_cache
        from loguru import logger

        logger.info("💾 Сохранение модели в кэш MinIO")

        try:
            model_uri = save_model_to_cache(
                model_path=train_result["model_path"],
                model_name="random_forest_cached",
                params=train_result["params"],
                data_path=train_result["data_path"],
                metrics=train_result["metrics"],
                bucket_name="airflow-cache",
            )

            logger.success(f"✅ Модель сохранена в кэш: {model_uri}")

            return {
                "status": "saved",
                "cache_uri": model_uri,
                "metrics": train_result["metrics"],
            }

        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения в кэш: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metrics": train_result["metrics"],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Использование кэшированной модели
    # ─────────────────────────────────────────────────────────────────────────
    @task(trigger_rule="none_failed")
    def use_cached_model(data_path: str) -> dict:
        """
        Использует модель из кэша (если обучение было пропущено).
        trigger_rule="none_failed" позволяет выполниться даже если
        upstream задачи были пропущены.
        """
        from minio_cache import MinIOCache
        from loguru import logger

        logger.info("📦 Попытка использования кэшированной модели")

        try:
            cache = MinIOCache(bucket_name="airflow-cache")
            prefix = "models/random_forest_cached"
            exists, cache_key = cache.check_cache(prefix, MODEL_PARAMS, data_path)

            if exists:
                # Скачиваем метаданные
                metadata = cache.get_json(f"{cache_key}_metadata.json")

                logger.success("✅ Используется кэшированная модель")
                logger.info(f"   R² Score: {metadata['metrics']['r2_score']:.4f}")

                return {
                    "status": "from_cache",
                    "cache_key": cache_key,
                    "metrics": metadata["metrics"],
                }
            else:
                logger.info("❌ Кэш не найден")
                return {"status": "no_cache"}

        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения кэша: {e}")
            return {"status": "error", "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Финальный отчёт
    # ─────────────────────────────────────────────────────────────────────────
    @task(trigger_rule="none_failed_min_one_success")
    def generate_summary(
        data_path: str,
        cache_result: dict = None,
        train_save_result: dict = None,
    ) -> dict:
        """Генерирует итоговый отчёт."""
        from loguru import logger

        logger.info("📝 Генерация итогового отчёта")

        # Определяем источник результата
        if train_save_result and train_save_result.get("status") == "saved":
            source = "trained"
            metrics = train_save_result.get("metrics", {})
        elif cache_result and cache_result.get("status") == "from_cache":
            source = "cached"
            metrics = cache_result.get("metrics", {})
        else:
            source = "unknown"
            metrics = {}

        summary = {
            "source": source,
            "data_path": data_path,
            "params": MODEL_PARAMS,
            "metrics": metrics,
        }

        if source == "cached":
            logger.success("✅ Результат получен из кэша (обучение пропущено)")
        elif source == "trained":
            logger.success("✅ Модель обучена и сохранена в кэш")
        else:
            logger.warning("⚠️ Не удалось определить источник результата")

        if metrics:
            logger.info(f"📊 R² Score: {metrics.get('r2_score', 'N/A')}")

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # ПОСТРОЕНИЕ DAG
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Загрузка данных
    data_path = download_data()

    # 2. Проверка кэша (ShortCircuitOperator)
    cache_check = ShortCircuitOperator(
        task_id="check_cache",
        python_callable=check_cache_exists,
        op_kwargs={"data_path": data_path},
        provide_context=True,
    )

    # 3. Обучение модели (пропускается если кэш найден)
    train_result = train_model(data_path)

    # 4. Сохранение в кэш
    save_result = save_to_cache(train_result)

    # 5. Использование кэшированной модели (всегда выполняется)
    cached_result = use_cached_model(data_path)

    # 6. Финальный отчёт
    summary = generate_summary(
        data_path=data_path,
        cache_result=cached_result,
        train_save_result=save_result,
    )

    # Определение зависимостей
    data_path >> cache_check >> train_result >> save_result >> summary
    data_path >> cached_result >> summary


# Создание DAG
boston_housing_cached_dag()
