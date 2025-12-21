"""
Демо эксперимент с логированием в MLflow.

Запускает несколько базовых моделей и логирует результаты в MLflow.
"""

import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Загружаем переменные окружения из .env
PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")

# Пути к данным
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
HOUSING_DATA_FILE = "housing.csv"


def setup_mlflow():
    """Настройка MLflow."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"🔗 MLflow Tracking URI: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = "boston_housing_demo"
    mlflow.set_experiment(experiment_name)
    logger.info(f"📁 Эксперимент: {experiment_name}")

    return mlflow_uri


def load_data():
    """Загрузка датасета."""
    data_path = RAW_DATA_DIR / HOUSING_DATA_FILE
    if not data_path.exists():
        logger.error(f"❌ Файл данных не найден: {data_path}")
        logger.info("   Выполните: make download-data")
        sys.exit(1)

    df = pd.read_csv(data_path, sep=r"\s+", header=None)
    cols = [
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
    df.columns = cols

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_experiment(name, model, X_train, X_test, y_train, y_test):
    """Запуск одного эксперимента с логированием в MLflow."""

    with mlflow.start_run(run_name=name):
        # Логируем параметры
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Логируем гиперпараметры модели
        for param, value in model.get_params().items():
            if value is not None and not callable(value):
                mlflow.log_param(param, value)

        # Обучение
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Предсказание
        y_pred = model.predict(X_test)

        # Метрики
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Логируем метрики
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("train_time_seconds", train_time)

        # Логируем модель
        mlflow.sklearn.log_model(model, "model")

        # Теги
        mlflow.set_tag("experiment_type", "demo")
        mlflow.set_tag("dataset", "boston_housing")

        logger.info(
            f"  ✅ {name:<25} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
        )

        return {"name": name, "r2": r2, "rmse": rmse, "mae": mae}


def main():
    """Запуск демо эксперимента с MLflow."""

    print("\n" + "═" * 70)
    print("🏠 BOSTON HOUSING - ДЕМО ЭКСПЕРИМЕНТ С MLFLOW")
    print("═" * 70 + "\n")

    # Настройка MLflow
    mlflow_uri = setup_mlflow()

    # Загрузка данных
    logger.info("\n📂 Загрузка данных...")
    X_train, X_test, y_train, y_test = load_data()

    print(
        f"\n📊 Датасет: {len(X_train) + len(X_test)} записей, {X_train.shape[1]} признаков"
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}\n")

    # Модели для тестирования
    models = [
        ("Ridge_alpha_1.0", Ridge(alpha=1.0)),
        ("Lasso_alpha_0.1", Lasso(alpha=0.1)),
        (
            "RandomForest_n100_d10",
            RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
        ),
        (
            "GradientBoosting_n100",
            GradientBoostingRegressor(n_estimators=100, random_state=42),
        ),
    ]

    print("🔬 ЗАПУСК ЭКСПЕРИМЕНТОВ:\n")
    print("-" * 70)

    results = []
    for name, model in models:
        try:
            result = run_experiment(name, model, X_train, X_test, y_train, y_test)
            results.append(result)
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")

    print("-" * 70)

    # Лучшая модель
    if results:
        best = max(results, key=lambda x: x["r2"])

        print("\n" + "═" * 70)
        print(f"🏆 Лучшая модель: {best['name']} (R² = {best['r2']:.4f})")
        print("═" * 70)

    print("\n✅ Эксперименты завершены!")
    print(f"   📊 MLflow UI: {mlflow_uri}")
    print("   📁 Эксперимент: boston_housing_demo")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
