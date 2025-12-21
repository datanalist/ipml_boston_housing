"""
Скрипт для запуска множественных экспериментов с разными алгоритмами ML.

Логирует метрики, параметры и артефакты в MLflow.
"""

import os
import pickle
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Загружаем переменные окружения и добавляем путь к src
PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")
sys.path.insert(0, str(PROJ_ROOT))

from src.config import RAW_DATA_DIR, HOUSING_DATA_FILE  # noqa: E402
from src.ml_models.model_loader import MODEL_REGISTRY, create_model  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ 19 ЭКСПЕРИМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS_CONFIG = [
    # ─────────────────────────────────────────────────────────────────────────
    # Линейные модели (7 экспериментов)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "linear_regression",
        "params": {},
        "description": "Baseline линейная регрессия",
    },
    {
        "name": "ridge",
        "params": {"alpha": 0.1},
        "description": "Ridge со слабой регуляризацией",
    },
    {"name": "ridge", "params": {"alpha": 1.0}, "description": "Ridge стандартный"},
    {
        "name": "ridge",
        "params": {"alpha": 10.0},
        "description": "Ridge с сильной регуляризацией",
    },
    {
        "name": "lasso",
        "params": {"alpha": 0.1},
        "description": "Lasso для отбора признаков",
    },
    {
        "name": "elastic_net",
        "params": {"alpha": 0.5, "l1_ratio": 0.5},
        "description": "Elastic Net комбинированный",
    },
    {
        "name": "huber",
        "params": {"epsilon": 1.35},
        "description": "Huber робастная регрессия",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # Древовидные модели и ансамбли (9 экспериментов)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "decision_tree",
        "params": {"max_depth": 5},
        "description": "Дерево решений (shallow)",
    },
    {
        "name": "decision_tree",
        "params": {"max_depth": 10},
        "description": "Дерево решений (deep)",
    },
    {
        "name": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10},
        "description": "Random Forest стандартный",
    },
    {
        "name": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 15},
        "description": "Random Forest большой",
    },
    {
        "name": "extra_trees",
        "params": {"n_estimators": 100, "max_depth": 10},
        "description": "Extra Trees",
    },
    {
        "name": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.1},
        "description": "Gradient Boosting стандартный",
    },
    {
        "name": "gradient_boosting",
        "params": {"n_estimators": 200, "learning_rate": 0.05},
        "description": "Gradient Boosting медленный",
    },
    {
        "name": "adaboost",
        "params": {"n_estimators": 50, "learning_rate": 1.0},
        "description": "AdaBoost",
    },
    {
        "name": "bagging",
        "params": {"n_estimators": 20},
        "description": "Bagging регрессор",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # Другие модели (3 эксперимента)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "name": "svr",
        "params": {"kernel": "rbf", "C": 1.0},
        "description": "SVR с RBF ядром",
    },
    {
        "name": "knn",
        "params": {"n_neighbors": 5, "weights": "uniform"},
        "description": "KNN k=5 uniform",
    },
    {
        "name": "knn",
        "params": {"n_neighbors": 10, "weights": "distance"},
        "description": "KNN k=10 distance",
    },
]


def load_data():
    """Загрузка и подготовка данных Boston Housing."""
    data_file = RAW_DATA_DIR / HOUSING_DATA_FILE

    if not data_file.exists():
        logger.error(f"Файл данных не найден: {data_file}")
        logger.info("Выполните 'dvc pull' для загрузки данных")
        sys.exit(1)

    df = pd.read_csv(data_file, sep=r"\s+", header=None)

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

    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_algorithm_family(model_name: str) -> str:
    """Определение семейства алгоритма для тегирования."""
    linear_models = [
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "huber",
        "sgd",
    ]
    tree_models = [
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "adaboost",
        "bagging",
    ]

    if model_name in linear_models:
        return "linear"
    elif model_name in tree_models:
        return "tree_ensemble"
    else:
        return "other"


def evaluate_model(model, X_test, y_test):
    """Оценка модели и расчёт метрик."""
    y_pred = model.predict(X_test)

    return {
        "r2_score": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }, y_pred


def create_plots(model, model_name, X_train, y_test, y_pred, temp_dir):
    """Создание графиков для артефактов."""
    plots = []

    # 1. График предсказаний vs реальных значений
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Идеальное предсказание",
    )
    ax.set_xlabel("Реальные значения (MEDV)", fontsize=12)
    ax.set_ylabel("Предсказанные значения", fontsize=12)
    ax.set_title(f"Предсказания vs Реальные значения\n{model_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    scatter_path = os.path.join(temp_dir, "predictions_scatter.png")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    plots.append(("plots", scatter_path))

    # 2. График остатков
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Предсказанные значения", fontsize=12)
    ax.set_ylabel("Остатки (Residuals)", fontsize=12)
    ax.set_title(f"График остатков\n{model_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    residuals_path = os.path.join(temp_dir, "residuals.png")
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=150)
    plt.close(fig)
    plots.append(("plots", residuals_path))

    # 3. Feature Importance (для tree-based моделей)
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importances)))
        ax.barh(importances["feature"], importances["importance"], color=colors)
        ax.set_xlabel("Важность признака", fontsize=12)
        ax.set_title(f"Важность признаков\n{model_name}", fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")

        importance_path = os.path.join(temp_dir, "feature_importance.png")
        fig.tight_layout()
        fig.savefig(importance_path, dpi=150)
        plt.close(fig)
        plots.append(("plots", importance_path))

    return plots


def run_single_experiment(
    config, X_train, X_test, y_train, y_test, experiment_idx, total_experiments
):
    """Запуск одного эксперимента с полным логированием в MLflow."""

    model_name = config["name"]
    custom_params = config["params"]
    description = config.get("description", "")

    # Генерация уникального имени run
    param_str = "_".join([f"{k}={v}" for k, v in custom_params.items()])
    run_name = f"{model_name}_{param_str}" if param_str else model_name

    logger.info(f"\n{'═' * 60}")
    logger.info(f"[{experiment_idx}/{total_experiments}] 🚀 {run_name}")
    logger.info(f"Описание: {description}")
    logger.info(f"{'═' * 60}")

    with mlflow.start_run(run_name=run_name):
        # ═══════════════════════════════════════════════════════════════════
        # 1. ЛОГИРОВАНИЕ ПАРАМЕТРОВ
        # ═══════════════════════════════════════════════════════════════════

        # Основные параметры модели
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("model_description", MODEL_REGISTRY[model_name]["description"])
        mlflow.log_param("experiment_description", description)

        # Кастомные параметры модели
        for param_name, param_value in custom_params.items():
            mlflow.log_param(param_name, param_value)

        # Параметры данных
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_split_ratio", 0.2)

        # ═══════════════════════════════════════════════════════════════════
        # 2. ОБУЧЕНИЕ МОДЕЛИ С ЗАМЕРОМ ВРЕМЕНИ
        # ═══════════════════════════════════════════════════════════════════

        model = create_model(model_name, custom_params)

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_inference = time.time()
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        inference_time = time.time() - start_inference

        # ═══════════════════════════════════════════════════════════════════
        # 3. ЛОГИРОВАНИЕ МЕТРИК
        # ═══════════════════════════════════════════════════════════════════

        # Метрики качества
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Метрики производительности
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("inference_time_seconds", inference_time)
        mlflow.log_metric(
            "predictions_per_second",
            len(X_test) / inference_time if inference_time > 0 else 0,
        )

        logger.info(f"  📊 R² Score:  {metrics['r2_score']:.4f}")
        logger.info(f"  📊 RMSE:      {metrics['rmse']:.4f}")
        logger.info(f"  📊 MAE:       {metrics['mae']:.4f}")
        logger.info(f"  📊 MAPE:      {metrics['mape']:.2f}%")
        logger.info(f"  ⏱️  Train:     {train_time:.3f}s")

        # ═══════════════════════════════════════════════════════════════════
        # 4. ЛОГИРОВАНИЕ АРТЕФАКТОВ
        # ═══════════════════════════════════════════════════════════════════

        with tempfile.TemporaryDirectory() as temp_dir:
            # 4.1 Модель в формате MLflow sklearn
            # Примечание: registered_model_name=None отключает автоматическую регистрацию
            # в Model Registry (требует MLflow 3.x на сервере)
            try:
                mlflow.sklearn.log_model(
                    model,
                    "sklearn_model",
                    registered_model_name=None,  # Не регистрируем в Model Registry
                )
            except Exception as e:
                logger.warning(f"  ⚠️  Не удалось залогировать sklearn модель: {e}")

            # 4.2 Модель в pickle формате
            model_pkl_path = os.path.join(temp_dir, f"{model_name}.pkl")
            with open(model_pkl_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_pkl_path, "model_artifacts")

            # 4.3 CSV с предсказаниями
            predictions_df = pd.DataFrame(
                {
                    "actual": y_test.values,
                    "predicted": y_pred,
                    "error": y_test.values - y_pred,
                    "abs_error": np.abs(y_test.values - y_pred),
                    "pct_error": np.abs((y_test.values - y_pred) / y_test.values) * 100,
                }
            )
            predictions_csv = os.path.join(temp_dir, "predictions.csv")
            predictions_df.to_csv(predictions_csv, index=False)
            mlflow.log_artifact(predictions_csv, "predictions")

            # 4.4 Графики
            plots = create_plots(model, run_name, X_train, y_test, y_pred, temp_dir)
            for artifact_path, plot_path in plots:
                mlflow.log_artifact(plot_path, artifact_path)

            # 4.5 Конфигурация эксперимента (JSON)
            import json

            config_data = {
                "model_type": model_name,
                "params": custom_params,
                "description": description,
                "data": {
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "features": list(X_train.columns),
                },
                "metrics": metrics,
                "performance": {
                    "train_time_seconds": train_time,
                    "inference_time_seconds": inference_time,
                },
                "timestamp": datetime.now().isoformat(),
            }
            config_path = os.path.join(temp_dir, "experiment_config.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            mlflow.log_artifact(config_path, "config")

        # ═══════════════════════════════════════════════════════════════════
        # 5. ЛОГИРОВАНИЕ ТЕГОВ
        # ═══════════════════════════════════════════════════════════════════

        mlflow.set_tag("algorithm_family", get_algorithm_family(model_name))
        mlflow.set_tag("experiment_type", "model_comparison")
        mlflow.set_tag("dataset", "boston_housing")
        mlflow.set_tag("author", "data_scientist")
        mlflow.set_tag("environment", "development")
        mlflow.set_tag(
            "mlflow.note.content",
            f"""
## Эксперимент: {run_name}

**Описание:** {description}

**Модель:** {MODEL_REGISTRY[model_name]["description"]}

**Параметры:** {custom_params}

**Результаты:**
- R² Score: {metrics["r2_score"]:.4f}
- RMSE: {metrics["rmse"]:.4f}
- MAE: {metrics["mae"]:.4f}
- MAPE: {metrics["mape"]:.2f}%
""",
        )

        # Явно завершаем run как успешный (важно для MLflow 3.x + сервер 2.x)
        mlflow.end_run(status="FINISHED")

        logger.success("  ✅ Эксперимент завершён!")

    # Возвращаем результаты после закрытия контекста mlflow
    return {
        "run_name": run_name,
        "model_type": model_name,
        **metrics,
        "train_time": train_time,
    }


def print_summary(results):
    """Печать итоговой сводки экспериментов."""

    df = pd.DataFrame(results)
    df = df.sort_values("r2_score", ascending=False)

    logger.info("\n")
    logger.info("=" * 80)
    logger.info("📊 ИТОГОВАЯ СВОДКА ЭКСПЕРИМЕНТОВ")
    logger.info("=" * 80)

    logger.info("\n🏆 ТОП-5 МОДЕЛЕЙ ПО R² SCORE:\n")
    for i, row in df.head(5).iterrows():
        logger.info(
            f"  {row['run_name'][:40]:40} | R²: {row['r2_score']:.4f} | RMSE: {row['rmse']:.4f}"
        )

    logger.info("\n📈 СТАТИСТИКА ПО СЕМЕЙСТВАМ АЛГОРИТМОВ:\n")

    # Группировка по семействам
    df["family"] = df["model_type"].apply(get_algorithm_family)

    for family in df["family"].unique():
        family_df = df[df["family"] == family]
        best = family_df.loc[family_df["r2_score"].idxmax()]
        logger.info(
            f"  {family.upper():15} | Best R²: {best['r2_score']:.4f} | Model: {best['model_type']}"
        )

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Всего проведено {len(results)} экспериментов")
    logger.info("=" * 80)

    return df


def main():
    """Основная функция запуска экспериментов."""

    logger.info("\n")
    logger.info("🔬" * 30)
    logger.info("  ЗАПУСК МАССОВЫХ ЭКСПЕРИМЕНТОВ ML")
    logger.info("🔬" * 30)
    logger.info("\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Настройка MLflow
    # ═══════════════════════════════════════════════════════════════════════

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"🔗 MLflow Tracking URI: {mlflow_uri}")

    mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = "boston_housing_model_comparison"
    mlflow.set_experiment(experiment_name)
    logger.info(f"📁 Эксперимент: {experiment_name}")

    # ═══════════════════════════════════════════════════════════════════════
    # Загрузка данных
    # ═══════════════════════════════════════════════════════════════════════

    logger.info("\n📂 Загрузка данных Boston Housing...")
    X_train, X_test, y_train, y_test = load_data()
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    logger.info(f"  Features: {list(X_train.columns)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Запуск всех экспериментов
    # ═══════════════════════════════════════════════════════════════════════

    logger.info(f"\n🚀 Запуск {len(EXPERIMENTS_CONFIG)} экспериментов...\n")

    results = []
    total = len(EXPERIMENTS_CONFIG)

    for i, config in enumerate(EXPERIMENTS_CONFIG, 1):
        try:
            result = run_single_experiment(
                config, X_train, X_test, y_train, y_test, i, total
            )
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Ошибка в эксперименте {config['name']}: {e}")
            continue

    # ═══════════════════════════════════════════════════════════════════════
    # Сводка результатов
    # ═══════════════════════════════════════════════════════════════════════

    results_df = print_summary(results)

    # Сохранение результатов в CSV
    results_path = (
        Path(__file__).parent.parent / "data" / "experiments" / "results_summary.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n💾 Результаты сохранены: {results_path}")

    logger.info(f"\n🌐 Откройте MLflow UI: {mlflow_uri}")
    logger.info("   Для просмотра и сравнения экспериментов\n")


if __name__ == "__main__":
    main()
