"""
DAG: Boston Housing Experiments Pipeline
========================================
Параллельное обучение 19 моделей машинного обучения с различными алгоритмами.

Архитектура:
1. download_data - Загрузка и подготовка данных
2. validate_data - Валидация данных
3. Параллельное обучение моделей (3 группы):
   - Linear Models (7 моделей)
   - Tree Models (9 моделей)
   - Other Models (3 модели)
4. aggregate_results - Агрегация результатов
5. generate_report - Генерация итогового отчёта
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

# Пути к данным
DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
HOUSING_DATA_FILE = "housing.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ 19 ЭКСПЕРИМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════════

# Линейные модели (7 экспериментов)
LINEAR_MODELS = [
    {
        "name": "linear_regression",
        "params": {},
        "description": "Baseline линейная регрессия",
    },
    {"name": "ridge", "params": {"alpha": 0.1}, "description": "Ridge α=0.1"},
    {"name": "ridge", "params": {"alpha": 1.0}, "description": "Ridge α=1.0"},
    {"name": "ridge", "params": {"alpha": 10.0}, "description": "Ridge α=10.0"},
    {"name": "lasso", "params": {"alpha": 0.1}, "description": "Lasso α=0.1"},
    {
        "name": "elastic_net",
        "params": {"alpha": 0.5, "l1_ratio": 0.5},
        "description": "ElasticNet",
    },
    {"name": "huber", "params": {"epsilon": 1.35}, "description": "Huber Regressor"},
]

# Древовидные модели и ансамбли (9 экспериментов)
TREE_MODELS = [
    {
        "name": "decision_tree",
        "params": {"max_depth": 5},
        "description": "Decision Tree d=5",
    },
    {
        "name": "decision_tree",
        "params": {"max_depth": 10},
        "description": "Decision Tree d=10",
    },
    {
        "name": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 10},
        "description": "RF n=100",
    },
    {
        "name": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 15},
        "description": "RF n=200",
    },
    {
        "name": "extra_trees",
        "params": {"n_estimators": 100, "max_depth": 10},
        "description": "ExtraTrees",
    },
    {
        "name": "gradient_boosting",
        "params": {"n_estimators": 100, "learning_rate": 0.1},
        "description": "GBM lr=0.1",
    },
    {
        "name": "gradient_boosting",
        "params": {"n_estimators": 200, "learning_rate": 0.05},
        "description": "GBM lr=0.05",
    },
    {
        "name": "adaboost",
        "params": {"n_estimators": 50, "learning_rate": 1.0},
        "description": "AdaBoost",
    },
    {"name": "bagging", "params": {"n_estimators": 20}, "description": "Bagging"},
]

# Другие модели (3 эксперимента)
OTHER_MODELS = [
    {"name": "svr", "params": {"kernel": "rbf", "C": 1.0}, "description": "SVR RBF"},
    {
        "name": "knn",
        "params": {"n_neighbors": 5, "weights": "uniform"},
        "description": "KNN k=5",
    },
    {
        "name": "knn",
        "params": {"n_neighbors": 10, "weights": "distance"},
        "description": "KNN k=10",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════


def create_model(model_name: str, params: dict):
    """Создаёт модель по имени и параметрам."""
    from sklearn.ensemble import (
        AdaBoostRegressor,
        BaggingRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        Ridge,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "huber": HuberRegressor,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "adaboost": AdaBoostRegressor,
        "bagging": BaggingRegressor,
        "svr": SVR,
        "knn": KNeighborsRegressor,
    }

    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model_class(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════


@dag(
    dag_id="boston_housing_experiments",
    default_args=default_args,
    description="Параллельное обучение 19 ML моделей с агрегацией результатов",
    schedule=None,  # Ручной запуск
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "boston_housing", "experiments", "parallel"],
    max_active_tasks=8,  # Максимум 8 параллельных задач
)
def boston_housing_experiments_dag():
    """
    DAG для параллельного обучения множества ML моделей
    на датасете Boston Housing с агрегацией результатов.
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
            data_lines = []
            for line in lines[1:]:
                parts = line.split(",")
                data_lines.append(" ".join(parts))
            content = "\n".join(data_lines)

        with open(output_path, "w") as f:
            f.write(content)

        logger.success(f"✅ Данные сохранены: {output_path}")
        return str(output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Валидация данных
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def validate_data(data_path: str) -> dict:
        """Валидация данных и подготовка train/test split."""
        import pandas as pd
        from loguru import logger
        from sklearn.model_selection import train_test_split

        logger.info(f"🔍 Валидация данных: {data_path}")

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

        # Сохраняем подготовленные данные
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_path = EXPERIMENTS_DIR / "train_data.csv"
        test_path = EXPERIMENTS_DIR / "test_data.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.success(
            f"✅ Данные подготовлены: train={len(X_train)}, test={len(X_test)}"
        )

        return {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": list(X.columns),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Обучение одной модели (используется для expand)
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def train_single_model(model_config: dict, data_info: dict) -> dict:
        """
        Обучает одну модель по заданной конфигурации.
        Используется с expand() для параллельного обучения.
        """
        import pickle
        import time

        import numpy as np
        import pandas as pd
        from loguru import logger
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model_name = model_config["name"]
        params = model_config["params"]
        description = model_config["description"]

        # Генерируем уникальный ID
        param_str = "_".join([f"{k}={v}" for k, v in params.items()])
        run_id = f"{model_name}_{param_str}" if param_str else model_name

        logger.info(f"🚀 Обучение: {run_id}")

        # Загружаем данные
        train_df = pd.read_csv(data_info["train_path"])
        test_df = pd.read_csv(data_info["test_path"])

        X_train = train_df.drop("MEDV", axis=1)
        y_train = train_df["MEDV"]
        X_test = test_df.drop("MEDV", axis=1)
        y_test = test_df["MEDV"]

        # Создаём и обучаем модель
        model = create_model(model_name, params)

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Предсказания и метрики
        y_pred = model.predict(X_test)

        metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        }

        # Сохраняем модель
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{run_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(
            f"✅ {run_id}: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}"
        )

        return {
            "run_id": run_id,
            "model_name": model_name,
            "description": description,
            "params": params,
            "metrics": metrics,
            "train_time": train_time,
            "model_path": str(model_path),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Агрегация результатов
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def aggregate_results(
        linear_results: list[dict], tree_results: list[dict], other_results: list[dict]
    ) -> dict:
        """Агрегирует результаты всех экспериментов."""

        import pandas as pd
        from loguru import logger

        # Преобразуем LazyXComAccess в списки
        linear_results = list(linear_results) if linear_results else []
        tree_results = list(tree_results) if tree_results else []
        other_results = list(other_results) if other_results else []

        # Преобразуем LazyXComAccess в списки
        linear_results = list(linear_results) if linear_results else []
        tree_results = list(tree_results) if tree_results else []
        other_results = list(other_results) if other_results else []

        all_results = linear_results + tree_results + other_results

        logger.info(f"📊 Агрегация {len(all_results)} экспериментов")

        # Создаём DataFrame
        rows = []
        for r in all_results:
            rows.append(
                {
                    "run_id": r["run_id"],
                    "model_name": r["model_name"],
                    "description": r["description"],
                    "r2_score": r["metrics"]["r2_score"],
                    "rmse": r["metrics"]["rmse"],
                    "mae": r["metrics"]["mae"],
                    "mape": r["metrics"]["mape"],
                    "train_time": r["train_time"],
                    "model_path": r["model_path"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("r2_score", ascending=False)

        # Сохраняем результаты
        results_path = EXPERIMENTS_DIR / "all_results.csv"
        df.to_csv(results_path, index=False)

        # Лучшие модели по семействам
        best_overall = df.iloc[0].to_dict()

        logger.success(
            f"🏆 Лучшая модель: {best_overall['run_id']} (R²={best_overall['r2_score']:.4f})"
        )

        return {
            "total_experiments": len(all_results),
            "results_path": str(results_path),
            "best_model": best_overall,
            "summary": {
                "mean_r2": float(df["r2_score"].mean()),
                "std_r2": float(df["r2_score"].std()),
                "best_r2": float(df["r2_score"].max()),
                "worst_r2": float(df["r2_score"].min()),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TASK: Генерация отчёта
    # ─────────────────────────────────────────────────────────────────────────
    @task
    def generate_report(aggregated: dict) -> dict:
        """Генерирует итоговый отчёт и сохраняет в MinIO."""
        from datetime import datetime

        import boto3
        import pandas as pd
        from botocore.client import Config
        from loguru import logger

        logger.info("📝 Генерация отчёта")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Читаем результаты
        df = pd.read_csv(aggregated["results_path"])

        # Генерируем markdown отчёт
        report = f"""# Boston Housing ML Experiments Report
## Дата: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Сводка
- Всего экспериментов: {aggregated["total_experiments"]}
- Лучший R²: {aggregated["summary"]["best_r2"]:.4f}
- Средний R²: {aggregated["summary"]["mean_r2"]:.4f} ± {aggregated["summary"]["std_r2"]:.4f}

### Лучшая модель
- **Модель:** {aggregated["best_model"]["run_id"]}
- **R² Score:** {aggregated["best_model"]["r2_score"]:.4f}
- **RMSE:** {aggregated["best_model"]["rmse"]:.4f}
- **MAE:** {aggregated["best_model"]["mae"]:.4f}

### Топ-5 моделей

| Ранг | Модель | R² | RMSE | MAE |
|------|--------|-----|------|-----|
"""
        for i, row in df.head(5).iterrows():
            report += f"| {i + 1} | {row['run_id'][:30]} | {row['r2_score']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} |\n"

        report += """
### Результаты по семействам алгоритмов

#### Линейные модели
"""
        linear_df = df[
            df["model_name"].isin(
                ["linear_regression", "ridge", "lasso", "elastic_net", "huber"]
            )
        ]
        if not linear_df.empty:
            best_linear = linear_df.iloc[0]
            report += f"- Лучшая: {best_linear['run_id']} (R²={best_linear['r2_score']:.4f})\n"

        report += """
#### Древовидные модели
"""
        tree_names = [
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "bagging",
        ]
        tree_df = df[df["model_name"].isin(tree_names)]
        if not tree_df.empty:
            best_tree = tree_df.iloc[0]
            report += (
                f"- Лучшая: {best_tree['run_id']} (R²={best_tree['r2_score']:.4f})\n"
            )

        report += """
#### Другие модели
"""
        other_df = df[df["model_name"].isin(["svr", "knn"])]
        if not other_df.empty:
            best_other = other_df.iloc[0]
            report += (
                f"- Лучшая: {best_other['run_id']} (R²={best_other['r2_score']:.4f})\n"
            )

        # Сохраняем локально
        report_path = EXPERIMENTS_DIR / f"report_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.success(f"✅ Отчёт сохранён: {report_path}")

        # Загружаем в MinIO
        try:
            minio_endpoint = os.environ.get(
                "MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"
            )
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

            s3_client = boto3.client(
                "s3",
                endpoint_url=minio_endpoint,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=Config(signature_version="s3v4"),
            )

            bucket_name = "airflow-artifacts"

            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except Exception:
                s3_client.create_bucket(Bucket=bucket_name)

            # Загружаем отчёт
            s3_client.upload_file(
                str(report_path), bucket_name, f"reports/report_{timestamp}.md"
            )

            # Загружаем результаты
            s3_client.upload_file(
                aggregated["results_path"],
                bucket_name,
                f"results/all_results_{timestamp}.csv",
            )

            # Загружаем лучшую модель
            s3_client.upload_file(
                aggregated["best_model"]["model_path"],
                bucket_name,
                f"models/best_model_{timestamp}.pkl",
            )

            logger.success("✅ Артефакты загружены в MinIO")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки в MinIO: {e}")

        # Логирование в MLflow
        try:
            import mlflow

            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("boston_housing_experiments")

            with mlflow.start_run(run_name=f"experiments_summary_{timestamp}"):
                mlflow.log_metric("total_experiments", aggregated["total_experiments"])
                mlflow.log_metric("best_r2", aggregated["summary"]["best_r2"])
                mlflow.log_metric("mean_r2", aggregated["summary"]["mean_r2"])
                mlflow.log_param("best_model", aggregated["best_model"]["run_id"])
                mlflow.log_artifact(str(report_path), "reports")
                mlflow.log_artifact(aggregated["results_path"], "results")
                mlflow.set_tag("source", "airflow")
                mlflow.set_tag("dag", "boston_housing_experiments")

            logger.success("✅ Результаты залогированы в MLflow")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка логирования в MLflow: {e}")

        return {
            "status": "success",
            "report_path": str(report_path),
            "total_experiments": aggregated["total_experiments"],
            "best_model": aggregated["best_model"]["run_id"],
            "best_r2": aggregated["summary"]["best_r2"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ И ПАРАЛЛЕЛЬНОГО ВЫПОЛНЕНИЯ
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Загрузка данных
    data_path = download_data()

    # 2. Валидация и подготовка данных
    data_info = validate_data(data_path)

    # 3. Параллельное обучение моделей с использованием expand()

    # Линейные модели
    linear_results = train_single_model.expand(
        model_config=LINEAR_MODELS,
        data_info=[data_info] * len(LINEAR_MODELS),
    )

    # Древовидные модели
    tree_results = train_single_model.expand(
        model_config=TREE_MODELS,
        data_info=[data_info] * len(TREE_MODELS),
    )

    # Другие модели
    other_results = train_single_model.expand(
        model_config=OTHER_MODELS,
        data_info=[data_info] * len(OTHER_MODELS),
    )

    # 4. Агрегация результатов (ждёт завершения всех моделей)
    aggregated = aggregate_results(
        linear_results=linear_results,
        tree_results=tree_results,
        other_results=other_results,
    )

    # 5. Генерация отчёта
    generate_report(aggregated)


# Создание DAG
boston_housing_experiments_dag()
