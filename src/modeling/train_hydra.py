"""
Обучение моделей с использованием Hydra для управления конфигурациями.

Использование:
    # Базовый запуск (Random Forest по умолчанию)
    uv run python src/modeling/train_hydra.py

    # Смена модели
    uv run python src/modeling/train_hydra.py model=gradient_boosting

    # Переопределение параметров
    uv run python src/modeling/train_hydra.py model=random_forest model.n_estimators=500

    # Готовый эксперимент
    uv run python src/modeling/train_hydra.py +experiment=tuned

    # Multirun (несколько моделей)
    uv run python src/modeling/train_hydra.py --multirun model=ridge,lasso,elastic_net
"""

import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dvclive import Live
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# Добавляем путь проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR, RAW_DATA_DIR, HOUSING_DATA_FILE
from src.ml_models.model_loader import create_model as create_sklearn_model
from src.schemas import ExperimentConfig


def load_data(data_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Загрузка данных Boston Housing с учётом конфигурации."""
    raw_path = data_config.get("raw_path", "data/raw/housing.csv")

    # Определяем путь к данным
    data_path = Path(raw_path)
    if not data_path.is_absolute():
        # Относительный путь от корня проекта
        project_root = Path(__file__).resolve().parents[2]
        data_path = project_root / raw_path

    if not data_path.exists():
        # Fallback на стандартный путь
        data_path = RAW_DATA_DIR / HOUSING_DATA_FILE

    logger.info(f"Загрузка данных из {data_path}")

    # Чтение CSV
    separator = data_config.get("separator", r"\s+")
    header = data_config.get("header", None)
    df = pd.read_csv(data_path, sep=separator, header=header)

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

    # Целевая переменная
    target_column = data_config.get("target_column", "MEDV")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    logger.info(f"Загружено {len(df)} записей, {len(X.columns)} признаков")
    return X, y


def validate_config(cfg: DictConfig) -> ExperimentConfig:
    """Валидация конфигурации через Pydantic."""
    try:
        # Конвертируем OmegaConf в dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Создаём ExperimentConfig для валидации
        exp_config = ExperimentConfig(
            model=config_dict.get("model", {}),
            data=config_dict.get("data", {}),
            training=config_dict.get("training", {}),
            name=config_dict.get("name", "default"),
            description=config_dict.get("description", ""),
            tags=config_dict.get("tags", []),
        )

        # Валидируем вложенные конфигурации
        model_config, data_config, training_config = exp_config.validate_all()

        logger.success(f"Конфигурация валидна: model={model_config.name}")
        return exp_config

    except Exception as e:
        logger.error(f"Ошибка валидации конфигурации: {e}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Оценка модели и расчёт метрик."""
    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
    }

    return metrics


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Основная функция обучения с Hydra.

    Args:
        cfg: Hydra конфигурация

    Returns:
        R² score для оптимизации гиперпараметров
    """
    # Логируем конфигурацию
    logger.info("=" * 60)
    logger.info("HYDRA CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Валидация конфигурации через Pydantic
    exp_config = validate_config(cfg)
    model_config = exp_config.get_validated_model_config()

    # Извлекаем параметры
    model_name = model_config.name
    model_params = model_config.get_params()
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)

    # Параметры обучения
    test_size = training_dict.get("test_size", 0.2)
    random_state = training_dict.get("random_state", 42)
    use_cv = training_dict.get("cross_validation", False)
    cv_folds = training_dict.get("cv_folds", 5)
    use_dvclive = training_dict.get("use_dvclive", True)

    # Загрузка данных
    X, y = load_data(data_config)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Создание модели через model_loader
    logger.info(f"Создание модели: {model_name}")
    logger.info(f"Параметры: {model_params}")

    model = create_sklearn_model(model_name, custom_params=model_params)

    # DVCLive для логирования
    if use_dvclive:
        live = Live(save_dvc_exp=training_dict.get("save_dvc_exp", True))
    else:
        live = None

    try:
        # Логируем параметры
        if live:
            live.log_param("model_name", model_name)
            for key, value in model_params.items():
                live.log_param(f"model.{key}", value)
            live.log_param("test_size", test_size)
            live.log_param("random_state", random_state)
            live.log_param("n_samples", len(X))
            live.log_param("n_features", len(X.columns))

        # Кросс-валидация (если включена)
        if use_cv:
            logger.info(f"Кросс-валидация: {cv_folds} фолдов")
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv_folds, scoring="r2"
            )
            logger.info(f"CV R² scores: {cv_scores}")
            logger.info(f"CV R² mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if live:
                live.log_metric("cv_r2_mean", float(cv_scores.mean()))
                live.log_metric("cv_r2_std", float(cv_scores.std()))

        # Обучение модели
        logger.info("Обучение модели...")
        model.fit(X_train, y_train)
        logger.success("Модель обучена!")

        # Оценка модели
        metrics = evaluate_model(model, X_test, y_test)

        # Логирование метрик
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
            if live:
                live.log_metric(metric_name, metric_value)

        # Важность признаков (если доступна)
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            logger.info("\n📊 Важность признаков:")
            for _, row in feature_importance.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Сохранение модели
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_filename = f"{model_name}_hydra.pkl"
        model_path = MODELS_DIR / model_filename

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"Модель сохранена: {model_path}")

        if live:
            live.log_artifact(str(model_path), type="model", name=model_name)

        # Итоговые метрики
        logger.info("\n" + "=" * 50)
        logger.info("📈 ИТОГОВЫЕ МЕТРИКИ:")
        logger.info(f"  R² Score:  {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE:      {metrics['rmse']:.4f}")
        logger.info(f"  MAE:       {metrics['mae']:.4f}")
        logger.info(f"  MAPE:      {metrics['mape']:.2f}%")
        logger.info("=" * 50)

        # Возвращаем R² для Hydra multirun optimization
        return metrics["r2_score"]

    finally:
        if live:
            live.end()


if __name__ == "__main__":
    main()
