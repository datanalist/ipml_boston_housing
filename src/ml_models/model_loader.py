"""
Загрузчик моделей регрессии scikit-learn.

Модуль предоставляет функции для создания, сохранения и загрузки
различных моделей регрессии из scikit-learn.
"""

import pickle
from pathlib import Path
from typing import Any

import click
from loguru import logger
from sklearn.base import RegressorMixin
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
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import MODELS_DIR


# Реестр доступных моделей регрессии с параметрами по умолчанию
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Линейные модели
    "linear_regression": {
        "class": LinearRegression,
        "params": {},
        "description": "Обычная линейная регрессия (МНК)",
    },
    "ridge": {
        "class": Ridge,
        "params": {"alpha": 1.0, "random_state": 42},
        "description": "Линейная регрессия с L2-регуляризацией",
    },
    "lasso": {
        "class": Lasso,
        "params": {"alpha": 1.0, "random_state": 42},
        "description": "Линейная регрессия с L1-регуляризацией",
    },
    "elastic_net": {
        "class": ElasticNet,
        "params": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42},
        "description": "Линейная регрессия с L1+L2 регуляризацией",
    },
    "huber": {
        "class": HuberRegressor,
        "params": {"epsilon": 1.35, "max_iter": 100},
        "description": "Робастная регрессия (устойчива к выбросам)",
    },
    "sgd": {
        "class": SGDRegressor,
        "params": {"max_iter": 1000, "tol": 1e-3, "random_state": 42},
        "description": "Стохастический градиентный спуск",
    },
    # Деревья и ансамбли
    "decision_tree": {
        "class": DecisionTreeRegressor,
        "params": {"max_depth": 10, "random_state": 42},
        "description": "Дерево решений для регрессии",
    },
    "random_forest": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "description": "Случайный лес",
    },
    "extra_trees": {
        "class": ExtraTreesRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "description": "Экстремально рандомизированные деревья",
    },
    "gradient_boosting": {
        "class": GradientBoostingRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        },
        "description": "Градиентный бустинг",
    },
    "adaboost": {
        "class": AdaBoostRegressor,
        "params": {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
        "description": "AdaBoost регрессор",
    },
    "bagging": {
        "class": BaggingRegressor,
        "params": {"n_estimators": 10, "random_state": 42, "n_jobs": -1},
        "description": "Бэггинг регрессор",
    },
    # Другие модели
    "svr": {
        "class": SVR,
        "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
        "description": "Опорные вектора для регрессии",
    },
    "knn": {
        "class": KNeighborsRegressor,
        "params": {"n_neighbors": 5, "weights": "uniform", "n_jobs": -1},
        "description": "K ближайших соседей",
    },
}


def get_available_models() -> list[str]:
    """Возвращает список доступных моделей."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> dict[str, Any] | None:
    """
    Возвращает информацию о модели.

    Args:
        model_name: Имя модели из реестра

    Returns:
        Словарь с информацией о модели или None, если модель не найдена
    """
    return MODEL_REGISTRY.get(model_name)


def create_model(
    model_name: str,
    custom_params: dict[str, Any] | None = None,
) -> RegressorMixin:
    """
    Создаёт экземпляр модели регрессии.

    Args:
        model_name: Имя модели из реестра
        custom_params: Пользовательские параметры (переопределяют параметры по умолчанию)

    Returns:
        Экземпляр модели scikit-learn

    Raises:
        ValueError: Если модель не найдена в реестре
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Модель '{model_name}' не найдена. Доступные модели: {available}"
        )

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info["class"]
    params = model_info["params"].copy()

    # Переопределяем параметры пользовательскими значениями
    if custom_params:
        params.update(custom_params)

    logger.info(f"Создание модели '{model_name}' с параметрами: {params}")
    return model_class(**params)


def save_model(
    model: RegressorMixin,
    model_name: str,
    models_dir: Path | None = None,
) -> Path:
    """
    Сохраняет модель в pickle-файл.

    Args:
        model: Обученная модель
        model_name: Имя для файла модели (без расширения)
        models_dir: Каталог для сохранения (по умолчанию: data/models/)

    Returns:
        Путь к сохранённому файлу
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.success(f"Модель сохранена: {model_path}")
    return model_path


def load_model(
    model_name: str,
    models_dir: Path | None = None,
) -> RegressorMixin:
    """
    Загружает модель из pickle-файла.

    Args:
        model_name: Имя файла модели (без расширения)
        models_dir: Каталог с моделями (по умолчанию: data/models/)

    Returns:
        Загруженная модель

    Raises:
        FileNotFoundError: Если файл модели не найден
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    model_path = models_dir / f"{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Модель загружена: {model_path}")
    return model


def create_and_save_model(
    model_name: str,
    custom_params: dict[str, Any] | None = None,
    models_dir: Path | None = None,
) -> tuple[RegressorMixin, Path]:
    """
    Создаёт и сохраняет модель (без обучения).

    Args:
        model_name: Имя модели из реестра
        custom_params: Пользовательские параметры
        models_dir: Каталог для сохранения

    Returns:
        Кортеж (модель, путь к файлу)
    """
    model = create_model(model_name, custom_params)
    path = save_model(model, model_name, models_dir)
    return model, path


def create_all_models(
    models_dir: Path | None = None,
    model_names: list[str] | None = None,
) -> dict[str, RegressorMixin]:
    """
    Создаёт и сохраняет все модели (или указанный список).

    Args:
        models_dir: Каталог для сохранения
        model_names: Список моделей для создания (по умолчанию: все модели)

    Returns:
        Словарь {имя_модели: экземпляр_модели}
    """
    if model_names is None:
        model_names = get_available_models()

    models = {}
    for name in model_names:
        try:
            model, _ = create_and_save_model(name, models_dir=models_dir)
            models[name] = model
        except Exception as e:
            logger.error(f"Ошибка создания модели '{name}': {e}")

    logger.success(f"Создано и сохранено {len(models)} моделей")
    return models


def load_all_models(
    models_dir: Path | None = None,
) -> dict[str, RegressorMixin]:
    """
    Загружает все сохранённые модели из каталога.

    Args:
        models_dir: Каталог с моделями

    Returns:
        Словарь {имя_модели: загруженная_модель}
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models = {}
    for pkl_file in models_dir.glob("*.pkl"):
        model_name = pkl_file.stem
        try:
            models[model_name] = load_model(model_name, models_dir)
        except Exception as e:
            logger.error(f"Ошибка загрузки модели '{model_name}': {e}")

    logger.info(f"Загружено {len(models)} моделей")
    return models


# CLI интерфейс
@click.group()
def cli():
    """Утилита для управления моделями регрессии."""
    pass


@cli.command("list")
def list_models_cmd():
    """Показать список доступных моделей."""
    click.echo("\n📋 Доступные модели регрессии:\n")
    click.echo("-" * 60)

    for name, info in MODEL_REGISTRY.items():
        click.echo(f"  {name:20} - {info['description']}")

    click.echo("-" * 60)
    click.echo(f"\nВсего: {len(MODEL_REGISTRY)} моделей\n")


@cli.command("create")
@click.argument("model_name", required=False)
@click.option(
    "--all", "-a", "create_all", is_flag=True, help="Создать все доступные модели"
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    type=click.Path(),
    help="Каталог для сохранения (по умолчанию: data/models/)",
)
def create_model_cmd(model_name: str | None, create_all: bool, output_dir: str | None):
    """Создать и сохранить модель(и)."""
    models_dir = Path(output_dir) if output_dir else MODELS_DIR

    if create_all:
        click.echo(f"\n🔧 Создание всех моделей в {models_dir}...\n")
        create_all_models(models_dir=models_dir)
    elif model_name:
        if model_name not in MODEL_REGISTRY:
            click.echo(f"❌ Модель '{model_name}' не найдена.")
            click.echo("Используйте 'python model_loader.py list' для списка моделей.")
            return
        click.echo(f"\n🔧 Создание модели '{model_name}'...\n")
        create_and_save_model(model_name, models_dir=models_dir)
    else:
        click.echo("❌ Укажите имя модели или используйте флаг --all")
        click.echo("Пример: python model_loader.py create random_forest")


@cli.command("info")
@click.argument("model_name")
def model_info_cmd(model_name: str):
    """Показать информацию о модели."""
    info = get_model_info(model_name)

    if info is None:
        click.echo(f"❌ Модель '{model_name}' не найдена.")
        return

    click.echo(f"\n📊 Информация о модели '{model_name}':\n")
    click.echo(f"  Описание: {info['description']}")
    click.echo(f"  Класс:    {info['class'].__name__}")
    click.echo("  Параметры по умолчанию:")
    for param, value in info["params"].items():
        click.echo(f"    - {param}: {value}")
    click.echo()


if __name__ == "__main__":
    cli()
