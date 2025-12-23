"""Pydantic схемы для конфигураций моделей ML."""

from typing import Any, Literal

from pydantic import Field, field_validator

from src.schemas.base import BaseConfig


class ModelConfig(BaseConfig):
    """Базовая конфигурация модели."""

    name: str = Field(
        ...,
        description="Имя модели из реестра",
    )
    random_state: int | None = Field(
        default=42,
        ge=0,
        description="Seed для воспроизводимости",
    )

    def get_params(self) -> dict[str, Any]:
        """Возвращает параметры для создания модели (без name)."""
        params = self.model_dump(exclude={"name"}, exclude_none=True)
        return params


# ═══════════════════════════════════════════════════════════════════════════════
# ЛИНЕЙНЫЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════


class LinearRegressionConfig(ModelConfig):
    """Конфигурация линейной регрессии (МНК)."""

    name: Literal["linear_regression"] = "linear_regression"
    fit_intercept: bool = Field(
        default=True,
        description="Вычислять свободный член",
    )
    random_state: int | None = Field(
        default=None,
        description="Не используется в LinearRegression",
    )


class RidgeConfig(ModelConfig):
    """Конфигурация Ridge регрессии (L2-регуляризация)."""

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(
        default=1.0,
        gt=0,
        le=1000,
        description="Коэффициент регуляризации (0 < alpha <= 1000)",
    )
    fit_intercept: bool = Field(
        default=True,
        description="Вычислять свободный член",
    )
    solver: Literal["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"] = (
        Field(
            default="auto",
            description="Алгоритм решения",
        )
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"alpha должен быть положительным, получено: {v}")
        return v


class LassoConfig(ModelConfig):
    """Конфигурация Lasso регрессии (L1-регуляризация)."""

    name: Literal["lasso"] = "lasso"
    alpha: float = Field(
        default=1.0,
        gt=0,
        le=1000,
        description="Коэффициент регуляризации",
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Максимальное число итераций",
    )
    tol: float = Field(
        default=1e-4,
        gt=0,
        description="Допуск для критерия остановки",
    )
    selection: Literal["cyclic", "random"] = Field(
        default="cyclic",
        description="Метод выбора коэффициентов для обновления",
    )


class ElasticNetConfig(ModelConfig):
    """Конфигурация ElasticNet (L1+L2 регуляризация)."""

    name: Literal["elastic_net"] = "elastic_net"
    alpha: float = Field(
        default=1.0,
        gt=0,
        le=1000,
        description="Общий коэффициент регуляризации",
    )
    l1_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Соотношение L1/L2 (0=Ridge, 1=Lasso)",
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Максимальное число итераций",
    )
    tol: float = Field(
        default=1e-4,
        gt=0,
        description="Допуск для критерия остановки",
    )

    @field_validator("l1_ratio")
    @classmethod
    def validate_l1_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"l1_ratio должен быть между 0 и 1, получено: {v}")
        return v


class HuberConfig(ModelConfig):
    """Конфигурация Huber регрессора (робастная регрессия)."""

    name: Literal["huber"] = "huber"
    epsilon: float = Field(
        default=1.35,
        gt=1.0,
        le=10.0,
        description="Порог для определения выбросов (>1.0)",
    )
    max_iter: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Максимальное число итераций",
    )
    alpha: float = Field(
        default=0.0001,
        ge=0,
        description="Коэффициент L2-регуляризации",
    )
    fit_intercept: bool = Field(
        default=True,
        description="Вычислять свободный член",
    )
    random_state: int | None = Field(
        default=None,
        description="Не используется в HuberRegressor",
    )

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        if v <= 1.0:
            raise ValueError(f"epsilon должен быть > 1.0, получено: {v}")
        return v


class SGDConfig(ModelConfig):
    """Конфигурация SGD регрессора."""

    name: Literal["sgd"] = "sgd"
    loss: Literal["squared_error", "huber", "epsilon_insensitive"] = Field(
        default="squared_error",
        description="Функция потерь",
    )
    penalty: Literal["l2", "l1", "elasticnet", None] = Field(
        default="l2",
        description="Тип регуляризации",
    )
    alpha: float = Field(
        default=0.0001,
        gt=0,
        description="Коэффициент регуляризации",
    )
    max_iter: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Максимальное число итераций",
    )
    tol: float = Field(
        default=1e-3,
        gt=0,
        description="Допуск для критерия остановки",
    )
    learning_rate: Literal["constant", "optimal", "invscaling", "adaptive"] = Field(
        default="invscaling",
        description="Стратегия скорости обучения",
    )
    eta0: float = Field(
        default=0.01,
        gt=0,
        description="Начальная скорость обучения",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ДРЕВОВИДНЫЕ МОДЕЛИ И АНСАМБЛИ
# ═══════════════════════════════════════════════════════════════════════════════


class DecisionTreeConfig(ModelConfig):
    """Конфигурация дерева решений."""

    name: Literal["decision_tree"] = "decision_tree"
    max_depth: int | None = Field(
        default=10,
        ge=1,
        le=100,
        description="Максимальная глубина дерева",
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Минимум образцов для разбиения узла",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Минимум образцов в листе",
    )
    criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = (
        Field(
            default="squared_error",
            description="Критерий качества разбиения",
        )
    )
    splitter: Literal["best", "random"] = Field(
        default="best",
        description="Стратегия разбиения",
    )


class RandomForestConfig(ModelConfig):
    """Конфигурация Random Forest."""

    name: Literal["random_forest"] = "random_forest"
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Количество деревьев в лесу",
    )
    max_depth: int | None = Field(
        default=10,
        ge=1,
        le=100,
        description="Максимальная глубина деревьев",
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Минимум образцов для разбиения узла",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Минимум образцов в листе",
    )
    max_features: Literal["sqrt", "log2"] | float | None = Field(
        default="sqrt",
        description="Число признаков для поиска лучшего разбиения",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Число параллельных потоков (-1 = все ядра)",
    )
    bootstrap: bool = Field(
        default=True,
        description="Использовать bootstrap выборки",
    )

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        if v < 10:
            raise ValueError(f"n_estimators должен быть >= 10, получено: {v}")
        return v


class ExtraTreesConfig(ModelConfig):
    """Конфигурация Extra Trees."""

    name: Literal["extra_trees"] = "extra_trees"
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Количество деревьев",
    )
    max_depth: int | None = Field(
        default=10,
        ge=1,
        le=100,
        description="Максимальная глубина деревьев",
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Минимум образцов для разбиения узла",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Минимум образцов в листе",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Число параллельных потоков",
    )


class GradientBoostingConfig(ModelConfig):
    """Конфигурация Gradient Boosting."""

    name: Literal["gradient_boosting"] = "gradient_boosting"
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Количество этапов бустинга",
    )
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Скорость обучения (0 < lr <= 1)",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Максимальная глубина деревьев",
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Минимум образцов для разбиения",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Минимум образцов в листе",
    )
    subsample: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Доля образцов для обучения каждого дерева",
    )
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = Field(
        default="squared_error",
        description="Функция потерь",
    )

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"learning_rate должен быть (0, 1], получено: {v}")
        return v


class AdaBoostConfig(ModelConfig):
    """Конфигурация AdaBoost."""

    name: Literal["adaboost"] = "adaboost"
    n_estimators: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Количество слабых классификаторов",
    )
    learning_rate: float = Field(
        default=1.0,
        gt=0,
        le=10.0,
        description="Вклад каждого классификатора",
    )
    loss: Literal["linear", "square", "exponential"] = Field(
        default="linear",
        description="Функция потерь для обновления весов",
    )


class BaggingConfig(ModelConfig):
    """Конфигурация Bagging."""

    name: Literal["bagging"] = "bagging"
    n_estimators: int = Field(
        default=10,
        ge=5,
        le=200,
        description="Количество базовых моделей",
    )
    max_samples: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Доля образцов для каждой модели",
    )
    max_features: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Доля признаков для каждой модели",
    )
    bootstrap: bool = Field(
        default=True,
        description="Использовать bootstrap для образцов",
    )
    bootstrap_features: bool = Field(
        default=False,
        description="Использовать bootstrap для признаков",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Число параллельных потоков",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ДРУГИЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════


class SVRConfig(ModelConfig):
    """Конфигурация Support Vector Regression."""

    name: Literal["svr"] = "svr"
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field(
        default="rbf",
        description="Ядро SVM",
    )
    C: float = Field(
        default=1.0,
        gt=0,
        le=1000,
        description="Параметр регуляризации",
    )
    epsilon: float = Field(
        default=0.1,
        ge=0,
        le=10,
        description="Ширина epsilon-tube",
    )
    gamma: Literal["scale", "auto"] | float = Field(
        default="scale",
        description="Коэффициент ядра для rbf/poly/sigmoid",
    )
    degree: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Степень полиномиального ядра",
    )
    random_state: int | None = Field(
        default=None,
        description="Не используется в SVR",
    )


class KNNConfig(ModelConfig):
    """Конфигурация K-Nearest Neighbors."""

    name: Literal["knn"] = "knn"
    n_neighbors: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Количество соседей",
    )
    weights: Literal["uniform", "distance"] = Field(
        default="uniform",
        description="Весовая функция для прогноза",
    )
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = Field(
        default="auto",
        description="Алгоритм поиска соседей",
    )
    leaf_size: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Размер листа для BallTree/KDTree",
    )
    p: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Степень метрики Минковского (1=Manhattan, 2=Euclidean)",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Число параллельных потоков",
    )
    random_state: int | None = Field(
        default=None,
        description="Не используется в KNN",
    )

    @field_validator("n_neighbors")
    @classmethod
    def validate_n_neighbors(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_neighbors должен быть >= 1, получено: {v}")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# ФАБРИКА КОНФИГУРАЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG_REGISTRY: dict[str, type[ModelConfig]] = {
    "linear_regression": LinearRegressionConfig,
    "ridge": RidgeConfig,
    "lasso": LassoConfig,
    "elastic_net": ElasticNetConfig,
    "huber": HuberConfig,
    "sgd": SGDConfig,
    "decision_tree": DecisionTreeConfig,
    "random_forest": RandomForestConfig,
    "extra_trees": ExtraTreesConfig,
    "gradient_boosting": GradientBoostingConfig,
    "adaboost": AdaBoostConfig,
    "bagging": BaggingConfig,
    "svr": SVRConfig,
    "knn": KNNConfig,
}


def get_model_config_class(model_name: str) -> type[ModelConfig]:
    """Возвращает класс конфигурации для указанной модели."""
    if model_name not in MODEL_CONFIG_REGISTRY:
        available = ", ".join(MODEL_CONFIG_REGISTRY.keys())
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {available}")
    return MODEL_CONFIG_REGISTRY[model_name]
