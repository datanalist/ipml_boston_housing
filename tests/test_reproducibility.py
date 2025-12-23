"""Тесты воспроизводимости ML-пайплайна.

Проверяет:
- Фиксацию random_state для всех моделей
- Идентичность результатов при повторном запуске
- Сохранение/загрузку моделей
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════════
# Фикстуры
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def synthetic_data():
    """Синтетические данные для тестирования."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        noise=0.1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def boston_data():
    """Данные Boston Housing (если доступны)."""
    data_path = Path("data/raw/housing.csv")
    if not data_path.exists():
        pytest.skip("Boston Housing data not available")

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

    X = df.drop("MEDV", axis=1).values
    y = df["MEDV"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты детерминизма
# ═══════════════════════════════════════════════════════════════════════════════


class TestRandomStateFixation:
    """Тесты фиксации random_state."""

    def test_random_forest_determinism(self, synthetic_data):
        """Тест детерминизма Random Forest."""
        from sklearn.ensemble import RandomForestRegressor

        X_train, X_test, y_train, _ = synthetic_data

        # Обучаем две модели с одинаковым random_state
        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred1,
            pred2,
            err_msg="Random Forest predictions should be identical with same random_state",
        )

    def test_gradient_boosting_determinism(self, synthetic_data):
        """Тест детерминизма Gradient Boosting."""
        from sklearn.ensemble import GradientBoostingRegressor

        X_train, X_test, y_train, _ = synthetic_data

        model1 = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model2 = GradientBoostingRegressor(n_estimators=10, random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred1,
            pred2,
            err_msg="Gradient Boosting predictions should be identical with same random_state",
        )

    def test_ridge_determinism(self, synthetic_data):
        """Тест детерминизма Ridge."""
        from sklearn.linear_model import Ridge

        X_train, X_test, y_train, _ = synthetic_data

        model1 = Ridge(alpha=1.0, random_state=42)
        model2 = Ridge(alpha=1.0, random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(
            pred1, pred2, err_msg="Ridge predictions should be identical"
        )

    def test_train_test_split_determinism(self, synthetic_data):
        """Тест детерминизма train_test_split."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)


class TestResultsReproducibility:
    """Тесты воспроизводимости результатов."""

    def test_metrics_reproducibility(self, synthetic_data):
        """Тест воспроизводимости метрик."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        X_train, X_test, y_train, y_test = synthetic_data

        metrics_runs = []

        for _ in range(3):
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics_runs.append(
                {
                    "r2": r2_score(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                }
            )

        # Все метрики должны быть идентичны
        for key in ["r2", "rmse"]:
            values = [m[key] for m in metrics_runs]
            assert all(v == values[0] for v in values), (
                f"Metrics {key} should be identical across runs"
            )

    def test_feature_importance_reproducibility(self, synthetic_data):
        """Тест воспроизводимости feature importance."""
        from sklearn.ensemble import RandomForestRegressor

        X_train, _, y_train, _ = synthetic_data

        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        np.testing.assert_array_almost_equal(
            model1.feature_importances_,
            model2.feature_importances_,
            err_msg="Feature importances should be identical",
        )


class TestModelPersistence:
    """Тесты сохранения и загрузки моделей."""

    def test_pickle_save_load(self, synthetic_data):
        """Тест сохранения/загрузки модели через pickle."""
        from sklearn.ensemble import RandomForestRegressor

        X_train, X_test, y_train, _ = synthetic_data

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        original_predictions = model.predict(X_test)

        # Сохраняем и загружаем
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name

        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        loaded_predictions = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(
            original_predictions,
            loaded_predictions,
            err_msg="Loaded model should produce same predictions",
        )

        # Очистка
        Path(model_path).unlink()

    def test_model_params_preserved(self, synthetic_data):
        """Тест сохранения параметров модели."""
        from sklearn.ensemble import RandomForestRegressor

        X_train, _, y_train, _ = synthetic_data

        original_params = {
            "n_estimators": 50,
            "max_depth": 10,
            "random_state": 42,
        }

        model = RandomForestRegressor(**original_params)
        model.fit(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name

        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        for param, value in original_params.items():
            assert getattr(loaded_model, param) == value, (
                f"Parameter {param} should be preserved"
            )

        Path(model_path).unlink()


class TestDataReproducibility:
    """Тесты воспроизводимости обработки данных."""

    def test_data_loading_consistency(self, boston_data):
        """Тест консистентности загрузки данных."""
        X_train1, X_test1, y_train1, y_test1 = boston_data

        # Загружаем данные повторно
        data_path = Path("data/raw/housing.csv")
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

        X = df.drop("MEDV", axis=1).values
        y = df["MEDV"].values

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_feature_scaling_reproducibility(self, synthetic_data):
        """Тест воспроизводимости масштабирования признаков."""
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, _, _ = synthetic_data

        scaler1 = StandardScaler()
        scaler2 = StandardScaler()

        X_train_scaled1 = scaler1.fit_transform(X_train)
        X_train_scaled2 = scaler2.fit_transform(X_train)

        np.testing.assert_array_almost_equal(
            X_train_scaled1, X_train_scaled2, err_msg="Scaled data should be identical"
        )


class TestEndToEndReproducibility:
    """End-to-end тесты воспроизводимости."""

    def test_full_pipeline_reproducibility(self, synthetic_data):
        """Тест полной воспроизводимости пайплайна."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = synthetic_data

        results = []

        for run in range(3):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            results.append(
                {
                    "run": run,
                    "r2": r2_score(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "predictions": y_pred.copy(),
                }
            )

        # Проверяем идентичность метрик
        r2_values = [r["r2"] for r in results]
        rmse_values = [r["rmse"] for r in results]
        mae_values = [r["mae"] for r in results]

        assert len(set(r2_values)) == 1, "R² should be identical across runs"
        assert len(set(rmse_values)) == 1, "RMSE should be identical across runs"
        assert len(set(mae_values)) == 1, "MAE should be identical across runs"

        # Проверяем идентичность предсказаний
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0]["predictions"],
                results[i]["predictions"],
                err_msg=f"Predictions should be identical between run 0 and run {i}",
            )

    def test_cross_validation_reproducibility(self, synthetic_data):
        """Тест воспроизводимости кросс-валидации."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        X_train, _, y_train, _ = synthetic_data

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        scores1 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        scores2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        np.testing.assert_array_almost_equal(
            scores1, scores2, err_msg="Cross-validation scores should be reproducible"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты с использованием model_loader (если доступен)
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelLoaderReproducibility:
    """Тесты воспроизводимости с model_loader."""

    @pytest.fixture
    def model_loader(self):
        """Загрузка model_loader."""
        try:
            from src.ml_models.model_loader import create_model

            return create_model
        except ImportError:
            pytest.skip("model_loader not available")

    def test_all_models_have_random_state(self, model_loader, synthetic_data):
        """Тест наличия random_state у всех моделей."""
        X_train, X_test, y_train, _ = synthetic_data

        model_names = [
            "random_forest",
            "gradient_boosting",
            "ridge",
            "lasso",
            "extra_trees",
            "decision_tree",
        ]

        for model_name in model_names:
            try:
                model1 = model_loader(model_name, custom_params={"random_state": 42})
                model2 = model_loader(model_name, custom_params={"random_state": 42})

                model1.fit(X_train, y_train)
                model2.fit(X_train, y_train)

                pred1 = model1.predict(X_test)
                pred2 = model2.predict(X_test)

                np.testing.assert_array_almost_equal(
                    pred1,
                    pred2,
                    err_msg=f"Model {model_name} should be reproducible with random_state=42",
                )
            except Exception as e:
                pytest.fail(f"Model {model_name} failed reproducibility test: {e}")
