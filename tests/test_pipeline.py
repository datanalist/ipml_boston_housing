"""Тесты пайплайна обучения.

Проверяет:
- Полный цикл обучения
- Воспроизводимость метрик
- Сохранение артефактов
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
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты полного цикла обучения
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingPipeline:
    """Тесты полного цикла обучения."""

    def test_basic_training_pipeline(self, synthetic_data):
        """Тест базового пайплайна обучения."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Создание пайплайна
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        # Обучение
        pipeline.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
        )

        # Предсказание
        y_pred = pipeline.predict(synthetic_data["X_test"])

        # Метрики
        r2 = r2_score(synthetic_data["y_test"], y_pred)
        rmse = np.sqrt(mean_squared_error(synthetic_data["y_test"], y_pred))

        assert r2 > 0.5, "R² should be reasonably good"
        # RMSE зависит от масштаба данных, проверяем что метрика вычислена
        assert rmse > 0, "RMSE should be positive"

    def test_training_with_model_loader(self, synthetic_data):
        """Тест обучения с model_loader."""
        try:
            from src.ml_models.model_loader import create_model
        except ImportError:
            pytest.skip("model_loader not available")

        from sklearn.metrics import r2_score

        model = create_model("random_forest", custom_params={"random_state": 42})
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        y_pred = model.predict(synthetic_data["X_test"])
        r2 = r2_score(synthetic_data["y_test"], y_pred)

        assert r2 > 0.5, "Model should have reasonable performance"

    def test_multiple_models_training(self, synthetic_data):
        """Тест обучения нескольких моделей."""
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        models = {
            "random_forest": RandomForestRegressor(n_estimators=10, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=10, random_state=42
            ),
            "ridge": Ridge(random_state=42),
        }

        results = {}

        for name, model in models.items():
            model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
            y_pred = model.predict(synthetic_data["X_test"])
            r2 = r2_score(synthetic_data["y_test"], y_pred)
            results[name] = r2

        # Все модели должны иметь положительный R²
        for name, r2 in results.items():
            assert r2 > 0, f"Model {name} should have positive R²"

        # Должны быть результаты для всех моделей
        assert len(results) == 3


class TestMetricsCalculation:
    """Тесты расчёта метрик."""

    def test_all_metrics_computed(self, synthetic_data):
        """Тест расчёта всех метрик."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        y_pred = model.predict(synthetic_data["X_test"])

        metrics = {
            "r2_score": r2_score(synthetic_data["y_test"], y_pred),
            "rmse": np.sqrt(mean_squared_error(synthetic_data["y_test"], y_pred)),
            "mae": mean_absolute_error(synthetic_data["y_test"], y_pred),
            "mape": np.mean(
                np.abs((synthetic_data["y_test"] - y_pred) / synthetic_data["y_test"])
            )
            * 100,
        }

        # Проверяем, что все метрики вычислены
        for name, value in metrics.items():
            assert value is not None, f"Metric {name} should be computed"
            assert not np.isnan(value), f"Metric {name} should not be NaN"
            assert not np.isinf(value), f"Metric {name} should not be Inf"

    def test_metrics_in_valid_range(self, synthetic_data):
        """Тест валидности диапазонов метрик."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        y_pred = model.predict(synthetic_data["X_test"])

        r2 = r2_score(synthetic_data["y_test"], y_pred)
        rmse = np.sqrt(mean_squared_error(synthetic_data["y_test"], y_pred))
        mae = mean_absolute_error(synthetic_data["y_test"], y_pred)

        # R² обычно между -1 и 1 (может быть отрицательным для плохих моделей)
        assert -10 < r2 <= 1, "R² should be in reasonable range"

        # RMSE и MAE должны быть положительными
        assert rmse >= 0, "RMSE should be non-negative"
        assert mae >= 0, "MAE should be non-negative"

        # MAE должен быть меньше или равен RMSE
        assert mae <= rmse + 1e-6, "MAE should be <= RMSE"


class TestArtifactSaving:
    """Тесты сохранения артефактов."""

    def test_model_saving(self, synthetic_data, temp_dir):
        """Тест сохранения модели."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        model_path = temp_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        assert model_path.exists(), "Model file should exist"
        assert model_path.stat().st_size > 0, "Model file should not be empty"

    def test_model_loading_and_prediction(self, synthetic_data, temp_dir):
        """Тест загрузки модели и предсказания."""
        from sklearn.ensemble import RandomForestRegressor

        # Обучаем и сохраняем
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        original_preds = model.predict(synthetic_data["X_test"])

        model_path = temp_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Загружаем и предсказываем
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        loaded_preds = loaded_model.predict(synthetic_data["X_test"])

        np.testing.assert_array_almost_equal(
            original_preds,
            loaded_preds,
            err_msg="Predictions should be identical after loading",
        )

    def test_metrics_saving(self, temp_dir):
        """Тест сохранения метрик."""
        import json

        metrics = {
            "r2_score": 0.95,
            "rmse": 2.5,
            "mae": 1.8,
            "mape": 5.2,
        }

        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        assert metrics_path.exists(), "Metrics file should exist"

        # Проверяем чтение
        with open(metrics_path) as f:
            loaded_metrics = json.load(f)

        assert loaded_metrics == metrics, "Metrics should be preserved"

    def test_results_dataframe_saving(self, temp_dir):
        """Тест сохранения результатов в DataFrame."""
        results = [
            {"model": "random_forest", "r2": 0.95, "rmse": 2.5},
            {"model": "gradient_boosting", "r2": 0.92, "rmse": 2.8},
            {"model": "ridge", "r2": 0.88, "rmse": 3.2},
        ]

        df = pd.DataFrame(results)
        csv_path = temp_dir / "results.csv"
        df.to_csv(csv_path, index=False)

        assert csv_path.exists(), "CSV file should exist"

        # Проверяем чтение
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)


class TestPipelineReproducibility:
    """Тесты воспроизводимости пайплайна."""

    def test_pipeline_determinism(self, synthetic_data):
        """Тест детерминизма пайплайна."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        results = []

        for _ in range(3):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
                ]
            )

            pipeline.fit(synthetic_data["X_train"], synthetic_data["y_train"])
            y_pred = pipeline.predict(synthetic_data["X_test"])
            r2 = r2_score(synthetic_data["y_test"], y_pred)

            results.append(
                {
                    "r2": r2,
                    "predictions": y_pred.copy(),
                }
            )

        # Проверяем идентичность R²
        r2_values = [r["r2"] for r in results]
        assert len(set(r2_values)) == 1, "R² should be identical across runs"

        # Проверяем идентичность предсказаний
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0]["predictions"],
                results[i]["predictions"],
            )

    def test_saved_model_reproducibility(self, synthetic_data, temp_dir):
        """Тест воспроизводимости сохранённой модели."""
        from sklearn.ensemble import RandomForestRegressor

        # Первый запуск
        model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        model1.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        path1 = temp_dir / "model1.pkl"
        with open(path1, "wb") as f:
            pickle.dump(model1, f)

        # Второй запуск
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        path2 = temp_dir / "model2.pkl"
        with open(path2, "wb") as f:
            pickle.dump(model2, f)

        # Загрузка и сравнение
        with open(path1, "rb") as f:
            loaded1 = pickle.load(f)
        with open(path2, "rb") as f:
            loaded2 = pickle.load(f)

        preds1 = loaded1.predict(synthetic_data["X_test"])
        preds2 = loaded2.predict(synthetic_data["X_test"])

        np.testing.assert_array_almost_equal(preds1, preds2)


class TestPipelineWithMonitoring:
    """Тесты пайплайна с мониторингом."""

    def test_pipeline_with_monitor(self, synthetic_data, temp_dir):
        """Тест пайплайна с Pipeline Monitor."""
        try:
            from src.monitoring.pipeline_monitor import PipelineMonitor
        except ImportError:
            pytest.skip("PipelineMonitor not available")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        monitor = PipelineMonitor(
            pipeline_name="test_training",
            history_dir=temp_dir,
        )

        with monitor.context(run_id="test_001"):
            # Этап загрузки данных
            with monitor.stage_context("data_loading") as stage:
                X_train = synthetic_data["X_train"]
                y_train = synthetic_data["y_train"]
                stage.log_metric("samples", len(X_train))

            # Этап обучения
            with monitor.stage_context("training") as stage:
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                stage.log_metric("n_estimators", 10)

            # Этап оценки
            with monitor.stage_context("evaluation") as stage:
                y_pred = model.predict(synthetic_data["X_test"])
                r2 = r2_score(synthetic_data["y_test"], y_pred)
                stage.log_metric("r2", r2)

        # Проверяем историю
        history = monitor.get_history()
        assert len(history) > 0, "Should have history records"

        last_run = history[0]
        assert last_run["status"] == "success"
        assert len(last_run["stages"]) == 3

    def test_pipeline_with_notifications(self, synthetic_data, temp_dir):
        """Тест пайплайна с уведомлениями."""
        try:
            from src.notifications.notifier import (
                NotificationChannel,
                notify_pipeline_complete,
            )
        except ImportError:
            pytest.skip("Notifier not available")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        # Обучение
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        y_pred = model.predict(synthetic_data["X_test"])
        r2 = r2_score(synthetic_data["y_test"], y_pred)
        rmse = np.sqrt(mean_squared_error(synthetic_data["y_test"], y_pred))

        # Уведомление
        result = notify_pipeline_complete(
            pipeline_name="test_pipeline",
            run_id="test_001",
            duration_seconds=10.5,
            metrics={"r2": r2, "rmse": rmse},
            best_model="random_forest",
            stages_completed=3,
            stages_total=3,
            channels=[NotificationChannel.FILE],
        )

        assert result["success"] is True


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_empty_predictions(self):
        """Тест обработки пустых данных."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        model.fit(X_train, y_train)

        # Пустой массив вызывает ошибку в sklearn - это ожидаемое поведение
        X_empty = np.array([]).reshape(0, 5)
        with pytest.raises(ValueError):
            model.predict(X_empty)

    def test_single_sample_prediction(self, synthetic_data):
        """Тест предсказания для одного образца."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        # Один образец
        single_sample = synthetic_data["X_test"][:1]
        pred = model.predict(single_sample)

        assert len(pred) == 1, "Single sample should produce single prediction"
        assert not np.isnan(pred[0]), "Prediction should not be NaN"

    def test_constant_target(self):
        """Тест с константной целевой переменной."""
        from sklearn.ensemble import RandomForestRegressor

        X = np.random.randn(100, 5)
        y = np.ones(100)  # Константа

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Для константы предсказания должны быть близки к этой константе
        assert np.allclose(y_pred, 1.0, atol=0.1), "Should predict close to constant"
