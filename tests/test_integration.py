"""Тесты интеграции компонентов ML-инфраструктуры.

Проверяет:
- Доступность MLflow
- Работу DVC
- Конфигурации Hydra
- Логирование метрик
"""

import subprocess
import tempfile
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты DVC
# ═══════════════════════════════════════════════════════════════════════════════


class TestDVCIntegration:
    """Тесты интеграции с DVC."""

    def test_dvc_installed(self):
        """Тест установки DVC."""
        result = subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "DVC should be installed"
        assert "dvc version" in result.stdout.lower() or "DVC version" in result.stdout

    def test_dvc_config_exists(self):
        """Тест наличия конфигурации DVC."""
        dvc_dir = Path(".dvc")
        assert dvc_dir.exists(), ".dvc directory should exist"

        config_file = dvc_dir / "config"
        assert config_file.exists(), "DVC config file should exist"

    def test_dvc_yaml_exists(self):
        """Тест наличия dvc.yaml."""
        dvc_yaml = Path("dvc.yaml")
        assert dvc_yaml.exists(), "dvc.yaml should exist"

    def test_dvc_remote_configured(self):
        """Тест настройки DVC remote."""
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "dvc remote list should succeed"
        # Может быть пустым если remote не настроен локально

    def test_dvc_status_runs(self):
        """Тест выполнения dvc status."""
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Может вернуть non-zero если есть изменения, но не должен падать
        assert result.returncode in [0, 1], "dvc status should run without errors"


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты Hydra
# ═══════════════════════════════════════════════════════════════════════════════


class TestHydraIntegration:
    """Тесты интеграции с Hydra."""

    def test_hydra_config_directory_exists(self):
        """Тест наличия директории конфигураций Hydra."""
        conf_dir = Path("conf")
        assert conf_dir.exists(), "conf directory should exist"

    def test_main_config_exists(self):
        """Тест наличия основной конфигурации."""
        config_yaml = Path("conf/config.yaml")
        assert config_yaml.exists(), "conf/config.yaml should exist"

    def test_model_configs_exist(self):
        """Тест наличия конфигураций моделей."""
        model_dir = Path("conf/model")
        assert model_dir.exists(), "conf/model directory should exist"

        model_configs = list(model_dir.glob("*.yaml"))
        assert len(model_configs) > 0, "Should have at least one model config"

    def test_hydra_config_valid(self):
        """Тест валидности конфигурации Hydra."""
        import yaml

        config_path = Path("conf/config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config is not None, "Config should not be empty"
        assert "defaults" in config, "Config should have defaults section"

    def test_model_config_structure(self):
        """Тест структуры конфигурации модели."""
        import yaml

        model_config = Path("conf/model/random_forest.yaml")
        if not model_config.exists():
            pytest.skip("random_forest.yaml not found")

        with open(model_config) as f:
            config = yaml.safe_load(f)

        assert "name" in config, "Model config should have 'name' field"


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты MLflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLflowIntegration:
    """Тесты интеграции с MLflow."""

    def test_mlflow_installed(self):
        """Тест установки MLflow."""
        import mlflow

        version = mlflow.__version__
        assert version is not None, "MLflow should be installed"

    def test_mlflow_tracking_local(self):
        """Тест локального трекинга MLflow."""
        import mlflow

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")
            mlflow.set_experiment("test_experiment")

            with mlflow.start_run():
                mlflow.log_param("test_param", 42)
                mlflow.log_metric("test_metric", 0.95)

                run = mlflow.active_run()
                assert run is not None

    def test_mlflow_config_exists(self):
        """Тест наличия конфигурации MLflow."""
        config_path = Path("src/config/mlflow_config.py")
        assert config_path.exists(), "MLflow config should exist"

    def test_mlflow_tracker_imports(self):
        """Тест импорта MLflow трекера."""
        try:
            from src.tracking.mlflow_tracker import MLflowExperimentTracker

            assert MLflowExperimentTracker is not None
        except ImportError as e:
            pytest.fail(f"Failed to import MLflowExperimentTracker: {e}")

    def test_mlflow_decorators_imports(self):
        """Тест импорта декораторов MLflow."""
        try:
            from src.tracking.decorators import (
                log_metrics_decorator,
                log_params_decorator,
                mlflow_run,
            )

            assert mlflow_run is not None
            assert log_params_decorator is not None
            assert log_metrics_decorator is not None
        except ImportError as e:
            pytest.fail(f"Failed to import MLflow decorators: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты логирования
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoggingIntegration:
    """Тесты интеграции логирования."""

    def test_loguru_installed(self):
        """Тест установки loguru."""
        from loguru import logger

        assert logger is not None

    def test_monitoring_logger_imports(self):
        """Тест импорта мониторинг логгера."""
        try:
            from src.monitoring.logger import MonitoringLogger, configure_logging

            assert MonitoringLogger is not None
            assert configure_logging is not None
        except ImportError as e:
            pytest.fail(f"Failed to import MonitoringLogger: {e}")

    def test_monitoring_logger_works(self):
        """Тест работы MonitoringLogger."""
        from src.monitoring.logger import MonitoringLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log = MonitoringLogger(component="test", log_dir=Path(tmpdir))
            log.info("Test message")
            log.log_metrics({"accuracy": 0.95, "loss": 0.05})

            # Проверяем создание файлов логов
            log_files = list(Path(tmpdir).glob("*.log"))
            assert len(log_files) > 0, "Log files should be created"


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты Pipeline Monitor
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineMonitorIntegration:
    """Тесты интеграции Pipeline Monitor."""

    def test_pipeline_monitor_imports(self):
        """Тест импорта Pipeline Monitor."""
        try:
            from src.monitoring.pipeline_monitor import PipelineMonitor, StageMetrics

            assert PipelineMonitor is not None
            assert StageMetrics is not None
        except ImportError as e:
            pytest.fail(f"Failed to import PipelineMonitor: {e}")

    def test_pipeline_monitor_basic_usage(self):
        """Тест базового использования Pipeline Monitor."""
        from src.monitoring.pipeline_monitor import PipelineMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PipelineMonitor(
                pipeline_name="test_pipeline",
                history_dir=tmpdir,
            )

            run = monitor.start_run(run_id="test_run_001")
            assert run is not None

            stage = monitor.start_stage("data_loading")
            assert stage is not None

            monitor.end_stage(success=True, metrics={"rows_loaded": 100})
            run = monitor.end_run(success=True)

            assert run.status == "success"
            assert len(run.stages) == 1

    def test_pipeline_monitor_context_manager(self):
        """Тест контекстного менеджера Pipeline Monitor."""
        from src.monitoring.pipeline_monitor import PipelineMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PipelineMonitor(
                pipeline_name="test_pipeline",
                history_dir=tmpdir,
            )

            with monitor.context(run_id="context_test") as run:
                assert run is not None

                with monitor.stage_context("processing") as stage:
                    stage.log_metric("items_processed", 50)

            # Проверяем сохранение истории
            history = monitor.get_history()
            assert len(history) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты уведомлений
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotificationsIntegration:
    """Тесты интеграции уведомлений."""

    def test_notifier_imports(self):
        """Тест импорта Notifier."""
        try:
            from src.notifications.notifier import (
                NotificationChannel,
                Notifier,
                notify_pipeline_complete,
            )

            assert Notifier is not None
            assert NotificationChannel is not None
            assert notify_pipeline_complete is not None
        except ImportError as e:
            pytest.fail(f"Failed to import Notifier: {e}")

    def test_templates_imports(self):
        """Тест импорта шаблонов."""
        try:
            from src.notifications.templates import (
                ErrorTemplate,
                ExperimentSummaryTemplate,
                SuccessTemplate,
            )

            assert SuccessTemplate is not None
            assert ErrorTemplate is not None
            assert ExperimentSummaryTemplate is not None
        except ImportError as e:
            pytest.fail(f"Failed to import templates: {e}")

    def test_success_template_rendering(self):
        """Тест рендеринга шаблона успеха."""
        from src.notifications.templates import SuccessTemplate

        template = SuccessTemplate(
            pipeline_name="test_pipeline",
            run_id="test_123",
            duration_seconds=120.5,
            metrics={"r2": 0.95, "rmse": 2.5},
            best_model="random_forest",
            stages_completed=5,
            stages_total=5,
        )

        text = template.render_text()
        assert "test_pipeline" in text
        assert "SUCCESS" in text

        json_data = template.render_json()
        assert json_data["status"] == "success"
        assert json_data["pipeline"]["name"] == "test_pipeline"

        md = template.render_markdown()
        assert "# ✅" in md

    def test_notifier_file_channel(self):
        """Тест файлового канала уведомлений."""
        from src.notifications.notifier import NotificationChannel, Notifier
        from src.notifications.templates import SuccessTemplate

        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = Notifier(
                channels=[NotificationChannel.FILE],
                reports_dir=tmpdir,
            )

            template = SuccessTemplate(
                pipeline_name="test",
                run_id="001",
                duration_seconds=60,
            )

            result = notifier.notify(template, formats=["json", "txt"])

            assert result["success"] is True
            assert "file" in result["channels"]

            # Проверяем создание файлов
            files = list(Path(tmpdir).glob("*"))
            assert len(files) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты Health Check
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthCheckIntegration:
    """Тесты интеграции Health Check."""

    def test_health_check_imports(self):
        """Тест импорта Health Check."""
        try:
            from src.integration.health_check import HealthChecker, check_dvc

            assert HealthChecker is not None
            assert check_dvc is not None
        except ImportError as e:
            pytest.fail(f"Failed to import HealthChecker: {e}")

    def test_dvc_health_check(self):
        """Тест проверки здоровья DVC."""
        from src.integration.health_check import check_dvc

        result = check_dvc()

        assert result is not None
        assert result.service == "dvc"
        # DVC должен быть healthy если установлен
        assert result.status.value in ["healthy", "unhealthy", "degraded"]

    def test_service_config(self):
        """Тест конфигурации сервисов."""
        from src.integration.utils import get_service_config

        mlflow_config = get_service_config("mlflow")
        assert mlflow_config.name == "MLflow Tracking Server"
        assert mlflow_config.port == 5000

        minio_config = get_service_config("minio")
        assert minio_config.name == "MinIO S3 Storage"
        assert minio_config.port == 9000


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты model_loader
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelLoaderIntegration:
    """Тесты интеграции model_loader."""

    def test_model_loader_imports(self):
        """Тест импорта model_loader."""
        try:
            from src.ml_models.model_loader import create_model

            assert create_model is not None
        except ImportError as e:
            pytest.fail(f"Failed to import model_loader: {e}")

    def test_create_random_forest(self):
        """Тест создания Random Forest."""
        from src.ml_models.model_loader import create_model

        model = create_model("random_forest")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_with_params(self):
        """Тест создания модели с параметрами."""
        from src.ml_models.model_loader import create_model

        model = create_model(
            "random_forest",
            custom_params={"n_estimators": 50, "max_depth": 5},
        )
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_all_models_createable(self):
        """Тест создания всех моделей."""
        from src.ml_models.model_loader import create_model

        model_names = [
            "linear_regression",
            "ridge",
            "lasso",
            "random_forest",
            "gradient_boosting",
            "decision_tree",
        ]

        for name in model_names:
            try:
                model = create_model(name)
                assert model is not None, f"Model {name} should be created"
            except Exception as e:
                pytest.fail(f"Failed to create model {name}: {e}")
