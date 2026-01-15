"""Мониторинг выполнения ML-пайплайнов."""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class StageMetrics:
    """Метрики выполнения этапа пайплайна."""

    stage_name: str
    start_time: float
    end_time: float | None = None
    status: str = "running"
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Время выполнения в секундах."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "stage_name": self.stage_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time
            else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class PipelineRun:
    """Информация о запуске пайплайна."""

    run_id: str
    pipeline_name: str
    start_time: float
    stages: list[StageMetrics] = field(default_factory=list)
    status: str = "running"
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Общее время выполнения."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat()
            if self.end_time
            else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata,
        }


class PipelineMonitor:
    """
    Мониторинг выполнения ML-пайплайнов.

    Отслеживает:
    - Состояние всех этапов пайплайна
    - Время выполнения каждого этапа
    - Метрики и ошибки
    - Историю запусков
    """

    def __init__(
        self,
        pipeline_name: str,
        history_dir: Path | str | None = None,
        max_history: int = 100,
    ):
        """
        Инициализация монитора.

        Args:
            pipeline_name: Название пайплайна
            history_dir: Директория для истории запусков
            max_history: Максимальное количество записей истории
        """
        self.pipeline_name = pipeline_name
        self.history_dir = Path(
            history_dir or os.environ.get("MONITOR_HISTORY_DIR", "logs/pipeline_runs")
        )
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history

        self.current_run: PipelineRun | None = None
        self.current_stage: StageMetrics | None = None

    def start_run(
        self,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """
        Начало нового запуска пайплайна.

        Args:
            run_id: ID запуска (генерируется автоматически если не указан)
            metadata: Метаданные запуска

        Returns:
            PipelineRun
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_run = PipelineRun(
            run_id=run_id,
            pipeline_name=self.pipeline_name,
            start_time=time.time(),
            metadata=metadata or {},
        )

        logger.info(f"🚀 Pipeline started: {self.pipeline_name} (run_id: {run_id})")
        return self.current_run

    def start_stage(self, stage_name: str) -> StageMetrics:
        """
        Начало нового этапа.

        Args:
            stage_name: Название этапа

        Returns:
            StageMetrics
        """
        if self.current_run is None:
            raise RuntimeError("Pipeline run not started. Call start_run() first.")

        self.current_stage = StageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
        )

        logger.info(f"  ▶️ Stage started: {stage_name}")
        return self.current_stage

    def end_stage(
        self,
        success: bool = True,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> StageMetrics | None:
        """
        Завершение текущего этапа.

        Args:
            success: Успешность выполнения
            metrics: Метрики этапа
            error: Сообщение об ошибке

        Returns:
            StageMetrics
        """
        if self.current_stage is None:
            logger.warning("No active stage to end")
            return None

        self.current_stage.end_time = time.time()
        self.current_stage.status = "success" if success else "failed"
        self.current_stage.metrics = metrics or {}
        self.current_stage.error = error

        if self.current_run:
            self.current_run.stages.append(self.current_stage)

        status_icon = "✅" if success else "❌"
        logger.info(
            f"  {status_icon} Stage ended: {self.current_stage.stage_name} "
            f"({self.current_stage.duration_seconds:.2f}s)"
        )

        if metrics:
            for name, value in metrics.items():
                logger.info(f"    📊 {name}: {value:.4f}")

        stage = self.current_stage
        self.current_stage = None
        return stage

    def end_run(
        self,
        success: bool = True,
        save_history: bool = True,
    ) -> PipelineRun | None:
        """
        Завершение запуска пайплайна.

        Args:
            success: Успешность выполнения
            save_history: Сохранить в историю

        Returns:
            PipelineRun
        """
        if self.current_run is None:
            logger.warning("No active run to end")
            return None

        self.current_run.end_time = time.time()
        self.current_run.status = "success" if success else "failed"

        status_icon = "✅" if success else "❌"
        logger.info(
            f"{status_icon} Pipeline ended: {self.pipeline_name} "
            f"({self.current_run.duration_seconds:.2f}s)"
        )

        # Сводка по этапам
        successful_stages = sum(
            1 for s in self.current_run.stages if s.status == "success"
        )
        total_stages = len(self.current_run.stages)
        logger.info(f"  📊 Stages: {successful_stages}/{total_stages} successful")

        if save_history:
            self._save_to_history()

        run = self.current_run
        self.current_run = None
        return run

    def _save_to_history(self) -> None:
        """Сохранение запуска в историю."""
        if self.current_run is None:
            return

        # Сохраняем текущий запуск
        run_file = (
            self.history_dir / f"{self.pipeline_name}_{self.current_run.run_id}.json"
        )
        with open(run_file, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2, ensure_ascii=False)

        # Очищаем старые записи
        self._cleanup_history()

    def _cleanup_history(self) -> None:
        """Очистка старых записей истории."""
        history_files = sorted(
            self.history_dir.glob(f"{self.pipeline_name}_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Удаляем старые файлы
        for old_file in history_files[self.max_history :]:
            old_file.unlink()
            logger.debug(f"Deleted old history file: {old_file}")

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Получение истории запусков.

        Args:
            limit: Максимальное количество записей

        Returns:
            Список запусков
        """
        history_files = sorted(
            self.history_dir.glob(f"{self.pipeline_name}_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )[:limit]

        history = []
        for file_path in history_files:
            with open(file_path) as f:
                history.append(json.load(f))

        return history

    def get_statistics(self) -> dict[str, Any]:
        """
        Получение статистики по запускам.

        Returns:
            Словарь со статистикой
        """
        history = self.get_history(limit=self.max_history)

        if not history:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "average_duration": 0,
            }

        successful = sum(1 for h in history if h["status"] == "success")
        durations = [h["duration_seconds"] for h in history if h["duration_seconds"]]

        return {
            "total_runs": len(history),
            "successful_runs": successful,
            "failed_runs": len(history) - successful,
            "success_rate": successful / len(history) * 100,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "last_run": history[0] if history else None,
        }

    def context(
        self,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PipelineMonitorContext":
        """
        Контекстный менеджер для мониторинга пайплайна.

        Args:
            run_id: ID запуска
            metadata: Метаданные

        Returns:
            Контекстный менеджер
        """
        return PipelineMonitorContext(self, run_id, metadata)

    def stage_context(self, stage_name: str) -> "StageMonitorContext":
        """
        Контекстный менеджер для мониторинга этапа.

        Args:
            stage_name: Название этапа

        Returns:
            Контекстный менеджер
        """
        return StageMonitorContext(self, stage_name)


class PipelineMonitorContext:
    """Контекстный менеджер для мониторинга пайплайна."""

    def __init__(
        self,
        monitor: PipelineMonitor,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.monitor = monitor
        self.run_id = run_id
        self.metadata = metadata
        self.success = True

    def __enter__(self) -> PipelineRun:
        return self.monitor.start_run(self.run_id, self.metadata)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.success = False
            logger.exception(f"Pipeline failed with error: {exc_val}")
        self.monitor.end_run(success=self.success)


class StageMonitorContext:
    """Контекстный менеджер для мониторинга этапа."""

    def __init__(self, monitor: PipelineMonitor, stage_name: str):
        self.monitor = monitor
        self.stage_name = stage_name
        self.success = True
        self.metrics: dict[str, float] = {}

    def __enter__(self) -> "StageMonitorContext":
        self.monitor.start_stage(self.stage_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.success = False
            error_msg = str(exc_val)
        else:
            error_msg = None

        self.monitor.end_stage(
            success=self.success,
            metrics=self.metrics,
            error=error_msg,
        )

    def log_metric(self, name: str, value: float) -> None:
        """Добавление метрики."""
        self.metrics[name] = value
