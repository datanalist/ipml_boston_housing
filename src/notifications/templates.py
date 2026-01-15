"""Шаблоны уведомлений о результатах ML-пайплайнов."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NotificationTemplate(ABC):
    """Базовый класс для шаблонов уведомлений."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "boston_housing_pipeline"

    @abstractmethod
    def render_text(self) -> str:
        """Рендеринг в текстовый формат."""
        ...

    @abstractmethod
    def render_json(self) -> dict[str, Any]:
        """Рендеринг в JSON формат."""
        ...

    def render_markdown(self) -> str:
        """Рендеринг в Markdown формат."""
        return self.render_text()


@dataclass
class SuccessTemplate(NotificationTemplate):
    """Шаблон успешного завершения пайплайна."""

    pipeline_name: str = ""
    run_id: str = ""
    duration_seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    best_model: str | None = None
    artifacts: list[str] = field(default_factory=list)
    stages_completed: int = 0
    stages_total: int = 0

    def render_text(self) -> str:
        """Рендеринг в текстовый формат."""
        lines = [
            "=" * 60,
            "✅ PIPELINE COMPLETED SUCCESSFULLY",
            "=" * 60,
            "",
            f"Pipeline: {self.pipeline_name}",
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Stages: {self.stages_completed}/{self.stages_total}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        if self.metrics:
            lines.append("📊 Metrics:")
            for name, value in self.metrics.items():
                lines.append(f"   • {name}: {value:.4f}")
            lines.append("")

        if self.best_model:
            lines.append(f"🏆 Best Model: {self.best_model}")
            lines.append("")

        if self.artifacts:
            lines.append("📦 Artifacts:")
            for artifact in self.artifacts:
                lines.append(f"   • {artifact}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_json(self) -> dict[str, Any]:
        """Рендеринг в JSON формат."""
        return {
            "status": "success",
            "notification_type": "pipeline_complete",
            "timestamp": self.timestamp,
            "source": self.source,
            "pipeline": {
                "name": self.pipeline_name,
                "run_id": self.run_id,
                "duration_seconds": self.duration_seconds,
                "stages_completed": self.stages_completed,
                "stages_total": self.stages_total,
            },
            "metrics": self.metrics,
            "best_model": self.best_model,
            "artifacts": self.artifacts,
        }

    def render_markdown(self) -> str:
        """Рендеринг в Markdown формат."""
        lines = [
            "# ✅ Pipeline Completed Successfully",
            "",
            "## Overview",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Pipeline | {self.pipeline_name} |",
            f"| Run ID | `{self.run_id}` |",
            f"| Duration | {self.duration_seconds:.2f}s |",
            f"| Stages | {self.stages_completed}/{self.stages_total} |",
            f"| Timestamp | {self.timestamp} |",
            "",
        ]

        if self.metrics:
            lines.extend(
                [
                    "## 📊 Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )
            for name, value in self.metrics.items():
                lines.append(f"| {name} | {value:.4f} |")
            lines.append("")

        if self.best_model:
            lines.extend(
                [
                    "## 🏆 Best Model",
                    "",
                    f"**{self.best_model}**",
                    "",
                ]
            )

        if self.artifacts:
            lines.extend(
                [
                    "## 📦 Artifacts",
                    "",
                ]
            )
            for artifact in self.artifacts:
                lines.append(f"- `{artifact}`")
            lines.append("")

        return "\n".join(lines)


@dataclass
class ErrorTemplate(NotificationTemplate):
    """Шаблон ошибки пайплайна."""

    pipeline_name: str = ""
    run_id: str = ""
    stage_name: str = ""
    error_type: str = ""
    error_message: str = ""
    traceback: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def render_text(self) -> str:
        """Рендеринг в текстовый формат."""
        lines = [
            "=" * 60,
            "❌ PIPELINE FAILED",
            "=" * 60,
            "",
            f"Pipeline: {self.pipeline_name}",
            f"Run ID: {self.run_id}",
            f"Failed Stage: {self.stage_name}",
            f"Timestamp: {self.timestamp}",
            "",
            "⚠️ Error:",
            f"   Type: {self.error_type}",
            f"   Message: {self.error_message}",
            "",
        ]

        if self.traceback:
            lines.extend(
                [
                    "📋 Traceback:",
                    self.traceback,
                    "",
                ]
            )

        if self.context:
            lines.append("📝 Context:")
            for key, value in self.context.items():
                lines.append(f"   • {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_json(self) -> dict[str, Any]:
        """Рендеринг в JSON формат."""
        return {
            "status": "error",
            "notification_type": "pipeline_error",
            "timestamp": self.timestamp,
            "source": self.source,
            "pipeline": {
                "name": self.pipeline_name,
                "run_id": self.run_id,
                "failed_stage": self.stage_name,
            },
            "error": {
                "type": self.error_type,
                "message": self.error_message,
                "traceback": self.traceback,
            },
            "context": self.context,
        }

    def render_markdown(self) -> str:
        """Рендеринг в Markdown формат."""
        lines = [
            "# ❌ Pipeline Failed",
            "",
            "## Overview",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Pipeline | {self.pipeline_name} |",
            f"| Run ID | `{self.run_id}` |",
            f"| Failed Stage | {self.stage_name} |",
            f"| Timestamp | {self.timestamp} |",
            "",
            "## ⚠️ Error",
            "",
            f"**Type:** `{self.error_type}`",
            "",
            f"**Message:** {self.error_message}",
            "",
        ]

        if self.traceback:
            lines.extend(
                [
                    "## 📋 Traceback",
                    "",
                    "```",
                    self.traceback,
                    "```",
                    "",
                ]
            )

        if self.context:
            lines.extend(
                [
                    "## 📝 Context",
                    "",
                ]
            )
            for key, value in self.context.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class ExperimentSummaryTemplate(NotificationTemplate):
    """Шаблон сводки экспериментов."""

    experiment_name: str = ""
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    best_model: dict[str, Any] = field(default_factory=dict)
    all_results: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def render_text(self) -> str:
        """Рендеринг в текстовый формат."""
        lines = [
            "=" * 60,
            "📊 EXPERIMENT SUMMARY",
            "=" * 60,
            "",
            f"Experiment: {self.experiment_name}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Timestamp: {self.timestamp}",
            "",
            "📈 Results:",
            f"   • Total: {self.total_experiments}",
            f"   • Successful: {self.successful_experiments}",
            f"   • Failed: {self.failed_experiments}",
            "",
        ]

        if self.best_model:
            lines.extend(
                [
                    "🏆 Best Model:",
                    f"   • Name: {self.best_model.get('name', 'N/A')}",
                    f"   • R² Score: {self.best_model.get('r2_score', 0):.4f}",
                    f"   • RMSE: {self.best_model.get('rmse', 0):.4f}",
                    f"   • MAE: {self.best_model.get('mae', 0):.4f}",
                    "",
                ]
            )

        if self.all_results:
            lines.append("📋 Top 5 Models:")
            for i, result in enumerate(self.all_results[:5], 1):
                name = result.get("name", result.get("run_id", "Unknown"))
                r2 = result.get("r2_score", 0)
                lines.append(f"   {i}. {name}: R²={r2:.4f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_json(self) -> dict[str, Any]:
        """Рендеринг в JSON формат."""
        return {
            "status": "complete",
            "notification_type": "experiment_summary",
            "timestamp": self.timestamp,
            "source": self.source,
            "experiment": {
                "name": self.experiment_name,
                "duration_seconds": self.duration_seconds,
            },
            "summary": {
                "total": self.total_experiments,
                "successful": self.successful_experiments,
                "failed": self.failed_experiments,
                "success_rate": self.successful_experiments
                / max(self.total_experiments, 1)
                * 100,
            },
            "best_model": self.best_model,
            "all_results": self.all_results,
        }

    def render_markdown(self) -> str:
        """Рендеринг в Markdown формат."""
        success_rate = (
            self.successful_experiments / max(self.total_experiments, 1) * 100
        )

        lines = [
            "# 📊 Experiment Summary",
            "",
            "## Overview",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Experiment | {self.experiment_name} |",
            f"| Duration | {self.duration_seconds:.2f}s |",
            f"| Total | {self.total_experiments} |",
            f"| Successful | {self.successful_experiments} |",
            f"| Failed | {self.failed_experiments} |",
            f"| Success Rate | {success_rate:.1f}% |",
            "",
        ]

        if self.best_model:
            lines.extend(
                [
                    "## 🏆 Best Model",
                    "",
                    f"**{self.best_model.get('name', 'N/A')}**",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| R² Score | {self.best_model.get('r2_score', 0):.4f} |",
                    f"| RMSE | {self.best_model.get('rmse', 0):.4f} |",
                    f"| MAE | {self.best_model.get('mae', 0):.4f} |",
                    "",
                ]
            )

        if self.all_results:
            lines.extend(
                [
                    "## 📋 All Results",
                    "",
                    "| Rank | Model | R² Score | RMSE | MAE |",
                    "|------|-------|----------|------|-----|",
                ]
            )
            for i, result in enumerate(self.all_results, 1):
                name = result.get("name", result.get("run_id", "Unknown"))[:25]
                r2 = result.get("r2_score", 0)
                rmse = result.get("rmse", 0)
                mae = result.get("mae", 0)
                lines.append(f"| {i} | {name} | {r2:.4f} | {rmse:.4f} | {mae:.4f} |")
            lines.append("")

        return "\n".join(lines)
