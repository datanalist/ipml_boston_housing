"""
Automatic Experiment Report Generator.

Генерирует отчеты об экспериментах в формате Markdown с визуализациями.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Настройка стиля визуализаций
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class ExperimentReportGenerator:
    """Генератор отчетов об экспериментах."""

    def __init__(
        self,
        experiments_dir: str | Path = "data/experiments",
        reports_dir: str | Path = "docs/reports/generated",
        plots_dir: str | Path = "docs/reports/plots",
    ) -> None:
        """
        Инициализация генератора.

        Args:
            experiments_dir: Каталог с данными экспериментов.
            reports_dir: Каталог для сохранения отчетов.
            plots_dir: Каталог для сохранения графиков.
        """
        self.experiments_dir = Path(experiments_dir)
        self.reports_dir = Path(reports_dir)
        self.plots_dir = Path(plots_dir)

        # Создаем каталоги
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_experiments_data(self) -> pd.DataFrame | None:
        """
        Загрузка данных экспериментов.

        Returns:
            DataFrame с данными экспериментов или None.
        """
        results_file = self.experiments_dir / "results_summary.csv"

        if not results_file.exists():
            print(f"Warning: {results_file} not found")
            return None

        df = pd.read_csv(results_file)
        return df

    def load_dvclive_metrics(self) -> dict[str, Any] | None:
        """
        Загрузка метрик из DVCLive.

        Returns:
            Словарь с метриками или None.
        """
        metrics_file = Path("dvclive/metrics.json")

        if not metrics_file.exists():
            return None

        with open(metrics_file) as f:
            metrics = json.load(f)

        return metrics

    def create_metrics_comparison_plot(
        self, df: pd.DataFrame, output_file: str = "metrics_comparison.png"
    ) -> Path:
        """
        Создание графика сравнения метрик.

        Args:
            df: DataFrame с данными экспериментов.
            output_file: Имя файла для сохранения.

        Returns:
            Путь к созданному графику.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Сравнение метрик моделей", fontsize=16, fontweight="bold")

        # Поддержка разных форматов колонок
        rmse_col = "test_rmse" if "test_rmse" in df.columns else "rmse"
        model_col = "model" if "model" in df.columns else "run_name"
        r2_col = "test_r2" if "test_r2" in df.columns else "r2_score"
        mae_col = "test_mae" if "test_mae" in df.columns else "mae"

        # RMSE
        df_sorted = df.sort_values(rmse_col)
        axes[0, 0].barh(df_sorted[model_col], df_sorted[rmse_col], color="skyblue")
        axes[0, 0].set_xlabel("RMSE")
        axes[0, 0].set_title("Root Mean Squared Error (меньше - лучше)")
        axes[0, 0].grid(axis="x", alpha=0.3)

        # R² Score
        df_sorted = df.sort_values(r2_col, ascending=False)
        axes[0, 1].barh(df_sorted[model_col], df_sorted[r2_col], color="lightgreen")
        axes[0, 1].set_xlabel("R² Score")
        axes[0, 1].set_title("R² Score (больше - лучше)")
        axes[0, 1].grid(axis="x", alpha=0.3)

        # MAE
        df_sorted = df.sort_values(mae_col)
        axes[1, 0].barh(df_sorted[model_col], df_sorted[mae_col], color="lightcoral")
        axes[1, 0].set_xlabel("MAE")
        axes[1, 0].set_title("Mean Absolute Error (меньше - лучше)")
        axes[1, 0].grid(axis="x", alpha=0.3)

        # Training Time
        time_col = "training_time" if "training_time" in df.columns else "train_time"
        if time_col in df.columns:
            df_sorted = df.sort_values(time_col)
            axes[1, 1].barh(df_sorted[model_col], df_sorted[time_col], color="plum")
            axes[1, 1].set_xlabel("Время (сек)")
            axes[1, 1].set_title("Время обучения")
            axes[1, 1].grid(axis="x", alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Данные о времени\nобучения\nотсутствуют",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1, 1].axis("off")

        plt.tight_layout()

        output_path = self.plots_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_scatter_plot(
        self, df: pd.DataFrame, output_file: str = "rmse_vs_r2.png"
    ) -> Path:
        """
        Создание scatter plot RMSE vs R².

        Args:
            df: DataFrame с данными экспериментов.
            output_file: Имя файла для сохранения.

        Returns:
            Путь к созданному графику.
        """
        plt.figure(figsize=(12, 8))

        # Поддержка разных форматов колонок
        rmse_col = "test_rmse" if "test_rmse" in df.columns else "rmse"
        model_col = "model" if "model" in df.columns else "run_name"
        r2_col = "test_r2" if "test_r2" in df.columns else "r2_score"

        # Определение категорий моделей
        linear_models = ["linear_regression", "ridge", "lasso", "elastic_net", "huber"]
        tree_models = [
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
        ]

        colors = []
        for model in df[model_col]:
            if any(lm in model.lower() for lm in linear_models):
                colors.append("blue")
            elif any(tm in model.lower() for tm in tree_models):
                colors.append("green")
            else:
                colors.append("orange")

        plt.scatter(
            df[rmse_col],
            df[r2_col],
            c=colors,
            s=200,
            alpha=0.6,
            edgecolors="black",
        )

        # Подписи точек
        for idx, row in df.iterrows():
            plt.annotate(
                row[model_col],
                (row[rmse_col], row[r2_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        plt.xlabel("RMSE (меньше - лучше)", fontsize=12)
        plt.ylabel("R² Score (больше - лучше)", fontsize=12)
        plt.title("Сравнение моделей: RMSE vs R²", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        # Легенда
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", label="Линейные модели"),
            Patch(facecolor="green", label="Древовидные модели"),
            Patch(facecolor="orange", label="Другие модели"),
        ]
        plt.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        output_path = self.plots_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_comparison_table(self, df: pd.DataFrame) -> str:
        """
        Генерация Markdown таблицы сравнения.

        Args:
            df: DataFrame с данными экспериментов.

        Returns:
            Markdown строка с таблицей.
        """
        # Поддержка разных форматов колонок
        rmse_col = "test_rmse" if "test_rmse" in df.columns else "rmse"
        model_col = "model" if "model" in df.columns else "run_name"
        r2_col = "test_r2" if "test_r2" in df.columns else "r2_score"
        mae_col = "test_mae" if "test_mae" in df.columns else "mae"
        mape_col = "test_mape" if "test_mape" in df.columns else "mape"

        # Сортируем по R² (по убыванию)
        df_sorted = df.sort_values(r2_col, ascending=False)

        # Создаем Markdown таблицу
        table = "| Модель | RMSE ↓ | R² ↑ | MAE ↓ | MAPE (%) ↓ |\n"
        table += "|--------|--------|------|-------|------------|\n"

        for _, row in df_sorted.iterrows():
            model_name = row[model_col]
            rmse = row[rmse_col]
            r2 = row[r2_col]
            mae = row[mae_col]
            mape = row.get(mape_col, 0) * 100 if mape_col in row else 0

            # Выделяем лучшие значения
            rmse_str = (
                f"**{rmse:.3f}**" if rmse == df[rmse_col].min() else f"{rmse:.3f}"
            )
            r2_str = f"**{r2:.4f}**" if r2 == df[r2_col].max() else f"{r2:.4f}"
            mae_str = f"**{mae:.3f}**" if mae == df[mae_col].min() else f"{mae:.3f}"
            mape_str = f"{mape:.2f}"

            table += (
                f"| {model_name} | {rmse_str} | {r2_str} | {mae_str} | {mape_str} |\n"
            )

        return table

    def generate_report(
        self, title: str = "Отчет об экспериментах", output_file: str | None = None
    ) -> Path:
        """
        Генерация полного отчета.

        Args:
            title: Заголовок отчета.
            output_file: Имя файла для сохранения (по умолчанию: experiment_report_YYYY-MM-DD.md).

        Returns:
            Путь к созданному отчету.
        """
        # Загрузка данных
        df = self.load_experiments_data()

        if df is None or df.empty:
            print("No experiments data found. Skipping report generation.")
            return None

        # Создание графиков
        print("Generating plots...")
        metrics_plot = self.create_metrics_comparison_plot(df)
        scatter_plot = self.create_scatter_plot(df)

        # Создание таблицы сравнения
        comparison_table = self.generate_comparison_table(df)

        # Поддержка разных форматов колонок
        rmse_col = "test_rmse" if "test_rmse" in df.columns else "rmse"
        model_col = "model" if "model" in df.columns else "run_name"
        r2_col = "test_r2" if "test_r2" in df.columns else "r2_score"
        mae_col = "test_mae" if "test_mae" in df.columns else "mae"

        # Статистика
        best_model = df.loc[df[r2_col].idxmax()]
        worst_model = df.loc[df[r2_col].idxmin()]

        # Шаблон отчета
        template_str = """# {{ title }}

**Дата создания:** {{ date }}

---

## 📊 Общая статистика

- **Всего экспериментов:** {{ total_experiments }}
- **Лучшая модель:** `{{ best_model[model_col] }}` (R² = {{ "%.4f"|format(best_model[r2_col]) }})
- **Худшая модель:** `{{ worst_model[model_col] }}` (R² = {{ "%.4f"|format(worst_model[r2_col]) }})

### Средние метрики

| Метрика | Значение |
|---------|----------|
| **Средний RMSE** | {{ "%.3f"|format(avg_rmse) }} |
| **Средний R²** | {{ "%.4f"|format(avg_r2) }} |
| **Средний MAE** | {{ "%.3f"|format(avg_mae) }} |

---

## 🏆 Лучшая модель: {{ best_model[model_col] }}

| Метрика | Значение |
|---------|----------|
| **RMSE** | {{ "%.3f"|format(best_model[rmse_col]) }} |
| **R²** | {{ "%.4f"|format(best_model[r2_col]) }} |
| **MAE** | {{ "%.3f"|format(best_model[mae_col]) }} |

{% if best_model.get('train_time') %}
**Время обучения:** {{ "%.2f"|format(best_model.train_time) }} сек
{% endif %}

---

## 📈 Визуализации

### Сравнение метрик

![Сравнение метрик](plots/{{ metrics_plot_name }})

### RMSE vs R² Score

![RMSE vs R²](plots/{{ scatter_plot_name }})

---

## 📋 Сравнительная таблица

{{ comparison_table }}

---

## 💡 Выводы

1. **Лучшая модель:** `{{ best_model[model_col] }}` показала наилучший результат с R² = {{ "%.4f"|format(best_model[r2_col]) }}
2. **Разброс результатов:** Разница между лучшей и худшей моделью составляет {{ "%.4f"|format(best_model[r2_col] - worst_model[r2_col]) }} по R²

---

## 🔍 Рекомендации

1. Для production рекомендуется использовать модель `{{ best_model[model_col] }}`
2. Рассмотреть ансамблирование top-3 моделей для улучшения результатов
3. Провести дополнительный hyperparameter tuning для лучших моделей

---

*Отчет сгенерирован автоматически с помощью `ExperimentReportGenerator`*
"""

        template = Template(template_str)

        # Рендеринг отчета
        report_content = template.render(
            title=title,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_experiments=len(df),
            best_model=best_model,
            worst_model=worst_model,
            avg_rmse=df[rmse_col].mean(),
            avg_r2=df[r2_col].mean(),
            avg_mae=df[mae_col].mean(),
            comparison_table=comparison_table,
            metrics_plot_name=metrics_plot.name,
            scatter_plot_name=scatter_plot.name,
            model_col=model_col,
            rmse_col=rmse_col,
            r2_col=r2_col,
            mae_col=mae_col,
        )

        # Сохранение отчета
        if output_file is None:
            output_file = f"experiment_report_{datetime.now().strftime('%Y-%m-%d')}.md"

        report_path = self.reports_dir / output_file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✓ Report generated: {report_path}")
        print(f"✓ Plots saved: {self.plots_dir}")

        return report_path


def main() -> None:
    """Главная функция для генерации отчета."""
    generator = ExperimentReportGenerator()

    report_path = generator.generate_report(
        title="Отчет о сравнении моделей Boston Housing",
        output_file="latest_experiments.md",
    )

    if report_path:
        print(f"\n✅ Отчет успешно создан: {report_path}")
        print("\nОткройте отчет:")
        print(f"  cat {report_path}")
        print("\nИли в браузере (после сборки docs):")
        print("  mkdocs serve")
    else:
        print("\n❌ Не удалось создать отчет (нет данных экспериментов)")


if __name__ == "__main__":
    main()
