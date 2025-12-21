"""
Скрипт для загрузки датасета Boston Housing из интернета.

Источники данных:
1. Statlib (CMU) - оригинальный источник датасета
2. Резервный вариант через scikit-learn (если доступен)
"""

import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from loguru import logger

# Добавляем путь к корню проекта
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# Определяем пути локально для избежания конфликтов импортов
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
HOUSING_DATA_FILE = "housing.csv"

# URL источников данных
DATA_SOURCES = [
    {
        "name": "Statlib (CMU)",
        "url": "http://lib.stat.cmu.edu/datasets/boston",
        "type": "statlib",
    },
    {
        "name": "GitHub Mirror (selva86)",
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "type": "csv",
    },
]


def download_from_statlib(url: str) -> str | None:
    """
    Загрузка данных из формата Statlib.

    Statlib формат содержит заголовки и данные в особом формате:
    - Каждая запись занимает 2 строки (11 значений + 3 значения = 14 признаков)
    - Нужно объединить строки попарно
    """
    logger.info(f"📥 Загрузка из Statlib: {url}")

    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read().decode("latin-1")

        # Ищем начало данных (после описания)
        lines = content.strip().split("\n")
        raw_data_lines = []
        in_data = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Пропускаем строки описания, ищем числовые данные
            parts = line.split()
            if parts and len(parts) >= 2:
                try:
                    float(parts[0])
                    in_data = True
                except ValueError:
                    continue

            if in_data:
                raw_data_lines.append(line)

        # Объединяем строки попарно (каждая запись = 2 строки)
        if raw_data_lines:
            combined_lines = []
            for i in range(0, len(raw_data_lines), 2):
                if i + 1 < len(raw_data_lines):
                    # Объединяем две строки в одну
                    combined = raw_data_lines[i] + " " + raw_data_lines[i + 1]
                    combined_lines.append(combined)
            return "\n".join(combined_lines)
        return None

    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning(f"⚠️ Не удалось загрузить из Statlib: {e}")
        return None


def download_csv(url: str) -> str | None:
    """Загрузка CSV файла напрямую."""
    logger.info(f"📥 Загрузка CSV: {url}")

    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8")

        # Конвертируем CSV с заголовками в формат без заголовков (пробельный разделитель)
        lines = content.strip().split("\n")
        if lines and "," in lines[0]:
            # Это CSV с заголовками, пропускаем первую строку
            data_lines = []
            for line in lines[1:]:
                # Заменяем запятые на пробелы
                parts = line.split(",")
                data_lines.append(" ".join(parts))
            return "\n".join(data_lines)
        return content

    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning(f"⚠️ Не удалось загрузить CSV: {e}")
        return None


def load_from_sklearn() -> str | None:
    """
    Резервный вариант: загрузка через scikit-learn.

    Примечание: в современных версиях sklearn датасет удалён из-за
    этических вопросов, но может работать на старых версиях.
    """
    try:
        logger.info("📥 Попытка загрузки через scikit-learn...")
        from sklearn.datasets import load_boston

        data = load_boston()
        lines = []
        for i in range(len(data.data)):
            row = list(data.data[i]) + [data.target[i]]
            lines.append(" ".join(map(str, row)))
        return "\n".join(lines)

    except ImportError:
        logger.warning("⚠️ scikit-learn не установлен или load_boston недоступен")
        return None
    except AttributeError:
        logger.warning("⚠️ load_boston удалён из scikit-learn (версия >= 1.2)")
        return None


def download_data() -> bool:
    """
    Основная функция загрузки данных.

    Пробует разные источники по очереди.
    Returns:
        True если данные успешно загружены, False в противном случае
    """
    logger.info("🏠 Загрузка датасета Boston Housing\n")

    # Создаём директорию если не существует
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / HOUSING_DATA_FILE

    # Проверяем, существует ли файл
    if output_path.exists():
        logger.info(f"✅ Файл уже существует: {output_path}")
        logger.info("   Используйте --force для перезаписи")
        return True

    data_content = None

    # Пробуем разные источники
    for source in DATA_SOURCES:
        logger.info(f"\n🔗 Источник: {source['name']}")

        if source["type"] == "statlib":
            data_content = download_from_statlib(source["url"])
        elif source["type"] == "csv":
            data_content = download_csv(source["url"])

        if data_content:
            logger.success(f"✅ Данные загружены из {source['name']}")
            break

    # Резервный вариант через sklearn
    if not data_content:
        data_content = load_from_sklearn()

    if not data_content:
        logger.error("❌ Не удалось загрузить данные ни из одного источника")
        return False

    # Сохраняем данные
    with open(output_path, "w") as f:
        f.write(data_content)

    # Проверяем результат
    lines = data_content.strip().split("\n")
    logger.success(f"\n✅ Датасет сохранён: {output_path}")
    logger.info(f"   Записей: {len(lines)}")

    # Показываем первые строки
    logger.info("\n📋 Первые 3 записи:")
    for line in lines[:3]:
        logger.info(f"   {line[:80]}...")

    return True


def main():
    """Точка входа."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Загрузка датасета Boston Housing из интернета"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Перезаписать существующий файл"
    )
    args = parser.parse_args()

    output_path = RAW_DATA_DIR / HOUSING_DATA_FILE

    # Удаляем файл если указан --force
    if args.force and output_path.exists():
        output_path.unlink()
        logger.info(f"🗑️ Удалён существующий файл: {output_path}")

    success = download_data()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
