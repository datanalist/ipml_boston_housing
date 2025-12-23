"""Конфигурация проекта."""

from pathlib import Path

from dotenv import load_dotenv
from src.config.mlflow_config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_TRACKING_URI,
    setup_mlflow_env,
)

# Load environment variables from .env file if it exists
load_dotenv()

# Paths - воспроизводим из src/config.py для совместимости
PROJ_ROOT = Path(__file__).resolve().parents[2]

# Data directories (версионируются через DVC)
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# MinIO storage (внутреннее хранилище)
MINIO_DATA_DIR = PROJ_ROOT / "minio_data"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Названия файлов данных
HOUSING_DATA_FILE = "housing.csv"

__all__ = [
    # Пути
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "EXPERIMENTS_DIR",
    "INTERIM_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MINIO_DATA_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "HOUSING_DATA_FILE",
    # MLflow
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "setup_mlflow_env",
]
