from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# MinIO Data directories (версионируются через DVC)
MINIO_DATA_DIR = PROJ_ROOT / "minio_data"
RAW_DATA_DIR = MINIO_DATA_DIR / "raw"
PROCESSED_DATA_DIR = MINIO_DATA_DIR / "processed"
MODELS_DIR = MINIO_DATA_DIR / "models"
EXPERIMENTS_DIR = MINIO_DATA_DIR / "experiments"

# Legacy data directories
DATA_DIR = PROJ_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Названия файлов данных
HOUSING_DATA_FILE = "housing.csv"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
