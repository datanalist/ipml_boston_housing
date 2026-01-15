"""
MinIO Cache Utilities для Airflow
=================================
Модуль для кэширования артефактов в MinIO с проверкой хэшей.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from loguru import logger


class MinIOCache:
    """
    Класс для кэширования артефактов в MinIO.

    Позволяет:
    - Проверять существование файлов
    - Вычислять хэши для проверки изменений
    - Загружать/скачивать артефакты
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: str = "airflow-cache",
    ):
        """
        Инициализация клиента MinIO.

        Args:
            endpoint_url: URL MinIO сервера
            access_key: Access Key
            secret_key: Secret Key
            bucket_name: Имя бакета для кэша
        """
        self.endpoint_url = endpoint_url or os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"
        )
        self.access_key = access_key or os.environ.get(
            "AWS_ACCESS_KEY_ID", "minioadmin"
        )
        self.secret_key = secret_key or os.environ.get(
            "AWS_SECRET_ACCESS_KEY", "minioadmin"
        )
        self.bucket_name = bucket_name

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Создаёт бакет если не существует."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)
            logger.info(f"📦 Создан бакет: {self.bucket_name}")

    def compute_file_hash(self, file_path: str) -> str:
        """
        Вычисляет MD5 хэш файла.

        Args:
            file_path: Путь к файлу

        Returns:
            MD5 хэш в hex формате
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def compute_params_hash(self, params: dict) -> str:
        """
        Вычисляет хэш параметров (для кэширования по параметрам).

        Args:
            params: Словарь параметров

        Returns:
            MD5 хэш параметров
        """
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def exists(self, key: str) -> bool:
        """
        Проверяет существование объекта в MinIO.

        Args:
            key: Ключ объекта

        Returns:
            True если объект существует
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def get_cache_key(
        self, prefix: str, params: dict, data_hash: Optional[str] = None
    ) -> str:
        """
        Генерирует ключ кэша на основе параметров и хэша данных.

        Args:
            prefix: Префикс ключа (например, "models/random_forest")
            params: Параметры модели/эксперимента
            data_hash: Хэш входных данных (опционально)

        Returns:
            Ключ кэша
        """
        params_hash = self.compute_params_hash(params)
        if data_hash:
            return f"{prefix}_{params_hash}_{data_hash}"
        return f"{prefix}_{params_hash}"

    def check_cache(
        self,
        prefix: str,
        params: dict,
        data_path: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Проверяет наличие кэшированного результата.

        Args:
            prefix: Префикс ключа
            params: Параметры
            data_path: Путь к входным данным (для вычисления хэша)

        Returns:
            (exists, cache_key) - существует ли кэш и ключ кэша
        """
        data_hash = None
        if data_path and Path(data_path).exists():
            data_hash = self.compute_file_hash(data_path)

        cache_key = self.get_cache_key(prefix, params, data_hash)
        exists = self.exists(cache_key)

        if exists:
            logger.info(f"✅ Кэш найден: {cache_key}")
        else:
            logger.info(f"❌ Кэш не найден: {cache_key}")

        return exists, cache_key

    def upload(self, local_path: str, key: str) -> str:
        """
        Загружает файл в MinIO.

        Args:
            local_path: Локальный путь к файлу
            key: Ключ в MinIO

        Returns:
            S3 URI загруженного файла
        """
        self.client.upload_file(local_path, self.bucket_name, key)
        s3_uri = f"s3://{self.bucket_name}/{key}"
        logger.info(f"📤 Загружено: {s3_uri}")
        return s3_uri

    def download(self, key: str, local_path: str) -> str:
        """
        Скачивает файл из MinIO.

        Args:
            key: Ключ в MinIO
            local_path: Локальный путь для сохранения

        Returns:
            Локальный путь к скачанному файлу
        """
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket_name, key, local_path)
        logger.info(f"📥 Скачано: {local_path}")
        return local_path

    def put_json(self, key: str, data: dict) -> str:
        """
        Сохраняет JSON в MinIO.

        Args:
            key: Ключ в MinIO
            data: Словарь для сохранения

        Returns:
            S3 URI
        """
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json_bytes,
            ContentType="application/json",
        )
        return f"s3://{self.bucket_name}/{key}"

    def get_json(self, key: str) -> dict:
        """
        Читает JSON из MinIO.

        Args:
            key: Ключ в MinIO

        Returns:
            Словарь с данными
        """
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))


def check_model_cache(
    model_name: str,
    params: dict,
    data_path: str,
    bucket_name: str = "airflow-cache",
) -> bool:
    """
    Функция для использования с ShortCircuitOperator.

    Проверяет, есть ли кэшированная модель с заданными параметрами.
    Если модель найдена в кэше - возвращает False (пропуск downstream задач).

    Args:
        model_name: Имя модели
        params: Параметры модели
        data_path: Путь к данным
        bucket_name: Имя бакета

    Returns:
        True если нужно обучать (кэш не найден),
        False если можно пропустить (кэш найден)
    """
    cache = MinIOCache(bucket_name=bucket_name)
    prefix = f"models/{model_name}"
    exists, _ = cache.check_cache(prefix, params, data_path)

    # ShortCircuitOperator: True = продолжить, False = пропустить
    return not exists


def get_cached_model(
    model_name: str,
    params: dict,
    data_path: str,
    local_path: str,
    bucket_name: str = "airflow-cache",
) -> Optional[str]:
    """
    Получает кэшированную модель из MinIO.

    Args:
        model_name: Имя модели
        params: Параметры модели
        data_path: Путь к данным
        local_path: Локальный путь для сохранения
        bucket_name: Имя бакета

    Returns:
        Локальный путь к модели или None если не найдена
    """
    cache = MinIOCache(bucket_name=bucket_name)
    prefix = f"models/{model_name}"
    exists, cache_key = cache.check_cache(prefix, params, data_path)

    if exists:
        return cache.download(f"{cache_key}.pkl", local_path)
    return None


def save_model_to_cache(
    model_path: str,
    model_name: str,
    params: dict,
    data_path: str,
    metrics: dict,
    bucket_name: str = "airflow-cache",
) -> str:
    """
    Сохраняет обученную модель в кэш MinIO.

    Args:
        model_path: Локальный путь к модели
        model_name: Имя модели
        params: Параметры модели
        data_path: Путь к данным
        metrics: Метрики модели
        bucket_name: Имя бакета

    Returns:
        S3 URI сохранённой модели
    """
    cache = MinIOCache(bucket_name=bucket_name)

    # Вычисляем хэш данных
    data_hash = cache.compute_file_hash(data_path) if Path(data_path).exists() else None

    # Генерируем ключ
    prefix = f"models/{model_name}"
    cache_key = cache.get_cache_key(prefix, params, data_hash)

    # Сохраняем модель
    model_uri = cache.upload(model_path, f"{cache_key}.pkl")

    # Сохраняем метаданные
    metadata = {
        "model_name": model_name,
        "params": params,
        "data_hash": data_hash,
        "metrics": metrics,
    }
    cache.put_json(f"{cache_key}_metadata.json", metadata)

    return model_uri
