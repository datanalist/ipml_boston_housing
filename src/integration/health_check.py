"""Проверка доступности всех компонентов ML-инфраструктуры."""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from loguru import logger

from src.integration.utils import (
    ServiceConfig,
    ServiceStatus,
    get_service_config,
    retry_with_backoff,
    unified_logger,
)


@dataclass
class HealthCheckResult:
    """Результат проверки здоровья сервиса."""

    service: str
    status: ServiceStatus
    message: str
    response_time_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "service": self.service,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Проверка здоровья всех компонентов ML-инфраструктуры.

    Проверяет доступность и работоспособность:
    - MLflow Tracking Server
    - MinIO S3 Storage
    - Apache Airflow
    - DVC Remote
    """

    def __init__(self):
        """Инициализация чекера."""
        self.log = unified_logger("health_check")
        self.results: list[HealthCheckResult] = []

    def check_all(self) -> list[HealthCheckResult]:
        """
        Проверка всех сервисов.

        Returns:
            Список результатов проверки
        """
        self.log.info("Начало проверки всех сервисов...")
        self.results = []

        checks = [
            ("mlflow", self.check_mlflow),
            ("minio", self.check_minio),
            ("airflow", self.check_airflow),
            ("dvc", self.check_dvc),
        ]

        for service_name, check_func in checks:
            try:
                result = check_func()
                self.results.append(result)
            except Exception as e:
                self.results.append(
                    HealthCheckResult(
                        service=service_name,
                        status=ServiceStatus.UNHEALTHY,
                        message=f"Ошибка проверки: {e!s}",
                    )
                )

        self._log_summary()
        return self.results

    def _log_summary(self) -> None:
        """Логирование сводки проверок."""
        healthy = sum(1 for r in self.results if r.status == ServiceStatus.HEALTHY)
        total = len(self.results)

        if healthy == total:
            self.log.success(f"Все сервисы работают ({healthy}/{total})")
        else:
            self.log.warning(f"Работающие сервисы: {healthy}/{total}")
            for result in self.results:
                if result.status != ServiceStatus.HEALTHY:
                    self.log.error(f"  - {result.service}: {result.message}")

    def check_mlflow(self) -> HealthCheckResult:
        """Проверка MLflow Tracking Server."""
        config = get_service_config("mlflow")
        return _check_http_service(config)

    def check_minio(self) -> HealthCheckResult:
        """Проверка MinIO S3 Storage."""
        config = get_service_config("minio")
        result = _check_http_service(config)

        # Дополнительная проверка доступа к bucket
        if result.status == ServiceStatus.HEALTHY:
            try:
                import boto3
                from botocore.client import Config
                from botocore.exceptions import ClientError

                s3_client = boto3.client(
                    "s3",
                    endpoint_url=f"http://{config.host}:{config.port}",
                    aws_access_key_id=config.extra["access_key"],
                    aws_secret_access_key=config.extra["secret_key"],
                    config=Config(signature_version="s3v4"),
                )

                bucket_name = config.extra["bucket"]
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                    result.details["bucket_accessible"] = True
                    result.details["bucket_name"] = bucket_name
                except ClientError:
                    # Bucket не существует, но MinIO работает
                    result.details["bucket_accessible"] = False
                    result.details["bucket_name"] = bucket_name
                    result.message += " (bucket не найден)"
                    result.status = ServiceStatus.DEGRADED

            except Exception as e:
                result.details["s3_client_error"] = str(e)

        return result

    def check_airflow(self) -> HealthCheckResult:
        """Проверка Apache Airflow."""
        config = get_service_config("airflow")
        return _check_http_service(config)

    def check_dvc(self) -> HealthCheckResult:
        """Проверка DVC."""
        import time

        start_time = time.time()

        try:
            # Проверяем установку DVC
            result = subprocess.run(
                ["dvc", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return HealthCheckResult(
                    service="dvc",
                    status=ServiceStatus.UNHEALTHY,
                    message="DVC не установлен или не доступен",
                )

            dvc_version = result.stdout.strip().split("\n")[0]

            # Проверяем статус DVC
            status_result = subprocess.run(
                ["dvc", "status"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            response_time = (time.time() - start_time) * 1000

            # Проверяем remote
            remote_result = subprocess.run(
                ["dvc", "remote", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            remotes = [
                line.strip()
                for line in remote_result.stdout.strip().split("\n")
                if line.strip()
            ]

            return HealthCheckResult(
                service="dvc",
                status=ServiceStatus.HEALTHY,
                message=f"DVC работает ({dvc_version})",
                response_time_ms=response_time,
                details={
                    "version": dvc_version,
                    "remotes": remotes,
                    "status_output": status_result.stdout[:200]
                    if status_result.stdout
                    else "No changes",
                },
            )

        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                service="dvc",
                status=ServiceStatus.UNHEALTHY,
                message="Таймаут при проверке DVC",
            )
        except FileNotFoundError:
            return HealthCheckResult(
                service="dvc",
                status=ServiceStatus.UNHEALTHY,
                message="DVC не установлен",
            )
        except Exception as e:
            return HealthCheckResult(
                service="dvc",
                status=ServiceStatus.UNHEALTHY,
                message=f"Ошибка проверки DVC: {e!s}",
            )

    def get_summary(self) -> dict[str, Any]:
        """
        Получение сводки проверок.

        Returns:
            Словарь со сводной информацией
        """
        if not self.results:
            self.check_all()

        healthy = sum(1 for r in self.results if r.status == ServiceStatus.HEALTHY)
        degraded = sum(1 for r in self.results if r.status == ServiceStatus.DEGRADED)
        unhealthy = sum(1 for r in self.results if r.status == ServiceStatus.UNHEALTHY)

        overall_status = ServiceStatus.HEALTHY
        if unhealthy > 0:
            overall_status = ServiceStatus.UNHEALTHY
        elif degraded > 0:
            overall_status = ServiceStatus.DEGRADED

        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
            },
            "services": [r.to_dict() for r in self.results],
        }


@retry_with_backoff(
    max_retries=2, base_delay=0.5, exceptions=(requests.RequestException,)
)
def _check_http_service(config: ServiceConfig) -> HealthCheckResult:
    """
    Проверка HTTP-сервиса.

    Args:
        config: Конфигурация сервиса

    Returns:
        Результат проверки
    """
    import time

    start_time = time.time()

    try:
        auth = None
        if "username" in config.extra and "password" in config.extra:
            auth = (config.extra["username"], config.extra["password"])

        response = requests.get(
            config.url,
            timeout=config.timeout,
            auth=auth,
        )

        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            return HealthCheckResult(
                service=config.name,
                status=ServiceStatus.HEALTHY,
                message=f"Сервис доступен ({response.status_code})",
                response_time_ms=response_time,
                details={"url": config.url, "status_code": response.status_code},
            )
        else:
            return HealthCheckResult(
                service=config.name,
                status=ServiceStatus.DEGRADED,
                message=f"Сервис вернул код {response.status_code}",
                response_time_ms=response_time,
                details={"url": config.url, "status_code": response.status_code},
            )

    except requests.Timeout:
        return HealthCheckResult(
            service=config.name,
            status=ServiceStatus.UNHEALTHY,
            message=f"Таймаут при подключении ({config.timeout}с)",
            details={"url": config.url},
        )
    except requests.ConnectionError as e:
        return HealthCheckResult(
            service=config.name,
            status=ServiceStatus.UNHEALTHY,
            message=f"Ошибка подключения: {e!s}",
            details={"url": config.url},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Функции для использования без класса
# ═══════════════════════════════════════════════════════════════════════════════


def check_all_services() -> dict[str, Any]:
    """
    Проверка всех сервисов.

    Returns:
        Сводка проверок
    """
    checker = HealthChecker()
    checker.check_all()
    return checker.get_summary()


def check_mlflow() -> HealthCheckResult:
    """Проверка MLflow."""
    return HealthChecker().check_mlflow()


def check_minio() -> HealthCheckResult:
    """Проверка MinIO."""
    return HealthChecker().check_minio()


def check_dvc() -> HealthCheckResult:
    """Проверка DVC."""
    return HealthChecker().check_dvc()


def check_airflow() -> HealthCheckResult:
    """Проверка Airflow."""
    return HealthChecker().check_airflow()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI интерфейс
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    logger.info("=" * 60)
    logger.info("Проверка ML-инфраструктуры")
    logger.info("=" * 60)

    summary = check_all_services()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("=" * 60)

    for service in summary["services"]:
        status_icon = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "unknown": "❓",
        }.get(service["status"], "❓")

        time_str = (
            f" ({service['response_time_ms']:.0f}ms)"
            if service["response_time_ms"]
            else ""
        )
        print(f"{status_icon} {service['service']}: {service['message']}{time_str}")

    print("\n" + "-" * 60)
    s = summary["summary"]
    print(
        f"Итого: {s['healthy']}/{s['total']} здоровых, "
        f"{s['degraded']} деградированных, {s['unhealthy']} недоступных"
    )
    print(f"Общий статус: {summary['overall_status'].upper()}")
    print("=" * 60)

    # Выводим JSON для интеграции
    if "--json" in sys.argv:
        print("\nJSON:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Код возврата
    sys.exit(0 if summary["overall_status"] == "healthy" else 1)
