"""
Environment Check Script for Boston Housing Project.

Проверяет корректность установки всех компонентов проекта.
"""

import sys
import subprocess
from pathlib import Path

# ANSI цвета для вывода
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Вывод заголовка."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(text: str) -> None:
    """Вывод успешного результата."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text: str) -> None:
    """Вывод ошибки."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str) -> None:
    """Вывод предупреждения."""
    print(f"{YELLOW}⚠{RESET} {text}")


def check_python_version() -> bool:
    """Проверка версии Python."""
    required_version = (3, 13)
    current_version = sys.version_info[:2]

    if current_version >= required_version:
        print_success(
            f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        return True
    else:
        print_error(
            f"Python version {current_version[0]}.{current_version[1]} < required {required_version[0]}.{required_version[1]}"
        )
        return False


def check_command(command: str, name: str | None = None) -> bool:
    """Проверка доступности команды."""
    name = name or command
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            print_success(f"{name} installed: {version}")
            return True
        else:
            print_error(f"{name} not found")
            return False
    except FileNotFoundError:
        print_error(f"{name} not found")
        return False


def check_virtual_env() -> bool:
    """Проверка виртуального окружения."""
    if sys.prefix != sys.base_prefix:
        print_success(f"Virtual environment active: {sys.prefix}")
        return True
    else:
        print_warning("Virtual environment not active")
        return False


def check_dependencies() -> tuple[bool, int, int]:
    """Проверка установленных зависимостей."""
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "mlflow",
        "dvc",
        "hydra-core",
        "loguru",
        "click",
        "pydantic",
        "pytest",
    ]

    installed = 0
    total = len(required_packages)

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            installed += 1
        except ImportError:
            print_error(f"Package not installed: {package}")

    if installed == total:
        print_success(f"All dependencies installed: {installed}/{total}")
        return True, installed, total
    else:
        print_warning(f"Dependencies installed: {installed}/{total}")
        return False, installed, total


def check_data_files() -> bool:
    """Проверка наличия файлов данных."""
    data_file = Path("data/raw/housing.csv")

    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print_success(f"Data file present: {data_file} ({size_mb:.2f} MB)")
        return True
    else:
        print_error(f"Data file not found: {data_file}")
        return False


def check_config_files() -> bool:
    """Проверка конфигурационных файлов."""
    config_files = [
        "conf/config.yaml",
        "pyproject.toml",
        "dvc.yaml",
        ".dvc/config",
    ]

    all_present = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print_success(f"Config present: {config_file}")
        else:
            print_error(f"Config not found: {config_file}")
            all_present = False

    return all_present


def check_docker() -> bool:
    """Проверка Docker."""
    docker_available = check_command("docker", "Docker")

    if docker_available:
        # Проверка Docker Compose
        compose_v2 = check_command("docker", "Docker Compose (проверка)")
        if not compose_v2:
            compose_v2 = check_command("docker-compose", "Docker Compose")

        # Проверка запущенных контейнеров
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                containers = result.stdout.strip().split("\n")
                if containers and containers[0]:
                    print_success(f"Docker containers running: {len(containers)}")
                    for container in containers[:5]:  # показываем первые 5
                        print(f"  - {container}")
                else:
                    print_warning("No Docker containers running")
            return True
        except Exception as e:
            print_warning(f"Could not check Docker containers: {e}")
            return True
    else:
        return False


def check_precommit_hooks() -> bool:
    """Проверка pre-commit хуков."""
    precommit_config = Path(".pre-commit-config.yaml")
    git_hooks = Path(".git/hooks/pre-commit")

    if precommit_config.exists():
        print_success("Pre-commit config present")
        if git_hooks.exists():
            print_success("Pre-commit hooks installed")
            return True
        else:
            print_warning("Pre-commit hooks not installed (run: pre-commit install)")
            return False
    else:
        print_error("Pre-commit config not found")
        return False


def check_project_structure() -> bool:
    """Проверка структуры проекта."""
    required_dirs = [
        "src",
        "conf",
        "data",
        "airflow",
        "docker",
        "docs",
        "tests",
    ]

    all_present = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists() and path.is_dir():
            print_success(f"Directory present: {directory}/")
        else:
            print_error(f"Directory not found: {directory}/")
            all_present = False

    return all_present


def main() -> None:
    """Главная функция проверки."""
    print_header("Boston Housing Environment Check")

    checks = {
        "Python Version": check_python_version,
        "Virtual Environment": check_virtual_env,
        "Project Structure": check_project_structure,
        "Configuration Files": check_config_files,
        "Data Files": check_data_files,
        "Dependencies": lambda: check_dependencies()[0],
        "Pre-commit Hooks": check_precommit_hooks,
    }

    # Опциональные проверки
    optional_checks = {
        "Docker": check_docker,
        "uv Package Manager": lambda: check_command("uv", "uv"),
        "Git": lambda: check_command("git", "Git"),
        "DVC": lambda: check_command("dvc", "DVC"),
    }

    print_header("Required Checks")
    results = {}
    for name, check_func in checks.items():
        print(f"\n{BLUE}Checking {name}...{RESET}")
        results[name] = check_func()

    print_header("Optional Checks")
    optional_results = {}
    for name, check_func in optional_checks.items():
        print(f"\n{BLUE}Checking {name}...{RESET}")
        optional_results[name] = check_func()

    # Итоговая сводка
    print_header("Summary")

    passed = sum(results.values())
    total = len(results)
    optional_passed = sum(optional_results.values())
    optional_total = len(optional_results)

    print(f"\nRequired checks: {passed}/{total} passed")
    print(f"Optional checks: {optional_passed}/{optional_total} passed")

    if passed == total:
        print(f"\n{GREEN}All required checks passed! ✓{RESET}")
        print(f"\n{BLUE}You're ready to start!{RESET}")
        print("\nNext steps:")
        print("  1. Run experiments: uv run python src/modeling/train_hydra.py")
        print("  2. Start Docker: make docker-up")
        print("  3. Open Airflow UI: http://localhost:8080")
        sys.exit(0)
    else:
        print(f"\n{RED}Some checks failed! ✗{RESET}")
        print(f"\n{YELLOW}Please fix the issues above.{RESET}")
        print("\nQuick fixes:")
        if not results.get("Virtual Environment", False):
            print("  - Activate venv: source .venv/bin/activate")
        if not results.get("Dependencies", False):
            print("  - Install deps: make requirements")
        if not results.get("Data Files", False):
            print("  - Download data: make dvc-pull or make download-data")
        if not results.get("Pre-commit Hooks", False):
            print("  - Install hooks: make pre-commit")
        sys.exit(1)


if __name__ == "__main__":
    main()
