#!/usr/bin/env python3
"""
Скрипт для исправления прав доступа на файлы Airflow
"""

import os
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    airflow_dags = project_root / "airflow" / "dags"
    airflow_plugins = project_root / "airflow" / "plugins"

    current_uid = os.getuid()
    current_gid = os.getgid()

    files_to_check = []
    if airflow_dags.exists():
        files_to_check.extend(airflow_dags.glob("*.py"))
    if airflow_plugins.exists():
        files_to_check.extend(airflow_plugins.glob("*.py"))

    files_with_issues = []

    for file_path in files_to_check:
        file_stat = file_path.stat()
        file_uid = file_stat.st_uid
        file_gid = file_stat.st_gid

        if file_uid != current_uid:
            files_with_issues.append((file_path, file_uid, file_gid))

    if files_with_issues:
        print(f"Найдено {len(files_with_issues)} файлов с проблемами прав доступа:")
        for file_path, file_uid, file_gid in files_with_issues:
            print(f"  {file_path} (UID: {file_uid}, GID: {file_gid})")

        print("\nИсправление прав доступа...")
        for file_path, _, _ in files_with_issues:
            try:
                os.chown(file_path, current_uid, current_gid)
                print(f"  ✓ {file_path}")
            except PermissionError as e:
                print(f"  ✗ {file_path}: {e}")

        # Исправление прав на директории
        for dir_path in [airflow_dags, airflow_plugins]:
            if dir_path.exists():
                try:
                    os.chown(dir_path, current_uid, current_gid)
                    print(f"  ✓ {dir_path}")
                except PermissionError as e:
                    print(f"  ✗ {dir_path}: {e}")
    else:
        print("Все файлы имеют правильные права доступа.")

    # Проверка после исправления
    if files_with_issues:
        print("\nПроверка прав доступа после исправления...")
        all_fixed = True
        for file_path, _, _ in files_with_issues:
            try:
                file_stat = file_path.stat()
                file_uid = file_stat.st_uid
                can_write_now = os.access(file_path, os.W_OK)
                if file_uid == current_uid and can_write_now:
                    print(f"  ✓ {file_path} - исправлено")
                else:
                    print(
                        f"  ✗ {file_path} - все еще требует исправления (UID: {file_uid}, can_write: {can_write_now})"
                    )
                    all_fixed = False
            except Exception as e:
                print(f"  ✗ {file_path} - ошибка проверки: {e}")
                all_fixed = False

        if not all_fixed:
            print(
                "\n⚠️  Некоторые файлы не были исправлены. Возможно, требуется выполнить команду с sudo:"
            )
            print(
                f"   sudo chown -R {current_uid}:{current_gid} {project_root}/airflow/dags {project_root}/airflow/plugins"
            )
        else:
            print("\n✅ Все файлы успешно исправлены!")


if __name__ == "__main__":
    main()
