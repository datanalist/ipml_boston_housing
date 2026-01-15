#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = boston_housing
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Run all tests
.PHONY: test
test:
	uv run pytest tests -v

## Run reproducibility tests
.PHONY: test-reproducibility
test-reproducibility:
	uv run pytest tests/test_reproducibility.py -v

## Run integration tests
.PHONY: test-integration
test-integration:
	uv run pytest tests/test_integration.py -v

## Run pipeline tests
.PHONY: test-pipeline
test-pipeline:
	uv run pytest tests/test_pipeline.py -v

## Run tests with coverage
.PHONY: test-coverage
test-coverage:
	uv run pytest tests -v --cov=src --cov-report=html --cov-report=term

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@if [ -d ".venv" ]; then \
		echo ">>> Virtual environment already exists in .venv"; \
		echo ">>> Skipping creation. To recreate, run: rm -rf .venv && make create_environment"; \
	else \
		uv venv --python $(PYTHON_VERSION); \
		echo ">>> New uv virtual environment created. Activate with:"; \
		echo ">>> Windows: .\\.venv\\Scripts\\activate"; \
		echo ">>> Unix/macOS: source ./.venv/bin/activate"; \
	fi

## Install all pre-commit hooks (pre-commit, commit-msg, pre-push)
.PHONY: pre-commit
pre-commit:
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type commit-msg
	uv run pre-commit install --hook-type pre-push
	uv run pre-commit run --all-files

#################################################################################
# DVC COMMANDS                                                                  #
#################################################################################

## Pull data from remote DVC storage
.PHONY: dvc-pull
dvc-pull:
	uv run dvc pull

## Push data to remote DVC storage
.PHONY: dvc-push
dvc-push:
	uv run dvc push

## Check DVC status
.PHONY: dvc-status
dvc-status:
	uv run dvc status

#################################################################################
# INTEGRATION & MONITORING                                                      #
#################################################################################

## Check all services health (MLflow, MinIO, DVC, Airflow)
.PHONY: health-check
health-check:
	$(PYTHON_INTERPRETER) -m src.integration.health_check

## Check services health with JSON output
.PHONY: health-check-json
health-check-json:
	$(PYTHON_INTERPRETER) -m src.integration.health_check --json

## Verify all integrations are working
.PHONY: verify-integration
verify-integration: health-check test-integration
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "✅ ИНТЕГРАЦИЯ ПРОВЕРЕНА"
	@echo "═══════════════════════════════════════════════════════════════════════════════"

## Run full verification: health check + all tests
.PHONY: verify-all
verify-all: health-check test
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ"
	@echo "═══════════════════════════════════════════════════════════════════════════════"

#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Start infrastructure (MinIO, MLflow, Nginx)
.PHONY: docker-up
docker-up:
	@docker-compose up -d 2>&1 || (echo ">>> Recreating Docker network..." && docker-compose down && docker network rm ipml_boston_housing_boston_housing_network 2>/dev/null; docker-compose up -d)

## Stop infrastructure
.PHONY: docker-down
docker-down:
	docker-compose down

## View infrastructure logs
.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

## Restart infrastructure
.PHONY: docker-restart
docker-restart:
	docker-compose restart

## Build Docker images
.PHONY: docker-build
docker-build:
	docker-compose build

## Check infrastructure status
.PHONY: docker-status
docker-status:
	docker-compose ps

## Recreate Docker network (use if "network needs to be recreated" error)
.PHONY: docker-recreate
docker-recreate:
	docker-compose down
	docker network rm ipml_boston_housing_boston_housing_network 2>/dev/null || true
	docker-compose up -d

## Fix Airflow data permissions
.PHONY: airflow-fix-permissions
airflow-fix-permissions:
	bash scripts/fix_airflow_data_permissions.sh

## Trigger Airflow DAG
.PHONY: airflow-trigger-dag
airflow-trigger-dag:
	docker exec boston_housing_airflow_scheduler airflow dags trigger boston_housing_experiments

## List Airflow DAGs
.PHONY: airflow-list-dags
airflow-list-dags:
	docker exec boston_housing_airflow_scheduler airflow dags list

## Check Airflow DAG status
.PHONY: airflow-dag-status
airflow-dag-status:
	docker exec boston_housing_airflow_scheduler airflow dags list-runs -d boston_housing_experiments

#################################################################################
# DATA ACQUISITION                                                              #
#################################################################################

## Download dataset from the internet
.PHONY: download-data
download-data:
	$(PYTHON_INTERPRETER) scripts/download_data.py

## Force re-download dataset (overwrite existing)
.PHONY: download-data-force
download-data-force:
	$(PYTHON_INTERPRETER) scripts/download_data.py --force

#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

## Install documentation dependencies
.PHONY: requirements-docs
requirements-docs:
	uv sync --group docs

## Serve documentation locally with live reload
.PHONY: docs-serve
docs-serve:
	uv run mkdocs serve

## Build documentation
.PHONY: docs-build
docs-build:
	uv run mkdocs build --clean --strict

## Generate experiment reports
.PHONY: generate-reports
generate-reports:
	@echo "📊 Generating experiment reports..."
	$(PYTHON_INTERPRETER) scripts/generate_experiment_report.py
	@echo "✅ Reports generated successfully!"

## Generate reports and serve documentation
.PHONY: docs-with-reports
docs-with-reports: generate-reports docs-serve

## Deploy documentation to GitHub Pages (manual)
.PHONY: docs-deploy
docs-deploy:
	uv run mkdocs gh-deploy --force

## Check environment setup
.PHONY: check-env
check-env:
	$(PYTHON_INTERPRETER) scripts/check_environment.py

#################################################################################
# ML EXPERIMENTS                                                                #
#################################################################################

## Run ML experiments (requires MLflow server running)
.PHONY: experiments
experiments:
	$(PYTHON_INTERPRETER) scripts/run_experiments.py

## Run training in Docker container
.PHONY: train
train:
	docker-compose --profile train up train

## Run main application
.PHONY: run
run:
	$(PYTHON_INTERPRETER) main.py

#################################################################################
# FULL EXPERIMENT PIPELINE                                                      #
#################################################################################

## Run complete experiment: download data → start infra → run experiments
.PHONY: experiment-full
experiment-full: download-data docker-up
	@echo "⏳ Waiting for MLflow server to start..."
	@sleep 5
	$(PYTHON_INTERPRETER) scripts/run_experiments.py
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "✅ ЭКСПЕРИМЕНТ ЗАВЕРШЁН"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "📊 MLflow UI:        http://localhost:5000"
	@echo "📦 MinIO Console:    http://localhost:9001"
	@echo "📁 Результаты:       data/experiments/results_summary.csv"
	@echo "═══════════════════════════════════════════════════════════════════════════════"

## Run experiment without Docker (local MLflow required)
.PHONY: experiment-local
experiment-local: download-data
	@echo "🔬 Running experiments with local/external MLflow..."
	$(PYTHON_INTERPRETER) scripts/run_experiments.py

## Quick experiment demo: download data + start MLflow + run 4 models
.PHONY: experiment-demo
experiment-demo: download-data docker-up
	@echo "⏳ Waiting for MLflow server to start..."
	@sleep 5
	$(PYTHON_INTERPRETER) scripts/experiment_demo.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py

#################################################################################
# QUICK START                                                                   #
#################################################################################

## Full setup: create env, install deps, start infra, pull data
.PHONY: setup
setup: create_environment requirements pre-commit docker-up dvc-pull
	@echo ">>> Setup complete!"
	@echo ">>> MLflow UI available at: http://localhost:5000"
	@echo ">>> MinIO Console available at: http://localhost:9001"

## Quick start: download data, start infra, run experiments
.PHONY: quickstart
quickstart: requirements download-data docker-up
	@echo "⏳ Waiting for MLflow server to start..."
	@sleep 5
	$(PYTHON_INTERPRETER) scripts/run_experiments.py
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "🎉 QUICKSTART ЗАВЕРШЁН!"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "📊 MLflow UI:        http://localhost:5000"
	@echo "📦 MinIO Console:    http://localhost:9001"
	@echo "📁 Результаты:       data/experiments/results_summary.csv"
	@echo "═══════════════════════════════════════════════════════════════════════════════"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
