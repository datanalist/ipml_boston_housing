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

## Run tests
.PHONY: test
test:
	uv run pytest tests -v

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
# DOCKER COMMANDS                                                               #
#################################################################################

## Start infrastructure (MinIO, MLflow, Nginx)
.PHONY: docker-up
docker-up:
	docker-compose up -d

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
