#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME := holod
PYTHON_VERSION := 3.13
PYTHON_INTERPRETER := python

# Which torch build to install: cu126 (CUDA) or cpu. These are mutually
# exclusive extras, so one must always be selected -- otherwise `uv run`
# re-syncs the environment without torch. Override per-invocation:
#   make test TORCH_EXTRA=cpu
TORCH_EXTRA ?= cu126
UV_RUN := uv run --extra $(TORCH_EXTRA)

# Directories
SRC_DIR      := src
DB_TESTS     := $(SRC_DIR)/tests/check_database.py
TESTS     	 := $(SRC_DIR)/tests/verbose_tests/check_training.py $(SRC_DIR)/tests/verbose_tests/check_dataset_config.py $(SRC_DIR)/tests/verbose_tests/check_overfit.py $(DB_TESTS)
BUILD_DIRS   := build dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies (sync lock + install)
.PHONY: requirements
requirements:
	uv sync --extra $(TORCH_EXTRA)

## Delete all checkpoints, reports
.PHONY: clear
clear:
	find reports/ -type f -name "*.json" -delete
	find reports/ -type f -name "*.html" -delete
	find reports/ -type f -name "*.csv" -delete
	find src/checkpoints/ -type f -name "*.pth" -delete
	find src/checkpoints/  -type f -name "*.tar" -delete

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Update all dependencies to latest allowed by constraints (refresh lockfile)
.PHONY: bump-deps
bump-deps:
	uv lock --upgrade
	uv sync --extra $(TORCH_EXTRA)

## Format code (apply fixes)
.PHONY: format
format:
	$(UV_RUN) ruff check --fix
	$(UV_RUN) ruff format

## Static type checking 
.PHONY: typecheck
typecheck:
	$(UV_RUN) mypy $(SRC_DIR) || true

## Run tests (skips slow training tests)
.PHONY: test
test:
	$(UV_RUN) pytest -q --show-capture=stdout -m "not slow" $(TESTS)

## Run tests and save results (also skips slow training tests)
.PHONY: test-report
test-report:
	$(UV_RUN) pytest --junitxml=reports/test-results.xml -m "not slow" $(TESTS)

## Run slow tests (single-batch overfit per backbone)
.PHONY: test-slow
test-slow:
	$(UV_RUN) pytest -q -m slow $(TESTS)

## Run tests with coverage HTML report
.PHONY: coverage
coverage:
	$(UV_RUN) pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html $(TESTS)
	@echo "Open htmlcov/index.html"

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


## One command that does requirements, lint, typecheck, and test
.PHONY: check
check: requirements typecheck test

#################################################################################
# PROJECT DATABASE                                                              #
#################################################################################

## start the local Postgres used by the database tests
.PHONY: db-up
db-up:
	docker compose up -d db

## stop the local Postgres
.PHONY: db-down
db-down:
	docker compose down

## run only the database tests (needs no torch, so plain `uv run`)
.PHONY: test-db
test-db:
	uv run pytest -q $(DB_TESTS)

## migrate db to latest schema
.PHONY: db-migrate
db-migrate:
	uv run alembic upgrade head

## create blank revision with some message
.PHONY: db-revision
db-revision:
	uv run alembic revision -m "$(m)"

## check what version the database is
.PHONY: db-current
db-current:
	uv run alembic current

## runs sql command without executing it
.PHONY: db-sql
db-sql:
	uv run alembic upgrade head --sql
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Train model 
.PHONY: train
train: requirements
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py train

## Compare model backbones under one shared configuration
.PHONY: compare
compare: requirements
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py compare

## Generate plots
.PHONY: plot
plot: requirements
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py plot-train

## Train and plot
.PHONY: do
do: requirements
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py train
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py plot-train

## Preform Reconstruction on sample data
.PHONY: recon
recon: requirements
	$(UV_RUN) $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py reconstruction "./src/data/MW_Dataset_Sample/405/10_Skeletal_muscle/z15/1.bmp"


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
