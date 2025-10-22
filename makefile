#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME := holod
PYTHON_VERSION := 3.13
PYTHON_INTERPRETER := python

# Directories
SRC_DIR      := src
TESTS     	 := $(SRC_DIR)/tests/check_training.py
BUILD_DIRS   := build dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies (sync lock + install)
.PHONY: requirements
requirements:
	uv sync	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Update all dependencies to latest allowed by constraints (refresh lockfile)
.PHONY: bump-deps
bump-deps:
	uv lock --upgrade
	uv sync


## Format code (apply fixes)
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Static type checking 
.PHONY: typecheck
typecheck:
	uv run mypy $(SRC_DIR) || true

## Run tests
.PHONY: test
test:
	uv run pytest -q $(TESTS)

## Run tests with coverage HTML report
.PHONY: coverage
coverage:
	uv run pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html $(TESTS)
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
# PROJECT RULES                                                                 #
#################################################################################


## Train model 
.PHONY: train
train: requirements
	uv run $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py train

## Generate plots 
.PHONY: plot
plot: requirements
	uv run $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py plot-train

## Train and plot
.PHONY: do
do: requirements
	uv run $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py train
	uv run $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py plot-train

## Preform Reconstruction on sample data
.PHONY: recon
recon: requirements
	uv run $(PYTHON_INTERPRETER) $(SRC_DIR)/holod/cli.py reconstruction "./src/data/MW_Dataset_Sample/405/10_Skeletal_muscle/z15/1.bmp"


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
