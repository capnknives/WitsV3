.PHONY: help install install-dev test test-q lint format clean web cli

# Prefer the project venv when present
PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))

help:
	@echo "WitsV3 Make targets (local venv preferred)"
	@echo "  make install       pip install -r requirements.txt"
	@echo "  make install-dev   + requirements-dev.txt"
	@echo "  make web           run Web UI (run_web.py)"
	@echo "  make cli           run CLI (run.py)"
	@echo "  make test          pytest -v"
	@echo "  make test-q        pytest -q --no-cov"
	@echo "  make lint          ruff check (critical paths)"
	@echo "  make format        black + isort (if installed)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

web:
	$(PYTHON) run_web.py

cli:
	$(PYTHON) run.py

test:
	$(PYTHON) -m pytest tests/ -v --no-cov

test-q:
	$(PYTHON) -m pytest tests/ -q --no-cov

lint:
	$(PYTHON) -m ruff check agents core tools web --select E9,F63,F7,F82

format:
	$(PYTHON) -m black agents core tools web tests
	$(PYTHON) -m isort agents core tools web tests

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
