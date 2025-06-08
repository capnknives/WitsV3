.PHONY: install install-dev test lint format clean build publish docs

# Python version
PYTHON := python3.10

# Package name
PACKAGE := witsv3

# Default target
all: install-dev

# Install production dependencies
install:
	$(PYTHON) -m pip install -e .

# Install development dependencies
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install -r requirements-dev.txt

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html

# Run linting
lint:
	$(PYTHON) -m flake8 $(PACKAGE) tests
	$(PYTHON) -m mypy $(PACKAGE) tests
	$(PYTHON) -m black --check $(PACKAGE) tests
	$(PYTHON) -m isort --check-only $(PACKAGE) tests

# Format code
format:
	$(PYTHON) -m black $(PACKAGE) tests
	$(PYTHON) -m isort $(PACKAGE) tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf coverage_html/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Build package
build: clean
	$(PYTHON) setup.py sdist bdist_wheel

# Publish package to PyPI
publish: build
	$(PYTHON) -m twine upload dist/*

# Build documentation
docs:
	$(PYTHON) -m sphinx-build -b html docs/ docs/_build/html

# Run development server
dev:
	$(PYTHON) -m $(PACKAGE).cli

# Run background agent
agent:
	$(PYTHON) -m $(PACKAGE).background_agent

# Help
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  publish      - Publish package to PyPI"
	@echo "  docs         - Build documentation"
	@echo "  dev          - Run development server"
	@echo "  agent        - Run background agent"
	@echo "  help         - Show this help message" 