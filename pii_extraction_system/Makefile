# PII Extraction System - Development and CI/CD Makefile
# Agent 5: DevOps & CI/CD Specialist

.PHONY: help install install-dev test test-unit test-integration test-e2e test-coverage lint format security-scan type-check build clean docker-build docker-run deploy-dev deploy-staging deploy-prod

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	poetry install --only main

install-dev: ## Install all dependencies including dev tools
	poetry install --with dev,docs
	poetry run pre-commit install

# Testing pipeline
test: test-unit test-integration test-e2e ## Run all tests

test-unit: ## Run unit tests
	poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov/unit

test-integration: ## Run integration tests
	poetry run pytest tests/integration/ -v --cov=src --cov-append --cov-report=html:htmlcov/integration

test-e2e: ## Run end-to-end tests
	poetry run pytest tests/e2e/ -v --cov=src --cov-append --cov-report=html:htmlcov/e2e

test-coverage: ## Generate comprehensive test coverage report
	poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov/full --cov-report=xml:coverage.xml --cov-fail-under=90

# Code quality
lint: ## Run all linting tools
	poetry run flake8 src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

format: ## Format code with black and isort
	poetry run black src/ tests/
	poetry run isort src/ tests/

type-check: ## Run mypy type checking
	poetry run mypy src/

# Security
security-scan: ## Run security vulnerability scanning
	poetry run bandit -r src/ -f json -o security-report.json || true
	poetry run safety check --json --output safety-report.json || true

# Build and distribution
build: ## Build distribution packages
	poetry build

clean: ## Clean build artifacts and cache
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker operations
docker-build: ## Build Docker image
	docker build -t pii-extraction-system:latest .

docker-run: ## Run Docker container locally
	docker run -p 8501:8501 -v $(PWD)/data:/app/data pii-extraction-system:latest

# Deployment
deploy-dev: ## Deploy to development environment
	@echo "Deploying to development environment..."
	# Add deployment commands here

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging environment..."
	# Add deployment commands here

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production environment..."
	# Add deployment commands here

# CI/CD pipeline
ci-pipeline: install-dev lint type-check security-scan test-coverage ## Complete CI pipeline
	@echo "✅ CI Pipeline completed successfully"

cd-pipeline: build docker-build ## Complete CD pipeline
	@echo "✅ CD Pipeline completed successfully"

# Performance testing
perf-test: ## Run performance tests
	poetry run pytest tests/performance/ -v --benchmark-only

# Database and data management
migrate-db: ## Run database migrations
	@echo "Running database migrations..."
	# Add migration commands here

seed-data: ## Seed database with test data
	@echo "Seeding database with test data..."
	# Add data seeding commands here

# Monitoring and health checks
health-check: ## Run health check on running application
	curl -f http://localhost:8501/health || exit 1

# Development utilities
dev-setup: install-dev ## Set up development environment
	@echo "Setting up development environment..."
	mkdir -p logs/
	mkdir -p data/processed/
	mkdir -p data/models/

pre-commit: lint type-check test-unit ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed"