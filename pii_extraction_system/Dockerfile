# Multi-stage Dockerfile for PII Extraction System
# Agent 5: DevOps & CI/CD Specialist

# Base image with Python 3.9
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Development stage
FROM base as development

WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --with dev,docs && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Set up development environment
RUN mkdir -p logs/ data/processed/ data/models/

# Expose port for development
EXPOSE 8501

# Development command
CMD ["poetry", "run", "streamlit", "run", "src/dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install only production dependencies
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs/ data/processed/ data/models/ && \
    chmod -R 755 logs/ data/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port
EXPOSE 8501

# Production command
CMD ["poetry", "run", "streamlit", "run", "src/dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests
RUN poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

# Security scanning stage
FROM production as security

# Install security scanning tools
USER root
RUN poetry add --group dev bandit safety
USER appuser

# Run security scans
RUN poetry run bandit -r src/ -f json -o security-report.json || true
RUN poetry run safety check --json --output safety-report.json || true