# Dockerfile for MLOps Bike Sharing Project
# Multi-stage build for optimized image size

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 mlops && \
    mkdir -p /app && \
    chown -R mlops:mlops /app

WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Switch to non-root user
USER mlops

# Copy project files
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops params.yaml .
COPY --chown=mlops:mlops dvc.yaml .
COPY --chown=mlops:mlops dvc.lock .
COPY --chown=mlops:mlops setup.py .
COPY --chown=mlops:mlops .dvc/config ./.dvc/config
COPY --chown=mlops:mlops data/raw.dvc ./data/
COPY --chown=mlops:mlops models.dvc .

# Copy entrypoint script
COPY --chown=mlops:mlops docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p data/raw data/processed metrics reports models

# Expose port for FastAPI
EXPOSE 8000

# Health check (will work once FastAPI is implemented)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command (placeholder until FastAPI is implemented)
# When FastAPI is ready, this will be: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
CMD ["python", "-c", "import time; print('🚀 Container ready for FastAPI implementation'); print('📡 Port 8000 exposed and waiting for requests'); print('⏳ Keeping container alive...'); time.sleep(infinity)"]

