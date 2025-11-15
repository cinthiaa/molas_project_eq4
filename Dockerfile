# Dockerfile for MLOps Bike Sharing Project
# Single container with application and MLflow

FROM python:3.11-slim

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
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 mlops && \
    mkdir -p /app /mlflow && \
    chown -R mlops:mlops /app /mlflow

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER mlops

# Initialize git repository (required by DVC)
RUN git init && \
    git config --global user.email "docker@mlops.local" && \
    git config --global user.name "Docker MLOps"

# Copy project files
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops params.yaml .
COPY --chown=mlops:mlops dvc.yaml .
COPY --chown=mlops:mlops dvc.lock .
COPY --chown=mlops:mlops setup.py .
COPY --chown=mlops:mlops .dvc/config ./.dvc/config
COPY --chown=mlops:mlops data/raw.dvc ./data/
COPY --chown=mlops:mlops models.dvc .

# Add files to git (DVC requires git tracking)
RUN git add . && \
    git commit -m "Initial commit for Docker container" || true

# Copy entrypoint and supervisor config
COPY --chown=mlops:mlops docker-entrypoint.sh .
COPY --chown=mlops:mlops supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p data/raw data/processed metrics reports models

# Expose ports
EXPOSE 8000 5000

# Health check (will work once FastAPI is implemented)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:5000/health || exit 1

# Switch back to root for supervisor
USER root

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Start supervisor to manage both MLflow and app
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
