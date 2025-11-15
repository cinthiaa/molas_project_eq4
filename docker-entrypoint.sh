#!/bin/bash
# Docker entrypoint script for MLOps Bike Sharing Project

set -e

echo "🚀 Starting MLOps Bike Sharing Container..."

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  WARNING: AWS credentials not set. DVC operations will fail."
    echo "   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
fi

# Check if models exist, if not try to download from DVC
if [ ! -f "models/random_forest.pkl" ]; then
    echo "📥 Models not found locally. Attempting to download from S3..."
    if dvc pull models.dvc 2>/dev/null; then
        echo "✅ Models downloaded successfully from S3"
    else
        echo "⚠️  Could not download models from S3. Will need to train."
        echo "   Run: dvc repro --force"
    fi
fi

# Check if data exists, if not try to download from DVC
if [ ! -f "data/raw/bike_sharing_modified.csv" ]; then
    echo "📥 Raw data not found. Attempting to download from S3..."
    if dvc pull data/raw.dvc 2>/dev/null; then
        echo "✅ Raw data downloaded successfully from S3"
    else
        echo "⚠️  Could not download raw data from S3."
        echo "   Data will be downloaded when pipeline runs."
        echo "   Or mount data volume: -v \$(pwd)/data:/app/data"
    fi
fi

echo "✅ Container initialized successfully"
echo "🌐 Ready to accept requests on port 8000 (FastAPI)"
echo "📊 MLflow UI available on port 5000 (external: 5001)"
echo ""
echo "Available commands:"
echo "  - Run pipeline: dvc repro"
echo "  - Train models: python -m src.main --stage=train"
echo "  - Make predictions: (FastAPI endpoint - to be implemented)"
echo "  - Access MLflow: http://localhost:5001"
echo ""

# Execute the command passed to docker run (supervisor will start both services)
exec "$@"

