#!/bin/bash
# Start PM2.5 Forecasting Pipeline
# This script activates venv and runs the pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting PM2.5 Forecasting Pipeline"
echo "========================================"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies if needed
if ! python3 -c "import httpx" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install --upgrade pip -q
    pip install httpx requests pandas pyarrow numpy scipy -q
    echo "✅ Dependencies installed"
fi

# Run pipeline
echo ""
echo "▶️  Running pipeline..."
echo "========================================"
python3 run_pipeline.py

echo ""
echo "✅ Pipeline execution completed!"

