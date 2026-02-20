#!/bin/bash
# Wait for training data to be ready

cd "$(dirname "$0")"

echo "🔍 Monitoring Pipeline - Waiting for Training Data"
echo "=================================================="
echo ""

# Check if pipeline is running
if [ -f pipeline.pid ]; then
    PID=$(cat pipeline.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Pipeline running (PID: $PID)"
    else
        echo "⚠️  Pipeline not running (may have finished or crashed)"
    fi
else
    echo "⚠️  No pipeline.pid found"
fi

echo ""
echo "📊 Starting continuous monitoring..."
echo "Press Ctrl+C to stop"
echo ""

# Run monitor script
python3 monitor_pipeline.py

