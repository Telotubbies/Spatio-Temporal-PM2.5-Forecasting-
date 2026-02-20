#!/bin/bash
# Installation script for PM2.5 Forecasting Pipeline
# Optimized for AMD 7800XT with ROCm support

set -e

echo "🚀 PM2.5 Forecasting Pipeline Installation"
echo "=========================================="

# Check Python version
echo "📦 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with ROCm (for AMD 7800XT)
echo "📦 Installing PyTorch with ROCm support..."
echo "   This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify PyTorch installation
echo "📦 Verifying PyTorch installation..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   ROCm available: {torch.cuda.is_available()}')" || echo "   ⚠️  Warning: ROCm may not be available"

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{raw/{pm25,weather,fire},processed/{station_level,grid_level},features,tensors,models,logs}

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "   python pipeline.py"
echo ""
echo "Or use the Jupyter notebook:"
echo "   jupyter notebook pipline.ipynb"

