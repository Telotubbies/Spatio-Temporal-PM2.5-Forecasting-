# 🚀 Quick Start Guide

## Installation (AMD 7800XT)

### Option 1: Automated Installation

```bash
cd "/home/a/Desktop/pm2.5 forcasting"
bash install.sh
source venv/bin/activate
```

### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with ROCm (for AMD 7800XT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}')"
```

## Run Pipeline

### Method 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook pipline.ipynb
```

Then run all cells.

### Method 2: Python Script

```bash
python pipeline.py
```

### Method 3: Python Interactive

```python
from pipeline import PM25Pipeline
from config import PipelineConfig
from datetime import datetime, timedelta

# Initialize
config = PipelineConfig()
pipeline = PM25Pipeline(config)

# Run (last 365 days)
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=365)

result = pipeline.run(start_date=start_date, end_date=end_date)
```

## Expected Output

```
🚀 PM2.5 Forecasting Data Pipeline
============================================================
[Step 1] Fetching PM2.5 stations...
Found X stations in Bangkok area
[Step 2] Fetching weather data...
[Step 3] Fetching fire data...
[Step 4] Extracting static features...
[Step 5] Merging data sources...
[Step 6] Engineering features...
[Step 7] Cleaning data...
[Step 8] Saving processed data...
============================================================
Pipeline completed successfully!
Final dataset size: X records
Stations: Y
============================================================
```

## Data Location

Processed data will be saved to:
```
data/processed/station_level/year=YYYY/month=MM/station_id=XXX.parquet
```

## Next Steps

1. **Inspect data**: Check `data/processed/` directory
2. **Create sequences**: Use `utils.sliding_window.create_sliding_window()`
3. **Train ST-UNN**: Implement model training (next phase)

## Troubleshooting

### ROCm Not Available

```bash
# Check ROCm installation
rocminfo

# If not installed, install ROCm:
# Ubuntu: sudo apt install rocm-dev rocm-libs
```

### API Errors

- **Air4Thai**: Usually no issues, free API
- **Open-Meteo**: Free tier, may have rate limits
- **FIRMS**: May require API key (currently placeholder)

### Memory Issues

Reduce batch size in `config.py`:
```python
DataConfig.BATCH_SIZE = 25  # Default: 50
```

## Support

See `README.md` for full documentation.

