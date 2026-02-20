# PM2.5 Forecasting Data Pipeline

Production-ready data pipeline for PM2.5 forecasting in Bangkok using ST-UNN model.

## 🎯 Features

- **Multi-source data collection**: Air4Thai (PM2.5), Open-Meteo (Weather), NASA FIRMS (Fire), WorldCover (Land Use), WorldPop (Population)
- **Optimized API calls**: Batch processing to minimize Open-Meteo requests
- **Feature engineering**: Wind encoding (u, v components), time features, data cleaning
- **Parquet storage**: Partitioned by year/month/station_id
- **Ready for ST-UNN**: Sliding window creation for spatio-temporal model
- **AMD GPU support**: Configured for ROCm (7800XT)

## 📦 Installation

### 1. Install System Dependencies (ROCm for AMD GPU)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y rocm-dev rocm-libs rocblas rocfft rocrand rocsparse rocthrust

# Verify ROCm installation
rocminfo
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm support
# Check https://pytorch.org/get-started/locally/ for latest version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}')"
```

## 🚀 Usage

### Command Line

```bash
python pipeline.py
```

### Python Script

```python
from pipeline import PM25Pipeline
from config import PipelineConfig
from datetime import datetime, timedelta

config = PipelineConfig()
pipeline = PM25Pipeline(config)

# Run pipeline
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=365)

result = pipeline.run(start_date=start_date, end_date=end_date)
```

### Jupyter Notebook

See `pipline.ipynb` for interactive usage.

## 📁 Project Structure

```
pm2.5 forcasting/
├── config.py                 # Configuration dataclasses
├── pipeline.py               # Main pipeline orchestrator
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data_collectors/          # Data source collectors
│   ├── __init__.py
│   ├── pm25_collector.py     # Air4Thai API
│   ├── weather_collector.py  # Open-Meteo API (batch optimized)
│   ├── fire_collector.py     # NASA FIRMS
│   └── static_collector.py   # WorldCover, WorldPop
│
├── features/                 # Feature engineering
│   ├── __init__.py
│   ├── time_features.py      # Time encoding (hour_sin, hour_cos, etc.)
│   ├── wind_features.py      # Wind u, v components
│   └── data_cleaner.py       # Missing values, outliers
│
├── utils/                    # Utilities
│   ├── __init__.py
│   ├── sliding_window.py    # Create sequences for ST-UNN
│   └── logger.py             # Logging setup
│
└── data/                     # Data storage (created automatically)
    ├── raw/                  # Raw data
    ├── processed/            # Processed data (partitioned)
    ├── features/             # Feature datasets
    ├── tensors/              # Training tensors
    └── models/               # Saved models
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Bangkok bounding box**: `DataConfig.BANGKOK_BBOX`
- **Historical days**: `DataConfig.HISTORICAL_DAYS`
- **Batch size**: `DataConfig.BATCH_SIZE` (for Open-Meteo)
- **Missing value threshold**: `FeatureConfig.MAX_MISSING_PCT`
- **Model parameters**: `ModelConfig` (input/output hours, grid size)

## 📊 Data Flow

1. **PM2.5 Stations** → Air4Thai API → Filter Bangkok area
2. **Weather Data** → Open-Meteo API (batch calls) → Merge by timestamp
3. **Fire Data** → NASA FIRMS → Aggregate by 25km radius
4. **Static Features** → WorldCover/WorldPop rasters → Extract per station
5. **Feature Engineering** → Wind encoding, time features
6. **Data Cleaning** → Missing value handling, outlier removal
7. **Storage** → Parquet partitioned by year/month/station_id

## 🧠 ST-UNN Preparation

The pipeline prepares data for ST-UNN model:

```python
from utils.sliding_window import create_sliding_window

# Load processed data
df = pd.read_parquet("data/processed/station_level/...")

# Create sequences
X, y = create_sliding_window(
    df,
    input_hours=24,
    output_hours=6,
    feature_cols=["temperature", "humidity", "u_wind", "v_wind", ...],
    target_col="pm25"
)

# X shape: (batch, 24, features)
# y shape: (batch, 6, 1)
```

## 🐛 Troubleshooting

### ROCm Issues

```bash
# Check ROCm installation
rocminfo

# Check PyTorch ROCm support
python -c "import torch; print(torch.version.hip)"
```

### API Rate Limits

- Open-Meteo: Free tier allows reasonable usage
- Air4Thai: No authentication required
- FIRMS: May require API key for production

### Memory Issues

- Reduce `BATCH_SIZE` in `DataConfig`
- Process data in chunks
- Use Polars instead of Pandas for large datasets

## 📝 Notes

- **Bangkok focus**: Currently configured for Bangkok area only
- **Historical data**: Air4Thai API may not support historical queries directly
- **Fire data**: FIRMS integration is placeholder (implement actual API call)
- **Static features**: WorldCover/WorldPop require raster files (not included)

## 🔄 Next Steps

1. Implement actual FIRMS API integration
2. Add grid interpolation (station → 32x32 grid)
3. Implement ST-UNN model training
4. Add evaluation metrics
5. Deploy as FastAPI service

## 📄 License

MIT License

## 👤 Author

Senior Data Engineer + ML Engineer

