# PM2.5 Forecasting System for Thailand

Production-ready AI forecasting system for PM2.5 prediction in Thailand, supporting multiple model architectures.

## Features

- **Multiple Model Architectures**:
  - Station-based LSTM
  - Grid-based ConvLSTM
  - Grid-based ST-UNN (Spatio-Temporal UNet)

- **Complete Data Pipeline**:
  - PM2.5 data from Air4Thai
  - Weather data from Open-Meteo (batch optimized)
  - Fire hotspots from NASA FIRMS
  - Static features (Land Use, Population)

- **Production-Ready**:
  - Async API calls
  - Retry logic with rate limiting
  - Data validation
  - Comprehensive error handling
  - GPU support (CUDA/ROCm)

## Installation

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# For AMD GPU (ROCm)
# See install.sh for ROCm installation
```

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For AMD GPU (ROCm)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

## Configuration

Edit `config.yaml` to configure:

- Data sources and API settings
- Grid size (16×16, 32×32, 64×64)
- Model architecture and hyperparameters
- Training parameters
- Evaluation metrics

## Usage

### 1. Data Collection

```bash
# Run data pipeline
python pipeline.py

# Or use Jupyter notebook
jupyter notebook pipline.ipynb
```

### 2. Training

```bash
# Train ST-UNN model (default)
python train.py --config config.yaml --model-type st_unn

# Train LSTM model
python train.py --config config.yaml --model-type lstm

# Train ConvLSTM model
python train.py --config config.yaml --model-type conv_lstm
```

### 3. Inference

```python
from inference.forecaster import Forecaster
import torch

# Load model
forecaster = Forecaster(
    model_path="data/models/checkpoints/best.pt",
    model_type="st_unn",
    config=config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Generate forecast
forecast = forecaster.forecast(input_data, feature_cols)
```

## Project Structure

```
pm2.5 forcasting/
├── config.py              # Python configuration
├── config.yaml            # YAML configuration
├── pipeline.py            # Data collection pipeline
├── train.py               # Training script
├── requirements.txt       # Python dependencies
├── install.sh            # System setup script
│
├── data_collectors/      # Data collection modules
│   ├── pm25_collector.py
│   ├── weather_collector.py
│   ├── fire_collector.py
│   └── static_collector.py
│
├── features/             # Feature engineering
│   ├── wind_features.py
│   ├── time_features.py
│   └── data_cleaner.py
│
├── preprocessing/        # Data preprocessing
│   ├── normalizer.py
│   └── grid_interpolation.py
│
├── models/              # Model architectures
│   ├── lstm_model.py
│   ├── conv_lstm_model.py
│   └── st_unn_model.py
│
├── training/            # Training utilities
│   ├── trainer.py
│   └── data_loader.py
│
├── evaluation/          # Evaluation metrics
│   ├── metrics.py
│   └── visualization.py
│
├── inference/           # Inference pipeline
│   └── forecaster.py
│
└── utils/               # Utilities
    ├── sliding_window.py
    └── tensor_builder.py
```

## Model Architectures

### LSTM (Station-based)
- Input: (batch, 24, features)
- Output: (batch, 6)
- Architecture: LSTM(64) → Dropout → Dense(32) → Dense(6)

### ConvLSTM (Grid-based)
- Input: (batch, 24, channels, H, W)
- Output: (batch, 6, 1, H, W)
- Architecture: ConvLSTM2D(32) → BatchNorm → ConvLSTM2D(64) → Conv3D(1)

### ST-UNN (Grid-based)
- Input: (batch, 24, channels, H, W)
- Output: (batch, 6, 1, H, W)
- Architecture: Encoder → Bottleneck → Decoder with skip connections
- Loss: 0.7×MSE + 0.3×MAE

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **Pixel MAE**: Pixel-wise MAE (grid models)
- **SSIM**: Structural Similarity Index (grid models)

## Data Format

### Input Data Structure
- Parquet format, partitioned by year/month/station_id
- Columns: timestamp, station_id, lat, lon, pm25, temperature, humidity, u_wind, v_wind, pressure, rain, solar, fire_count, land_use, population_density, hour_sin, hour_cos, day_of_week, month

### Output Forecast
- Parquet format with forecast timestamps
- GeoJSON export for grid-based models

## Performance

- **API Rate Limiting**: Automatic retry with exponential backoff
- **Batch Processing**: Optimized Open-Meteo calls (10 locations per batch)
- **Memory Efficient**: Streaming data processing
- **GPU Acceleration**: CUDA/ROCm support with mixed precision training

## Troubleshooting

### Rate Limiting
If you encounter 429 errors:
1. Reduce `BATCH_SIZE` in config.yaml
2. Increase `REQUEST_DELAY_SECONDS`
3. Wait 1-2 hours before retrying

### GPU Issues
For AMD GPUs (ROCm):
```bash
# Install ROCm-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

### Data Collection
Check data status:
```bash
python check_data.py
```

## License

MIT License

## Contributing

1. Follow the engineering standards in the project rules
2. Ensure all code is modular and testable
3. Add docstrings to all public functions
4. Run linter before committing

## Citation

If you use this system in your research, please cite:

```bibtex
@software{pm25_forecasting,
  title={PM2.5 Forecasting System for Thailand},
  author={Your Name},
  year={2024}
}
```
