# Project Structure

## Directory Organization

```
pm2.5 forcasting/
├── README.md                 # Main documentation
├── config.py                 # Python configuration
├── config.yaml               # YAML configuration
├── requirements.txt          # Python dependencies
├── install.sh               # Installation script
│
├── data_collectors/         # Data collection modules
│   ├── __init__.py
│   ├── pm25_collector.py    # Air4Thai PM2.5 data
│   ├── weather_collector.py  # Open-Meteo weather data
│   ├── fire_collector.py     # NASA FIRMS fire data
│   └── static_collector.py   # WorldCover/WorldPop static features
│
├── features/                # Feature engineering
│   ├── __init__.py
│   ├── wind_features.py     # Wind encoding (u, v)
│   ├── time_features.py     # Time features (hour_sin, hour_cos, etc.)
│   └── data_cleaner.py      # Data cleaning and outlier removal
│
├── preprocessing/           # Data preprocessing
│   ├── __init__.py
│   ├── normalizer.py        # Data normalization (StandardScaler, etc.)
│   └── grid_interpolation.py # Station to grid interpolation (IDW/Kriging)
│
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── lstm_model.py        # Station-based LSTM
│   ├── conv_lstm_model.py   # Grid-based ConvLSTM
│   ├── st_unn_model.py      # Grid-based ST-UNN
│   ├── attention.py         # Attention mechanisms
│   ├── ensemble.py          # Ensemble models
│   └── mc_dropout.py        # Monte Carlo Dropout
│
├── training/                # Training utilities
│   ├── __init__.py
│   ├── trainer.py           # Training pipeline
│   └── data_loader.py       # Data loaders for PyTorch
│
├── evaluation/              # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics (MAE, RMSE, R², etc.)
│   └── visualization.py    # Visualization and reporting
│
├── inference/               # Inference pipeline
│   ├── __init__.py
│   └── forecaster.py       # Forecast generation
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── logger.py           # Logging utilities
│   ├── sliding_window.py   # Sliding window creation
│   └── tensor_builder.py   # Tensor building for different models
│
├── core/                    # Core utilities
│   ├── __init__.py
│   ├── exceptions.py       # Custom exceptions
│   └── validators.py       # Data validation
│
├── docs/                    # Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── HISTORICAL_DATA.md
│   └── ...
│
├── scripts/                 # Utility scripts
│   ├── push_to_github.sh
│   ├── prepare_push.sh
│   └── ...
│
├── data/                    # Data storage (gitignored)
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── features/           # Feature data
│   ├── tensors/            # Tensor data
│   └── models/             # Trained models
│
├── logs/                    # Log files (gitignored)
│
├── pipeline.py             # Main data pipeline
├── train.py                # Training script
├── run_inference.py        # Inference script
├── hyperparameter_tuning.py # Optuna hyperparameter tuning
├── check_data.py           # Data checking utility
├── monitor_pipeline.py     # Pipeline monitoring
└── pipline.ipynb          # Jupyter notebook for pipeline
```

## File Categories

### Core Files
- `config.py`, `config.yaml`: Configuration
- `pipeline.py`: Data collection pipeline
- `train.py`: Model training
- `run_inference.py`: Inference

### Models
- `models/lstm_model.py`: Station-based LSTM
- `models/conv_lstm_model.py`: Grid-based ConvLSTM
- `models/st_unn_model.py`: Grid-based ST-UNN
- `models/attention.py`: Attention mechanisms
- `models/ensemble.py`: Ensemble models
- `models/mc_dropout.py`: Uncertainty estimation

### Data Processing
- `data_collectors/`: API clients
- `features/`: Feature engineering
- `preprocessing/`: Data preprocessing
- `utils/`: Utility functions

### Training & Evaluation
- `training/`: Training pipeline
- `evaluation/`: Metrics and visualization

### Documentation
- `README.md`: Main documentation
- `docs/`: Additional documentation

### Scripts
- `scripts/`: Utility scripts

## Ignored Files

The following are ignored by git (see `.gitignore`):
- `venv/`: Virtual environment
- `data/`: Data files
- `logs/`: Log files
- `*.pid`: Process ID files
- `*.log`: Log files
- `*.parquet`: Parquet data files
- `*.pt`, `*.pth`: Model checkpoints
- Status and guide markdown files

