"""
Configuration module for PM2.5 forecasting pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta


@dataclass
class DataConfig:
    """Data source configuration."""
    # Bangkok bounding box (approximate)
    BANGKOK_BBOX: Tuple[float, float, float, float] = (100.3, 13.5, 100.8, 13.9)  # lon_min, lat_min, lon_max, lat_max
    
    # API endpoints
    AIR4THAI_API: str = "https://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
    OPEN_METEO_API: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HISTORICAL_API: str = "https://archive-api.open-meteo.com/v1/archive"
    FIRMS_API: str = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
    
    # Data collection
    HISTORICAL_DAYS: int = 365  # Days to fetch (default)
    HISTORICAL_START_YEAR: int = 2010  # Start year for historical data
    BATCH_SIZE: int = 10  # Stations per Open-Meteo batch call (reduced to avoid rate limit)
    CHUNK_SIZE_DAYS: int = 365  # Fetch data in chunks to avoid large API calls
    REQUEST_DELAY_SECONDS: float = 3.0  # Delay between API requests to avoid rate limiting
    
    # Time features
    TIMEZONE: str = "Asia/Bangkok"
    UTC_OFFSET: int = 7


@dataclass
class StorageConfig:
    """Storage configuration."""
    BASE_DIR: Path = Path("data")
    RAW_DIR: Path = BASE_DIR / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    FEATURES_DIR: Path = BASE_DIR / "features"
    TENSORS_DIR: Path = BASE_DIR / "tensors"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Parquet settings
    PARQUET_ENGINE: str = "pyarrow"
    COMPRESSION: str = "snappy"
    
    # Partitioning
    PARTITION_COLS: List[str] = None
    
    def __post_init__(self):
        """Initialize partition columns."""
        if self.PARTITION_COLS is None:
            self.PARTITION_COLS = ["year", "month", "station_id"]


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Wind encoding
    WIND_ENCODE: bool = True
    
    # Time features
    TIME_FEATURES: bool = True
    
    # Missing value handling
    MAX_MISSING_PCT: float = 0.20  # Remove station if >20% missing
    INTERPOLATION_LIMIT_HOURS: int = 3  # Linear interpolation for <3hr gaps
    
    # Outlier removal
    USE_IQR: bool = True
    Z_SCORE_THRESHOLD: float = 3.0
    
    # Fire aggregation
    FIRE_RADIUS_KM: float = 25.0  # 25km radius for fire count


@dataclass
class ModelConfig:
    """Model configuration for ST-UNN."""
    INPUT_HOURS: int = 24
    OUTPUT_HOURS: int = 6
    
    # Grid settings (for future grid interpolation)
    GRID_SIZE: Tuple[int, int] = (32, 32)  # H, W
    BANGKOK_GRID_BBOX: Tuple[float, float, float, float] = (100.3, 13.5, 100.8, 13.9)
    
    # Training
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    data: DataConfig = None
    storage: StorageConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("logs")
    
    def __post_init__(self):
        """Initialize sub-configs."""
        if self.data is None:
            self.data = DataConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        
        # Create directories
        self.storage.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.TENSORS_DIR.mkdir(parents=True, exist_ok=True)
        self.storage.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

