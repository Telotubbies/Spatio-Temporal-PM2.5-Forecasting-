"""
Run inference on new data.
"""
import logging
import yaml
import torch
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta

from inference.forecaster import Forecaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_latest_data(data_dir: Path, hours: int = 24) -> pd.DataFrame:
    """Load latest N hours of data."""
    processed_dir = Path(data_dir) / "processed"
    
    # Load all parquet files
    parquet_files = list(processed_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {processed_dir}")
    
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
    
    if not dfs:
        raise ValueError("No data loaded")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter to last N hours
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_time = df['timestamp'].max()
        cutoff_time = latest_time - timedelta(hours=hours)
        df = df[df['timestamp'] >= cutoff_time]
    
    logger.info(f"Loaded {len(df)} records from last {hours} hours")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run PM2.5 forecast inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model-type', type=str, choices=['lstm', 'conv_lstm', 'st_unn'],
                       default='st_unn', help='Model type')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--hours', type=int, default=24, help='Hours of input data')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load latest data
    df = load_latest_data(args.data_dir, hours=args.hours)
    
    # Feature columns
    exclude_cols = ["timestamp", "station_id", "lat", "lon", "year", "month"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != "pm25"]
    
    # Create forecaster
    forecaster = Forecaster(
        model_path=Path(args.model_path),
        model_type=args.model_type,
        config=config,
        device=device
    )
    
    # Generate forecast
    logger.info("Generating forecast...")
    forecast = forecaster.forecast(df, feature_cols)
    
    # Generate forecast timestamps
    if 'timestamp' in df.columns:
        latest_time = df['timestamp'].max()
        forecast_timestamps = [
            latest_time + timedelta(hours=i+1)
            for i in range(len(forecast))
        ]
    else:
        forecast_timestamps = [
            datetime.now() + timedelta(hours=i+1)
            for i in range(len(forecast))
        ]
    
    # Save forecast
    output_dir = Path(config.get('inference', {}).get('forecast_dir', 'data/forecasts'))
    forecaster.save_forecast(forecast, forecast_timestamps, output_dir)
    
    logger.info("Forecast completed!")
    logger.info(f"Forecast shape: {forecast.shape}")
    logger.info(f"Forecast saved to {output_dir}")


if __name__ == "__main__":
    main()

