"""
Inference pipeline for PM2.5 forecasting.
"""
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import json

from models import LSTMPredictor, ConvLSTMPredictor, STUNNPredictor
from preprocessing.normalizer import DataNormalizer
from preprocessing.grid_interpolation import GridInterpolator
from utils.tensor_builder import TensorBuilder

logger = logging.getLogger(__name__)


class Forecaster:
    """Forecast PM2.5 using trained model."""
    
    def __init__(
        self,
        model_path: Path,
        model_type: str,
        config: dict,
        device: torch.device
    ):
        """
        Initialize forecaster.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Model type (lstm, conv_lstm, st_unn)
            config: Configuration dictionary
            device: Device (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.config = config
        self.device = device
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load scaler if available
        self.scaler = None
        scaler_path = self.model_path.parent / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = DataNormalizer()
            self.scaler.load(scaler_path)
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        
        if self.model_type == "lstm":
            model = LSTMPredictor(
                input_features=model_config.get('input_features', 10),
                output_hours=model_config.get('output_hours', 6),
                hidden_size=model_config.get('hidden_size', 64),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2)
            )
        elif self.model_type == "conv_lstm":
            model = ConvLSTMPredictor(
                input_channels=model_config.get('input_channels', 10),
                output_hours=model_config.get('output_hours', 6),
                filters=model_config.get('filters', [32, 64]),
                grid_size=model_config.get('grid_size', (32, 32))
            )
        elif self.model_type == "st_unn":
            model = STUNNPredictor(
                input_channels=model_config.get('input_channels', 10),
                output_hours=model_config.get('output_hours', 6),
                encoder_filters=model_config.get('encoder_filters', 32),
                bottleneck_filters=model_config.get('bottleneck_filters', 64),
                grid_size=model_config.get('grid_size', (32, 32))
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        logger.info(f"Loaded {self.model_type} model from {self.model_path}")
        return model
    
    def forecast(
        self,
        input_data: pd.DataFrame,
        feature_cols: list
    ) -> np.ndarray:
        """
        Generate forecast.
        
        Args:
            input_data: Input DataFrame (last 24 hours)
            feature_cols: Feature columns
            
        Returns:
            Forecast array
        """
        # Normalize if scaler available
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        
        # Build input tensor
        if self.model_type == "lstm":
            # Station-based: aggregate or use last station
            X = self._build_lstm_input(input_data, feature_cols)
        else:
            # Grid-based
            X = self._build_grid_input(input_data, feature_cols)
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # Add batch dimension
            pred = self.model(X_tensor)
            pred = pred.cpu().numpy().squeeze(0)  # Remove batch dimension
        
        # Inverse transform if scaler available
        if self.scaler and 'pm25' in self.scaler.feature_cols:
            # Inverse transform PM2.5 predictions
            pm25_idx = self.scaler.feature_cols.index('pm25')
            # This is simplified - full implementation would need proper inverse transform
            pass
        
        return pred
    
    def _build_lstm_input(
        self,
        df: pd.DataFrame,
        feature_cols: list
    ) -> np.ndarray:
        """Build LSTM input tensor."""
        # Sort by timestamp
        df = df.sort_values('timestamp').tail(24)  # Last 24 hours
        
        # Aggregate by averaging stations (or use specific station)
        X = df[feature_cols].values  # (time, features)
        
        # If multiple stations, average
        if 'station_id' in df.columns:
            X = df.groupby('timestamp')[feature_cols].mean().values
        
        return X  # (24, features)
    
    def _build_grid_input(
        self,
        df: pd.DataFrame,
        feature_cols: list
    ) -> np.ndarray:
        """Build grid input tensor."""
        # Create grid interpolator
        bbox = self.config.get('bbox', (100.3, 13.5, 100.8, 13.9))
        grid_size = self.config.get('grid_size', (32, 32))
        
        interpolator = GridInterpolator(
            bbox=bbox,
            grid_size=grid_size,
            method=self.config.get('interpolation_method', 'idw')
        )
        
        # Get last 24 timestamps
        timestamps = sorted(df['timestamp'].unique())[-24:]
        
        grids = []
        for ts in timestamps:
            grid = interpolator.create_grid_tensor(
                df=df,
                feature_cols=feature_cols,
                timestamp=pd.Timestamp(ts)
            )
            grids.append(grid)
        
        X = np.stack(grids, axis=0)  # (24, H, W, channels)
        return X
    
    def save_forecast(
        self,
        forecast: np.ndarray,
        timestamps: list,
        output_dir: Path
    ):
        """
        Save forecast to Parquet and GeoJSON.
        
        Args:
            forecast: Forecast array
            timestamps: List of forecast timestamps
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet (simplified - would need proper structure)
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'pm25_forecast': forecast.flatten() if len(forecast.shape) > 1 else forecast
        })
        
        parquet_path = output_dir / f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        forecast_df.to_parquet(parquet_path)
        logger.info(f"Saved forecast to {parquet_path}")
        
        # Export GeoJSON if grid-based
        if self.model_type in ["conv_lstm", "st_unn"]:
            self._export_geojson(forecast, timestamps, output_dir)
    
    def _export_geojson(
        self,
        forecast: np.ndarray,
        timestamps: list,
        output_dir: Path
    ):
        """Export forecast as GeoJSON grid."""
        # This is a simplified version
        # Full implementation would create proper GeoJSON with grid cells
        
        bbox = self.config.get('bbox', (100.3, 13.5, 100.8, 13.9))
        grid_size = self.config.get('grid_size', (32, 32))
        
        lon_min, lat_min, lon_max, lat_max = bbox
        lon_grid = np.linspace(lon_min, lon_max, grid_size[1])
        lat_grid = np.linspace(lat_max, lat_min, grid_size[0])
        
        features = []
        
        # Create grid cells
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_lon = lon_grid[j]
                cell_lat = lat_grid[i]
                
                # Get forecast value for this cell
                if len(forecast.shape) == 3:
                    value = forecast[0, i, j]  # First timestep
                else:
                    value = forecast[i, j]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [cell_lon, cell_lat]
                    },
                    "properties": {
                        "pm25": float(value),
                        "timestamp": timestamps[0] if timestamps else None
                    }
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        geojson_path = output_dir / f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Exported GeoJSON to {geojson_path}")

