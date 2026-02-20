"""
Tensor builder for different model types (LSTM, ConvLSTM, ST-UNN).
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path

from preprocessing.grid_interpolation import GridInterpolator
from utils.sliding_window import create_sliding_window

logger = logging.getLogger(__name__)


class TensorBuilder:
    """Build tensors for different model types."""
    
    def __init__(
        self,
        model_type: str,
        input_hours: int = 24,
        output_hours: int = 6,
        grid_size: Tuple[int, int] = (32, 32),
        bbox: Tuple[float, float, float, float] = None,
        interpolation_method: str = "idw"
    ):
        """
        Initialize tensor builder.
        
        Args:
            model_type: Model type (lstm, conv_lstm, st_unn)
            input_hours: Input sequence length
            output_hours: Output sequence length
            grid_size: Grid dimensions (H, W)
            bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
            interpolation_method: Interpolation method (idw, kriging)
        """
        self.model_type = model_type
        self.input_hours = input_hours
        self.output_hours = output_hours
        self.grid_size = grid_size
        
        # Grid interpolator (for grid-based models)
        if model_type in ["conv_lstm", "st_unn"]:
            if bbox is None:
                raise ValueError("bbox required for grid-based models")
            self.interpolator = GridInterpolator(
                bbox=bbox,
                grid_size=grid_size,
                method=interpolation_method
            )
        else:
            self.interpolator = None
    
    def build_tensors(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "pm25"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build tensors for training.
        
        Args:
            df: Input DataFrame
            feature_cols: Feature columns
            target_col: Target column
            
        Returns:
            Tuple of (X, y) tensors
        """
        if self.model_type == "lstm":
            return self._build_lstm_tensors(df, feature_cols, target_col)
        else:
            return self._build_grid_tensors(df, feature_cols, target_col)
    
    def _build_lstm_tensors(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build tensors for LSTM (station-based).
        
        Returns:
            X: (batch, input_hours, features)
            y: (batch, output_hours)
        """
        X, y = create_sliding_window(
            df=df,
            input_hours=self.input_hours,
            output_hours=self.output_hours,
            feature_cols=feature_cols,
            target_col=target_col
        )
        
        # Reshape y from (batch, output_hours, 1) to (batch, output_hours)
        if len(y.shape) == 3 and y.shape[2] == 1:
            y = y.squeeze(axis=2)
        
        logger.info(f"Built LSTM tensors: X={X.shape}, y={y.shape}")
        return X, y
    
    def _build_grid_tensors(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build tensors for grid-based models (ConvLSTM, ST-UNN).
        
        Returns:
            X: (batch, input_hours, H, W, channels)
            y: (batch, output_hours, H, W, 1)
        """
        # Sort by timestamp
        df = df.sort_values(["timestamp", "station_id"]).copy()
        
        # Get unique timestamps
        timestamps = df['timestamp'].unique()
        timestamps = sorted(timestamps)
        
        if len(timestamps) < self.input_hours + self.output_hours:
            logger.warning(f"Insufficient timestamps: {len(timestamps)} < {self.input_hours + self.output_hours}")
            return np.array([]), np.array([])
        
        sequences_X = []
        sequences_y = []
        
        # Create sliding windows
        for i in range(len(timestamps) - self.input_hours - self.output_hours + 1):
            input_timestamps = timestamps[i:i + self.input_hours]
            output_timestamps = timestamps[i + self.input_hours:i + self.input_hours + self.output_hours]
            
            # Build input grid sequence
            input_grids = []
            for ts in input_timestamps:
                grid = self.interpolator.create_grid_tensor(
                    df=df,
                    feature_cols=feature_cols,
                    timestamp=pd.Timestamp(ts)
                )
                input_grids.append(grid)
            
            # Build output grid sequence
            output_grids = []
            for ts in output_timestamps:
                grid = self.interpolator.create_grid_tensor(
                    df=df,
                    feature_cols=[target_col],
                    timestamp=pd.Timestamp(ts)
                )
                # Take only first channel (PM2.5)
                if grid.shape[2] > 0:
                    output_grids.append(grid[:, :, 0:1])  # (H, W, 1)
                else:
                    output_grids.append(np.zeros((*self.grid_size, 1)))
            
            # Stack sequences
            X_seq = np.stack(input_grids, axis=0)  # (input_hours, H, W, channels)
            y_seq = np.stack(output_grids, axis=0)  # (output_hours, H, W, 1)
            
            sequences_X.append(X_seq)
            sequences_y.append(y_seq)
        
        if not sequences_X:
            logger.warning("No sequences created")
            return np.array([]), np.array([])
        
        X = np.array(sequences_X)  # (batch, input_hours, H, W, channels)
        y = np.array(sequences_y)  # (batch, output_hours, H, W, 1)
        
        logger.info(f"Built grid tensors: X={X.shape}, y={y.shape}")
        return X, y

