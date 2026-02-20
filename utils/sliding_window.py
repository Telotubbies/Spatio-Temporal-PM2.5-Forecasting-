"""
Sliding window creation for time series data.
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def create_sliding_window(
    df: pd.DataFrame,
    input_hours: int = 24,
    output_hours: int = 6,
    feature_cols: list = None,
    target_col: str = "pm25",
    timestamp_col: str = "timestamp",
    station_col: str = "station_id"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for ST-UNN model.
    
    Args:
        df: Input DataFrame with time series data
        input_hours: Number of input hours
        output_hours: Number of output hours
        feature_cols: List of feature column names
        target_col: Target column name
        timestamp_col: Timestamp column name
        station_col: Station ID column name
        
    Returns:
        Tuple of (X, y) arrays
        X shape: (batch, input_hours, features)
        y shape: (batch, output_hours, 1)
    """
    logger.info(f"Creating sliding windows: input={input_hours}h, output={output_hours}h")
    
    # Sort by station and timestamp
    df = df.sort_values([station_col, timestamp_col]).copy()
    
    # Default feature columns (exclude metadata)
    if feature_cols is None:
        exclude_cols = [timestamp_col, station_col, "lat", "lon", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group by station
    sequences_X = []
    sequences_y = []
    
    for station_id, group in df.groupby(station_col):
        group = group.sort_values(timestamp_col)
        
        # Ensure hourly data (no gaps)
        group = group.set_index(timestamp_col).resample("H").first().reset_index()
        
        if len(group) < input_hours + output_hours:
            continue
        
        # Create sliding windows
        for i in range(len(group) - input_hours - output_hours + 1):
            input_window = group.iloc[i:i + input_hours]
            output_window = group.iloc[i + input_hours:i + input_hours + output_hours]
            
            # Extract features
            X_seq = input_window[feature_cols].values
            y_seq = output_window[[target_col]].values
            
            sequences_X.append(X_seq)
            sequences_y.append(y_seq)
    
    if not sequences_X:
        logger.warning("No sequences created")
        return np.array([]), np.array([])
    
    X = np.array(sequences_X)
    y = np.array(sequences_y)
    
    logger.info(f"Created {len(sequences_X)} sequences")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

