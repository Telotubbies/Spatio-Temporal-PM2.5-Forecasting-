"""
Time feature engineering.
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeFeatureEngineer:
    """Engineer time-based features."""
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Add time-based features to DataFrame.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with added time features
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Column {timestamp_col} not found")
            return df
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Hour features (cyclic encoding)
        hour = df[timestamp_col].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        
        # Month (1-12)
        df["month"] = df[timestamp_col].dt.month
        
        logger.info("Added time features: hour_sin, hour_cos, day_of_week, month")
        return df

