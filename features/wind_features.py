"""
Wind feature engineering (u, v components).
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class WindFeatureEngineer:
    """Engineer wind features (u, v components)."""
    
    @staticmethod
    def encode_wind(
        df: pd.DataFrame,
        speed_col: str = "wind_speed",
        direction_col: str = "wind_direction"
    ) -> pd.DataFrame:
        """
        Encode wind speed and direction to u, v components.
        
        Formula:
            u = -speed * sin(direction_rad)
            v = -speed * cos(direction_rad)
        
        Args:
            df: Input DataFrame
            speed_col: Wind speed column name
            direction_col: Wind direction column name (degrees)
            
        Returns:
            DataFrame with u_wind and v_wind columns
        """
        df = df.copy()
        
        if speed_col not in df.columns or direction_col not in df.columns:
            logger.warning("Wind columns not found, skipping wind encoding")
            return df
        
        # Convert direction from degrees to radians
        direction_rad = np.radians(df[direction_col])
        
        # Calculate u, v components
        # Note: Negative sign because meteorological convention
        df["u_wind"] = -df[speed_col] * np.sin(direction_rad)
        df["v_wind"] = -df[speed_col] * np.cos(direction_rad)
        
        logger.info("Added wind features: u_wind, v_wind")
        return df

