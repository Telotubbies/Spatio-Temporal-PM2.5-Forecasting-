"""
Data validation utilities.
"""
import logging
from typing import Optional, List
import pandas as pd
import numpy as np

from .exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data at pipeline boundaries."""
    
    @staticmethod
    def validate_stations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate station data.
        
        Args:
            df: Station DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            raise DataValidationError("Station DataFrame is empty")
        
        required_cols = ["station_id", "lat", "lon"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            )
        
        # Validate coordinates
        if df["lat"].isna().any() or df["lon"].isna().any():
            raise DataValidationError("Missing lat/lon coordinates")
        
        # Validate coordinate ranges (Bangkok area)
        if not ((13.0 <= df["lat"]).all() and (df["lat"] <= 14.5).all()):
            raise DataValidationError("Latitude out of Bangkok range")
        
        if not ((100.0 <= df["lon"]).all() and (df["lon"] <= 101.0).all()):
            raise DataValidationError("Longitude out of Bangkok range")
        
        logger.info(f"Validated {len(df)} stations")
        return df
    
    @staticmethod
    def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate weather data.
        
        Args:
            df: Weather DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            logger.warning("Weather DataFrame is empty")
            return df
        
        required_cols = ["timestamp", "lat", "lon"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            )
        
        # Validate timestamp
        if df["timestamp"].isna().any():
            raise DataValidationError("Missing timestamps")
        
        logger.info(f"Validated {len(df)} weather records")
        return df
    
    @staticmethod
    def validate_merged_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate merged dataset.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            raise DataValidationError("Merged DataFrame is empty")
        
        required_cols = ["timestamp", "station_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            )
        
        # Check for duplicate records
        duplicates = df.duplicated(subset=["station_id", "timestamp"]).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate records, removing...")
            df = df.drop_duplicates(subset=["station_id", "timestamp"])
        
        logger.info(f"Validated merged dataset: {len(df)} records")
        return df
    
    @staticmethod
    def validate_numeric_range(
        series: pd.Series,
        col_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> pd.Series:
        """
        Validate numeric values are within range.
        
        Args:
            series: Series to validate
            col_name: Column name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated Series
            
        Raises:
            DataValidationError: If values out of range
        """
        if min_val is not None and (series < min_val).any():
            invalid = (series < min_val).sum()
            raise DataValidationError(
                f"{col_name}: {invalid} values below minimum {min_val}"
            )
        
        if max_val is not None and (series > max_val).any():
            invalid = (series > max_val).sum()
            raise DataValidationError(
                f"{col_name}: {invalid} values above maximum {max_val}"
            )
        
        return series

