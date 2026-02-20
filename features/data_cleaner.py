"""
Data cleaning and preprocessing.
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from config import FeatureConfig


logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess data."""
    
    def __init__(self, feature_config: FeatureConfig):
        """
        Initialize data cleaner.
        
        Args:
            feature_config: Feature configuration
        """
        self.config = feature_config
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        target_col: str = "pm25"
    ) -> pd.DataFrame:
        """
        Handle missing values with interpolation and forward fill.
        
        Args:
            df: Input DataFrame
            target_col: Target column to check missing values
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values(["station_id", "timestamp"])
        
        # Calculate missing percentage per station
        if "station_id" in df.columns and target_col in df.columns:
            missing_pct = df.groupby("station_id")[target_col].apply(
                lambda x: x.isna().sum() / len(x)
            )
            
            # Remove stations with >20% missing
            bad_stations = missing_pct[missing_pct > self.config.MAX_MISSING_PCT].index
            if len(bad_stations) > 0:
                logger.info(f"Removing {len(bad_stations)} stations with >{self.config.MAX_MISSING_PCT*100}% missing")
                df = df[~df["station_id"].isin(bad_stations)]
        
        # Interpolate for small gaps (<3 hours)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == target_col or col in ["lat", "lon"]:
                continue
            
            # Group by station and interpolate
            if "station_id" in df.columns:
                df[col] = df.groupby("station_id")[col].apply(
                    lambda x: x.interpolate(
                        method="linear",
                        limit=self.config.INTERPOLATION_LIMIT_HOURS
                    )
                )
            else:
                df[col] = df[col].interpolate(
                    method="linear",
                    limit=self.config.INTERPOLATION_LIMIT_HOURS
                )
        
        # Forward fill remaining missing values
        df[numeric_cols] = df[numeric_cols].ffill()
        
        logger.info("Handled missing values")
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        target_col: str = "pm25"
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR and z-score methods.
        
        Args:
            df: Input DataFrame
            target_col: Target column to check outliers
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return df
        
        initial_count = len(df)
        
        if self.config.USE_IQR:
            # IQR method
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[target_col].dropna()))
        df = df[z_scores < self.config.Z_SCORE_THRESHOLD]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.2f}%)")
        
        return df
    
    def clean(self, df: pd.DataFrame, target_col: str = "pm25") -> pd.DataFrame:
        """
        Apply all cleaning steps.
        
        Args:
            df: Input DataFrame
            target_col: Target column
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = self.handle_missing_values(df, target_col)
        df = self.remove_outliers(df, target_col)
        logger.info(f"Cleaning complete. Final size: {len(df)} records")
        return df

