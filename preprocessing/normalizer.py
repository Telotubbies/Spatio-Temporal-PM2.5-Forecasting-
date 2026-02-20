"""
Data normalization and scaling.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, List
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize data using various scalers."""
    
    def __init__(
        self,
        scaler_type: str = "StandardScaler",
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize normalizer.
        
        Args:
            scaler_type: Type of scaler (StandardScaler, MinMaxScaler, RobustScaler)
            feature_cols: List of feature columns to normalize
        """
        self.scaler_type = scaler_type
        self.feature_cols = feature_cols
        
        # Create scaler
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "RobustScaler":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """
        Fit scaler on data.
        
        Args:
            df: DataFrame to fit on
            feature_cols: Feature columns (if None, use self.feature_cols)
        """
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        if feature_cols is None:
            # Auto-detect numeric columns (exclude metadata)
            exclude_cols = ["timestamp", "station_id", "lat", "lon", "year", "month"]
            feature_cols = [col for col in df.columns 
                          if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No feature columns available for normalization")
            return
        
        # Fit scaler
        self.scaler.fit(df[available_cols].values)
        self.feature_cols = available_cols
        self.fitted = True
        
        logger.info(f"Fitted {self.scaler_type} on {len(available_cols)} features")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Transform features
        if self.feature_cols:
            available_cols = [col for col in self.feature_cols if col in df.columns]
            if available_cols:
                df[available_cols] = self.scaler.transform(df[available_cols].values)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform data.
        
        Args:
            df: DataFrame
            feature_cols: Feature columns
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_cols)
        return self.transform(df)
    
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Inverse transform values.
        
        Args:
            values: Normalized values
            
        Returns:
            Original scale values
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.scaler.inverse_transform(values)
    
    def save(self, path: Path):
        """
        Save scaler to disk.
        
        Args:
            path: Path to save
        """
        joblib.dump({
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'scaler_type': self.scaler_type,
            'fitted': self.fitted
        }, path)
        logger.info(f"Saved scaler to {path}")
    
    def load(self, path: Path):
        """
        Load scaler from disk.
        
        Args:
            path: Path to load from
        """
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        self.scaler_type = data['scaler_type']
        self.fitted = data['fitted']
        logger.info(f"Loaded scaler from {path}")

