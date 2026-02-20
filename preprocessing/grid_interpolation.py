"""
Station to grid interpolation (IDW, Kriging).
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class GridInterpolator:
    """
    Interpolate station data to grid using IDW or Kriging.
    """
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        grid_size: Tuple[int, int] = (32, 32),
        method: str = "idw",
        idw_power: float = 2.0
    ):
        """
        Initialize grid interpolator.
        
        Args:
            bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
            grid_size: Grid dimensions (H, W)
            method: Interpolation method (idw, kriging)
            idw_power: Power parameter for IDW
        """
        self.bbox = bbox
        self.grid_size = grid_size
        self.method = method
        self.idw_power = idw_power
        
        # Create grid coordinates
        lon_min, lat_min, lon_max, lat_max = bbox
        self.lon_grid = np.linspace(lon_min, lon_max, grid_size[1])
        self.lat_grid = np.linspace(lat_max, lat_min, grid_size[0])  # Reverse for image coordinates
        
        # Create meshgrid
        self.lon_mesh, self.lat_mesh = np.meshgrid(self.lon_grid, self.lat_grid)
        
        logger.info(f"Initialized {method} interpolator: {grid_size} grid, bbox={bbox}")
    
    def interpolate(
        self,
        stations: pd.DataFrame,
        values: np.ndarray,
        timestamp: pd.Timestamp
    ) -> np.ndarray:
        """
        Interpolate station values to grid.
        
        Args:
            stations: DataFrame with lat, lon columns
            values: Array of values per station (n_stations,)
            timestamp: Timestamp for this interpolation
            
        Returns:
            Grid array (H, W)
        """
        if self.method == "idw":
            return self._idw_interpolate(stations, values)
        elif self.method == "kriging":
            return self._kriging_interpolate(stations, values)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _idw_interpolate(
        self,
        stations: pd.DataFrame,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation.
        
        Args:
            stations: DataFrame with lat, lon
            values: Station values
            
        Returns:
            Grid array (H, W)
        """
        # Station coordinates
        station_coords = stations[['lon', 'lat']].values  # (n_stations, 2)
        
        # Grid coordinates
        grid_coords = np.column_stack([
            self.lon_mesh.ravel(),
            self.lat_mesh.ravel()
        ])  # (H*W, 2)
        
        # Calculate distances
        distances = cdist(grid_coords, station_coords)  # (H*W, n_stations)
        
        # Avoid division by zero
        distances = np.where(distances < 1e-10, 1e-10, distances)
        
        # IDW weights
        weights = 1.0 / (distances ** self.idw_power)  # (H*W, n_stations)
        weights_sum = weights.sum(axis=1, keepdims=True)  # (H*W, 1)
        weights = weights / weights_sum  # Normalize
        
        # Interpolate
        grid_values = np.dot(weights, values)  # (H*W,)
        
        # Reshape to grid
        grid = grid_values.reshape(self.grid_size)  # (H, W)
        
        return grid
    
    def _kriging_interpolate(
        self,
        stations: pd.DataFrame,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Kriging interpolation (simplified version).
        
        Note: Full Kriging requires variogram fitting.
        This is a simplified implementation.
        
        Args:
            stations: DataFrame with lat, lon
            values: Station values
            
        Returns:
            Grid array (H, W)
        """
        # For now, use IDW as placeholder
        # Full Kriging implementation would require:
        # 1. Variogram fitting
        # 2. Kriging system solving
        logger.warning("Kriging not fully implemented, using IDW instead")
        return self._idw_interpolate(stations, values)
    
    def create_grid_tensor(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        timestamp: pd.Timestamp
    ) -> np.ndarray:
        """
        Create grid tensor for a single timestamp.
        
        Args:
            df: DataFrame with station data
            feature_cols: Feature columns to interpolate
            timestamp: Timestamp
            
        Returns:
            Grid tensor (H, W, channels)
        """
        # Filter by timestamp
        df_t = df[df['timestamp'] == timestamp].copy()
        
        if df_t.empty:
            logger.warning(f"No data for timestamp {timestamp}")
            return np.zeros((*self.grid_size, len(feature_cols)))
        
        # Interpolate each feature
        grid_channels = []
        
        for col in feature_cols:
            if col not in df_t.columns:
                logger.warning(f"Column {col} not found, using zeros")
                grid_channels.append(np.zeros(self.grid_size))
                continue
            
            values = df_t[col].values
            stations = df_t[['lat', 'lon']].copy()
            
            # Interpolate
            grid = self.interpolate(stations, values, timestamp)
            grid_channels.append(grid)
        
        # Stack channels
        grid_tensor = np.stack(grid_channels, axis=-1)  # (H, W, channels)
        
        return grid_tensor

