"""
Fire hotspot data collector from NASA FIRMS.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import httpx
from pathlib import Path
import numpy as np

# Optional imports for geospatial operations
try:
    from shapely.geometry import Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

from config import DataConfig, StorageConfig, FeatureConfig


logger = logging.getLogger(__name__)


class FireCollector:
    """Collect fire hotspot data from NASA FIRMS."""
    
    def __init__(
        self,
        data_config: DataConfig,
        storage_config: StorageConfig,
        feature_config: FeatureConfig
    ):
        """
        Initialize fire collector.
        
        Args:
            data_config: Data configuration
            storage_config: Storage configuration
            feature_config: Feature configuration
        """
        self.data_config = data_config
        self.storage_config = storage_config
        self.feature_config = feature_config
        self.fire_radius = feature_config.FIRE_RADIUS_KM * 1000  # Convert to meters
        
    def fetch_fire_data(
        self,
        start_date: datetime,
        end_date: datetime,
        country_code: str = "THA"
    ) -> pd.DataFrame:
        """
        Fetch fire data from NASA FIRMS.
        
        Note: FIRMS API may require authentication.
        This is a simplified version.
        
        Args:
            start_date: Start date
            end_date: End date
            country_code: Country code (THA for Thailand)
            
        Returns:
            DataFrame with fire hotspot data
        """
        logger.info(f"Fetching fire data from {start_date} to {end_date}...")
        
        # FIRMS API endpoint (may need API key)
        # For now, return empty DataFrame as placeholder
        # In production, implement actual FIRMS API call
        
        logger.warning("FIRMS API integration not fully implemented. Returning empty data.")
        return pd.DataFrame(columns=["timestamp", "lat", "lon", "fire_count"])
    
    def aggregate_fire_by_station(
        self,
        fire_data: pd.DataFrame,
        stations: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Aggregate fire counts per station within radius.
        
        Args:
            fire_data: Fire hotspot DataFrame with lat, lon, timestamp
            stations: Station DataFrame with lat, lon
            timestamp: Target timestamp
            
        Returns:
            DataFrame with fire_count per station
        """
        if fire_data.empty:
            # Return zeros for all stations
            result = stations[["station_id", "lat", "lon"]].copy()
            result["fire_count"] = 0
            result["timestamp"] = timestamp
            return result
        
        # Filter fire data by timestamp (hourly aggregation)
        fire_hour = fire_data[
            fire_data["timestamp"].dt.floor("H") == timestamp.floor("H")
        ].copy()
        
        if fire_hour.empty:
            result = stations[["station_id", "lat", "lon"]].copy()
            result["fire_count"] = 0
            result["timestamp"] = timestamp
            return result
        
        # Spatial join: count fires within radius
        fire_counts = []
        
        for _, station in stations.iterrows():
            station_point = Point(station["lon"], station["lat"])
            
            # Count fires within radius
            count = 0
            for _, fire in fire_hour.iterrows():
                fire_point = Point(fire["lon"], fire["lat"])
                distance = self._haversine_distance(
                    station["lat"], station["lon"],
                    fire["lat"], fire["lon"]
                )
                
                if distance <= self.fire_radius:
                    count += 1
            
            fire_counts.append({
                "station_id": station["station_id"],
                "lat": station["lat"],
                "lon": station["lon"],
                "fire_count": count,
                "timestamp": timestamp
            })
        
        return pd.DataFrame(fire_counts)
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points in meters.
        
        Args:
            lat1, lon1: First point
            lat2, lon2: Second point
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi / 2) ** 2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def save_raw(self, df: pd.DataFrame, filename: str = "fire_raw.parquet"):
        """
        Save raw fire data to Parquet.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return
        
        output_path = self.storage_config.RAW_DIR / "fire" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_path,
            engine=self.storage_config.PARQUET_ENGINE,
            compression=self.storage_config.COMPRESSION
        )
        logger.info(f"Saved raw fire data to {output_path}")

