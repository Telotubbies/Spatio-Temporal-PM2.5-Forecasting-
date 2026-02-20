"""
Static feature collector (Land Use, Population).
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Optional imports for raster operations
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from shapely.geometry import Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from config import StorageConfig


logger = logging.getLogger(__name__)


class StaticCollector:
    """Collect static features (land use, population)."""
    
    def __init__(self, storage_config: StorageConfig):
        """
        Initialize static collector.
        
        Args:
            storage_config: Storage configuration
        """
        self.storage_config = storage_config
    
    def extract_land_use(
        self,
        stations: pd.DataFrame,
        worldcover_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Extract land use class for each station from WorldCover.
        
        Args:
            stations: DataFrame with lat, lon
            worldcover_path: Path to WorldCover raster (optional)
            
        Returns:
            DataFrame with land_use column
        """
        logger.info("Extracting land use features...")
        
        # Placeholder: In production, load actual WorldCover raster
        # For now, return default value
        result = stations[["station_id", "lat", "lon"]].copy()
        result["land_use"] = 0  # Default: unknown
        
        if worldcover_path and worldcover_path.exists() and RASTERIO_AVAILABLE:
            try:
                with rasterio.open(worldcover_path) as src:
                    for idx, row in stations.iterrows():
                        # Sample raster at station location
                        lon, lat = row["lon"], row["lat"]
                        row_idx, col_idx = rasterio.transform.rowcol(
                            src.transform, lon, lat
                        )
                        
                        if (0 <= row_idx < src.height and 
                            0 <= col_idx < src.width):
                            value = src.read(1)[row_idx, col_idx]
                            result.loc[idx, "land_use"] = int(value)
                            
            except Exception as e:
                logger.error(f"Error reading WorldCover: {e}")
        
        logger.info(f"Extracted land use for {len(result)} stations")
        return result
    
    def extract_population(
        self,
        stations: pd.DataFrame,
        worldpop_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Extract population density for each station from WorldPop.
        
        Args:
            stations: DataFrame with lat, lon
            worldpop_path: Path to WorldPop raster (optional)
            
        Returns:
            DataFrame with population_density column
        """
        logger.info("Extracting population features...")
        
        # Placeholder: In production, load actual WorldPop raster
        result = stations[["station_id", "lat", "lon"]].copy()
        result["population_density"] = 0.0  # Default: unknown
        
        if worldpop_path and worldpop_path.exists() and RASTERIO_AVAILABLE:
            try:
                with rasterio.open(worldpop_path) as src:
                    for idx, row in stations.iterrows():
                        # Sample raster at station location
                        lon, lat = row["lon"], row["lat"]
                        row_idx, col_idx = rasterio.transform.rowcol(
                            src.transform, lon, lat
                        )
                        
                        if (0 <= row_idx < src.height and 
                            0 <= col_idx < src.width):
                            value = src.read(1)[row_idx, col_idx]
                            result.loc[idx, "population_density"] = float(value)
                            
            except Exception as e:
                logger.error(f"Error reading WorldPop: {e}")
        
        logger.info(f"Extracted population for {len(result)} stations")
        return result
    
    def save_static(self, df: pd.DataFrame, filename: str = "static_features.parquet"):
        """
        Save static features to Parquet.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return
        
        output_path = self.storage_config.PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_path,
            engine=self.storage_config.PARQUET_ENGINE,
            compression=self.storage_config.COMPRESSION
        )
        logger.info(f"Saved static features to {output_path}")

