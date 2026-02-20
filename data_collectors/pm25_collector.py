"""
PM2.5 data collector from Air4Thai API.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import httpx
from pathlib import Path

from config import DataConfig, StorageConfig


logger = logging.getLogger(__name__)


class PM25Collector:
    """Collect PM2.5 data from Air4Thai API."""
    
    def __init__(
        self,
        data_config: DataConfig,
        storage_config: StorageConfig
    ):
        """
        Initialize PM2.5 collector.
        
        Args:
            data_config: Data configuration
            storage_config: Storage configuration
        """
        self.data_config = data_config
        self.storage_config = storage_config
        self.api_url = data_config.AIR4THAI_API
        
    def fetch_stations(self) -> pd.DataFrame:
        """
        Fetch all PM2.5 stations from Air4Thai API.
        
        Returns:
            DataFrame with columns: station_id, lat, lon, name
        """
        logger.info("Fetching PM2.5 stations from Air4Thai...")
        
        try:
            # Air4Thai API may have SSL certificate issues, use verify=False for development
            with httpx.Client(timeout=30.0, verify=False) as client:
                response = client.get(self.api_url)
                response.raise_for_status()
                data = response.json()
            
            stations = []
            
            # Handle response format: {"stations": [...]}
            if isinstance(data, dict) and "stations" in data:
                data = data["stations"]
            elif not isinstance(data, list):
                logger.warning(f"Unexpected data format: {type(data)}")
                return pd.DataFrame()
            
            for record in data:
                # Skip if record is not a dict
                if not isinstance(record, dict):
                    continue
                
                # Get PM2.5 data from AQILast (current API format)
                aqi_last = record.get("AQILast", {})
                if not isinstance(aqi_last, dict):
                    continue
                
                pm25_data = aqi_last.get("PM25", {})
                if not isinstance(pm25_data, dict):
                    continue
                
                lat = record.get("lat")
                lon = record.get("long")
                
                # Validate coordinates
                try:
                    lat = float(lat) if lat else None
                    lon = float(lon) if lon else None
                except (ValueError, TypeError):
                    continue
                
                if lat is None or lon is None:
                    continue
                
                # Filter by Bangkok bounding box
                bbox = self.data_config.BANGKOK_BBOX
                if (bbox[0] <= lon <= bbox[2] and 
                    bbox[1] <= lat <= bbox[3]):
                    
                    # Get timestamp from AQILast
                    date_str = aqi_last.get("date", "")
                    time_str = aqi_last.get("time", "")
                    timestamp = None
                    if date_str and time_str:
                        try:
                            timestamp = pd.to_datetime(f"{date_str} {time_str}")
                            # Convert to UTC (Bangkok is UTC+7)
                            timestamp = timestamp - pd.Timedelta(hours=7)
                        except:
                            pass
                    
                    stations.append({
                        "station_id": record.get("stationID", ""),
                        "lat": lat,
                        "lon": lon,
                        "name": record.get("nameTH", ""),
                        "pm25": pm25_data.get("value"),
                        "timestamp": timestamp
                    })
            
            df = pd.DataFrame(stations)
            
            if df.empty:
                logger.warning("No stations found in Bangkok area")
                return df
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                # Convert to UTC (Bangkok is UTC+7)
                df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=7)
            
            logger.info(f"Found {len(df)} stations in Bangkok area")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stations: {e}")
            raise
    
    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        stations: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fetch historical PM2.5 data.
        
        Note: Air4Thai API may not support historical queries directly.
        This is a placeholder for when historical API is available.
        
        Args:
            start_date: Start date
            end_date: End date
            stations: Optional station list to filter
            
        Returns:
            DataFrame with historical PM2.5 data
        """
        logger.warning("Historical API not available. Using current data only.")
        return self.fetch_stations()
    
    def save_raw(self, df: pd.DataFrame, filename: str = "pm25_raw.parquet"):
        """
        Save raw PM2.5 data to Parquet.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return
        
        output_path = self.storage_config.RAW_DIR / "pm25" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_path,
            engine=self.storage_config.PARQUET_ENGINE,
            compression=self.storage_config.COMPRESSION
        )
        logger.info(f"Saved raw PM2.5 data to {output_path}")

