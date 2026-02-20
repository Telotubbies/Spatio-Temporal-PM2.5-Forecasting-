"""
Weather data collector from Open-Meteo API.
Optimized for batch calls to minimize API requests.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import httpx
import asyncio
import time
from pathlib import Path
import numpy as np

from config import DataConfig, StorageConfig


logger = logging.getLogger(__name__)


class WeatherCollector:
    """Collect weather data from Open-Meteo API with batch optimization."""
    
    def __init__(
        self,
        data_config: DataConfig,
        storage_config: StorageConfig
    ):
        """
        Initialize weather collector.
        
        Args:
            data_config: Data configuration
            storage_config: Storage configuration
        """
        self.data_config = data_config
        self.storage_config = storage_config
        self.api_url = data_config.OPEN_METEO_API
        self.historical_api_url = data_config.OPEN_METEO_HISTORICAL_API
        self.batch_size = data_config.BATCH_SIZE
        self.chunk_size_days = data_config.CHUNK_SIZE_DAYS
        self.request_delay = getattr(data_config, 'REQUEST_DELAY_SECONDS', 3.0)
        
    def _build_url(
        self,
        lats: List[float],
        lons: List[float],
        start_date: datetime,
        end_date: datetime,
        use_historical: bool = False
    ) -> str:
        """
        Build Open-Meteo API URL with multiple locations.
        
        Args:
            lats: List of latitudes
            lons: List of longitudes
            start_date: Start date
            end_date: End date
            use_historical: Use historical API endpoint
            
        Returns:
            API URL string
        """
        # Open-Meteo supports comma-separated coordinates
        lat_str = ",".join(map(str, lats))
        lon_str = ",".join(map(str, lons))
        
        # Choose API endpoint
        base_url = self.historical_api_url if use_historical else self.api_url
        
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,"
                      "wind_speed_10m,wind_direction_10m,precipitation,"
                      "shortwave_radiation",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "UTC"
        }
        
        # Build URL
        url = f"{base_url}?"
        url += "&".join([f"{k}={v}" for k, v in params.items()])
        
        return url
    
    async def _fetch_batch_async(
        self,
        lats: List[float],
        lons: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict]:
        """
        Fetch weather data for a batch of locations asynchronously.
        
        Args:
            lats: List of latitudes
            lons: List of longitudes
            start_date: Start date
            end_date: End date
            
        Returns:
            API response JSON or None
        """
        url = self._build_url(lats, lons, start_date, end_date)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching weather batch: {e}")
            return None
    
    def fetch_weather(
        self,
        stations: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch weather data for multiple stations with batch optimization.
        Automatically uses historical API for dates before today.
        
        Args:
            stations: DataFrame with columns: station_id, lat, lon
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with weather data
        """
        logger.info(f"Fetching weather data from {start_date.date()} to {end_date.date()}")
        logger.info(f"Number of stations: {len(stations)}")
        
        # Determine if we need historical API
        today = datetime.utcnow().date()
        use_historical = start_date.date() < today
        
        # Get unique lat/lon pairs
        unique_locs = stations[["lat", "lon"]].drop_duplicates()
        logger.info(f"Unique locations: {len(unique_locs)}")
        
        # Calculate date range
        total_days = (end_date - start_date).days
        logger.info(f"Total days to fetch: {total_days}")
        
        # Process in time chunks to avoid large API calls
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate chunk end date
            chunk_end = min(
                current_start + timedelta(days=self.chunk_size_days),
                end_date
            )
            
            chunk_days = (chunk_end - current_start).days
            logger.info(f"\nFetching chunk: {current_start.date()} to {chunk_end.date()} ({chunk_days} days)")
            
            # Determine if this chunk needs historical API
            chunk_use_historical = current_start.date() < today
            
            # Process locations in batches
            for i in range(0, len(unique_locs), self.batch_size):
                batch_locs = unique_locs.iloc[i:i + self.batch_size]
                lats = batch_locs["lat"].tolist()
                lons = batch_locs["lon"].tolist()
                
                batch_num = i//self.batch_size + 1
                total_batches = (len(unique_locs) + self.batch_size - 1) // self.batch_size
                logger.info(f"  Batch {batch_num}/{total_batches} ({len(lats)} locations)...")
                
                # Add delay before request (except first batch)
                if i > 0:
                    logger.info(f"  Waiting {self.request_delay} seconds to avoid rate limit...")
                    time.sleep(self.request_delay)
                
                # Build URL with appropriate endpoint
                url = self._build_url(
                    lats, lons, 
                    current_start, chunk_end,
                    use_historical=chunk_use_historical
                )
                
                try:
                    with httpx.Client(timeout=300.0) as client:  # Increased timeout for historical data
                        response = client.get(url)
                        
                        # Check for rate limit (429)
                        if response.status_code == 429:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.warning(f"  ⚠️  Rate limited! Waiting {retry_after} seconds...")
                            time.sleep(retry_after)
                            # Retry the request
                            response = client.get(url)
                        
                        response.raise_for_status()
                        data = response.json()
                    
                    # Parse response
                    if "hourly" in data:
                        hourly = data["hourly"]
                        time = hourly.get("time", [])
                        
                        if not time:
                            logger.warning(f"  No time data in response for batch {batch_num}")
                            continue
                        
                        # Process each location in batch
                        # Open-Meteo with multiple locations returns data sequentially
                        # Each location's data follows the time array
                        num_timesteps = len(time)
                        num_locations = len(batch_locs)
                        
                        # Get all hourly arrays
                        temp_arr = hourly.get("temperature_2m", [])
                        humidity_arr = hourly.get("relative_humidity_2m", [])
                        pressure_arr = hourly.get("surface_pressure", [])
                        wind_speed_arr = hourly.get("wind_speed_10m", [])
                        wind_dir_arr = hourly.get("wind_direction_10m", [])
                        precip_arr = hourly.get("precipitation", [])
                        solar_arr = hourly.get("shortwave_radiation", [])
                        
                        # Helper to safely get value from array
                        def safe_get(arr, idx):
                            if arr and isinstance(arr, list) and 0 <= idx < len(arr):
                                return arr[idx]
                            return None
                        
                        # Process each location
                        for loc_idx, (_, loc) in enumerate(batch_locs.iterrows()):
                            for t_idx, timestamp in enumerate(time):
                                # For multiple locations, data is: [loc1_t1, loc1_t2, ..., loc1_tN, loc2_t1, ...]
                                # So index = loc_idx * num_timesteps + t_idx
                                data_idx = loc_idx * num_timesteps + t_idx
                                
                                record = {
                                    "timestamp": pd.to_datetime(timestamp),
                                    "lat": loc["lat"],
                                    "lon": loc["lon"],
                                    "temperature": safe_get(temp_arr, data_idx),
                                    "humidity": safe_get(humidity_arr, data_idx),
                                    "pressure": safe_get(pressure_arr, data_idx),
                                    "wind_speed": safe_get(wind_speed_arr, data_idx),
                                    "wind_direction": safe_get(wind_dir_arr, data_idx),
                                    "precipitation": safe_get(precip_arr, data_idx),
                                    "solar": safe_get(solar_arr, data_idx)
                                }
                                all_data.append(record)
                    else:
                        logger.warning(f"  No 'hourly' key in response for batch {batch_num}")
                            
                except Exception as e:
                    logger.error(f"  Error in batch {batch_num}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Move to next chunk
            current_start = chunk_end + timedelta(days=1)
        
        if not all_data:
            logger.warning("No weather data fetched")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Remove duplicates (in case of overlapping chunks)
        df = df.drop_duplicates(subset=["lat", "lon", "timestamp"])
        
        # Merge with station IDs
        df = df.merge(
            stations[["station_id", "lat", "lon"]],
            on=["lat", "lon"],
            how="left"
        )
        
        logger.info(f"\n✅ Fetched {len(df)} weather records")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def save_raw(self, df: pd.DataFrame, filename: str = "weather_raw.parquet"):
        """
        Save raw weather data to Parquet.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return
        
        output_path = self.storage_config.RAW_DIR / "weather" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(
            output_path,
            engine=self.storage_config.PARQUET_ENGINE,
            compression=self.storage_config.COMPRESSION
        )
        logger.info(f"Saved raw weather data to {output_path}")

