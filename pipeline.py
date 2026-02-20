"""
Main data pipeline orchestrator.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from config import PipelineConfig
from data_collectors import (
    PM25Collector,
    WeatherCollector,
    FireCollector,
    StaticCollector
)
from features import (
    TimeFeatureEngineer,
    WindFeatureEngineer,
    DataCleaner
)
from utils.logger import setup_logging
from core.exceptions import (
    DataCollectionError,
    DataValidationError,
    FeatureEngineeringError,
    StorageError
)
from core.validators import DataValidator

logger = logging.getLogger(__name__)


class PM25Pipeline:
    """Main PM2.5 forecasting data pipeline."""
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        setup_logging(config)
        
        # Initialize collectors
        self.pm25_collector = PM25Collector(
            config.data,
            config.storage
        )
        self.weather_collector = WeatherCollector(
            config.data,
            config.storage
        )
        self.fire_collector = FireCollector(
            config.data,
            config.storage,
            config.features
        )
        self.static_collector = StaticCollector(config.storage)
        
        # Initialize feature engineers
        self.time_engineer = TimeFeatureEngineer()
        self.wind_engineer = WindFeatureEngineer()
        self.data_cleaner = DataCleaner(config.features)
    
    def run(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Run complete data pipeline.
        
        Args:
            start_date: Start date (default: today - historical_days)
            end_date: End date (default: today)
            save_intermediate: Save intermediate results
            
        Returns:
            Final processed DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting PM2.5 Forecasting Data Pipeline")
        logger.info("=" * 60)
        
        # Set default dates
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            # Use historical start year if specified, otherwise use HISTORICAL_DAYS
            if hasattr(self.config.data, 'HISTORICAL_START_YEAR'):
                start_date = datetime(self.config.data.HISTORICAL_START_YEAR, 1, 1)
                logger.info(f"Using historical start year: {self.config.data.HISTORICAL_START_YEAR}")
            else:
                start_date = end_date - timedelta(days=self.config.data.HISTORICAL_DAYS)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Step 1: Fetch PM2.5 stations
        logger.info("\n[Step 1] Fetching PM2.5 stations...")
        try:
            stations = self.pm25_collector.fetch_stations()
            
            if stations.empty:
                logger.error("No stations found. Exiting.")
                return pd.DataFrame()
            
            # Validate stations
            stations = DataValidator.validate_stations(stations)
            
            if save_intermediate:
                self.pm25_collector.save_raw(stations, "pm25_stations.parquet")
            
            logger.info(f"Found {len(stations)} stations")
        except Exception as e:
            logger.error(f"Error fetching stations: {e}", exc_info=True)
            raise DataCollectionError(f"Failed to fetch PM2.5 stations: {e}") from e
        
        # Step 2: Fetch weather data
        logger.info("\n[Step 2] Fetching weather data...")
        try:
            weather = self.weather_collector.fetch_weather(
                stations,
                start_date,
                end_date
            )
            
            # Validate weather data
            if not weather.empty:
                weather = DataValidator.validate_weather(weather)
            
            if save_intermediate and not weather.empty:
                self.weather_collector.save_raw(weather, "weather_raw.parquet")
        except Exception as e:
            logger.error(f"Error fetching weather: {e}", exc_info=True)
            raise DataCollectionError(f"Failed to fetch weather data: {e}") from e
        
        # Step 3: Fetch fire data (placeholder)
        logger.info("\n[Step 3] Fetching fire data...")
        fire_data = self.fire_collector.fetch_fire_data(start_date, end_date)
        
        if save_intermediate and not fire_data.empty:
            self.fire_collector.save_raw(fire_data, "fire_raw.parquet")
        
        # Step 4: Extract static features
        logger.info("\n[Step 4] Extracting static features...")
        land_use = self.static_collector.extract_land_use(stations)
        population = self.static_collector.extract_population(stations)
        
        static_features = land_use.merge(
            population[["station_id", "population_density"]],
            on="station_id",
            how="left"
        )
        
        if save_intermediate:
            self.static_collector.save_static(static_features, "static_features.parquet")
        
        # Step 5: Merge all data
        logger.info("\n[Step 5] Merging data sources...")
        
        # Merge PM2.5 with weather
        if not weather.empty:
            # Create hourly time range
            time_range = pd.date_range(start_date, end_date, freq="H")
            time_df = pd.DataFrame({"timestamp": time_range})
            
            # Merge stations with time
            station_time = stations[["station_id", "lat", "lon"]].merge(
                time_df.assign(key=1),
                how="cross"
            )
            
            # Merge with weather
            merged = station_time.merge(
                weather,
                on=["station_id", "lat", "lon", "timestamp"],
                how="left"
            )
        else:
            logger.warning("No weather data, creating time series from stations only")
            time_range = pd.date_range(start_date, end_date, freq="H")
            time_df = pd.DataFrame({"timestamp": time_range})
            merged = stations[["station_id", "lat", "lon"]].merge(
                time_df.assign(key=1),
                how="cross"
            )
        
        # Merge with PM2.5 (if available)
        if "pm25" in stations.columns:
            pm25_data = stations[["station_id", "timestamp", "pm25"]].copy()
            merged = merged.merge(
                pm25_data,
                on=["station_id", "timestamp"],
                how="left"
            )
        
        # Merge with static features
        merged = merged.merge(
            static_features[["station_id", "land_use", "population_density"]],
            on="station_id",
            how="left"
        )
        
        # Step 6: Feature engineering
        logger.info("\n[Step 6] Engineering features...")
        try:
            # Wind encoding
            merged = self.wind_engineer.encode_wind(merged)
            
            # Time features
            merged = self.time_engineer.add_time_features(merged)
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}", exc_info=True)
            raise FeatureEngineeringError(f"Feature engineering failed: {e}") from e
        
        # Step 7: Data cleaning
        logger.info("\n[Step 7] Cleaning data...")
        try:
            merged = self.data_cleaner.clean(merged, target_col="pm25")
            
            # Validate merged data
            merged = DataValidator.validate_merged_data(merged)
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}", exc_info=True)
            raise DataValidationError(f"Data cleaning failed: {e}") from e
        
        # Step 8: Add partition columns
        merged["year"] = merged["timestamp"].dt.year
        merged["month"] = merged["timestamp"].dt.month
        
        # Step 9: Save processed data
        logger.info("\n[Step 8] Saving processed data...")
        try:
            self._save_processed(merged)
        except Exception as e:
            logger.error(f"Error saving processed data: {e}", exc_info=True)
            raise StorageError(f"Failed to save processed data: {e}") from e
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final dataset size: {len(merged)} records")
        logger.info(f"Stations: {merged['station_id'].nunique()}")
        logger.info(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
        logger.info("=" * 60)
        
        return merged
    
    def _save_processed(self, df: pd.DataFrame):
        """
        Save processed data with partitioning.
        
        Args:
            df: Processed DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return
        
        output_dir = self.config.storage.PROCESSED_DIR / "station_level"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save partitioned by year/month/station_id
        for (year, month, station_id), group in df.groupby(["year", "month", "station_id"]):
            partition_dir = output_dir / f"year={year}" / f"month={month:02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = partition_dir / f"station_id={station_id}.parquet"
            
            group.to_parquet(
                output_path,
                engine=self.config.storage.PARQUET_ENGINE,
                compression=self.config.storage.COMPRESSION,
                index=False
            )
        
        logger.info(f"Saved processed data to {output_dir}")


def main():
    """Main entry point."""
    config = PipelineConfig()
    pipeline = PM25Pipeline(config)
    
    # Run pipeline
    result = pipeline.run()
    
    return result


if __name__ == "__main__":
    main()

