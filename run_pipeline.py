#!/usr/bin/env python3
"""
Main entry point for PM2.5 forecasting data pipeline.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

from pipeline import PM25Pipeline
from config import PipelineConfig
from core.exceptions import PipelineError

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        logger.info("=" * 70)
        logger.info("PM2.5 Forecasting Data Pipeline - Starting")
        logger.info("=" * 70)
        
        # Initialize configuration
        config = PipelineConfig()
        logger.info(f"Configuration loaded")
        logger.info(f"  - Data directory: {config.storage.BASE_DIR}")
        logger.info(f"  - Historical start year: {config.data.HISTORICAL_START_YEAR}")
        logger.info(f"  - Batch size: {config.data.BATCH_SIZE}")
        logger.info(f"  - Chunk size: {config.data.CHUNK_SIZE_DAYS} days")
        
        # Initialize pipeline
        pipeline = PM25Pipeline(config)
        
        # Set date range (ตั้งแต่ปี 2010)
        end_date = datetime.utcnow()
        start_date = datetime(2010, 1, 1)
        
        logger.info(f"\nDate range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Total days: {(end_date - start_date).days}")
        logger.info(f"Total years: {(end_date - start_date).days / 365:.1f}")
        
        # Run pipeline
        result = pipeline.run(
            start_date=start_date,
            end_date=end_date,
            save_intermediate=True
        )
        
        if result.empty:
            logger.error("Pipeline completed but result is empty")
            sys.exit(1)
        
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)
        logger.info(f"Final dataset: {len(result)} records")
        logger.info(f"Stations: {result['station_id'].nunique()}")
        logger.info(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        logger.info(f"Data saved to: {config.storage.PROCESSED_DIR}")
        logger.info("=" * 70)
        
        return 0
        
    except PipelineError as e:
        logger.error(f"Pipeline error: {e.message}", exc_info=True)
        if e.details:
            logger.error(f"Details: {e.details}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

