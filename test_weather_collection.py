#!/usr/bin/env python3
"""
Test script to verify weather collection with rate limit fixes.
"""
from datetime import datetime, timedelta
from pipeline import PM25Pipeline
from config import PipelineConfig

def test_weather_collection():
    """Test weather collection with small date range."""
    print("🧪 Testing Weather Collection with Rate Limit Fixes")
    print("=" * 60)
    
    config = PipelineConfig()
    pipeline = PM25Pipeline(config)
    
    # Test with small date range (last 7 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    print(f"\n📅 Test Date Range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Stations: 82")
    print(f"⚙️  Batch Size: {config.data.BATCH_SIZE}")
    print(f"⏱️  Delay: {config.data.REQUEST_DELAY_SECONDS} seconds")
    print("\n▶️  Starting test collection...")
    print("=" * 60)
    
    # Fetch stations first
    stations = pipeline.pm25_collector.fetch_stations()
    print(f"\n✅ Found {len(stations)} stations")
    
    # Test weather collection
    try:
        weather = pipeline.weather_collector.fetch_weather(
            stations,
            start_date,
            end_date
        )
        
        if not weather.empty:
            print(f"\n✅ Weather collection successful!")
            print(f"   Records: {len(weather)}")
            print(f"   Date range: {weather['timestamp'].min()} to {weather['timestamp'].max()}")
            print(f"   Columns: {', '.join(weather.columns)}")
        else:
            print("\n⚠️  No weather data collected (may be rate limited)")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check logs for details")

if __name__ == "__main__":
    test_weather_collection()

