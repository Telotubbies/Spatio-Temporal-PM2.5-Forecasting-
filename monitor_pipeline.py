#!/usr/bin/env python3
"""
Monitor pipeline progress and check if data is ready for training.
"""
import time
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def check_pipeline_status():
    """Check if pipeline is running."""
    pid_file = Path("pipeline.pid")
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        if os.path.exists(f"/proc/{pid}"):
            return True, pid
    return False, None

def check_data_status():
    """Check data collection status."""
    status = {
        "pm25_stations": False,
        "weather_data": False,
        "static_features": False,
        "processed_data": False,
        "ready_for_training": False
    }
    
    # PM2.5 Stations
    pm25_path = Path("data/raw/pm25/pm25_stations.parquet")
    if pm25_path.exists():
        df = pd.read_parquet(pm25_path)
        status["pm25_stations"] = len(df) > 0
    
    # Weather Data
    weather_dir = Path("data/raw/weather")
    if weather_dir.exists():
        weather_files = list(weather_dir.glob("*.parquet"))
        if weather_files:
            total_records = 0
            for f in weather_files:
                df = pd.read_parquet(f)
                total_records += len(df)
            status["weather_data"] = total_records > 0
    
    # Static Features
    static_path = Path("data/processed/static_features.parquet")
    if static_path.exists():
        df = pd.read_parquet(static_path)
        status["static_features"] = len(df) > 0
    
    # Processed Data
    processed_dir = Path("data/processed/station_level")
    if processed_dir.exists():
        processed_files = list(processed_dir.rglob("*.parquet"))
        if processed_files:
            status["processed_data"] = len(processed_files) > 0
    
    # Ready for training
    status["ready_for_training"] = (
        status["pm25_stations"] and
        status["weather_data"] and
        status["static_features"] and
        status["processed_data"]
    )
    
    return status

def get_latest_logs(n=20):
    """Get latest log entries."""
    log_files = [
        Path("pipeline_output.log"),
        Path("logs/pipeline.log")
    ]
    
    for log_file in log_files:
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return lines[-n:] if len(lines) > n else lines
    return []

def monitor_until_ready(check_interval=60):
    """Monitor pipeline until data is ready for training."""
    print("=" * 70)
    print("🔍 Monitoring Pipeline - Waiting for Training Data")
    print("=" * 70)
    print(f"⏱️  Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    iteration = 0
    last_weather_records = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n[{timestamp}] Check #{iteration}")
            print("-" * 70)
            
            # Check pipeline status
            is_running, pid = check_pipeline_status()
            if is_running:
                print(f"✅ Pipeline: Running (PID: {pid})")
            else:
                print("⚠️  Pipeline: Not running")
            
            # Check data status
            status = check_data_status()
            
            print("\n📊 Data Status:")
            print(f"   PM2.5 Stations: {'✅' if status['pm25_stations'] else '❌'}")
            print(f"   Weather Data: {'✅' if status['weather_data'] else '⏳ Collecting...'}")
            print(f"   Static Features: {'✅' if status['static_features'] else '❌'}")
            print(f"   Processed Data: {'✅' if status['processed_data'] else '⏳ Waiting...'}")
            
            # Weather data progress
            if status['weather_data']:
                weather_dir = Path("data/raw/weather")
                weather_files = list(weather_dir.glob("*.parquet"))
                if weather_files:
                    total_records = 0
                    for f in weather_files:
                        df = pd.read_parquet(f)
                        total_records += len(df)
                    
                    if total_records > last_weather_records:
                        print(f"   📈 Weather Records: {total_records:,} (+{total_records - last_weather_records:,} new)")
                        last_weather_records = total_records
                    else:
                        print(f"   📈 Weather Records: {total_records:,}")
            
            # Check if ready
            if status['ready_for_training']:
                print("\n" + "=" * 70)
                print("🎉 DATA READY FOR TRAINING!")
                print("=" * 70)
                
                # Get data summary
                processed_dir = Path("data/processed/station_level")
                processed_files = list(processed_dir.rglob("*.parquet"))
                total_records = 0
                for f in processed_files[:10]:  # Sample
                    df = pd.read_parquet(f)
                    total_records += len(df)
                
                print(f"\n✅ Processed Data:")
                print(f"   Files: {len(processed_files)}")
                print(f"   Sample records: {total_records:,}")
                
                print("\n🚀 Next Steps:")
                print("   1. Create sequences: python3 -c \"from utils.sliding_window import create_sliding_window; ...\"")
                print("   2. Or use notebook: jupyter notebook pipline.ipynb")
                print("   3. Start ST-UNN model training")
                
                break
            
            # Show latest logs
            print("\n📋 Latest Activity:")
            logs = get_latest_logs(5)
            for line in logs[-5:]:
                if any(keyword in line for keyword in ["Batch", "Fetched", "Rate", "Waiting", "ERROR", "WARNING"]):
                    print(f"   {line.strip()[:100]}")
            
            print(f"\n⏳ Waiting {check_interval} seconds until next check...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        print("Pipeline may still be running in background")
        print("Check status: ps aux | grep run_pipeline")

if __name__ == "__main__":
    monitor_until_ready(check_interval=60)

