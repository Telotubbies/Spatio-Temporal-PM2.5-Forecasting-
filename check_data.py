#!/usr/bin/env python3
"""
Script to check all collected data.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

def check_all_data():
    """Check all collected data."""
    print("=" * 70)
    print("📊 ตรวจสอบข้อมูลทั้งหมดที่ Collect มา")
    print("=" * 70)
    
    base_dir = Path("data")
    
    # 1. PM2.5 Stations
    print("\n1️⃣ PM2.5 Stations (Air4Thai)")
    print("-" * 70)
    pm25_path = base_dir / "raw" / "pm25" / "pm25_stations.parquet"
    if pm25_path.exists():
        df = pd.read_parquet(pm25_path)
        print(f"✅ ไฟล์: {pm25_path}")
        print(f"✅ จำนวน: {len(df)} stations")
        print(f"✅ ขนาดไฟล์: {pm25_path.stat().st_size / 1024:.2f} KB")
        print(f"\n📋 Columns: {', '.join(df.columns)}")
        print(f"\n📍 พิกัด:")
        print(f"   Lat: {df['lat'].min():.4f}°N - {df['lat'].max():.4f}°N")
        print(f"   Lon: {df['lon'].min():.4f}°E - {df['lon'].max():.4f}°E")
        if 'pm25' in df.columns:
            valid_pm25 = df['pm25'].dropna()
            if len(valid_pm25) > 0:
                print(f"\n📊 PM2.5 Values:")
                print(f"   Min: {valid_pm25.min():.1f} μg/m³")
                print(f"   Max: {valid_pm25.max():.1f} μg/m³")
                print(f"   Mean: {valid_pm25.mean():.1f} μg/m³")
                print(f"   Valid: {len(valid_pm25)}/{len(df)} stations")
        if 'timestamp' in df.columns:
            print(f"\n📅 Timestamp:")
            print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("❌ ไม่พบไฟล์")
    
    # 2. Weather Data
    print("\n2️⃣ Weather Data (Open-Meteo)")
    print("-" * 70)
    weather_dir = base_dir / "raw" / "weather"
    if weather_dir.exists():
        weather_files = list(weather_dir.glob("*.parquet"))
        if weather_files:
            total_records = 0
            total_size = 0
            for f in weather_files:
                df = pd.read_parquet(f)
                total_records += len(df)
                total_size += f.stat().st_size
            print(f"✅ จำนวนไฟล์: {len(weather_files)}")
            print(f"✅ Total Records: {total_records:,}")
            print(f"✅ Total Size: {total_size / 1024 / 1024:.2f} MB")
            if weather_files:
                df_sample = pd.read_parquet(weather_files[0])
                print(f"✅ Columns: {', '.join(df_sample.columns)}")
                if 'timestamp' in df_sample.columns:
                    print(f"✅ Date Range (sample): {df_sample['timestamp'].min()} to {df_sample['timestamp'].max()}")
        else:
            print("⚠️  ไม่พบไฟล์ weather data")
            print("   สาเหตุ: Open-Meteo collection อาจยังไม่เสร็จหรือมี error")
    else:
        print("⚠️  Directory ไม่มี")
    
    # 3. Fire Data
    print("\n3️⃣ Fire Data (NASA FIRMS)")
    print("-" * 70)
    fire_dir = base_dir / "raw" / "fire"
    if fire_dir.exists():
        fire_files = list(fire_dir.glob("*.parquet"))
        if fire_files:
            print(f"✅ จำนวนไฟล์: {len(fire_files)}")
            for f in fire_files:
                df = pd.read_parquet(f)
                print(f"   - {f.name}: {len(df)} records")
        else:
            print("⚠️  ไม่พบไฟล์ (placeholder implementation)")
    else:
        print("⚠️  Directory ไม่มี (placeholder implementation)")
    
    # 4. Static Features
    print("\n4️⃣ Static Features (WorldCover, WorldPop)")
    print("-" * 70)
    static_path = base_dir / "processed" / "static_features.parquet"
    if static_path.exists():
        df = pd.read_parquet(static_path)
        print(f"✅ ไฟล์: {static_path}")
        print(f"✅ จำนวน: {len(df)} records")
        print(f"✅ ขนาดไฟล์: {static_path.stat().st_size / 1024:.2f} KB")
        print(f"✅ Columns: {', '.join(df.columns)}")
        if 'land_use' in df.columns:
            print(f"\n📊 Land Use:")
            print(f"   Unique values: {df['land_use'].nunique()}")
        if 'population_density' in df.columns:
            print(f"\n📊 Population Density:")
            print(f"   Min: {df['population_density'].min():.2f}")
            print(f"   Max: {df['population_density'].max():.2f}")
            print(f"   Mean: {df['population_density'].mean():.2f}")
    else:
        print("❌ ไม่พบไฟล์")
    
    # 5. Processed Data
    print("\n5️⃣ Processed Data (Merged & Cleaned)")
    print("-" * 70)
    processed_dir = base_dir / "processed" / "station_level"
    if processed_dir.exists():
        processed_files = list(processed_dir.rglob("*.parquet"))
        if processed_files:
            total_records = 0
            total_size = 0
            stations = set()
            years = set()
            months = set()
            for f in processed_files:
                df = pd.read_parquet(f)
                total_records += len(df)
                total_size += f.stat().st_size
                if 'station_id' in df.columns:
                    stations.update(df['station_id'].unique())
                if 'year' in df.columns:
                    years.update(df['year'].unique())
                if 'month' in df.columns:
                    months.update(df['month'].unique())
            print(f"✅ จำนวนไฟล์: {len(processed_files)}")
            print(f"✅ Total Records: {total_records:,}")
            print(f"✅ Total Size: {total_size / 1024 / 1024:.2f} MB")
            print(f"✅ Stations: {len(stations)}")
            if years:
                print(f"✅ Years: {sorted(years)}")
            if months:
                print(f"✅ Months: {sorted(months)}")
        else:
            print("⚠️  ไม่พบไฟล์ processed data")
            print("   สาเหตุ: Pipeline อาจยังไม่ merge และ clean data เสร็จ")
    else:
        print("⚠️  Directory ไม่มี")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 สรุป")
    print("=" * 70)
    
    status = {
        "PM2.5 Stations": pm25_path.exists(),
        "Weather Data": weather_dir.exists() and len(list(weather_dir.glob("*.parquet"))) > 0,
        "Static Features": static_path.exists(),
        "Processed Data": processed_dir.exists() and len(list(processed_dir.rglob("*.parquet"))) > 0,
    }
    
    for name, exists in status.items():
        status_icon = "✅" if exists else "⚠️ "
        print(f"{status_icon} {name}: {'พร้อมใช้งาน' if exists else 'ยังไม่พร้อม'}")
    
    print("\n💡 ข้อเสนอแนะ:")
    if not status["Weather Data"]:
        print("   - Weather data ยังไม่ collect: ตรวจสอบ Open-Meteo API")
    if not status["Processed Data"]:
        print("   - Processed data ยังไม่ merge: รัน pipeline ให้เสร็จ")
    print("=" * 70)

if __name__ == "__main__":
    check_all_data()

