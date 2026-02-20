#!/usr/bin/env python3
"""
Check if data is ready for training.
"""
import pandas as pd
from pathlib import Path
from utils.sliding_window import create_sliding_window
from config import PipelineConfig

def check_training_ready():
    """Check if data is ready for ST-UNN training."""
    print("=" * 70)
    print("🎯 Checking Training Data Readiness")
    print("=" * 70)
    
    config = PipelineConfig()
    
    # Check processed data
    processed_dir = Path("data/processed/station_level")
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        return False
    
    processed_files = list(processed_dir.rglob("*.parquet"))
    if not processed_files:
        print("❌ No processed data files found")
        print("   Waiting for pipeline to complete...")
        return False
    
    print(f"✅ Found {len(processed_files)} processed data files")
    
    # Load sample data
    print("\n📊 Loading sample data...")
    sample_file = processed_files[0]
    df = pd.read_parquet(sample_file)
    
    print(f"   File: {sample_file}")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Check required columns
    required_cols = ["timestamp", "station_id", "pm25"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ Missing required columns: {missing_cols}")
        return False
    
    # Check feature columns
    feature_cols = [
        "temperature", "humidity", "pressure",
        "u_wind", "v_wind", "precipitation", "solar",
        "fire_count", "land_use", "population_density",
        "hour_sin", "hour_cos", "day_of_week", "month"
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\n✅ Available features: {len(available_features)}/{len(feature_cols)}")
    print(f"   {', '.join(available_features)}")
    
    # Try to create sequences
    print("\n🧪 Testing sequence creation...")
    try:
        X, y = create_sliding_window(
            df,
            input_hours=config.model.INPUT_HOURS,
            output_hours=config.model.OUTPUT_HOURS,
            feature_cols=available_features,
            target_col="pm25"
        )
        
        if len(X) > 0:
            print(f"✅ Sequences created successfully!")
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
            print(f"   Sequences: {len(X)}")
            print("\n🎉 DATA READY FOR TRAINING!")
            return True
        else:
            print("⚠️  No sequences created (may need more data)")
            return False
            
    except Exception as e:
        print(f"❌ Error creating sequences: {e}")
        return False

if __name__ == "__main__":
    ready = check_training_ready()
    if not ready:
        print("\n⏳ Data not ready yet. Continue monitoring...")
        print("   Run: python3 monitor_pipeline.py")

