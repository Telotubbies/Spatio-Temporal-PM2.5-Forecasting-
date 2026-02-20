# PM2.5 Forecasting Pipeline - Production Ready Implementation

## 🎯 Overview

Production-ready data pipeline for PM2.5 forecasting in Bangkok with:
- **16+ years** of historical data (2010-present)
- **AI Engineering standards** (error handling, validation, logging)
- **AMD GPU support** (ROCm for 7800XT)
- **Multi-source data collection** (Air4Thai, Open-Meteo, NASA FIRMS)

## ✨ Features

- ✅ Custom exception hierarchy
- ✅ Data validation at boundaries
- ✅ Structured logging
- ✅ Type hints throughout
- ✅ Modular architecture
- ✅ Historical data from 2010

## 📊 Data Sources

- **PM2.5**: Air4Thai API (82 stations in Bangkok)
- **Weather**: Open-Meteo Historical API (2010-present)
- **Fire**: NASA FIRMS (placeholder)
- **Static**: WorldCover, WorldPop (optional)

## 🔧 Technical Details

- **Language**: Python 3.11+
- **Storage**: Parquet (partitioned by year/month/station_id)
- **GPU**: ROCm support for AMD 7800XT
- **Architecture**: Clean separation (collectors, features, pipeline)

## 📝 Testing

- [x] Import tests passed
- [x] Station collection working (82 stations found)
- [x] Weather API integration working
- [x] Error handling tested
- [x] Data validation tested

## 🚀 Usage

```bash
# Run pipeline
python3 run_pipeline.py

# Or use notebook
jupyter notebook pipline.ipynb
```

## 📚 Documentation

- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `AI_ENGINEERING_CHECKLIST.md` - Standards compliance
- `HISTORICAL_DATA.md` - Historical data collection guide

