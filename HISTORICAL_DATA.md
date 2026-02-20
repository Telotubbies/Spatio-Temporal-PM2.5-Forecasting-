# 📊 Historical Data Collection (ตั้งแต่ปี 2010)

## การตั้งค่า

Pipeline ถูกตั้งค่าให้ดึงข้อมูลจาก Open-Meteo ตั้งแต่ปี 2010 โดยอัตโนมัติ

### การทำงาน

1. **Historical API**: ใช้ `archive-api.open-meteo.com` สำหรับข้อมูลย้อนหลัง
2. **Forecast API**: ใช้ `api.open-meteo.com` สำหรับข้อมูลปัจจุบัน
3. **Chunking**: แบ่งการดึงข้อมูลเป็นช่วงๆ (ปีละครั้ง) เพื่อลดขนาด API call
4. **Batch Processing**: รวมหลาย locations ใน call เดียว

### วันที่เริ่มต้น

- **Default**: 2010-01-01
- **Configurable**: แก้ไขใน `config.py` → `HISTORICAL_START_YEAR`

### ตัวอย่างการใช้งาน

```python
from pipeline import PM25Pipeline
from config import PipelineConfig
from datetime import datetime

config = PipelineConfig()
pipeline = PM25Pipeline(config)

# ดึงข้อมูลตั้งแต่ 2010
start_date = datetime(2010, 1, 1)
end_date = datetime.utcnow()

result = pipeline.run(start_date=start_date, end_date=end_date)
```

### ใน Jupyter Notebook

Notebook (`pipline.ipynb`) ถูกตั้งค่าให้เริ่มตั้งแต่ปี 2010 แล้ว:

```python
start_date = datetime(2010, 1, 1)  # เริ่มตั้งแต่ปี 2010
```

### ข้อมูลที่จะได้

- **Temperature** (temperature_2m)
- **Humidity** (relative_humidity_2m)
- **Pressure** (surface_pressure)
- **Wind Speed** (wind_speed_10m)
- **Wind Direction** (wind_direction_10m)
- **Precipitation** (precipitation)
- **Solar Radiation** (shortwave_radiation)

### หมายเหตุ

- **เวลา**: ข้อมูลทุกชั่วโมง (hourly)
- **Timezone**: UTC
- **Storage**: Parquet format, partitioned by year/month/station_id
- **Duration**: การดึงข้อมูล 14+ ปี อาจใช้เวลานาน (ขึ้นอยู่กับจำนวน stations)

### การปรับแต่ง

แก้ไขใน `config.py`:

```python
HISTORICAL_START_YEAR: int = 2010  # เปลี่ยนปีเริ่มต้น
CHUNK_SIZE_DAYS: int = 365  # ขนาด chunk (วัน)
BATCH_SIZE: int = 50  # จำนวน locations ต่อ batch call
```

### Troubleshooting

**ปัญหา**: API timeout
- **แก้ไข**: ลด `CHUNK_SIZE_DAYS` หรือ `BATCH_SIZE`

**ปัญหา**: ข้อมูลไม่ครบ
- **ตรวจสอบ**: Log files ใน `logs/` directory
- **แก้ไข**: รันใหม่เฉพาะช่วงที่ขาด

**ปัญหา**: Memory หมด
- **แก้ไข**: ลด `BATCH_SIZE` และ `CHUNK_SIZE_DAYS`

