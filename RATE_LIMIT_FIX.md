# 🔧 Rate Limit Fix - Open-Meteo API

## ✅ Changes Applied

### 1. Reduced Batch Size
- **Before**: `BATCH_SIZE = 50`
- **After**: `BATCH_SIZE = 10`
- **Impact**: ลดจำนวน locations ต่อ request จาก 50 เป็น 10

### 2. Added Request Delay
- **New Config**: `REQUEST_DELAY_SECONDS = 3.0`
- **Implementation**: เพิ่ม delay 3 วินาทีระหว่างแต่ละ batch request
- **Impact**: ลดความถี่ของ requests

### 3. Rate Limit Handling
- **Added**: Automatic retry when receiving 429 status code
- **Behavior**: อ่าน `Retry-After` header และรอตามที่ API ระบุ
- **Impact**: Handle rate limit gracefully

## 📊 Expected Impact

### Before (Rate Limited)
- 82 stations ÷ 50 per batch = ~2 batches per chunk
- 16 years × 365 days = 16 chunks
- Total: ~32 requests in quick succession
- **Result**: 429 Too Many Requests

### After (With Fix)
- 82 stations ÷ 10 per batch = ~9 batches per chunk
- Delay: 3 seconds between batches
- 16 chunks × 9 batches = 144 batches
- Total time: ~7-8 minutes (with delays)
- **Result**: Should avoid rate limit

## 🚀 Usage

รัน pipeline ตามปกติ:

```bash
python3 run_pipeline.py
```

หรือใช้ notebook:

```python
from pipeline import PM25Pipeline
from config import PipelineConfig

config = PipelineConfig()
pipeline = PM25Pipeline(config)

# Run pipeline
result = pipeline.run()
```

## ⚙️ Configuration

แก้ไขใน `config.py`:

```python
BATCH_SIZE: int = 10  # Adjust if needed (lower = slower but safer)
REQUEST_DELAY_SECONDS: float = 3.0  # Adjust delay (higher = safer)
```

## 📝 Notes

1. **Slower but Safer**: การ collect จะช้าลง แต่จะไม่โดน rate limit
2. **Automatic Retry**: ถ้ายังโดน rate limit ระบบจะ retry อัตโนมัติ
3. **Progress Tracking**: ดู logs เพื่อติดตามความคืบหน้า

## 🔍 Monitor Progress

```bash
# Watch logs
tail -f logs/pipeline.log

# Check for rate limit errors
grep -i "rate\|429" logs/pipeline.log
```

