# CPU Training Optimization Guide

## Problem Diagnosed
Training was hanging during the first batch processing when using CPU with suboptimal parameters.

## Root Cause
- **Large batch size (32)** on CPU causes memory pressure and slow processing
- **num_workers=2** for data loading causes deadlocks in containerized environments
- **Large model (MONAI UNet, 1.6M parameters)** on CPU is very slow

## ✅ AUTOMATIC FIX IMPLEMENTED
The system now automatically optimizes CPU training parameters:

### Auto-Applied Fixes
- `num_workers` is automatically set to 0 when device is 'cpu' or 'auto'
- Prevents deadlocks in Docker containers
- Applied in `training_utils.py` during command building

### Log Output
When the fix is applied, you'll see:
```
[CPU_OPTIMIZATION] Forcing num_workers=0 for CPU training (was 2)
```

## Recommended Settings for CPU Training

### Optimal Parameters
```json
{
  "batch_size": 4,           // Reduced from 32
  "num_workers": 0,          // Disable multiprocessing for CPU
  "epochs": 2,               // Reduce for testing
  "learning_rate": 0.01,     // Higher LR for faster convergence
  "resolution": "128",       // Smaller images for faster processing
  "crop_size": 128,          // Match resolution
  "device": "cpu"
}
```

### Why These Settings Work Better

1. **batch_size: 4**
   - Reduces memory usage significantly
   - Allows processing to complete without hanging
   - Still provides stable gradients

2. **num_workers: 0** 
   - Eliminates multiprocessing deadlocks in Docker
   - Simplifies data loading pipeline
   - Reduces resource contention

3. **resolution: 128**
   - 4x faster processing than 256x256
   - Reduces memory requirements
   - Still meaningful for training

4. **epochs: 2**
   - Quick testing to verify training works
   - Can be increased once stability is confirmed

## Detection of Hanging Training

Signs that training is hanging:
- Only system monitoring logs appear
- No epoch progress logs
- CPU usage is low but constant
- No batch processing logs

## Quick Fix Commands

If training hangs:

```bash
# 1. Check active tasks
docker exec -it celery-training celery -A core.celery_app inspect active

# 2. Find hanging process
docker exec -it celery-training ps aux

# 3. Kill hanging process (replace PID)
docker exec -it celery-training kill -9 <PID>

# 4. Update model status
docker exec -it web python core/manage.py shell -c "
from core.apps.ml_manager.models import MLModel
model = MLModel.objects.get(id=<MODEL_ID>)
model.status = 'failed'
model.save()
"
```

## GPU Training Alternative

For better performance, consider using GPU training with:
```yaml
# In docker-compose
services:
  celery-training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

With GPU, you can use:
- batch_size: 16-32
- num_workers: 2-4
- resolution: 256 or higher
