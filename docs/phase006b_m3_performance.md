# Phase 006B Performance on Apple M3

## MPS Acceleration ⚡

Your system has **MPS (Metal Performance Shaders) enabled**, which provides GPU acceleration on Apple Silicon.

```
✅ MPS available: True
✅ MPS built: True
✅ PyTorch version: 2.5.1
```

The code has been automatically configured to use MPS when available.

## Updated Time Estimates

### Full Dataset (90K train, 1.9K test, 30 epochs)

| Component | CPU (Estimated) | MPS (M3) | Speedup |
|-----------|----------------|----------|---------|
| BERT Embeddings | ~50 min | ~10-15 min | 3-5x |
| Training (30 epochs) | ~8 min | ~2-3 min | 3-4x |
| **Per Experiment** | ~60 min | **~15-20 min** | **~3-4x** |
| **All 5 Experiments** | ~5 hours | **~1.5-2 hours** | **~3x** |

### With Caching (Subsequent Runs)

Once BERT embeddings are cached:

| Configuration | Time |
|---------------|------|
| First run (compute embeddings) | ~15-20 min |
| Subsequent runs (cached) | **~2-3 min** |

## Performance Tips for M3

### 1. Enable Caching (Recommended)

For repeated experiments, enable caching to skip BERT computation:

```python
# In scripts/run_phase006b_text_ood.py, line 55
use_cache: bool = True  # Change from False
```

**Result**: First run ~15 min, subsequent runs ~2-3 min per experiment

### 2. Increase Batch Size

M3 has sufficient unified memory to use larger batches:

```bash
# Try batch size 128 or 256
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --batch-size 128 \
  --epochs 30
```

**Typical gains**: 10-20% faster training

### 3. Mixed Precision (Optional)

PyTorch on M3 supports float16 for additional speedup:

```python
# Add to training loop (advanced)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = ...
scaler.scale(loss).backward()
```

**Potential gains**: Additional 10-30% speedup

### 4. Parallel Data Loading

Re-enable parallel workers now that tokenizer warnings are handled:

```python
# In ExperimentConfig (line 54)
num_workers: int = 4  # Change from 0
```

**Result**: Faster data loading (minor impact with MPS)

## Realistic Timeline for Full Experiments

### Conservative Estimate (Safe)
- **All 5 experiments**: ~2 hours
- **Analysis + Visualization**: ~5 minutes
- **Total**: **~2 hours 5 minutes**

### Optimistic Estimate (With Optimizations)
- **First experiment** (compute embeddings): ~15 min
- **Remaining 4 experiments** (cached): ~2-3 min each = ~10 min
- **Total experiments**: **~25 minutes**
- **Analysis + Visualization**: ~5 minutes
- **Grand Total**: **~30 minutes** ⚡

## Recommended Workflow for M3

### Option 1: Quick Iteration (Recommended)

Enable caching and run experiments one by one:

```bash
# First run computes embeddings (~15 min)
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --epochs 30

# Subsequent runs use cache (~2-3 min each)
python scripts/run_phase006b_text_ood.py --experiment aps-T --epochs 30
python scripts/run_phase006b_text_ood.py --experiment aps-C --epochs 30
python scripts/run_phase006b_text_ood.py --experiment aps-TC --epochs 30
python scripts/run_phase006b_text_ood.py --experiment aps-full --epochs 30
```

**Total time**: ~25-30 minutes

### Option 2: Batch Run

Use the batch script (automatically uses MPS):

```bash
./scripts/run_phase006b_all.sh
```

**Total time**: ~1.5-2 hours (first run without cache)

## Monitoring Performance

### Check Device Usage

```bash
# Run experiment and check device
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --max-samples 100 \
  --epochs 1
```

Look for:
```
Device: mps
Using device: mps
```

### Monitor Memory

```bash
# Check memory usage during training
activity monitor  # Look for Python process
```

M3 unified memory: Models should use ~2-3GB

## Comparison: M3 vs Other Hardware

| Hardware | Time (All 5 Exp) | Notes |
|----------|------------------|-------|
| CPU only | ~5-10 hours | No acceleration |
| M3 (MPS) | **~1.5-2 hours** | Your setup |
| M3 (cached) | **~25-30 min** | After first run |
| NVIDIA GPU | ~20-30 min | High-end datacenter |

**Your M3 with caching is competitive with datacenter GPUs!** 🚀

## Troubleshooting

### Issue: Still using CPU

**Check**:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Solution**: Make sure PyTorch 2.0+ is installed:
```bash
pip install --upgrade torch torchvision
```

### Issue: MPS out of memory

**Solution**: Reduce batch size:
```bash
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --batch-size 32 \
  --epochs 30
```

### Issue: Slower than expected

**Check**:
1. Verify MPS is being used (look for "Device: mps" in output)
2. Close other memory-intensive apps
3. Ensure not using power saving mode

## Recommendations

For your M3 system:

✅ **Do**:
- Enable caching for repeated experiments
- Use batch size 64-128
- Run experiments individually for faster iteration
- Monitor first run to estimate remaining time

❌ **Don't**:
- Use CPU-only mode (MPS is much faster)
- Run other heavy tasks during experiments
- Use batch size > 256 (diminishing returns + memory risk)

## Expected Timeline

**Starting now with optimized settings**:

```
Time: 00:00 - Enable caching, start first experiment
Time: 00:15 - First experiment done (embeddings cached)
Time: 00:18 - Second experiment done
Time: 00:21 - Third experiment done
Time: 00:24 - Fourth experiment done
Time: 00:27 - Fifth experiment done
Time: 00:32 - Analysis and visualization complete
```

**Total: ~30 minutes from start to finish!** ⚡

## Conclusion

Your Apple M3 with MPS acceleration will complete Phase 006B experiments **significantly faster** than the original estimates:

- **Original estimate** (CPU): ~10-15 hours
- **Your M3 estimate** (no cache): ~1.5-2 hours
- **Your M3 estimate** (with cache): **~25-30 minutes** 🎯

You're ready to proceed! 🚀
