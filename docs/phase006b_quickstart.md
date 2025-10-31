# Phase 006B Quick Start Guide

## Overview

Phase 006B tests APS framework on **text domain OOD generalization** using AG News dataset with domain shift.

## Prerequisites

```bash
# Install required packages (if not already installed)
pip install transformers datasets torch pandas matplotlib seaborn
```

## Running Experiments

### Option 1: Run All Experiments (Recommended)

Run all 5 experiments sequentially:

```bash
cd /Users/freeman.hui/Documents/Coding/atlasing_pattern_space
./scripts/run_phase006b_all.sh
```

This will run:
1. **baseline** (λ_T=0, λ_C=0, λ_E=0) - No APS components
2. **aps-T** (λ_T=1.0) - Topology preservation only  
3. **aps-C** (λ_C=1.0) - Causal invariance only
4. **aps-TC** (λ_T=1.0, λ_C=0.5) - Topology + Causality
5. **aps-full** (λ_T=1.0, λ_C=0.5, λ_E=0.1) - Full APS framework

**Time estimate**: ~2-3 hours per experiment on CPU (~10-15 hours total)

### Option 2: Run Individual Experiments

Run specific experiments:

```bash
# Baseline
python scripts/run_phase006b_text_ood.py --experiment baseline --epochs 30

# Topology only
python scripts/run_phase006b_text_ood.py --experiment aps-T --epochs 30

# Causality only
python scripts/run_phase006b_text_ood.py --experiment aps-C --epochs 30

# Topology + Causality
python scripts/run_phase006b_text_ood.py --experiment aps-TC --epochs 30

# Full APS
python scripts/run_phase006b_text_ood.py --experiment aps-full --epochs 30
```

### Option 3: Quick Testing

For quick testing with smaller dataset:

```bash
# Quick test (500 samples/domain, 5 epochs)
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --quick-test \
  --epochs 5

# Custom sample size
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --max-samples 1000 \
  --epochs 10
```

## Analyzing Results

After experiments complete:

```bash
# Generate summary tables and statistics
python scripts/analyze_phase006b_results.py

# Create visualization plots
python scripts/visualize_phase006b_results.py
```

## Output Files

Results are saved to `outputs/phase006b/`:

```
outputs/phase006b/
├── baseline/
│   ├── config.json          # Experiment configuration
│   ├── history.json         # Training curves
│   ├── final_metrics.json   # Final metrics
│   └── best_model.pt        # Best model checkpoint
├── aps-T/
├── aps-C/
├── aps-TC/
├── aps-full/
├── results_summary.csv      # Comparison table (CSV)
├── results_summary_table.md # Comparison table (Markdown)
├── accuracy_curves.png      # Training/OOD accuracy plots
├── loss_curves.png          # Loss curves
├── ood_comparison.png       # Bar chart comparison
├── ood_gap.png              # Generalization gap
└── improvement_over_baseline.png
```

## Viewing Results

### Summary Table

```bash
cat outputs/phase006b/results_summary_table.md
```

### Metrics for Specific Experiment

```bash
cat outputs/phase006b/baseline/final_metrics.json
```

### Training History

```bash
cat outputs/phase006b/baseline/history.json
```

## Tips

### Speed Up Experiments

1. **Enable caching** (first run will be slow, subsequent runs fast):
   ```python
   # In ExperimentConfig (scripts/run_phase006b_text_ood.py)
   use_cache: bool = True  # Line 55
   ```

2. **Use GPU** (if available):
   - BERT embedding computation will be 10-20x faster
   - Training will be 5-10x faster

3. **Reduce dataset size** for iteration:
   ```bash
   python scripts/run_phase006b_text_ood.py \
     --experiment baseline \
     --max-samples 5000 \
     --epochs 20
   ```

### Monitor Progress

Since experiments take time, you can monitor progress:

```bash
# Watch log file
tail -f outputs/phase006b/experiments.log

# Check current experiment status
ls -lht outputs/phase006b/*/best_model.pt
```

### Troubleshooting

**Issue**: Disk space errors
- **Solution**: Disable caching (`use_cache=False`, default)
- Embeddings use ~200MB per 10K samples without caching

**Issue**: Out of memory
- **Solution**: Reduce batch size:
  ```bash
  python scripts/run_phase006b_text_ood.py \
    --experiment baseline \
    --batch-size 32 \
    --epochs 30
  ```

**Issue**: Slow BERT computation
- **Solution**: Use caching for repeated runs or smaller `--max-samples`

## Next Steps After Experiments

1. **Analyze Results**
   ```bash
   python scripts/analyze_phase006b_results.py
   python scripts/visualize_phase006b_results.py
   ```

2. **Review Plots**
   - Check `outputs/phase006b/*.png` for visualizations
   - Look for improvements in OOD accuracy
   - Check generalization gap

3. **Statistical Testing**
   - Compare OOD accuracies across experiments
   - Check if improvements are significant

4. **Write Paper Section**
   - Document findings in `paper/sections/experiments.md`
   - Include tables and plots
   - Discuss which APS components help most

## Expected Results

Based on hypothesis:

| Experiment | Expected OOD Acc | Improvement |
|------------|------------------|-------------|
| baseline   | 55-60%          | -           |
| aps-T      | 58-63%          | +3-5%       |
| aps-C      | 62-68%          | +7-10%      |
| aps-TC     | 65-70%          | +10-12%     |
| aps-full   | 68-72%          | +13-15%     |

## Questions?

See detailed documentation:
- Implementation details: `docs/phase006b_text_ood.md`
- Dataset info: `src/aps/data/ag_news_ood.py`
- Experiment script: `scripts/run_phase006b_text_ood.py`
