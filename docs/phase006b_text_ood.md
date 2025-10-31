# Phase 006B: Text Domain OOD Generalization

## Overview

Phase 006B tests the APS framework on **text domain shift** using the AG News dataset. This is a simpler alternative to WILDS datasets for validating OOD generalization capabilities.

## Dataset: AG News with Domain Shift

### Task Setup
- **Base Dataset**: AG News 4-class classification (World/Sports/Business/Sci-Tech)
- **Domain Definition**: News categories treated as domains
- **Task**: Binary sentiment classification (positive/negative)
- **Sentiment Labels**: Synthetically generated from keyword heuristics

### Domain Split
- **Training Domains**: Sports (1), Business (2), Sci-Tech (3)
- **Test Domain (OOD)**: World (0)
- **Hypothesis**: Can APS learn sentiment representations invariant to topic?

### Data Statistics
- Train: 90,000 samples (30,000 per domain)
- Test (OOD): 1,900 samples
- Input: BERT-base embeddings (768-dim)
- Latent: 32-dim (default)

## Implementation

### Files Created
1. **`src/aps/data/ag_news_ood.py`**: Dataset loader with BERT embeddings
2. **`scripts/run_phase006b_text_ood.py`**: Main experiment script

### Key Features
- **Efficient Storage**: Uses float16 for embeddings (saves ~50% disk space)
- **Batch Processing**: Computes BERT embeddings in batches
- **Optional Caching**: Can cache embeddings (disabled by default to save space)
- **Flexible Sampling**: Can limit samples per domain for faster testing

## Usage

### Quick Test (100 samples/domain, 1 epoch)
```bash
python scripts/run_phase006b_text_ood.py \
  --experiment baseline \
  --max-samples 100 \
  --epochs 1 \
  --batch-size 16
```

### Experiments

#### 1. Baseline (No APS)
```bash
python scripts/run_phase006b_text_ood.py --experiment baseline --epochs 30
```
- Pure supervised learning
- λ_T=0, λ_C=0, λ_E=0

#### 2. APS-T (Topology Only)
```bash
python scripts/run_phase006b_text_ood.py --experiment aps-T --epochs 30
```
- Topology preservation via kNN loss
- λ_T=1.0, λ_C=0, λ_E=0

#### 3. APS-C (Causality Only)
```bash
python scripts/run_phase006b_text_ood.py --experiment aps-C --epochs 30
```
- HSIC independence loss
- λ_T=0, λ_C=1.0, λ_E=0

#### 4. APS-TC (Topology + Causality)
```bash
python scripts/run_phase006b_text_ood.py --experiment aps-TC --epochs 30
```
- Combined T+C
- λ_T=1.0, λ_C=0.5, λ_E=0

#### 5. APS-Full (T+C+E)
```bash
python scripts/run_phase006b_text_ood.py --experiment aps-full --epochs 30
```
- Full APS framework
- λ_T=1.0, λ_C=0.5, λ_E=0.1

### CLI Options
```
--experiment {baseline,aps-T,aps-C,aps-TC,aps-full}
  Experiment configuration
  
--max-samples N
  Limit to N samples per domain (for testing)
  
--quick-test
  Use 500 samples per domain
  
--epochs N
  Number of training epochs (default: 30)
  
--batch-size N
  Batch size (default: 64)
  
--latent-dim N
  Latent space dimension (default: 32)
  
--seed N
  Random seed (default: 42)
  
--output-dir PATH
  Output directory (default: ./outputs/phase006b)
```

## Outputs

### Directory Structure
```
outputs/phase006b/
├── baseline/
│   ├── config.json          # Experiment configuration
│   ├── history.json         # Training history (loss/acc per epoch)
│   ├── final_metrics.json   # Final OOD performance
│   └── best_model.pt        # Best model checkpoint
├── aps-T/
├── aps-C/
├── aps-TC/
└── aps-full/
```

### Metrics Tracked
- **Train Accuracy**: In-distribution performance
- **OOD Accuracy**: Out-of-distribution performance
- **OOD Gap**: Train acc - OOD acc (lower is better)
- **kNN Preservation**: Topology preservation metric

## Implementation Notes

### Memory Optimization
1. **Float16 Storage**: Embeddings stored as float16 (50% reduction)
2. **Batch Processing**: BERT embedding computation in batches
3. **No Caching Default**: Caching disabled to avoid 12GB cache files
4. **Configurable Sampling**: Can limit dataset size for testing

### Known Issues
1. **Disk Space**: Full dataset with caching requires ~12GB
   - Solution: Use `--max-samples` or disable caching
2. **BERT Computation Time**: Computing embeddings for 90K samples takes time
   - Solution: Use caching for repeated runs or smaller samples
3. **Sample Limiting Logic**: Currently limits total samples, not per-domain
   - Fix needed in `_load_data()` method

### Fixes Applied
1. ✅ Added error handling for corrupted cache files
2. ✅ Optimized embedding storage with float16
3. ✅ Disabled caching by default
4. ✅ Added `--max-samples` CLI option
5. ✅ Batch BERT computation

### TODO
1. Fix sample limiting to properly limit per domain
2. Add proper IRM implementation for causality loss
3. Add visualization of learned representations
4. Add statistical significance testing
5. Create comparison plots across experiments

## Expected Results

### Hypothesis
APS components (especially causality) should improve OOD generalization by learning topic-invariant sentiment representations.

### Predictions
- **Baseline**: ~55-60% OOD accuracy (random baseline: 50%)
- **APS-T**: ~58-63% OOD accuracy (topology helps local structure)
- **APS-C**: ~62-68% OOD accuracy (causality learns invariances)
- **APS-TC**: ~65-70% OOD accuracy (combined benefits)
- **APS-Full**: ~68-72% OOD accuracy (energy adds attractor basins)

## Next Steps

1. **Run Full Experiments**: Execute all 5 configurations with full dataset
2. **Analyze Results**: Compare OOD accuracy and training curves
3. **Ablation Studies**: Test different λ values
4. **Visualization**: Create t-SNE/UMAP plots of learned representations
5. **Write Paper Section**: Document findings for Phase 006B

## Resources

- AG News Dataset: https://huggingface.co/datasets/ag_news
- BERT: `bert-base-uncased` (768-dim)
- Paper Reference: See `paper/sections/experiments.md`
