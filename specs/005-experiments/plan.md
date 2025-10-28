# Phase 005: Experiments & Validation

**Status**: In Progress  
**Branch**: `005-experiments`  
**Dependencies**: Phase 004 (Integration) ✅ Complete

## Overview

Phase 005 focuses on running comprehensive experiments to validate the APS framework, reproduce key results, and demonstrate the effectiveness of combining T+C+E components. This includes benchmark experiments, ablation studies, and real-world applications.

## Objectives

1. **Validate Framework**: Prove that APS works as intended
2. **Ablation Studies**: Show each component contributes meaningfully
3. **Benchmark Performance**: Compare against baselines
4. **Demonstrate Utility**: Show practical applications
5. **Generate Results**: Create publication-ready figures and tables

## Experiment Categories

### 1. **Qualitative Visualization Experiments**
**Goal**: Visual demonstration of latent space structure

#### MNIST Embeddings
- **Setup**: Train APS on MNIST with 2D latent space
- **Configurations**: 
  - Baseline (no regularization)
  - T-only (topology preservation)
  - E-only (energy basins)
  - T+E (combined)
  - Full APS (T+C+E)
- **Metrics**:
  - Visual cluster separation
  - Within-class compactness
  - Between-class margins
- **Deliverables**:
  - 2D scatter plots colored by class
  - Energy landscape overlays
  - Memory pattern markers

#### Text Embeddings (Optional)
- **Setup**: Simple text classification (e.g., AG News subset)
- **Purpose**: Show APS works beyond images
- **Visualizations**: t-SNE/UMAP of APS latent space

### 2. **Quantitative Topology Experiments**
**Goal**: Measure topology preservation quality

#### Metrics
- **Trustworthiness**: Are close neighbors in embedding also close in input?
- **Continuity**: Are close neighbors in input also close in embedding?
- **kNN Preservation**: Jaccard similarity of k-NN sets

#### Experimental Setup
- **Dataset**: MNIST (10k samples)
- **Configurations**: Baseline, T-only, T+E, Full APS
- **Measurement**: Compute metrics over latent dimensions (2D to 32D)

#### Expected Results
- T-only: +10-20% improvement in trustworthiness/continuity
- T+E: Additional improvement from basin structure
- Full APS: Best overall topology preservation

### 3. **OOD Generalization Experiments**
**Goal**: Demonstrate robustness to distribution shift

#### Colored MNIST
- **Train**: MNIST with spurious color correlation (e.g., red digits are 0-4, blue are 5-9)
- **Test**: Flip correlation (red digits are 5-9, blue are 0-4)
- **Configurations**:
  - Baseline: Will overfit to color
  - C-only: Should ignore color
  - Full APS: Should be robust
- **Metrics**:
  - Train accuracy (all should be high)
  - Test accuracy on flipped (C-only and Full should maintain)
  - HSIC(latent, color) (should be low for C-only and Full)

#### Expected Results
- Baseline: >90% train, <50% test (catastrophic drop)
- C-only: >90% train, >85% test (robust)
- Full APS: >90% train, >88% test (best)

### 4. **Energy Basin Utility Experiments**
**Goal**: Show that energy basins improve downstream tasks

#### Few-Shot Classification
- **Setup**: Use pre-trained APS latent space
- **Task**: Classify with only 5 examples per class
- **Method**: Nearest memory pattern voting
- **Configurations**: Baseline vs E-only vs Full APS
- **Metrics**: Few-shot accuracy (1-shot, 5-shot)

#### Clustering Quality
- **Setup**: K-means on latent space
- **Metrics**: 
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Silhouette score
- **Expected**: E-only and Full APS produce clearer clusters

### 5. **Comprehensive Ablation Study**
**Goal**: Systematic comparison of all component combinations

#### Configurations
1. **Baseline**: No regularization (pure autoencoder)
2. **T-only**: Topology preservation only
3. **C-only**: Causality only (with synthetic nuisance)
4. **E-only**: Energy basins only
5. **T+C**: Topology + Causality
6. **T+E**: Topology + Energy
7. **C+E**: Causality + Energy
8. **T+C+E**: Full APS (all components)

#### Measurements (for each configuration)
- Reconstruction quality (MSE)
- Topology metrics (trustworthiness, continuity)
- Clustering quality (ARI, NMI)
- Training time (compute overhead)
- Latent space structure (visualization)

#### Expected Insights
- Each component contributes uniquely
- Combinations work synergistically
- Full APS achieves best overall performance

## Implementation Plan

### Phase 5.1: Infrastructure & Utilities
**Goal**: Common utilities for all experiments

#### Tasks
- [ ] Create `experiments/` directory structure
- [ ] Implement `experiments/utils/datasets.py`:
  - MNIST loader with variations
  - Colored MNIST generator
  - Few-shot split creation
- [ ] Implement `experiments/utils/metrics.py`:
  - Topology metrics (trustworthiness, continuity)
  - Clustering metrics (ARI, NMI, silhouette)
  - HSIC calculator
- [ ] Implement `experiments/utils/visualization.py`:
  - 2D embedding plots
  - Energy landscape overlays
  - Comparison plots (ablation charts)
- [ ] Template experiment runner script

**Deliverables**: Shared utilities for all experiments

### Phase 5.2: Baseline MNIST Experiment
**Goal**: Establish baseline and validate pipeline

#### Tasks
- [ ] Create `experiments/mnist_baseline.py`
- [ ] Train baseline autoencoder (no regularization)
- [ ] Train T-only configuration
- [ ] Compare reconstruction quality
- [ ] Measure topology metrics
- [ ] Generate visualizations

**Success Criteria**: 
- Scripts run without errors
- Visualizations are clear
- T-only shows topology improvement

### Phase 5.3: Full Ablation Study
**Goal**: Run all 8 configurations systematically

#### Tasks
- [ ] Create `experiments/run_ablation.py`
- [ ] Train all 8 configurations (can parallelize)
- [ ] Collect all metrics
- [ ] Generate comparison plots:
  - Loss curves for each config
  - Metric comparison table
  - Visual embedding grid (8 plots)
- [ ] Statistical significance testing

**Deliverables**: 
- Complete ablation results
- Publication-ready figures

### Phase 5.4: OOD Robustness Experiment
**Goal**: Demonstrate causal robustness

#### Tasks
- [ ] Create `experiments/colored_mnist.py`
- [ ] Generate colored MNIST dataset
- [ ] Train with color as nuisance
- [ ] Test on flipped correlation
- [ ] Measure HSIC(latent, color)
- [ ] Create OOD performance plots

**Success Criteria**:
- C-only and Full APS maintain >85% OOD accuracy
- Baseline drops to ~50% (random chance)

### Phase 5.5: Energy Basin Applications
**Goal**: Show practical utility of energy basins

#### Tasks
- [ ] Create `experiments/few_shot.py`
- [ ] Train APS with energy component
- [ ] Extract memory patterns
- [ ] Implement nearest-pattern classifier
- [ ] Test with 1-shot, 5-shot, 10-shot
- [ ] Compare vs baseline k-NN

**Success Criteria**:
- Energy-based method outperforms baseline
- Clear improvement in few-shot settings

### Phase 5.6: Results Summary & Documentation
**Goal**: Package all results for publication

#### Tasks
- [ ] Create `experiments/README.md` with all results
- [ ] Generate summary tables (LaTeX format)
- [ ] Create figure directory with all plots
- [ ] Write interpretation of results
- [ ] Create Jupyter notebook walkthrough
- [ ] Update main README with highlights

**Deliverables**:
- Complete experimental results
- Publication-ready materials
- Reproducible scripts

## Directory Structure

```
experiments/
├── README.md                     # Results summary
├── utils/
│   ├── __init__.py
│   ├── datasets.py               # Data loading utilities
│   ├── metrics.py                # Evaluation metrics
│   └── visualization.py          # Plotting utilities
├── configs/                      # Experiment configurations
│   ├── baseline.yaml
│   ├── t_only.yaml
│   ├── full_aps.yaml
│   └── ...
├── mnist_baseline.py             # Phase 5.2
├── run_ablation.py               # Phase 5.3
├── colored_mnist.py              # Phase 5.4
├── few_shot.py                   # Phase 5.5
├── results/                      # Generated results
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
└── notebooks/
    └── results_analysis.ipynb    # Interactive analysis
```

## Success Metrics

| Experiment | Metric | Target | Status |
|------------|--------|--------|--------|
| Topology | Trustworthiness (T-only) | +10-20% vs baseline | 📅 |
| Topology | Continuity (T-only) | +10-20% vs baseline | 📅 |
| OOD | Accuracy drop (Colored MNIST) | <10% for C-only/Full | 📅 |
| OOD | HSIC(z, color) | <0.1 for C-only/Full | 📅 |
| Clustering | ARI (E-only) | >0.8 | 📅 |
| Few-shot | 5-shot accuracy | >baseline k-NN | 📅 |
| Ablation | Full APS | Best on aggregate | 📅 |

## Configuration Standards

All experiments should use consistent hyperparameters where possible:

```python
# Model architecture
LATENT_DIM = 2  # For visualization (can vary)
HIDDEN_DIMS = [128, 64]  # Standard MLP

# Training
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
OPTIMIZER = 'adam'

# Loss weights (vary by experiment)
LAMBDA_T = 1.0  # Topology
LAMBDA_C = 1.0  # Causality
LAMBDA_E = 0.5  # Energy

# Topology
TOPO_K = 10  # kNN neighbors

# Energy
N_MEM = 10  # Memory patterns
BETA = 5.0  # Basin sharpness

# Causality
HSIC_SIGMA = 1.0  # RBF bandwidth
```

## Timeline

| Sub-Phase | Duration | Dependencies |
|-----------|----------|--------------|
| 5.1 Infrastructure | 2 days | None |
| 5.2 Baseline | 1 day | 5.1 |
| 5.3 Ablation | 2 days | 5.1, 5.2 |
| 5.4 OOD | 2 days | 5.1 |
| 5.5 Energy Apps | 2 days | 5.1 |
| 5.6 Documentation | 2 days | 5.2-5.5 |

**Total**: ~2 weeks

## Notes

- **Reproducibility**: Set random seeds for all experiments
- **Efficiency**: Can run ablation configurations in parallel
- **Visualization**: Use consistent color schemes across all figures
- **Statistical Testing**: Report confidence intervals where applicable
- **Checkpoints**: Save all trained models for analysis

## References

Key papers for metrics and methods:
- Trustworthiness/Continuity: Venna & Kaski (2001)
- Colored MNIST: Arjovsky et al. (2019) - IRM paper
- Clustering metrics: Vinh et al. (2010) - NMI
- Few-shot learning: Snell et al. (2017) - Prototypical Networks
