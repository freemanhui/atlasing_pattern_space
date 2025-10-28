# APS Experiments

This directory contains experimental validation of the APS (Atlasing Pattern Space) framework.

## Overview

Phase 005 validates the APS framework through comprehensive experiments demonstrating:
- **Topology Preservation**: T component improves manifold structure
- **Causal Robustness**: C component prevents spurious correlations
- **Energy Basin Utility**: E component enables few-shot learning
- **Synergistic Effects**: Combined T+C+E achieves best results

## Directory Structure

```
experiments/
├── README.md                    # This file
├── utils/                       # Shared utilities
│   ├── datasets.py              # Data loaders (MNIST, Colored MNIST)
│   ├── metrics.py               # Evaluation metrics
│   └── __init__.py
├── mnist_baseline.py            # Phase 5.2 - Baseline experiment ✅
├── run_ablation.py              # Phase 5.3 - Full ablation (TODO)
├── colored_mnist.py             # Phase 5.4 - OOD robustness (TODO)
├── few_shot.py                  # Phase 5.5 - Few-shot learning (TODO)
├── configs/                     # Experiment configurations
├── results/                     # Experiment outputs
│   ├── figures/                 # Generated plots
│   ├── tables/                  # Metric tables
│   └── checkpoints/             # Trained models
└── notebooks/                   # Analysis notebooks
```

## Quick Start

### Install Dependencies

The experiments use the main APS package:
```bash
pip install -e .
```

### Run Baseline Experiment

Train a baseline autoencoder:
```bash
python experiments/mnist_baseline.py --config baseline --epochs 50
```

Train with topology preservation:
```bash
python experiments/mnist_baseline.py --config t-only --epochs 50
```

Train with energy basins:
```bash
python experiments/mnist_baseline.py --config e-only --epochs 50
```

### View Results

Results are saved to:
- **Figures**: `experiments/results/figures/`
- **Metrics**: `experiments/results/tables/`
- **Checkpoints**: `experiments/results/checkpoints/`

## Available Experiments

### ✅ Phase 5.2: MNIST Baseline (`mnist_baseline.py`)

**Status**: Complete

Validates the experimental pipeline with baseline MNIST experiments.

**Configurations**:
- `baseline`: Pure autoencoder (no regularization)
- `t-only`: Topology preservation only (λ_T=1.0)
- `e-only`: Energy basins only (λ_E=0.5)
- `t+e`: Combined topology + energy

**Usage**:
```bash
# Train baseline
python experiments/mnist_baseline.py --config baseline --epochs 50 --device cpu

# Train T-only
python experiments/mnist_baseline.py --config t-only --epochs 50 --device cpu

# Evaluate only (after training)
python experiments/mnist_baseline.py --config baseline --eval-only
```

**Metrics Computed**:
- Topology: trustworthiness, continuity, kNN preservation
- Clustering: ARI, NMI, silhouette score
- Reconstruction: MSE

**Outputs**:
- 2D embedding visualization (for latent_dim=2)
- Metrics JSON file
- Trained model checkpoint

### 📅 Phase 5.3: Full Ablation Study (`run_ablation.py`)

**Status**: Planned

Systematic comparison of all 8 configurations:
1. Baseline (no regularization)
2. T-only
3. C-only
4. E-only
5. T+C
6. T+E
7. C+E
8. T+C+E (Full APS)

**Expected Outputs**:
- Comparison table of all metrics
- Grid of 2D embeddings (8 plots)
- Statistical significance tests

### 📅 Phase 5.4: OOD Robustness (`colored_mnist.py`)

**Status**: Planned

Colored MNIST experiment to demonstrate causal robustness.

**Setup**:
- Train set: Spurious color-digit correlation (90%)
- Test set: Flipped correlation
- Measure: OOD accuracy drop, HSIC(latent, color)

**Expected Results**:
- Baseline: >90% train, <50% test (catastrophic)
- C-only/Full: >90% train, >85% test (robust)

### 📅 Phase 5.5: Few-Shot Learning (`few_shot.py`)

**Status**: Planned

Demonstrate utility of energy basins for few-shot classification.

**Method**:
- Train APS with energy component
- Extract memory patterns
- Classify with k-shot examples (k=1, 5, 10)
- Compare vs baseline kNN

### 📅 Phase 5.6: Documentation

**Status**: Planned

Package all results for publication:
- Results summary
- Publication-ready figures
- Jupyter notebook walkthrough
- LaTeX tables

## Experimental Infrastructure

### Datasets (`utils/datasets.py`)

**Standard MNIST**:
```python
from utils import get_mnist_dataloaders

train_loader, val_loader, test_loader = get_mnist_dataloaders(
    batch_size=128,
    val_split=0.1,
    flatten=True,
)
```

**Colored MNIST** (for OOD):
```python
from utils import get_colored_mnist_dataloaders

train_loader, val_loader, test_loader = get_colored_mnist_dataloaders(
    train_correlation=0.9,  # Strong spurious correlation
    test_correlation=0.1,   # Flipped for OOD test
)
```

**Few-Shot Split**:
```python
from utils import create_few_shot_split

support_indices, query_indices = create_few_shot_split(
    dataset, n_way=10, k_shot=5, n_query=15
)
```

### Metrics (`utils/metrics.py`)

**Topology Metrics**:
```python
from utils import compute_topology_metrics

metrics = compute_topology_metrics(X_orig, X_emb, k=10)
# Returns: trustworthiness, continuity, knn_preservation
```

**Clustering Metrics**:
```python
from utils import compute_clustering_metrics

metrics = compute_clustering_metrics(X_emb, labels_true)
# Returns: ARI, NMI, silhouette
```

**HSIC (Independence)**:
```python
from utils import compute_hsic

hsic_value = compute_hsic(Z, nuisance_vars)
# Lower = more independent
```

**Few-Shot Accuracy**:
```python
from utils import few_shot_accuracy

acc = few_shot_accuracy(
    support_embeddings, support_labels,
    query_embeddings, query_labels,
    method='prototype'  # or 'nearest'
)
```

## Results

### Baseline Results (5 epochs, validation)

| Metric | Baseline | Expected (50 epochs) |
|--------|----------|----------------------|
| Trustworthiness | 0.91 | ~0.93 |
| Continuity | 0.95 | ~0.96 |
| ARI | 0.36 | ~0.50 |
| NMI | 0.54 | ~0.65 |
| Reconstruction MSE | 0.040 | ~0.035 |

*Note: Full results pending 50-epoch runs*

### Expected Improvements (T-only vs Baseline)

Based on APS framework design:
- **Trustworthiness**: +10-20%
- **Continuity**: +10-20%
- **kNN Preservation**: +15-25%
- **Clustering**: Improved visual separation

## Reproducibility

All experiments use fixed random seeds for reproducibility:
```bash
# Default seed=42
python experiments/mnist_baseline.py --config baseline --seed 42
```

Hardware used:
- CPU: Training validated on CPU
- GPU: Compatible with CUDA/MPS for faster training

## Citation

If you use these experiments, please cite:

```bibtex
@software{aps_experiments,
  title={APS Framework Experimental Validation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/atlasing_pattern_space}
}
```

## Contributing

To add new experiments:
1. Create new script in `experiments/`
2. Use utilities from `utils/` for consistency
3. Save results to `results/` directory
4. Update this README

## License

[Your License]
