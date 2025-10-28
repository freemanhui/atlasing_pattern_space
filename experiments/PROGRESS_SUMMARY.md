# APS Experiments - Progress Summary

**Date**: January 2025  
**Branch**: `005-experiments`

## Overview

This document tracks the completion of the experimental infrastructure for the Atlasing Pattern Space (APS) framework. All core experimental scripts, utilities, and documentation are now production-ready.

---

## ✅ Completed Phases

### Phase 5.1: Infrastructure & Metrics ✓

**Objective**: Build foundational utilities for data loading and model evaluation.

**Completed Components**:

1. **Dataset Utilities** (`experiments/utils/datasets.py`)
   - `get_mnist_loaders()`: Standard MNIST train/test with normalization
   - `get_rotated_mnist()`: Rotated MNIST for OOD testing
   - `get_noisy_mnist()`: Gaussian noise injection at multiple σ levels
   - `get_fashion_mnist()`: FashionMNIST for transfer evaluation
   - `sample_k_shot()`: k-shot sampling for few-shot learning

2. **Metrics Module** (`experiments/utils/metrics.py`)
   - **Topology**: `trustworthiness()`, `continuity()`, `knn_preservation()`
   - **Clustering**: `clustering_metrics()` (ARI, NMI, silhouette)
   - **Independence**: `hsic_independence()` (RBF kernel HSIC)
   - **Task**: `reconstruction_error()`, `few_shot_accuracy()`
   - **Unified**: `evaluate_model()` - comprehensive metric aggregation

3. **Module Exports** (`experiments/utils/__init__.py`)
   - Clean API for importing datasets and metrics

**Validation**: All utilities tested with shape checks and basic functionality tests.

---

### Phase 5.2: Baseline MNIST Experiment ✓

**Objective**: Validate training and evaluation pipeline with a baseline experiment.

**Script**: `experiments/mnist_baseline.py`

**Features**:
- Train/eval modes
- 4 configurations: baseline, t-only, e-only, t+e
- Comprehensive metrics: reconstruction, topology, clustering
- 2D embedding visualization with scatter plots
- Auto-saves: checkpoints (`.pt`), metrics (`.json`), plots (`.png`)
- Device auto-detection (CPU/CUDA/MPS)

**Quick Test**:
```bash
python experiments/mnist_baseline.py --config baseline --epochs 5
```

**Validation**: Successfully trained baseline model in 5 epochs, generated metrics and plots. Pipeline confirmed working.

**Output Structure**:
```
outputs/baseline/
├── checkpoints/
│   └── {config}_model.pt
├── metrics/
│   └── {config}_metrics.json
└── plots/
    └── {config}_embedding.png
```

---

### Phase 5.3: Full Ablation Study ✓

**Objective**: Systematically evaluate all component combinations.

**Script**: `experiments/mnist_ablation.py`

**Configurations** (8 total):

| Config    | Topology (T) | Causality (C) | Energy (E) |
|-----------|--------------|---------------|------------|
| baseline  | ✗            | ✗             | ✗          |
| t_only    | ✓            | ✗             | ✗          |
| c_only    | ✗            | ✓             | ✗          |
| e_only    | ✗            | ✗             | ✓          |
| t_c       | ✓            | ✓             | ✗          |
| t_e       | ✓            | ✗             | ✓          |
| c_e       | ✗            | ✓             | ✓          |
| t_c_e     | ✓            | ✓             | ✓          |

**Evaluation Metrics**:
- Reconstruction error (MSE)
- Topology preservation (trustworthiness, continuity, kNN preservation)
- Clustering quality (ARI, NMI, silhouette)
- Independence (HSIC between latent and labels)

**Features**:
- Unified `AblationModel` with selective component activation
- Auto-saves: checkpoints, per-config metrics, embeddings, plots
- Summary table comparing all configurations
- Ready for full 50-epoch runs

**Usage**:
```bash
python experiments/mnist_ablation.py --epochs 50 --device mps
```

**Output Structure**:
```
outputs/ablation/
├── checkpoints/
│   ├── baseline.pt
│   ├── t_only.pt
│   ├── ...
│   └── t_c_e.pt
├── metrics/
│   ├── baseline_metrics.json
│   ├── ...
│   └── t_c_e_metrics.json
├── plots/
│   ├── baseline_embedding.png
│   ├── ...
│   └── t_c_e_embedding.png
└── ablation_summary.json
```

---

### Phase 5.4: OOD Robustness ✓

**Objective**: Test generalization under distribution shifts.

**Script**: `experiments/mnist_ood.py`

**OOD Scenarios**:

1. **Rotated MNIST**: 15°, 30°, 45°, 60°
2. **Noisy MNIST**: Gaussian noise σ = 0.1, 0.2, 0.3, 0.5
3. **FashionMNIST**: Zero-shot transfer evaluation

**Evaluation**:
- Reconstruction error (robustness to corruption)
- Topology preservation (structural degradation)
- Clustering quality (class separability)
- kNN accuracy (latent space classification)

**Features**:
- Loads trained checkpoints from ablation study
- Visualizes all OOD embeddings in grid layout
- Summary table showing degradation across scenarios
- Compares robustness across model configurations

**Usage**:
```bash
# Evaluate t_c_e configuration on all OOD scenarios
python experiments/mnist_ood.py --config t_c_e --device mps
```

**Output Structure**:
```
outputs/ood/
├── metrics/
│   └── {config}_ood_results.json
└── plots/
    └── {config}_ood_embeddings.png  # Grid: Original + 4 rotations + 4 noise + Fashion
```

---

### Phase 5.5: Few-Shot Learning ✓

**Objective**: Test if learned embeddings support efficient few-shot learning.

**Script**: `experiments/mnist_fewshot.py`

**k-shot Settings**: 1, 3, 5, 10 examples per class

**Classifiers**:
1. **Logistic Regression**: Linear decision boundary in latent space
2. **k-Nearest Neighbors**: Non-parametric local classification
3. **Prototypical Network**: Distance to class centroids

**Evaluation**:
- Accuracy averaged over multiple random trials (default: 5 trials)
- Standard deviation to measure robustness
- Confusion matrices for error analysis
- Learning curves showing accuracy vs. k-shot

**Hypothesis**: Topology-preserving and energy-shaped embeddings should enable better few-shot generalization through more structured latent spaces.

**Features**:
- Averages over multiple random k-shot samples
- Generates confusion matrices and learning curves
- Summary tables with mean ± std accuracy
- Compares all three classifier types

**Usage**:
```bash
# Evaluate t_c_e configuration with default k-shots
python experiments/mnist_fewshot.py --config t_c_e --device mps

# Custom k-shot values
python experiments/mnist_fewshot.py --config baseline --k-shots 1 5 10 20 --n-trials 10
```

**Output Structure**:
```
outputs/fewshot/
├── metrics/
│   └── {config}_fewshot_results.json
└── plots/
    ├── {config}_confusion_5shot.png
    └── {config}_learning_curves.png  # 3-panel: LogReg | kNN | Proto
```

---

## 📊 Experimental Pipeline

### Complete Workflow

```bash
# 1. Train all configurations (Phase 5.3)
python experiments/mnist_ablation.py --epochs 50 --device mps

# 2. Evaluate OOD robustness (Phase 5.4)
for config in baseline t_only c_only e_only t_c t_e c_e t_c_e; do
    python experiments/mnist_ood.py --config $config --device mps
done

# 3. Evaluate few-shot learning (Phase 5.5)
for config in baseline t_only c_only e_only t_c t_e c_e t_c_e; do
    python experiments/mnist_fewshot.py --config $config --device mps
done
```

### Quick Single-Config Test

```bash
# Train baseline for 5 epochs (quick validation)
python experiments/mnist_baseline.py --config baseline --epochs 5

# Train t_c_e for full 50 epochs
python experiments/mnist_ablation.py --epochs 50 --device mps

# Then evaluate t_c_e on all downstream tasks
python experiments/mnist_ood.py --config t_c_e --device mps
python experiments/mnist_fewshot.py --config t_c_e --device mps
```

---

## 📂 Repository Structure

```
experiments/
├── utils/
│   ├── __init__.py           # Module exports
│   ├── datasets.py           # Data loading utilities
│   └── metrics.py            # Evaluation metrics
├── mnist_baseline.py         # Phase 5.2: Baseline validation
├── mnist_ablation.py         # Phase 5.3: Full ablation study
├── mnist_ood.py              # Phase 5.4: OOD robustness
├── mnist_fewshot.py          # Phase 5.5: Few-shot learning
├── README.md                 # User-facing documentation
└── PROGRESS_SUMMARY.md       # This file

outputs/
├── baseline/                 # Phase 5.2 outputs
├── ablation/                 # Phase 5.3 outputs
├── ood/                      # Phase 5.4 outputs
└── fewshot/                  # Phase 5.5 outputs
```

---

## 🎯 Research Questions Addressed

### 1. Component Effectiveness (Phase 5.3)
- **Q**: How does each component (T, C, E) contribute to embedding quality?
- **A**: Systematic ablation quantifies individual and combined effects on reconstruction, topology, clustering, and independence.

### 2. Robustness (Phase 5.4)
- **Q**: Do TCE embeddings maintain structure under distribution shifts?
- **A**: OOD experiments measure degradation across rotations, noise, and domain transfer.

### 3. Few-Shot Learning (Phase 5.5)
- **Q**: Does better latent structure enable more efficient learning?
- **A**: k-shot experiments test if topology/energy shaping improves sample efficiency.

### 4. Synergy vs. Redundancy
- **Q**: Are components complementary or redundant?
- **A**: Pairwise configurations (t_c, t_e, c_e) reveal interaction effects.

---

## 🧪 Next Steps

### Immediate Actions (Ready to Run)

1. **Full Ablation Run** (Phase 5.3)
   ```bash
   python experiments/mnist_ablation.py --epochs 50 --device mps
   ```
   - Expected runtime: ~2-3 hours for 8 configs × 50 epochs
   - Generates all checkpoints needed for downstream tasks

2. **OOD Robustness Sweep** (Phase 5.4)
   ```bash
   for config in baseline t_c_e; do
       python experiments/mnist_ood.py --config $config --device mps
   done
   ```
   - Start with baseline and t_c_e for comparison

3. **Few-Shot Analysis** (Phase 5.5)
   ```bash
   for config in baseline t_c_e; do
       python experiments/mnist_fewshot.py --config $config --device mps
   done
   ```
   - Compare few-shot performance across configs

### Future Phases (Planned)

#### Phase 5.6: Publication Materials
- [ ] Generate publication-quality figures
- [ ] Create comparison tables across all experiments
- [ ] Statistical significance tests (paired t-tests, Wilcoxon)
- [ ] LaTeX tables for paper
- [ ] High-resolution plots with consistent styling

#### Phase 5.7: Extended Experiments
- [ ] Test on CIFAR-10 (convolutional encoder)
- [ ] Test on text data (GLOVE embeddings)
- [ ] Higher latent dimensions (16, 32, 64)
- [ ] Hyperparameter sensitivity analysis
- [ ] Runtime/memory profiling

#### Phase 5.8: Theoretical Analysis
- [ ] Topology: persistent homology validation
- [ ] Causality: independence test power analysis
- [ ] Energy: basin of attraction quantification
- [ ] Combined: emergent properties of TCE interaction

---

## 📈 Validation Status

| Phase | Script                  | Status | Tested | Notes                          |
|-------|-------------------------|--------|--------|--------------------------------|
| 5.1   | `utils/datasets.py`     | ✅     | ✅     | All loaders functional         |
| 5.1   | `utils/metrics.py`      | ✅     | ✅     | All metrics validated          |
| 5.2   | `mnist_baseline.py`     | ✅     | ✅     | 5-epoch test successful        |
| 5.3   | `mnist_ablation.py`     | ✅     | ⏳     | Ready for full 50-epoch run    |
| 5.4   | `mnist_ood.py`          | ✅     | ⏳     | Awaiting Phase 5.3 checkpoints |
| 5.5   | `mnist_fewshot.py`      | ✅     | ⏳     | Awaiting Phase 5.3 checkpoints |

**Legend**:
- ✅ Complete
- ⏳ Ready but awaiting dependencies
- ❌ Not started

---

## 🔬 Key Design Decisions

### 1. Unified Model Architecture
All experiments use the same `AblationModel` class with selective component activation. This ensures:
- Fair comparison (no architecture confounds)
- Code reusability
- Consistent hyperparameters

### 2. Separate Scripts for Each Task
Rather than one monolithic script, we have:
- **mnist_baseline.py**: Quick validation
- **mnist_ablation.py**: Systematic training
- **mnist_ood.py**: Robustness evaluation
- **mnist_fewshot.py**: Few-shot evaluation

**Benefits**:
- Modular and maintainable
- Can run experiments independently
- Easy to parallelize on compute cluster

### 3. JSON Metrics + Plots
Each experiment saves:
- **JSON files**: Structured metrics for programmatic analysis
- **PNG plots**: Visual inspection and publication figures

This dual format supports both automated analysis pipelines and manual review.

### 4. Checkpoint Reuse
Phases 5.4 and 5.5 load checkpoints from Phase 5.3, avoiding redundant training.

### 5. Reproducibility
- Fixed random seeds for dataset splits
- Multiple trials for few-shot (averages over randomness)
- Saved configs alongside results

---

## 🚀 Running Full Experiment Suite

### Prerequisites
```bash
# Ensure environment is active
source .venv/bin/activate  # or conda activate aps

# Verify installation
python -c "import aps; print('APS installed')"
```

### Full Pipeline (~3-4 hours on M1/M2 Mac)

```bash
#!/bin/bash
# run_all_experiments.sh

set -e  # Exit on error

echo "=== Phase 5.3: Training all configurations ==="
python experiments/mnist_ablation.py --epochs 50 --device mps

echo ""
echo "=== Phase 5.4: OOD robustness ==="
for config in baseline t_only c_only e_only t_c t_e c_e t_c_e; do
    echo "Testing $config on OOD scenarios..."
    python experiments/mnist_ood.py --config $config --device mps
done

echo ""
echo "=== Phase 5.5: Few-shot learning ==="
for config in baseline t_only c_only e_only t_c t_e c_e t_c_e; do
    echo "Testing $config on few-shot learning..."
    python experiments/mnist_fewshot.py --config $config --device mps
done

echo ""
echo "=== All experiments complete! ==="
echo "Results saved in outputs/ directory"
```

Make executable and run:
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## 📝 Documentation Status

- [x] Code-level docstrings in all modules
- [x] User-facing README.md
- [x] Progress tracking (this file)
- [x] Quick start guide
- [x] Example usage commands
- [ ] Jupyter notebook for result analysis (Future)
- [ ] Paper figures generation script (Future)

---

## 🎓 Citation

If you use this experimental infrastructure, please cite:

```bibtex
@software{aps_experiments_2025,
  title = {Atlasing Pattern Space: Experimental Infrastructure},
  author = {Freeman Hui},
  year = {2025},
  url = {https://github.com/yourusername/atlasing_pattern_space}
}
```

---

## 📞 Contact & Contribution

For questions or contributions:
1. Open an issue on GitHub
2. Submit a pull request with proposed changes
3. Contact: freeman.hui@example.com

---

**Last Updated**: January 2025  
**Branch**: `005-experiments`  
**Status**: ✅ Infrastructure Complete, Ready for Full Experiments
