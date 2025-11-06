# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Atlasing Pattern Space (APS)** is a research framework for pattern-space embeddings that integrate three key dimensions:
- **Topology (T)**: kNN-based topology preservation for structural relationships
- **Causality (C)**: IRM-style invariance and HSIC independence for causal factors
- **Energy/Attractors (E)**: Memory-based energy functions to shape latent space basins

The combined objective minimizes: `L_task + λ_T * L_topo + λ_C * L_causal + λ_G * L_energy`

## Development Commands

### Installation
```bash
# Basic installation
pip install -e .

# With all optional dependencies (topology, causality, dev tools)
pip install -e ".[topology,causality,dev]"

# Using Make shortcuts
make setup    # Basic setup with dev tools
make dev      # Full setup with all extras
```

### Running Experiments
```bash
# Run unified APS experiments with ablation studies
python scripts/run_aps_experiment.py --experiment full --epochs 100
python scripts/run_aps_experiment.py --experiment T-only --epochs 100
python scripts/run_aps_experiment.py --experiment baseline --epochs 100

# Custom configuration
python scripts/run_aps_experiment.py --latent 10 --lambda-T 0.5 --lambda-E 0.1 --epochs 50

# With tensorboard logging
python scripts/run_aps_experiment.py --experiment full --tensorboard --epochs 100

# ColoredMNIST causal learning experiments (NEW)
python scripts/run_colored_mnist.py --experiment all --epochs 50  # Run all ablations
python scripts/run_colored_mnist.py --experiment aps-c --epochs 50  # Causality only
python scripts/run_colored_mnist.py --experiment aps-full --epochs 50  # Full APS (T+C+E)
python scripts/run_colored_mnist.py --experiment baseline --epochs 30  # No APS

# Legacy demos (individual components)
python scripts/run_topo_ae.py --latent 2 --epochs 100 --topo-weight 1.0
python scripts/run_energy_demo.py

# Using CLI tool
aps fit-topo --latent 2 --epochs 80 --topo-k 8 --topo-weight 1.0

# Using Make shortcuts
make topo        # Run topology demo
make energy      # Run energy demo
make dashboard   # Create 2D interactive dashboards
make viz3d       # Create 3D surface visualizations (2D latent + energy height)
make embedding3d # Create true 3D latent space visualizations
```

### Testing & Quality
```bash
# Run tests
pytest -q

# Run linting
ruff check .

# Using Make shortcuts
make test
make lint
```

## Architecture

### Module Structure
```
src/aps/
├── models/         # Unified models (APSAutoencoder)
├── training/       # Training pipeline (Trainer, config, logging)
├── topology/       # Topology preservation
├── causality/      # Causal learning components
├── energy/         # Energy-based attractors
├── metrics/        # Evaluation metrics
└── utils/          # Data generation and visualization
```

### Key Components

#### Models (`aps.models`)
- **`APSAutoencoder`**: Unified autoencoder combining T+C+E losses
  - Loss: `L_total = L_recon + λ_T·L_topo + λ_C·L_hsic + λ_E·L_energy`
  - Modular MLP encoder/decoder with configurable hidden layers
  - Independent weight control for ablation studies
- **`APSConvAutoencoder`**: Convolutional version for RGB images (ColoredMNIST)
  - Conv2d encoder + ConvTranspose2d decoder
  - Same unified APS loss structure
  - Includes classifier head for digit recognition
- **`APSConfig`**: Comprehensive configuration dataclass
  - Architecture params: `in_dim`, `latent_dim`, `hidden_dims`
  - Loss weights: `lambda_T`, `lambda_C`, `lambda_E`
  - Component-specific: `topo_k`, `hsic_sigma`, `n_mem`, `beta`, `alpha`
- **`APSConvConfig`**: Configuration for convolutional models
  - Image params: `in_channels`, `img_size`, `hidden_channels`
  - Same loss weights and component params as APSConfig

#### Training (`aps.training`)
- **`Trainer`**: Unified training loop with progress tracking
  - Train/validation phases with tqdm progress bars
  - Automatic checkpointing (best, final, periodic)
  - Early stopping with patience
  - Learning rate scheduling
  - Gradient clipping
  - Resume from checkpoint
- **`TrainingConfig`**: Training configuration
  - Optimizer: Adam, AdamW, SGD
  - Scheduler: Step, Cosine, Exponential
  - Logging: tensorboard, wandb, JSON
- **`MetricsLogger`**: Tracks all loss components over time
  - Step-level and epoch-level metrics
  - Optional tensorboard/wandb integration
  - JSON export for analysis

#### Topology (`aps.topology`)
- **`KNNTopoLoss`**: BCE-based loss that preserves kNN adjacency between original and latent space
- **`TopologicalAutoencoder`**: Simple MLP autoencoder with combined reconstruction + topology loss
- **`knn_indices()`** and **`adjacency_from_knn()`**: Helper functions to compute kNN graphs from tensors

The topology loss compares kNN adjacency matrices in original vs. latent space using binary cross-entropy. This is an "offline-friendly surrogate" approach that doesn't require persistent homology computation during training.

#### Causality (`aps.causality`)
- **`HSICIndependenceLoss`**: Penalizes dependence between latent factors Z and nuisance variables N using HSIC (Hilbert-Schmidt Independence Criterion) with RBF kernels
- **`IRMLoss`**: Implements Invariant Risk Minimization penalty by computing gradients of per-environment risks w.r.t. a scaling parameter

#### Energy (`aps.energy`)
- **`MemoryEnergy`**: Log-sum-exp energy function over learnable memory patterns
  - Energy formula: `E(z) = 0.5*α*||z||² - log(Σ exp(β·z·mᵢ))`
  - Lower energy near memory patterns → creates attractor basins
  - `beta`: controls basin sharpness (higher = sharper)
  - Memory patterns are learned via `nn.Parameter`

#### Metrics (`aps.metrics`)
- **`knn_preservation()`**: Jaccard similarity of kNN sets between original and embedded space
- **`trustworthiness()`**: Measures if close neighbors in embedding were also close in original space
- **`basin_depth()`**: Gap between mean and minimum energy (deeper = stronger attractors)
- **`pointer_strength()`**: Metric for memory/pointer mechanisms (see `pointer_strength.py`)
- **`color_label_correlation()`**: Measures if latent codes correlate with spurious (color) or causal (label) features
- **`environment_invariance_score()`**: Measures prediction consistency across environments with different spurious correlations
- **`spurious_feature_reliance()`**: Quantifies model's reliance on spurious vs causal features
- **`compute_causal_metrics()`**: Comprehensive causal evaluation suite for ColoredMNIST

#### Utils (`aps.utils`)
- **`toy_corpus()`**: Generates small synthetic text data for testing
- **`cooc_ppmi()`**: Computes PPMI (Positive Pointwise Mutual Information) co-occurrence matrix from tokens
- **`svd_embed()`**: Creates SVD-based embeddings from co-occurrence matrix
- **`scatter_labels()`**: Visualization helper for 2D embeddings with word labels
- **`ColoredMNIST`**: Dataset where digit color is spuriously correlated with labels
  - Configurable correlation rates per environment (training vs test)
  - RGB colored digits for testing causal invariance learning
- **`create_colored_mnist_envs()`**: Creates multiple training environments with different color-label correlations

### Workflow Pattern

1. **Generate or load data** using `aps.utils` (toy corpus → PPMI → SVD embeddings)
2. **Compute target topology** via `knn_indices()` and `adjacency_from_knn()` from the original space
3. **Train model** with combined losses:
   - Reconstruction loss (MSE)
   - Topology loss (kNN adjacency preservation)
   - Optional: Energy loss (attractor shaping)
   - Optional: Causal losses (IRM/HSIC)
4. **Evaluate** using `aps.metrics` (preservation, trustworthiness, basin depth)
5. **Visualize** embeddings with `scatter_labels()` or custom plots

### Design Principles

- **Modular losses**: Each component (topology, causality, energy) is a standalone `nn.Module` that can be weighted and combined
- **Offline-friendly**: Topology loss uses kNN adjacency rather than requiring expensive persistent homology
- **Memory-based attractors**: Energy function uses learnable memory patterns instead of explicit clustering
- **Hybrid approach**: Combines reconstruction (autoencoder) with geometric constraints (topology), causal invariance, and energy shaping

## Configuration Patterns

Models use dataclass configs for hyperparameters:

```python
# Unified APS Model
from aps.models import APSConfig, APSAutoencoder
from aps.training import Trainer, TrainingConfig

model_cfg = APSConfig(
    in_dim=50,           # Input dimension
    latent_dim=2,        # Latent space dimension
    hidden_dims=[64],    # Hidden layer sizes
    lambda_T=1.0,        # Topology weight
    lambda_C=1.0,        # Causality weight
    lambda_E=1.0,        # Energy weight
    topo_k=8,            # k for kNN
    n_mem=8,             # Number of memory patterns
    beta=5.0,            # Energy sharpness
)

train_cfg = TrainingConfig(
    epochs=100,
    batch_size=32,
    optimizer=OptimizerConfig(name='adam', lr=1e-3),
    device='cpu',
    experiment_name='aps_full',
)

model = APSAutoencoder(model_cfg)
trainer = Trainer(model, train_cfg)
trainer.train(train_loader, val_loader)

# Legacy configs (individual components)
# Topology Autoencoder
TopoAEConfig(
    in_dim=50,           # Input dimension
    latent_dim=2,        # Latent space dimension
    hidden=64,           # Hidden layer size
    lr=1e-3,             # Learning rate
    topo_k=8,            # k for kNN graph
    topo_weight=1.0      # Weight for topology loss
)

# Memory Energy
MemoryEnergyConfig(
    latent_dim=2,        # Latent space dimension
    n_mem=8,             # Number of memory patterns
    beta=5.0,            # Sharpness (higher = sharper basins)
    alpha=0.0            # L2 regularization on z
)
```

## Testing Notes

- Test files live in `tests/` directory
- Current tests focus on shape validation and basic functionality
- When adding new losses or metrics, verify tensor shapes and gradient flow
- Use `torch.randn()` for quick synthetic test data

## Output Files

Scripts generate outputs in `outputs/` directory (created automatically):
- `embedding_topo_ae.csv`: Learned embeddings with word labels
- `topo_ae_scatter.png`: 2D scatter plot of selected words
- `energy_basins.png`: Energy landscape visualization

## Virtual Environment

The project assumes a `.venv` virtual environment. Makefile commands automatically activate it.

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```
