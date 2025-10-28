# Energy Framework Overview

## Introduction

The APS energy framework provides a comprehensive system for shaping latent spaces using energy-based models. This document covers the complete energy system including the base architecture, variants, initialization strategies, and visualization tools.

## Core Concept: Energy-Based Embeddings

Energy functions define "attraction basins" in latent space:
- **Low energy** = Points are well-represented by memory patterns
- **High energy** = Points don't match any pattern (outliers)
- **Gradients** = Direction of flow toward nearest basin

Formula: `E(z) = f(z, memory_patterns, parameters)`

---

## Architecture

### Base Energy Class

All energy models inherit from `BaseEnergy` (in `src/aps/energy/base.py`):

```python
from aps.energy import BaseEnergy

class CustomEnergy(BaseEnergy):
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        # Compute energy for points z
        pass
    
    def _init_patterns(self, init_method: str, **kwargs):
        # Initialize memory patterns
        pass
```

**Key methods**:
- `energy(z)` - Compute energy values
- `loss(z)` - Training loss (mean energy)
- `forward(z)` - Negative energy (score)
- `nearest_pattern(z)` - Hard assignment
- `basin_assignment(z)` - Soft assignment (probabilities)
- `gradient_at(z)` - Energy gradient
- `hessian_at(z)` - Second derivatives
- `local_curvature(z)` - Basin convexity

---

## Energy Variants

### 1. MemoryEnergy (Log-Sum-Exp)

**Formula**: `E(z) = 0.5*α*||z||² - log(Σ exp(β·z·mᵢ))`

**Characteristics**:
- Dot product similarity
- Directional basins (not perfectly circular)
- Global sharpness parameter β

**Configuration**:
```python
from aps.energy import MemoryEnergy, MemoryEnergyConfig

cfg = MemoryEnergyConfig(
    latent_dim=2,           # Embedding dimension
    n_mem=8,                # Number of memory patterns
    beta=5.0,               # Sharpness (higher = sharper)
    alpha=0.0,              # L2 regularization
    init_method='grid',     # Initialization strategy
    init_scale=0.5          # For random init
)

energy = MemoryEnergy(cfg)
```

**Use cases**:
- Text embeddings (directional word vectors)
- Asymmetric relationships
- When orientation matters

---

### 2. RBFEnergy (Gaussian Basins)

**Formula**: `E(z) = -log(Σ exp(-||z - mᵢ||² / 2σᵢ²))`

**Characteristics**:
- Euclidean distance-based
- Isotropic (circular) basins
- Per-pattern radius (σ)
- Optional learnable σ

**Configuration**:
```python
from aps.energy import RBFEnergy, RBFEnergyConfig

cfg = RBFEnergyConfig(
    latent_dim=2,
    n_mem=8,
    sigma=1.0,              # Basin width (or list for per-pattern)
    init_method='sphere',
    learnable_sigma=True    # Learn during training
)

energy = RBFEnergy(cfg)
```

**Use cases**:
- Natural clustering (customer segments)
- Spatial data
- When distance is symmetric

---

### 3. MixtureEnergy (Learnable Parameters)

**Formula**: `E(z) = -log(Σ wᵢ exp(βᵢ·z·mᵢ))`

**Characteristics**:
- Per-pattern weights (wᵢ)
- Per-pattern sharpness (βᵢ)
- More flexible than MemoryEnergy

**Configuration**:
```python
from aps.energy import MixtureEnergy, MixtureEnergyConfig

cfg = MixtureEnergyConfig(
    latent_dim=2,
    n_mem=8,
    init_method='grid',
    learnable_weights=True,     # Learn pattern importance
    learnable_sharpness=True,   # Learn per-pattern beta
    init_beta=5.0,
    init_weight=1.0
)

energy = MixtureEnergy(cfg)

# Access learned parameters
print(energy.weights)  # Pattern importance
print(energy.beta)     # Per-pattern sharpness
```

**Use cases**:
- Imbalanced categories
- Hierarchical patterns
- When patterns have different importance

---

## Initialization Strategies

Memory patterns can be initialized in 7 different ways:

### 1. Random (Default)
```python
cfg = MemoryEnergyConfig(init_method='random', init_scale=0.5)
```
- Gaussian noise
- Good baseline
- Fast

### 2. Grid
```python
cfg = MemoryEnergyConfig(init_method='grid')
```
- Evenly spaced patterns
- Works for 1D, 2D, 3D
- Visualizable structure

### 3. Sphere
```python
cfg = MemoryEnergyConfig(init_method='sphere')
```
- Points on hypersphere surface
- Uniform angular distribution
- Good for normalized data

### 4. Cube Corners
```python
cfg = MemoryEnergyConfig(
    latent_dim=2,
    n_mem=4,  # Must be 2^latent_dim
    init_method='cube'
)
```
- Vertices of hypercube
- Creates 2^d patterns for d dimensions
- Maximal separation

### 5. K-means
```python
from aps.energy import kmeans_init

# Initialize from data
data = torch.randn(1000, 2)
patterns = kmeans_init(data, n_mem=8)

# Manually set patterns
cfg = MemoryEnergyConfig(init_method='random')
energy = MemoryEnergy(cfg)
energy.mem.data.copy_(patterns)
```
- Data-driven initialization
- Finds natural clusters
- Requires data upfront

### 6. Hierarchical
```python
cfg = MemoryEnergyConfig(
    init_method='hierarchical',
    n_mem=9  # 3 levels × 3 patterns
)
```
- Multi-scale nested grids
- Coarse + fine patterns
- Good for hierarchical data

### 7. PCA
```python
from aps.energy import pca_init

data = torch.randn(1000, 2)
patterns = pca_init(data, n_mem=8)
```
- Along principal components
- Captures main variation axes
- Data-driven

---

## Usage Examples

### Basic Training Loop

```python
from aps.energy import MemoryEnergy, MemoryEnergyConfig
import torch
import torch.optim as optim

# Create model
cfg = MemoryEnergyConfig(latent_dim=2, n_mem=4, beta=5.0)
energy_model = MemoryEnergy(cfg)

# Optimizer
optimizer = optim.Adam(energy_model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    # Get latent representations (from autoencoder, etc.)
    z = get_embeddings()  # Shape: (batch_size, latent_dim)
    
    # Compute energy loss
    loss = energy_model.loss(z)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Energy = {loss.item():.4f}")
```

### Combining with Autoencoder

```python
# Full model
class EnergyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
        
        cfg = MemoryEnergyConfig(latent_dim=latent_dim, n_mem=8)
        self.energy = MemoryEnergy(cfg)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def loss_fn(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x)
        energy_loss = self.energy.loss(z)
        return recon_loss + 0.1 * energy_loss
```

### Inference: Basin Assignment

```python
# Get latent representations
z = encoder(x)

# Hard assignment (nearest pattern)
nearest = energy_model.nearest_pattern(z)
print(f"Nearest memory pattern: {nearest}")

# Soft assignment (probabilities)
probs = energy_model.basin_assignment(z, temperature=1.0)
print(f"Basin probabilities: {probs}")

# Energy value (confidence)
energy_val = energy_model.energy(z)
print(f"Energy (lower = better): {energy_val}")
```

### Gradient Descent to Nearest Basin

```python
# Simulate gradient descent
z = torch.randn(1, 2, requires_grad=True)
optimizer = optim.SGD([z], lr=0.1)

for step in range(50):
    energy = energy_model.energy(z)
    energy.backward()
    optimizer.step()
    optimizer.zero_grad()
    
print(f"Final position: {z}")
print(f"Nearest pattern: {energy_model.nearest_pattern(z)}")
```

---

## Comparison: When to Use Which?

| Variant | Best For | Basin Shape | Parameters |
|---------|----------|-------------|------------|
| **MemoryEnergy** | Text, directed relationships | Directional | Global β |
| **RBFEnergy** | Clustering, spatial data | Circular | Per-pattern σ |
| **MixtureEnergy** | Imbalanced, hierarchical | Directional | Per-pattern w, β |

### Decision Tree

```
Do you need isotropic (circular) basins?
├─ Yes → RBFEnergy
└─ No → Is directionality important?
    ├─ Yes → Do patterns have different importance?
    │   ├─ Yes → MixtureEnergy
    │   └─ No → MemoryEnergy
    └─ No → RBFEnergy
```

---

## Visualization

### 2D Heatmap Comparison
```bash
python test_energy_comparison.py
# Creates: outputs/energy_comparison.png
```

### 3D Surface Comparison
```bash
python test_energy_comparison_3d.py
# Creates: outputs/energy_comparison_3d.html
```

### Detailed 3D Views
```bash
python test_energy_detailed_3d.py
# Creates: 
#   - outputs/energy_detailed_memory_3d.html
#   - outputs/energy_detailed_rbf_3d.html
#   - outputs/energy_detailed_mixture_3d.html
```

All visualizations are interactive (Plotly):
- Rotate by clicking and dragging
- Zoom with scroll wheel
- Hover for exact values

---

## Advanced Features

### Custom Initialization from Data

```python
# K-means from data
from aps.energy import kmeans_init

train_data = get_training_embeddings()
patterns = kmeans_init(train_data, n_mem=8, max_iter=100)

# Set patterns manually
energy_model.mem.data.copy_(patterns)
```

### Learnable Basin Width (RBF)

```python
# Enable learning of sigma
cfg = RBFEnergyConfig(
    latent_dim=2,
    n_mem=8,
    sigma=1.0,
    learnable_sigma=True  # ← Learn during training
)

energy = RBFEnergy(cfg)

# After training
print(f"Learned sigmas: {energy.sigma}")
```

### Per-Pattern Analysis (Mixture)

```python
energy = MixtureEnergy(cfg)

# After training
info = energy.info_dict()
print(f"Pattern weights: {info['weights']}")
print(f"Pattern sharpness: {info['beta_values']}")
print(f"Weight entropy: {info['weight_entropy']}")  # Uniformity measure
```

### Energy Analysis

```python
# Compute gradient
grad = energy_model.gradient_at(z)
print(f"Flow direction: {grad}")

# Compute curvature (basin strength)
curv = energy_model.local_curvature(z[0])
print(f"Curvature: {curv}")  # Positive = convex basin
```

---

## Integration with APS Framework

Energy models integrate with topology and causality:

```python
# Combined loss
def aps_loss(x, model, energy_model, target_adj):
    # Reconstruction
    z = model.encode(x)
    x_recon = model.decode(z)
    recon_loss = F.mse_loss(x_recon, x)
    
    # Topology
    from aps.topology import KNNTopoLoss
    topo_loss_fn = KNNTopoLoss(k=8)
    topo_loss = topo_loss_fn(z, target_adj)
    
    # Energy
    energy_loss = energy_model.loss(z)
    
    # Combine
    total_loss = recon_loss + λ_T * topo_loss + λ_E * energy_loss
    return total_loss
```

---

## Testing

### Run Test Suite
```bash
pytest -q  # All 71 tests should pass
```

### Quick Verification
```bash
python -c "
from aps.energy import *
import torch

# Test all variants
cfg1 = MemoryEnergyConfig(latent_dim=2, n_mem=4)
cfg2 = RBFEnergyConfig(latent_dim=2, n_mem=4)
cfg3 = MixtureEnergyConfig(latent_dim=2, n_mem=4)

e1 = MemoryEnergy(cfg1)
e2 = RBFEnergy(cfg2)
e3 = MixtureEnergy(cfg3)

z = torch.randn(10, 2)
print('MemoryEnergy:', e1.energy(z).shape)
print('RBFEnergy:', e2.energy(z).shape)
print('MixtureEnergy:', e3.energy(z).shape)
print('✓ All variants working!')
"
```

---

## Performance Notes

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Energy computation | O(N × M) | N = batch size, M = n_mem |
| Gradient computation | O(N × M × D) | D = latent_dim |
| K-means init | O(N × M × iter) | iter ≈ 10-100 |
| Grid init | O(M) | Very fast |

**Recommendations**:
- Use grid/sphere init for visualization (predictable)
- Use k-means init for real applications (data-driven)
- Start with β=5.0, adjust based on visualization
- Use RBF for spatial/clustering, Memory for text/sequences

---

## Common Patterns

### Outlier Detection
```python
z = encoder(x)
energy = energy_model.energy(z)

# High energy = outlier
threshold = energy.quantile(0.95)
outliers = energy > threshold
```

### Confidence Scoring
```python
# Lower energy = higher confidence
confidence = torch.exp(-energy)
```

### Generation by Gradient Descent
```python
# Start from noise
z = torch.randn(1, latent_dim, requires_grad=True)

# Flow to nearest basin
for _ in range(100):
    energy = energy_model.energy(z)
    energy.backward()
    z.data -= 0.1 * z.grad
    z.grad.zero_()

# Decode to data space
x_generated = decoder(z)
```

---

## References

- Base implementation: `src/aps/energy/base.py`
- Variants: `src/aps/energy/variants.py`
- Initialization: `src/aps/energy/init.py`
- Visualization guide: `docs/VISUALIZATION_GUIDE.md`
- Main project docs: `WARP.md`
