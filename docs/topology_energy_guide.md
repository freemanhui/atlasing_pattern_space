# Topology-Aware Energy Function

## Overview

`TopologyEnergy` is an alternative energy function that **reinforces topology preservation** rather than competing with it. Unlike `MemoryEnergy` which creates arbitrary attractor basins, `TopologyEnergy` is data-driven and rewards latent representations that maintain k-NN adjacency relationships from the original space.

## Key Differences from MemoryEnergy

| Feature | MemoryEnergy | TopologyEnergy |
|---------|--------------|----------------|
| **Attractor Type** | Fixed learnable memory patterns | Data-driven k-NN structure |
| **Independence** | Independent of data topology | Directly based on data topology |
| **Objective Alignment** | Can compete with topology loss | Aligns with topology loss |
| **Expected ARI** | Lower (clusters may not align with labels) | Higher (preserves semantic structure) |
| **Silhouette Score** | Very high (tight basins) | Moderate (natural clustering) |

## When to Use TopologyEnergy

**Use TopologyEnergy when:**
- You want to **enhance** topology preservation, not compete with it
- Maintaining label-aligned clustering (high ARI) is important
- You need better trustworthiness and continuity metrics
- The task requires preserving semantic relationships

**Use MemoryEnergy when:**
- You explicitly want discrete attractor basins
- You're willing to sacrifice some topology preservation for tighter clusters
- You have prior knowledge about desired basin locations

## Basic Usage

```python
from aps.energy import TopologyEnergy, TopologyEnergyConfig
from aps.topology import knn_indices

# Configuration
cfg = TopologyEnergyConfig(
    latent_dim=2,
    k=8,                    # k for k-NN preservation
    mode='agreement',       # 'agreement', 'disagreement', 'jaccard'
    continuous=True,        # Use differentiable continuous k-NN
    scale=1.0              # Energy scaling factor
)

# Initialize
energy_fn = TopologyEnergy(cfg)

# IMPORTANT: Set target adjacency from original data
X_train = ...  # Your training data in original space
energy_fn.set_target_adjacency(X_train)

# Compute energy loss during training
z_latent = encoder(X_train)
energy_loss = energy_fn.loss(z_latent)  # Mean energy

# Total loss
total_loss = reconstruction_loss + λ_topo * topo_loss + λ_energy * energy_loss
```

## Energy Modes

### 1. Agreement Mode (Recommended)
```python
cfg = TopologyEnergyConfig(mode='agreement')
```
- Energy: `E(z) = -sum(A_orig ⊙ A_latent) / (n * k)`
- Lower energy = more k-NN adjacencies preserved
- Best balance between preservation and optimization

### 2. Disagreement Mode
```python
cfg = TopologyEnergyConfig(mode='disagreement')
```
- Energy: `E(z) = sum(|A_orig - A_latent|) / (n * k)`
- Penalizes adjacency differences
- More sensitive to topology changes

### 3. Jaccard Mode
```python
cfg = TopologyEnergyConfig(mode='jaccard')
```
- Energy: `E(z) = -intersection(A_orig, A_latent) / union(A_orig, A_latent)`
- Jaccard similarity-based
- Normalized comparison

## Continuous vs Discrete k-NN

### Continuous (Default, Recommended)
```python
cfg = TopologyEnergyConfig(continuous=True, temperature=10.0)
```
- **Differentiable** - gradients flow smoothly
- Uses sigmoid-based soft adjacency
- Better for gradient-based optimization
- Higher temperature = sharper transitions

### Discrete
```python
cfg = TopologyEnergyConfig(continuous=False)
```
- Binary adjacency (0 or 1)
- Non-differentiable
- May cause optimization issues
- Use only for evaluation/analysis

## Adaptive Weighting

Use `AdaptiveTopologyEnergy` for automatic weight adjustment:

```python
from aps.energy import AdaptiveTopologyEnergy

energy_fn = AdaptiveTopologyEnergy(
    cfg,
    min_weight=0.1,
    max_weight=2.0
)

# Update weight based on current preservation quality
energy_fn.update_adaptive_weight(z_latent)

# Weight increases when preservation is poor, decreases when good
```

## Integration with Existing Experiments

### Modify Ablation Configuration

In `experiments/configs/ablation.yaml`:

```yaml
t_c_e_topo:
  name: "Topology + Causality + TopologyEnergy"
  components:
    topology: true
    causality: true
    energy_type: "topology"  # Instead of "memory"
  
  topology:
    k: 8
    weight: 1.0
  
  causality:
    irm_weight: 0.5
    hsic_weight: 0.5
  
  energy:
    type: "topology"
    latent_dim: 2
    k: 8
    mode: "agreement"
    continuous: true
    scale: 1.0
    weight: 0.5  # Start with lower weight than MemoryEnergy
```

### Update Training Script

In your training code:

```python
from aps.energy import TopologyEnergy, TopologyEnergyConfig

# Setup
if config.energy.type == "topology":
    energy_cfg = TopologyEnergyConfig(
        latent_dim=config.latent_dim,
        k=config.energy.k,
        mode=config.energy.mode,
        continuous=config.energy.continuous,
        scale=config.energy.scale
    )
    energy_fn = TopologyEnergy(energy_cfg)
    
    # CRITICAL: Set target adjacency once before training
    energy_fn.set_target_adjacency(X_train)
    
elif config.energy.type == "memory":
    # Original MemoryEnergy setup
    ...

# During training loop
z_latent = model.encode(X_batch)
energy_loss = energy_fn.loss(z_latent)
total_loss += config.energy.weight * energy_loss
```

## Expected Results

Based on the issues observed with MemoryEnergy in T+C+E configuration:

### MemoryEnergy (T+C+E)
- ✅ Silhouette: +43.7% (very tight clusters)
- ❌ Reconstruction Error: +58.2% (poor reconstruction)
- ❌ Trustworthiness: -34.8% (poor preservation)
- ❌ ARI: -92.4% (clusters don't align with labels)

### TopologyEnergy (T+C+E_topo) - Expected
- ✅ Silhouette: +10-20% (moderate improvement)
- ✅ Reconstruction Error: ≤+5% (minimal degradation)
- ✅ Trustworthiness: ~0% or positive (maintained or improved)
- ✅ ARI: ~0% or positive (maintained or improved)

## Hyperparameter Tuning

### Energy Weight (`λ_energy`)
Start with **lower values** than MemoryEnergy:
- Initial: `0.1 - 0.5`
- MemoryEnergy typically uses: `1.0 - 10.0`
- TopologyEnergy is more aligned, so less force needed

### k (Number of Neighbors)
- Match with topology loss k: `k_energy = k_topology`
- Typical range: `5 - 15`
- Larger k = more global structure
- Smaller k = more local structure

### Mode Selection
- **Default: 'agreement'** (best balance)
- Use 'disagreement' if you want stronger penalty
- Use 'jaccard' for normalized comparison

### Scale Factor
- Default: `1.0`
- Increase if energy magnitude is too small
- Decrease if energy dominates other losses

## Troubleshooting

### Energy values are very negative
- This is expected for agreement mode
- The absolute value doesn't matter, only relative changes
- Focus on convergence and final metrics

### Energy doesn't decrease during training
- Check that `continuous=True` for differentiability
- Verify target adjacency was set via `set_target_adjacency()`
- Try increasing energy weight

### Topology preservation doesn't improve
- Topology loss may already be strong enough
- Try reducing topology loss weight and increasing energy weight
- Ensure k values match between topology loss and energy

### Memory issues with large datasets
- Target adjacency is stored as `(n, n)` matrix
- For n > 10,000, consider using sparse adjacency
- Or compute energy in mini-batches (requires modification)

## Example: Complete Training Setup

```python
import torch
from torch.utils.data import DataLoader
from aps.energy import TopologyEnergy, TopologyEnergyConfig
from aps.topology import KNNTopoLoss
from aps.causality import IRMLoss

# Configuration
latent_dim = 2
k = 8
energy_weight = 0.3
topo_weight = 1.0

# Setup topology loss
topo_loss_fn = KNNTopoLoss(k=k)

# Setup topology energy
energy_cfg = TopologyEnergyConfig(
    latent_dim=latent_dim,
    k=k,
    mode='agreement',
    continuous=True
)
energy_fn = TopologyEnergy(energy_cfg)

# CRITICAL: Set target adjacency ONCE before training
# Use full training set or representative subset
energy_fn.set_target_adjacency(X_train)

# Training loop
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        X_recon, z = model(X_batch)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(X_recon, X_batch)
        
        # Topology loss
        topo_loss = topo_loss_fn(X_batch, z)
        
        # Energy loss
        energy_loss = energy_fn.loss(z)
        
        # Combined loss
        total_loss = (
            recon_loss + 
            topo_weight * topo_loss + 
            energy_weight * energy_loss
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## Running Experiments

```bash
# Test TopologyEnergy with demo
python scripts/demo_topology_energy.py

# Run full ablation with TopologyEnergy
# (requires updating experiment configs first)
python experiments/run_ablation_training.py \
    --config experiments/configs/ablation_topo_energy.yaml \
    --output-dir outputs/ablation_topo_energy

# Compare results
python experiments/analysis/summarize_results.py \
    --results-dir outputs/
```

## References

- Chen et al. (2022) "Local Distance Preserving Auto-encoders using Continuous k-Nearest Neighbours Graphs"
- Topology preservation via k-NN graph matching
- Differentiable k-NN graphs for deep learning
