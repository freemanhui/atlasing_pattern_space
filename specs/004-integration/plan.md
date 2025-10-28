# Implementation Plan: Full APS Integration (Phase 004)

**Branch**: `004-integration` | **Date**: 2025-01-28  
**Status**: In Progress  
**Dependencies**: Phases 001 (Energy), 002 (Topology), 003 (Causality) - all complete

## Summary

Implement unified training framework that combines all three APS components (Topology, Causality, Energy) into a cohesive system with the full objective:

```
L_APS = L_task + λ_T·L_T + λ_C·L_C + λ_E·L_E
```

**Paper Reference**: Section 4 (Full APS Framework) and Section 5 (Experiments)

## Current State

### Completed Modules ✅

1. **Energy (E)** - `aps.energy`
   - `MemoryEnergy`: Energy-based attractors
   - ~10 tests passing
   - Visualization tools in `aps.viz`

2. **Topology (T)** - `aps.topology`
   - `KNNTopoLoss`: k-NN structure preservation
   - `TopologicalAutoencoder`: Combined recon + topo
   - 62 tests passing

3. **Causality (C)** - `aps.causality`
   - `HSICLoss`: Independence testing
   - `IRMLoss`: Environment-invariant learning
   - 77 tests passing

### Integration Goal

Create a unified model that:
1. Combines all three regularization losses
2. Provides clean API for training with configurable weights
3. Supports flexible architectures (autoencoders, classifiers)
4. Integrates with visualization for monitoring
5. Enables easy ablation studies (T-only, C-only, E-only, combinations)

## Architecture

### Unified APS Model

```python
class APSAutoencoder(nn.Module):
    """
    Autoencoder with full APS regularization: Topology + Causality + Energy.
    
    Loss: L = L_recon + λ_T·L_topo + λ_C·L_causal + λ_E·L_energy
    """
    
    def __init__(self, config: APSConfig):
        # Encoder/Decoder
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)
        
        # Topology
        self.topo_loss = KNNTopoLoss(k=config.topo_k)
        
        # Causality
        self.hsic_loss = HSICLoss(kernel=config.causal_kernel)
        self.irm_loss = IRMLoss() if config.use_irm else None
        
        # Energy
        self.energy_fn = MemoryEnergy(config.energy_config)
    
    def forward(self, x, nuisance=None, env=None):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def compute_loss(self, x, nuisance=None, envs=None):
        x_recon, z = self(x)
        
        losses = {}
        
        # Task loss (reconstruction)
        losses['recon'] = F.mse_loss(x_recon, x)
        
        # Topology loss
        if self.config.lambda_T > 0:
            losses['topo'] = self.topo_loss(x, z)
        
        # Causality losses
        if self.config.lambda_C > 0 and nuisance is not None:
            losses['hsic'] = self.hsic_loss(z, nuisance)
        
        if self.config.use_irm and envs is not None:
            losses['irm'] = self.irm_loss(self.encoder, envs)
        
        # Energy loss
        if self.config.lambda_E > 0:
            losses['energy'] = self.energy_fn(z)
        
        # Total loss
        losses['total'] = (
            losses['recon'] +
            self.config.lambda_T * losses.get('topo', 0) +
            self.config.lambda_C * losses.get('hsic', 0) +
            self.config.lambda_E * losses.get('energy', 0)
        )
        
        return losses
```

### Configuration System

```python
@dataclass
class APSConfig:
    """Configuration for full APS model."""
    
    # Architecture
    in_dim: int
    latent_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    
    # Training
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    
    # Topology (T)
    lambda_T: float = 1.0
    topo_k: int = 8
    
    # Causality (C)
    lambda_C: float = 1.0
    causal_kernel: str = 'rbf'
    causal_sigma: float = 1.0
    use_irm: bool = False
    
    # Energy (E)
    lambda_E: float = 1.0
    energy_n_mem: int = 8
    energy_beta: float = 5.0
    energy_alpha: float = 0.0
```

## Implementation Phases

### Phase 1: Core Integration (This PR)

**Goal**: Basic unified model with all three losses

**Tasks**:
1. Create `aps.models.aps_autoencoder.py`
   - `APSAutoencoder` class
   - `APSConfig` dataclass
   - `compute_loss()` method with all components

2. Create test suite `tests/test_models/test_aps_autoencoder.py`
   - Test initialization
   - Test forward pass
   - Test loss computation (each component)
   - Test ablations (T-only, C-only, E-only)
   - Test training loop integration

3. Create simple example `examples/train_aps_mnist.py`
   - Train on MNIST with full APS
   - Visualize learned embeddings
   - Compare with baseline autoencoder

**Success Criteria**:
- Model trains successfully with all losses
- Can disable individual losses (ablation)
- Loss values are reasonable (not NaN/Inf)
- Test suite passes (>20 tests)

### Phase 2: Training Utilities

**Goal**: Helper classes for training and monitoring

**Tasks**:
1. Create `aps.training.trainer.py`
   - `APSTrainer` class
   - Handles training loop, validation
   - Logging and checkpointing
   - Integration with `aps.viz` for monitoring

2. Create `aps.training.callbacks.py`
   - `Callback` base class
   - `EnergyVisualizationCallback`
   - `TopologyMetricsCallback`
   - `LossLoggingCallback`

3. Update examples to use trainer
   - Cleaner training code
   - Better logging/visualization

**Success Criteria**:
- Trainer handles full training pipeline
- Callbacks work correctly
- Examples are clean and documented

### Phase 3: Advanced Features

**Goal**: Curriculum learning, multi-objective optimization

**Tasks**:
1. Curriculum learning
   - Start with task loss only
   - Gradually increase λ_T, λ_C, λ_E
   - Configurable schedule

2. Multi-objective optimization
   - Balance between losses
   - Adaptive weighting
   - Pareto optimization (optional)

3. Advanced architectures
   - Support for ResNets, Transformers
   - Flexible encoder/decoder builders

**Success Criteria**:
- Curriculum improves training stability
- Works with different architectures
- Documented and tested

## Testing Strategy

### Unit Tests

**Model Tests** (`test_aps_autoencoder.py`):
```python
def test_init_with_all_losses()
def test_init_topology_only()
def test_init_causality_only()
def test_init_energy_only()
def test_forward_shape()
def test_compute_loss_all_components()
def test_compute_loss_ablation_T()
def test_compute_loss_ablation_C()
def test_compute_loss_ablation_E()
def test_gradients_flow_to_encoder()
def test_gradients_flow_to_decoder()
def test_training_reduces_losses()
```

**Trainer Tests** (`test_trainer.py`):
```python
def test_trainer_init()
def test_trainer_fit()
def test_trainer_validation()
def test_trainer_checkpointing()
def test_callbacks_called()
```

### Integration Tests

```python
def test_full_aps_training_mnist()
def test_ablation_study()
def test_with_energy_visualization()
def test_with_nuisance_variables()
def test_multi_environment_irm()
```

## Example Usage

### Basic Training

```python
from aps.models import APSAutoencoder, APSConfig
from aps.training import APSTrainer

# Configure model
config = APSConfig(
    in_dim=784,  # MNIST
    latent_dim=2,
    lambda_T=1.0,  # Topology weight
    lambda_C=0.5,  # Causality weight
    lambda_E=0.1,  # Energy weight
)

# Create model
model = APSAutoencoder(config)

# Create trainer
trainer = APSTrainer(model, config)

# Train
trainer.fit(train_loader, val_loader)

# Visualize
trainer.visualize_embeddings()
trainer.visualize_energy_basins()
```

### Ablation Study

```python
# Train models with different components
configs = {
    'baseline': APSConfig(lambda_T=0, lambda_C=0, lambda_E=0),
    'T-only': APSConfig(lambda_T=1.0, lambda_C=0, lambda_E=0),
    'C-only': APSConfig(lambda_T=0, lambda_C=1.0, lambda_E=0),
    'E-only': APSConfig(lambda_T=0, lambda_C=0, lambda_E=1.0),
    'T+C': APSConfig(lambda_T=1.0, lambda_C=1.0, lambda_E=0),
    'T+E': APSConfig(lambda_T=1.0, lambda_C=0, lambda_E=1.0),
    'C+E': APSConfig(lambda_T=0, lambda_C=1.0, lambda_E=1.0),
    'full': APSConfig(lambda_T=1.0, lambda_C=1.0, lambda_E=1.0),
}

results = {}
for name, config in configs.items():
    model = APSAutoencoder(config)
    trainer = APSTrainer(model, config)
    metrics = trainer.fit(train_loader, val_loader)
    results[name] = metrics

# Compare results
plot_ablation_results(results)
```

### With HSIC (Colored MNIST)

```python
# Configure for colored MNIST
config = APSConfig(
    in_dim=784,
    latent_dim=10,
    lambda_C=5.0,  # Strong independence from color
    causal_kernel='rbf',
)

model = APSAutoencoder(config)

# Training loop
for batch_x, batch_y, batch_color in dataloader:
    # Forward
    x_recon, z = model(batch_x)
    
    # Compute loss with nuisance variable (color)
    losses = model.compute_loss(batch_x, nuisance=batch_color)
    
    # Backward
    losses['total'].backward()
    optimizer.step()
    
    # Monitor HSIC(z, color)
    print(f"HSIC: {losses['hsic'].item():.4f}")
```

### With IRM (Multi-Environment)

```python
config = APSConfig(
    in_dim=784,
    latent_dim=10,
    use_irm=True,
    lambda_C=10.0,
)

model = APSAutoencoder(config)

# Prepare environments
envs = [
    (X_env1, y_env1),
    (X_env2, y_env2),
]

# Training
losses = model.compute_loss(X, envs=envs)
losses['total'].backward()
```

## Success Metrics

### Quantitative

1. **Training Stability**:
   - All losses converge (no NaN/Inf)
   - Training completes without errors

2. **Component Effectiveness**:
   - Topology: Trustworthiness > 0.9
   - Causality: HSIC(z, nuisance) < 0.01
   - Energy: Clear basin structure

3. **Performance**:
   - Training time < 2x baseline autoencoder
   - Memory usage reasonable

4. **Test Coverage**:
   - >90% code coverage
   - All tests pass

### Qualitative

1. **Embeddings**:
   - Clear clusters (Energy)
   - Smooth manifold (Topology)
   - Independent of nuisances (Causality)

2. **Usability**:
   - Clean API
   - Good documentation
   - Easy to extend

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loss weights hard to tune | High | Provide sensible defaults, auto-tuning |
| Training instability | High | Curriculum learning, gradient clipping |
| Memory overhead | Medium | Efficient implementations, batch gradients |
| Complex API | Medium | Clear documentation, simple examples |

## Timeline

- **Week 1**: Phase 1 (Core Integration)
  - Days 1-2: Implement `APSAutoencoder`
  - Days 3-4: Test suite
  - Day 5: Simple example

- **Week 2**: Phase 2 (Training Utilities)
  - Days 1-2: Implement `APSTrainer`
  - Days 3-4: Callbacks
  - Day 5: Update examples

- **Week 3**: Phase 3 (Advanced Features)
  - Days 1-2: Curriculum learning
  - Days 3-4: Advanced architectures
  - Day 5: Documentation

## References

- **APS Paper**: Section 4 (Full Framework)
- **Topology**: Chen et al. 2022
- **Causality**: Arjovsky et al. 2019 (IRM), Greenfeld & Shalit 2020 (HSIC)
- **Energy**: Memory networks literature

## Next Steps

1. ✅ Create this implementation plan
2. ⏳ Implement `APSAutoencoder` class
3. ⏳ Create test suite
4. ⏳ Build simple MNIST example
5. ⏳ Validate training works end-to-end
