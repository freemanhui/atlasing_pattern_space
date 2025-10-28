# Implementation Plan: Topology Preservation Module (T)

**Branch**: `002-topology` | **Date**: 2025-01-27  
**Status**: Draft - Aligned with "Atlasing Pattern Space (APS) v1" paper  
**Dependencies**: Builds on `aps.energy` (complete) and `aps.viz` (complete)

## Summary

Implement the **Topology Preservation (T)** component of the APS framework using continuous k-NN graph loss to maintain manifold structure in latent representations.

**Paper Reference**: Section 3.1 (Topology-Preserving Embeddings) and Section 4.1 (Topology Loss)

**Core Formula**:
```
L_T = BCE(A_latent, A_input)
```
Where A_input and A_latent are k-NN adjacency matrices in input and latent space respectively.

## Technical Context

**Language/Version**: Python 3.9+  
**Core Framework**: PyTorch (existing)  
**Primary Dependencies**:
- PyTorch (existing)
- NumPy (existing)
- scikit-learn (existing, for kNN computation)
- scipy (optional, for advanced metrics)

**Integration Points**:
- Can be used standalone or with `aps.energy` and future `aps.causality`
- Designed for integration with autoencoders, VAEs, or any encoder model

**Performance Goals**:
- kNN graph computation: < 300ms for batch of 128 samples
- Topology loss computation: < 200ms for batch of 128 samples
- Combined: < 500ms total per batch

**Constraints**:
- Must be differentiable (compatible with PyTorch autograd)
- Memory efficient for typical batch sizes (32-256 samples)
- Support both 2D and high-dimensional latent spaces

## Alignment with APS Paper

### Paper's Topology Component (Section 3.1 & 4.1)

**Related Work Referenced**:
- ✅ **Chen et al. 2022** - Local Distance Preserving Autoencoders (continuous k-NN graph) - **OUR CHOICE**
- Moor et al. 2020 - Topological Autoencoders (persistent homology)
- Lee et al. 2021 - Neighborhood Reconstructing Autoencoders (local reconstruction)

**Why k-NN Graph Approach**:
1. Straightforward and differentiable
2. Captures topology at all scales (per paper)
3. No expensive homology computation needed
4. Well-tested in literature

### Paper's Description (Page 7)

> "One implementation is a continuous k-NN graph loss: we construct a graph G on the batch (or dataset) in input space where edges connect each point to its k nearest neighbors. We then encourage the distances in latent space to be small for edges and, optionally, to be larger for non-neighbor pairs."

> "The continuous k-NN approach is more straightforward and differentiable; Chen et al. (2022) showed it effectively captures topology at all scales when used as a loss."

## Project Structure

```text
src/aps/
├── energy/              # EXISTING
├── viz/                 # EXISTING
└── topology/            # NEW - THIS PHASE
    ├── __init__.py      # Export KNNTopoLoss, topology metrics
    ├── losses.py        # KNNTopoLoss class
    ├── graph.py         # kNN graph construction utilities
    └── metrics.py       # Trustworthiness, continuity, preservation metrics

tests/
├── test_energy.py       # EXISTING
├── test_viz/            # EXISTING
└── test_topology/       # NEW - THIS PHASE
    ├── __init__.py
    ├── test_losses.py   # Test KNNTopoLoss
    ├── test_graph.py    # Test kNN graph construction
    └── test_metrics.py  # Test topology metrics

examples/                # NEW - THIS PHASE
├── train_topo_ae.py     # Train autoencoder with topology loss only
└── visualize_topology.py # Visualize topology preservation metrics
```

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│            User's Encoder/Decoder               │
│                  x → z → x'                     │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  aps.topology           │
        │  ┌───────────────────┐  │
        │  │ KNNTopoLoss       │  │
        │  │ - knn_graph()     │  │
        │  │ - forward()       │  │
        │  └───────────────────┘  │
        │  ┌───────────────────┐  │
        │  │ Topology Metrics  │  │
        │  │ - trustworthiness │  │
        │  │ - continuity      │  │
        │  │ - knn_preservation│  │
        │  └───────────────────┘  │
        └─────────────────────────┘
```

### Data Flow

```
Training with Topology Loss:

1. Forward pass: z = encoder(x)
2. Compute reconstruction loss: L_recon = MSE(decoder(z), x)
3. Compute topology loss:
   a. Build k-NN graph in input space X → A_input
   b. Build k-NN graph in latent space Z → A_latent
   c. L_topo = BCE(A_latent, A_input)
4. Total loss: L = L_recon + λ_T * L_topo
5. Backward: L.backward()
6. Update: optimizer.step()

Evaluation:
1. Encode all data: Z = encoder(X)
2. Compute trustworthiness(X, Z, k)
3. Compute continuity(X, Z, k)
4. Compute knn_preservation_ratio(X, Z, k)
```

## Implementation Phases

### Phase 1: Graph Utilities (Foundation)

**Goal**: Implement kNN graph construction utilities

**Tasks**:
1. Create `src/aps/topology/` module structure
2. Implement `graph.py`:
   - `knn_indices(X: Tensor, k: int) -> Tensor`: Compute k-NN indices
   - `adjacency_from_knn(indices: Tensor) -> Tensor`: Convert to binary adjacency
   - `knn_graph(X: Tensor, k: int) -> Tensor`: Combined helper
3. Support both CPU and GPU tensors
4. Handle edge cases (k > batch_size, duplicates, etc.)
5. Write comprehensive tests (`test_graph.py`)

**Deliverable**: Robust kNN graph construction utilities

**Success Criteria**:
- Correct k-NN computation verified on known data
- Handles edge cases gracefully
- Performance < 300ms for batch of 128
- All tests pass (>90% coverage)

### Phase 2: Topology Loss Implementation

**Goal**: Implement KNNTopoLoss class

**Tasks**:
1. Implement `losses.py`:
   ```python
   class KNNTopoLoss(nn.Module):
       def __init__(self, k=8, loss_type='bce'):
           # k: number of neighbors
           # loss_type: 'bce' or 'contrastive'
       
       def forward(self, X, Z):
           # Compute k-NN graphs
           # Compute loss
           # Return scalar loss
   ```
2. Implement BCE variant:
   - Compare adjacency matrices using binary cross-entropy
3. Implement contrastive variant (optional):
   - Pull neighbors close, push non-neighbors apart
4. Support optional weighting (distant neighbors matter less)
5. Write tests (`test_losses.py`):
   - Test loss computation on synthetic data
   - Test gradient flow (loss.backward() works)
   - Test both variants produce reasonable values

**Deliverable**: Working KNNTopoLoss ready for training

**Success Criteria**:
- Loss computation < 200ms for batch of 128
- Gradients flow correctly
- Loss decreases when Z better preserves X topology
- All tests pass (>90% coverage)

### Phase 3: Topology Metrics

**Goal**: Implement evaluation metrics for topology preservation

**Tasks**:
1. Implement `metrics.py`:
   ```python
   def trustworthiness(X, Z, k=8):
       """Measure if close neighbors in Z were close in X"""
   
   def continuity(X, Z, k=8):
       """Measure if close neighbors in X are close in Z"""
   
   def knn_preservation_ratio(X, Z, k=8):
       """Jaccard similarity of k-NN sets"""
   ```
2. Implement efficient computation (vectorized)
3. Support batched evaluation for large datasets
4. Write tests (`test_metrics.py`):
   - Test metrics on known good/bad embeddings
   - Verify perfect embedding → metric = 1.0
   - Verify random embedding → metric ≈ 0.0

**Deliverable**: Evaluation metrics for topology preservation

**Success Criteria**:
- Metrics correctly distinguish good vs bad topology
- Efficient computation (< 1s for 1000 samples)
- All tests pass (>90% coverage)

### Phase 4: Integration & Examples

**Goal**: Demonstrate topology module usage

**Tasks**:
1. Create example: `examples/train_topo_ae.py`
   - Train simple autoencoder on MNIST
   - Add topology loss
   - Compare with/without topology
   - Plot trustworthiness over epochs
2. Create example: `examples/visualize_topology.py`
   - Load trained model
   - Compute topology metrics
   - Generate UMAP visualization
   - Show k-NN preservation heatmap
3. Update main README with topology module
4. Add docstrings and type hints throughout

**Deliverable**: Working examples demonstrating topology preservation

**Success Criteria**:
- Examples run without errors
- Clear improvement in topology metrics with L_T
- Visual evidence of better structure

### Phase 5: Testing & Validation

**Goal**: Comprehensive testing and validation

**Tasks**:
1. **Unit tests** (each function):
   - Test edge cases
   - Test error handling
   - Test device compatibility (CPU/GPU)
2. **Integration tests**:
   - Train small model with topology loss
   - Verify loss decreases
   - Verify metrics improve
3. **Synthetic data validation**:
   - Swiss roll manifold
   - Verify topology preserved
   - Compare to baseline (no topology loss)
4. **Performance profiling**:
   - Benchmark kNN computation
   - Benchmark loss computation
   - Identify bottlenecks

**Deliverable**: Validated and tested topology module

**Success Criteria**:
- All tests pass (target: >90% coverage)
- Performance meets targets (<500ms/batch)
- Swiss roll experiment shows clear improvement

## Mathematical Formulation

### Topology Loss (from paper, Section 4.1, page 7)

**Input**:
- X: Input data (batch_size, input_dim)
- Z: Latent representations (batch_size, latent_dim)
- k: Number of nearest neighbors

**Computation**:
1. Compute k-NN indices in X:
   ```
   I_X = knn_indices(X, k)  # (batch_size, k)
   ```

2. Convert to adjacency matrix:
   ```
   A_X[i, j] = 1 if j ∈ I_X[i] else 0
   A_X: (batch_size, batch_size) binary matrix
   ```

3. Similarly for Z:
   ```
   I_Z = knn_indices(Z, k)
   A_Z[i, j] = 1 if j ∈ I_Z[i] else 0
   ```

4. Compute BCE loss:
   ```
   L_T = BCE(A_Z, A_X) = -mean(A_X * log(A_Z) + (1-A_X) * log(1-A_Z))
   ```

**Alternative: Contrastive Loss**:
```
For each point i:
  - Neighbors: N_i = {j | j is k-NN of i in X}
  - Non-neighbors: N̄_i = {j | j not k-NN of i in X}
  
L_T = Σ_i [ Σ_{j∈N_i} ||z_i - z_j||² + Σ_{j∈N̄_i} max(0, margin - ||z_i - z_j||²) ]
```

### Topology Metrics

**Trustworthiness** (measures if close in Z → close in X):
```
T(k) = 1 - (2 / (n*k*(2n-3k-1))) * Σ_i Σ_{j∈U_k(i)} (r(i,j) - k)

Where:
- U_k(i) = points in k-NN of i in Z but not in X
- r(i,j) = rank of j in sorted distances from i in X
```

**Continuity** (measures if close in X → close in Z):
```
C(k) = 1 - (2 / (n*k*(2n-3k-1))) * Σ_i Σ_{j∈V_k(i)} (r'(i,j) - k)

Where:
- V_k(i) = points in k-NN of i in X but not in Z
- r'(i,j) = rank of j in sorted distances from i in Z
```

**k-NN Preservation Ratio**:
```
P(k) = (1/n) * Σ_i |N_X(i) ∩ N_Z(i)| / k

Where:
- N_X(i) = k-NN of i in X
- N_Z(i) = k-NN of i in Z
- Intersection counts shared neighbors
```

## Dependencies

### Internal (Existing)
- `aps.energy`: Independent (topology can be used alone)
- `aps.viz`: Optional (for visualizing results)

### External
- **PyTorch** (existing): All computation and gradients
- **NumPy** (existing): Array operations
- **scikit-learn** (existing): NearestNeighbors for kNN
- **scipy** (optional): Optimized distance computations

### Development
- **pytest** (existing): Testing
- **pytest-cov**: Coverage reports

## Testing Strategy

### Unit Tests

**Graph Utilities** (`test_graph.py`):
```python
def test_knn_indices_correctness():
    # Known data → verify correct neighbors
    
def test_knn_indices_k_larger_than_batch():
    # Edge case: k > batch_size
    
def test_adjacency_from_knn_shape():
    # Verify output shape
    
def test_knn_graph_gpu():
    # Test on GPU tensors
```

**Loss** (`test_losses.py`):
```python
def test_topology_loss_perfect_preservation():
    # Z = X → loss should be minimal
    
def test_topology_loss_random_preservation():
    # Z random → loss should be high
    
def test_topology_loss_gradients():
    # Verify gradients flow correctly
    
def test_topology_loss_batch_sizes():
    # Test various batch sizes
```

**Metrics** (`test_metrics.py`):
```python
def test_trustworthiness_perfect():
    # Perfect embedding → T(k) ≈ 1.0
    
def test_trustworthiness_random():
    # Random embedding → T(k) ≈ 0.5
    
def test_continuity_perfect():
    # Perfect embedding → C(k) ≈ 1.0
    
def test_knn_preservation_identity():
    # Z = X → preservation = 1.0
```

### Integration Tests

```python
def test_train_with_topology_loss():
    # Train small autoencoder
    # Verify loss decreases
    # Verify metrics improve
    
def test_swiss_roll_preservation():
    # Generate Swiss roll manifold
    # Train with topology loss
    # Verify manifold structure preserved
```

### Performance Tests

```python
def test_knn_computation_speed():
    # Benchmark various batch sizes
    # Verify < 300ms for batch of 128
    
def test_loss_computation_speed():
    # Benchmark loss forward pass
    # Verify < 200ms for batch of 128
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| kNN computation too slow | High | Use sklearn's efficient NearestNeighbors, cache when possible |
| Memory issues with adjacency matrix | Medium | Use sparse matrices, batch-wise computation |
| Loss conflicts with reconstruction | Medium | Provide weight tuning guidelines, start with low λ_T |
| Gradients vanish/explode | Low | Implement gradient clipping, normalize inputs |
| k-NN gives different results on GPU | Low | Use deterministic algorithms, test thoroughly |

## Success Metrics

### Quantitative Metrics:
1. **Performance**:
   - kNN graph computation: < 300ms for batch of 128
   - Topology loss computation: < 200ms for batch of 128
   
2. **Accuracy**:
   - Trustworthiness improvement: +10-20% over baseline
   - Continuity improvement: +10-20% over baseline
   - k-NN preservation: >0.8 on Swiss roll

3. **Code Quality**:
   - Test coverage: >90%
   - All tests pass
   - Type hints throughout

### Qualitative Metrics:
1. Visual inspection: UMAP plots show better structure with L_T
2. User-friendly API: Easy to add to existing training
3. Good documentation: Examples work out of the box

## Next Steps

### Immediate Actions:

1. **Create feature branch**:
   ```bash
   git checkout -b 002-topology
   ```

2. **Start with Phase 1 (Graph Utilities)**:
   - Implement `graph.py` first (most foundational)
   - Write tests as you go
   - Verify performance early

3. **Incremental Development**:
   - Complete Phase 1 → test → commit
   - Complete Phase 2 → test → commit
   - Complete Phase 3 → test → commit
   - etc.

4. **Milestone Demo**:
   - After Phase 2: Train simple model with L_T
   - Show loss curves and metrics
   - Prepare for Phase 3 (Causality)

### Connection to Next Phase (003-causality):

The topology module is designed to be **independent** but **composable**:
- Can be used alone: `L = L_recon + λ_T * L_topo`
- Or with energy: `L = L_recon + λ_T * L_topo + λ_E * L_energy`
- Or with causality (Phase 003): `L = L_recon + λ_T * L_topo + λ_C * L_causal`
- Or all together: `L_APS = L + λ_T * L_T + λ_C * L_C + λ_E * L_E`

## Appendix: Code Snippets

### Example Usage

```python
from aps.topology import KNNTopoLoss
from aps.topology.metrics import trustworthiness, continuity

# Create topology loss
topo_loss = KNNTopoLoss(k=8, loss_type='bce')

# Training loop
for epoch in range(num_epochs):
    for batch_x in dataloader:
        # Forward pass
        z = encoder(batch_x)
        x_recon = decoder(z)
        
        # Compute losses
        recon_loss = F.mse_loss(x_recon, batch_x)
        topo_loss_val = topo_loss(batch_x, z)
        
        # Total loss
        loss = recon_loss + lambda_T * topo_loss_val
        
        # Backward
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    Z = encoder(X_test)
    trust = trustworthiness(X_test, Z, k=8)
    cont = continuity(X_test, Z, k=8)
    print(f"Trustworthiness: {trust:.3f}, Continuity: {cont:.3f}")
```

### Example Model

```python
class TopologyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, k=8, lambda_T=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.topo_loss = KNNTopoLoss(k=k)
        self.lambda_T = lambda_T
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def compute_loss(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x)
        topo_loss = self.topo_loss(x, z)
        return recon_loss + self.lambda_T * topo_loss
```

## Appendix: Expected Results

### Swiss Roll Experiment

**Setup**:
- Generate 2000 points on Swiss roll manifold (3D)
- Train autoencoder to 2D latent space
- Compare with/without topology loss

**Expected Results**:
| Metric | Baseline | With Topology Loss |
|--------|----------|-------------------|
| Trustworthiness | 0.65 | 0.85+ |
| Continuity | 0.60 | 0.80+ |
| k-NN Preservation | 0.55 | 0.80+ |

**Visual**: Unrolled Swiss roll with smooth structure (not tangled)

### MNIST Experiment

**Setup**:
- Train autoencoder on MNIST (784D → 2D)
- Compare with/without topology loss

**Expected Results**:
- With L_T: Clear digit clusters, smooth transitions
- Without L_T: Overlapping clusters, discontinuities
- Trustworthiness: +15-20% improvement

## References

- **Chen et al. 2022**: Local Distance Preserving Auto-encoders using Continuous k-Nearest Neighbours Graphs. arXiv:2206.05909
- **Moor et al. 2020**: Topological Autoencoders. ICML 2020
- **Lee et al. 2021**: Neighborhood Reconstructing Autoencoders. NeurIPS 2021
- **van der Maaten & Hinton 2008**: Visualizing Data using t-SNE. JMLR
- **McInnes et al. 2018**: UMAP: Uniform Manifold Approximation and Projection
