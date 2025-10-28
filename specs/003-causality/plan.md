# Implementation Plan: Causal Invariance Module (C)

**Branch**: `003-causality` | **Date**: 2025-01-27  
**Status**: Draft - Aligned with "Atlasing Pattern Space (APS) v1" paper  
**Dependencies**: Builds on `aps.energy` (complete), `aps.viz` (complete), `aps.topology` (Phase 002)

## Summary

Implement the **Causal Invariance (C)** component of the APS framework using HSIC independence and IRM invariance losses to learn representations that are invariant to nuisance factors and capture stable, generalizable features.

**Paper Reference**: Section 3.2 (Causal and Invariant Representation Learning) and Section 4.1 (Causality Loss)

**Core Formulas**:
```
# HSIC Independence:
L_C = HSIC(Z, V)  where V are nuisance variables

# IRM Penalty:
L_C = Σ_e ||∇_w risk_e(w ∘ f)||²  across environments e
```

## Technical Context

**Language/Version**: Python 3.9+  
**Core Framework**: PyTorch (existing)  
**Primary Dependencies**:
- PyTorch (existing)
- NumPy (existing)
- Optional: torchvision (for data augmentation as pseudo-environments)

**Integration Points**:
- Can be used standalone or with `aps.energy` and `aps.topology`
- Designed for both supervised and unsupervised scenarios
- Supports explicit nuisance variables or environment labels

**Performance Goals**:
- HSIC computation: < 100ms for batch of 128 samples
- IRM penalty computation: < 200ms for batch with 2-3 environments
- Combined: < 300ms total per batch

**Constraints**:
- Must be differentiable (compatible with PyTorch autograd)
- Support scenarios with/without explicit nuisance variables
- Memory efficient for typical batch sizes (32-256 samples)

## Alignment with APS Paper

### Paper's Causality Component (Section 3.2 & 4.1)

**Related Work Referenced**:
- ✅ **Arjovsky et al. 2019** - Invariant Risk Minimization (IRM) - **IMPLEMENTED**
- ✅ **Greenfeld & Shalit 2020** - HSIC for Robustness - **IMPLEMENTED**
- Higgins et al. 2017 - β-VAE (optional future extension for disentanglement)

**Why HSIC + IRM Approach**:
1. **HSIC**: Direct, differentiable independence measure (kernel-based)
2. **IRM**: Learns environment-invariant predictors (handles spurious correlations)
3. Complementary: HSIC for known nuisances, IRM for unknown environment shifts
4. Well-established in causal ML literature

### Paper's Description (Pages 7-8)

> "Multi-environment IRM loss: If we have data segmented into environments (or we create environments via augmentation), we can apply the IRM principle... We can incorporate a differentiable approximation of this condition."

> "HSIC loss for independence: If certain nuisance factors v are known or can be estimated (e.g. image background, speaker identity in text, or simply the environment index), we add a loss L_C = HSIC(Z, v) to minimize the HSIC between latent representation and the nuisance variable."

## Project Structure

```text
src/aps/
├── energy/              # EXISTING
├── viz/                 # EXISTING
├── topology/            # EXISTING (Phase 002)
└── causality/           # NEW - THIS PHASE
    ├── __init__.py      # Export HSICLoss, IRMLoss
    ├── hsic.py          # HSIC independence loss implementation
    ├── irm.py           # Invariant Risk Minimization loss
    ├── kernels.py       # RBF and other kernel functions
    └── utils.py         # Environment handling, data augmentation

tests/
├── test_energy.py       # EXISTING
├── test_viz/            # EXISTING
├── test_topology/       # EXISTING (Phase 002)
└── test_causality/      # NEW - THIS PHASE
    ├── __init__.py
    ├── test_hsic.py     # Test HSIC loss and kernels
    ├── test_irm.py      # Test IRM loss
    ├── test_kernels.py  # Test kernel functions
    └── test_utils.py    # Test environment utilities

examples/                # NEW - THIS PHASE
├── train_hsic_ae.py     # Train with HSIC loss (e.g., colored MNIST)
├── train_irm_ae.py      # Train with IRM loss (multi-environment)
└── ood_generalization.py # Demonstrate OOD generalization improvement
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
        │  aps.causality          │
        │  ┌───────────────────┐  │
        │  │ HSICLoss          │  │
        │  │ - compute_hsic()  │  │
        │  │ - forward(z, v)   │  │
        │  └───────────────────┘  │
        │  ┌───────────────────┐  │
        │  │ IRMLoss           │  │
        │  │ - irm_penalty()   │  │
        │  │ - forward(envs)   │  │
        │  └───────────────────┘  │
        │  ┌───────────────────┐  │
        │  │ Kernels           │  │
        │  │ - rbf_kernel()    │  │
        │  │ - linear_kernel() │  │
        │  └───────────────────┘  │
        └─────────────────────────┘
```

### Data Flow

```
Training with HSIC Loss:

1. Forward pass: z = encoder(x)
2. Extract nuisance variables: v (e.g., color in colored MNIST)
3. Compute task loss: L_task = MSE(decoder(z), x) or CE(classifier(z), y)
4. Compute HSIC loss:
   a. Compute kernel matrices K_Z and K_V
   b. Center kernels
   c. L_hsic = trace(K_Z_centered @ K_V_centered)
5. Total loss: L = L_task + λ_C * L_hsic
6. Backward: L.backward()
7. Update: optimizer.step()

Training with IRM Loss:

1. Split batch into environments (or use pre-defined env labels)
2. For each environment e:
   a. Forward pass: z_e = encoder(x_e)
   b. Predict: logits_e = w * classifier(z_e)  [w is dummy scalar]
   c. Compute risk: risk_e = CE(logits_e, y_e)
3. Compute IRM penalty: Σ_e ||∇_w risk_e||²
4. Total loss: L = L_task + λ_C * L_irm
5. Backward: L.backward()
6. Update: optimizer.step()
```

## Implementation Phases

### Phase 1: Kernel Functions (Foundation)

**Goal**: Implement kernel functions for HSIC computation

**Tasks**:
1. Create `src/aps/causality/` module structure
2. Implement `kernels.py`:
   ```python
   def rbf_kernel(X, Y, sigma=1.0):
       """RBF (Gaussian) kernel: K(x,y) = exp(-||x-y||²/2σ²)"""
   
   def linear_kernel(X, Y):
       """Linear kernel: K(x,y) = x·y"""
   
   def center_kernel(K):
       """Center kernel matrix: H @ K @ H"""
   ```
3. Support batched computation
4. Handle both CPU and GPU tensors
5. Write comprehensive tests (`test_kernels.py`)

**Deliverable**: Robust kernel implementations

**Success Criteria**:
- Correct kernel computation verified on known data
- Efficient (vectorized) computation
- All tests pass (>90% coverage)

### Phase 2: HSIC Loss Implementation

**Goal**: Implement HSIC independence loss

**Tasks**:
1. Implement `hsic.py`:
   ```python
   class HSICLoss(nn.Module):
       def __init__(self, kernel='rbf', sigma=1.0):
           # kernel: 'rbf' or 'linear'
           # sigma: bandwidth for RBF kernel
       
       def forward(self, Z, V):
           # Z: latent representations (batch, latent_dim)
           # V: nuisance variables (batch, nuisance_dim)
           # Returns: HSIC(Z, V) - to be minimized
   ```
2. Implement unbiased HSIC estimator:
   - Compute kernel matrices K_Z and K_V
   - Center both matrices
   - Return trace(K_Z_centered @ K_V_centered) / (n-1)²
3. Handle numerical stability (very small values)
4. Support different kernel types
5. Write tests (`test_hsic.py`):
   - Test HSIC on independent variables (should be ~0)
   - Test HSIC on dependent variables (should be > 0)
   - Test gradient flow
   - Test different batch sizes

**Deliverable**: Working HSICLoss ready for training

**Success Criteria**:
- HSIC computation < 100ms for batch of 128
- Correctly detects independence/dependence
- Gradients flow correctly
- All tests pass (>90% coverage)

### Phase 3: IRM Loss Implementation

**Goal**: Implement Invariant Risk Minimization penalty

**Tasks**:
1. Implement `utils.py`:
   ```python
   def split_into_environments(X, y, n_envs=2, method='random'):
       """Split data into environments for IRM"""
   
   def augment_as_environments(X, augmentation_fn):
       """Use data augmentation to create pseudo-environments"""
   ```
2. Implement `irm.py`:
   ```python
   class IRMLoss(nn.Module):
       def __init__(self, num_envs=2):
           # num_envs: number of environments
       
       def forward(self, encoder, classifier, envs, labels):
           # envs: list of (x_e, env_idx) tuples
           # Returns: IRM penalty Σ_e ||∇_w risk_e||²
   ```
3. Implement gradient computation for dummy classifier:
   - Use torch.autograd.grad with create_graph=True
   - Compute penalty efficiently across environments
4. Support both supervised (with labels) and unsupervised cases
5. Write tests (`test_irm.py`):
   - Test IRM penalty computation
   - Test with known invariant features
   - Test gradient flow
   - Test environment handling

**Deliverable**: Working IRMLoss ready for training

**Success Criteria**:
- IRM computation < 200ms for batch with 2-3 envs
- Encourages invariant representations
- Gradients flow correctly
- All tests pass (>90% coverage)

### Phase 4: Integration & Examples

**Goal**: Demonstrate causality module usage

**Tasks**:
1. Create example: `examples/train_hsic_ae.py`
   - Colored MNIST experiment (per paper Section 5)
   - Train autoencoder with HSIC(z, color)
   - Show color becomes independent of latent
   - Measure HSIC value over training
2. Create example: `examples/train_irm_ae.py`
   - Multi-environment training
   - Compare IRM vs no IRM
   - Show improved invariance
3. Create example: `examples/ood_generalization.py`
   - Colored MNIST with spurious correlation
   - Train with HSIC or IRM
   - Test on flipped correlation (OOD)
   - Compare accuracy drop vs baseline
4. Update main README with causality module
5. Add docstrings and type hints throughout

**Deliverable**: Working examples demonstrating causal invariance

**Success Criteria**:
- Examples run without errors
- Clear improvement in OOD generalization with L_C
- HSIC value decreases during training
- Visual evidence of independence

### Phase 5: Testing & Validation

**Goal**: Comprehensive testing and validation

**Tasks**:
1. **Unit tests** (each function):
   - Test edge cases
   - Test error handling
   - Test device compatibility (CPU/GPU)
2. **Integration tests**:
   - Train model with HSIC loss
   - Train model with IRM loss
   - Verify losses decrease
   - Verify OOD performance improves
3. **Synthetic data validation**:
   - Create data with known spurious correlations
   - Verify causality loss removes dependency
4. **Performance profiling**:
   - Benchmark HSIC computation
   - Benchmark IRM computation
   - Identify bottlenecks

**Deliverable**: Validated and tested causality module

**Success Criteria**:
- All tests pass (target: >90% coverage)
- Performance meets targets (<300ms/batch)
- Colored MNIST experiment shows OOD improvement

## Mathematical Formulation

### HSIC Loss (from paper, Section 4.1, page 8)

**Input**:
- Z: Latent representations (batch_size, latent_dim)
- V: Nuisance variables (batch_size, nuisance_dim)
- kernel: Kernel function (RBF or linear)
- sigma: Bandwidth parameter for RBF

**Computation**:
1. Compute kernel matrices:
   ```
   K_Z = kernel(Z, Z)  # (batch, batch)
   K_V = kernel(V, V)  # (batch, batch)
   ```

2. Center kernel matrices:
   ```
   H = I - (1/n) * 1·1ᵀ  # Centering matrix
   K_Z_c = H @ K_Z @ H
   K_V_c = H @ K_V @ H
   ```

3. Compute HSIC (unbiased estimator):
   ```
   HSIC(Z, V) = trace(K_Z_c @ K_V_c) / (n - 1)²
   ```

**Properties**:
- HSIC = 0 ⟺ Z ⊥ V (independence)
- HSIC > 0 indicates dependence
- Minimizing HSIC encourages Z to be independent of V

**RBF Kernel**:
```
K_RBF(x, y) = exp(-||x - y||² / (2σ²))
```

### IRM Loss (from paper, Section 4.1, pages 7-8)

**Input**:
- encoder f: Maps inputs to latent z
- classifier c: Maps latent to predictions
- environments E = {(X_e, y_e) for e ∈ {1,...,n_env}}
- w: Dummy scalar weight (trainable)

**Computation**:
1. For each environment e:
   ```
   z_e = f(X_e)
   logits_e = w * c(z_e)  # Dummy classifier with scalar w
   risk_e = CrossEntropy(logits_e, y_e)
   ```

2. Compute gradient penalty:
   ```
   grad_e = ∇_w risk_e  # Gradient w.r.t. dummy weight
   penalty = Σ_e ||grad_e||²
   ```

**Properties**:
- penalty = 0 ⟺ optimal w is same across all environments
- Encourages f to extract features that work invariantly
- Removes spurious correlations specific to environments

**Intuition**:
If the latent representation z captures true causal features, the same classifier should work across all environments. IRM enforces this by penalizing gradient variance.

## Dependencies

### Internal (Existing)
- `aps.energy`: Independent (causality can be used alone)
- `aps.topology`: Independent (can combine both)
- `aps.viz`: Optional (for visualizing results)

### External
- **PyTorch** (existing): All computation and gradients
- **NumPy** (existing): Array operations
- **torchvision** (optional): Data augmentation for pseudo-environments

### Development
- **pytest** (existing): Testing
- **pytest-cov**: Coverage reports

## Testing Strategy

### Unit Tests

**Kernels** (`test_kernels.py`):
```python
def test_rbf_kernel_identity():
    # K(x, x) should be 1.0
    
def test_rbf_kernel_symmetry():
    # K(x, y) = K(y, x)
    
def test_center_kernel():
    # Verify centering: row/col sums = 0
    
def test_kernels_gpu():
    # Test on GPU tensors
```

**HSIC** (`test_hsic.py`):
```python
def test_hsic_independent_variables():
    # Z, V independent → HSIC ≈ 0
    
def test_hsic_identical_variables():
    # Z = V → HSIC > 0 (max dependence)
    
def test_hsic_gradients():
    # Verify gradients flow correctly
    
def test_hsic_batch_sizes():
    # Test various batch sizes
```

**IRM** (`test_irm.py`):
```python
def test_irm_penalty_invariant_features():
    # Invariant features → penalty ≈ 0
    
def test_irm_penalty_spurious_features():
    # Spurious features → penalty > 0
    
def test_irm_gradients():
    # Verify gradients flow correctly
    
def test_irm_multiple_environments():
    # Test with 2, 3, 5 environments
```

### Integration Tests

```python
def test_train_with_hsic_loss():
    # Train model with HSIC loss
    # Verify HSIC decreases
    # Verify independence achieved
    
def test_train_with_irm_loss():
    # Train model with IRM loss
    # Verify penalty decreases
    # Verify invariance achieved
    
def test_colored_mnist_ood():
    # Colored MNIST experiment
    # Train with HSIC or IRM
    # Test OOD (flipped colors)
    # Verify accuracy improvement
```

### Performance Tests

```python
def test_hsic_computation_speed():
    # Benchmark various batch sizes
    # Verify < 100ms for batch of 128
    
def test_irm_computation_speed():
    # Benchmark with 2-3 environments
    # Verify < 200ms for batch of 128
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HSIC requires manual nuisance specification | High | Provide data augmentation as automatic nuisance, examples |
| IRM requires multiple environments | Medium | Support pseudo-environments via augmentation |
| Kernel bandwidth (sigma) sensitive | Medium | Provide auto-tuning heuristic, sensible default |
| IRM penalty conflicts with task loss | Medium | Provide weight tuning guidelines, curriculum |
| Numerical instability with small batches | Low | Add epsilon for stability, test thoroughly |

## Success Metrics

### Quantitative Metrics:
1. **Performance**:
   - HSIC computation: < 100ms for batch of 128
   - IRM computation: < 200ms for batch with 2-3 envs
   
2. **Effectiveness**:
   - HSIC(z, nuisance) < 0.1 after training
   - OOD accuracy drop < 10% (vs >30% baseline)
   - IRM penalty decreases during training

3. **Code Quality**:
   - Test coverage: >90%
   - All tests pass
   - Type hints throughout

### Qualitative Metrics:
1. Colored MNIST: visual confirmation color doesn't affect clustering
2. OOD generalization: clear improvement in examples
3. User-friendly API: Easy to add to existing training

## Next Steps

### Immediate Actions:

1. **Create feature branch** (after 002-topology complete):
   ```bash
   git checkout -b 003-causality
   ```

2. **Start with Phase 1 (Kernels)**:
   - Implement `kernels.py` first (most foundational)
   - Write tests as you go
   - Verify correctness early

3. **Incremental Development**:
   - Complete Phase 1 → test → commit
   - Complete Phase 2 (HSIC) → test → commit
   - Complete Phase 3 (IRM) → test → commit
   - Complete Phase 4-5 → test → commit

4. **Milestone Demo**:
   - After Phase 2: Colored MNIST with HSIC
   - After Phase 3: Multi-environment with IRM
   - Prepare for Phase 004 (Integration)

### Connection to Next Phase (004-integration):

The causality module is designed to be **independent** but **composable**:
- Can be used alone: `L = L_task + λ_C * L_causal`
- Or with energy: `L = L_task + λ_C * L_causal + λ_E * L_energy`
- Or with topology: `L = L_task + λ_C * L_causal + λ_T * L_topo`
- **Or all together** (Phase 004): `L_APS = L_task + λ_T * L_T + λ_C * L_C + λ_E * L_E`

## Appendix: Code Snippets

### Example Usage - HSIC

```python
from aps.causality import HSICLoss

# Create HSIC loss
hsic_loss = HSICLoss(kernel='rbf', sigma=1.0)

# Training loop (Colored MNIST example)
for epoch in range(num_epochs):
    for batch_x, batch_color in dataloader:
        # batch_color: nuisance variable (e.g., [0, 1] for red/blue)
        
        # Forward pass
        z = encoder(batch_x)
        x_recon = decoder(z)
        
        # Compute losses
        recon_loss = F.mse_loss(x_recon, batch_x)
        hsic_val = hsic_loss(z, batch_color.unsqueeze(1))
        
        # Total loss (minimize HSIC to enforce independence)
        loss = recon_loss + lambda_C * hsic_val
        
        # Backward
        loss.backward()
        optimizer.step()
        
        print(f"HSIC(z, color): {hsic_val.item():.4f}")
```

### Example Usage - IRM

```python
from aps.causality import IRMLoss

# Create IRM loss
irm_loss = IRMLoss(num_envs=2)

# Training loop (Multi-environment)
for epoch in range(num_epochs):
    for env1, env2 in dataloader:
        # env1, env2: (x, y) tuples for different environments
        
        # Prepare environments
        envs = [(env1[0], env1[1]), (env2[0], env2[1])]
        
        # Compute task loss and IRM penalty
        task_loss = compute_task_loss(encoder, classifier, envs)
        irm_penalty = irm_loss(encoder, classifier, envs, None)
        
        # Total loss
        loss = task_loss + lambda_C * irm_penalty
        
        # Backward
        loss.backward()
        optimizer.step()
        
        print(f"IRM penalty: {irm_penalty.item():.4f}")
```

## Appendix: Expected Results

### Colored MNIST Experiment (per paper Section 5)

**Setup**:
- MNIST digits colored red or blue
- Training: 90% spurious correlation (digit 0 → red, digit 1 → blue, etc.)
- Testing: Flipped correlation (digit 0 → blue, digit 1 → red)

**Expected Results**:
| Method | Train Acc | Test Acc (OOD) | Accuracy Drop |
|--------|-----------|----------------|---------------|
| Baseline | 95% | 60% | -35% |
| With HSIC | 93% | 85% | -8% |
| With IRM | 92% | 87% | -5% |

**Analysis**:
- Baseline relies on color (spurious correlation)
- HSIC/IRM learn digit shape (causal feature)
- Much better OOD generalization

### HSIC Value Over Training

**Expected Behavior**:
- Epoch 0: HSIC(z, color) ≈ 2.0 (high dependence)
- Epoch 50: HSIC(z, color) ≈ 0.5
- Epoch 100: HSIC(z, color) < 0.1 (near independence)

## References

- **Arjovsky et al. 2019**: Invariant Risk Minimization. arXiv:1907.02893
- **Greenfeld & Shalit 2020**: Robust Learning with the Hilbert-Schmidt Independence Criterion. ICML 2020
- **Schölkopf et al. 2021**: Toward Causal Representation Learning. Proceedings of the IEEE
- **Higgins et al. 2017**: β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2017
- **Gretton et al. 2005**: Measuring Statistical Dependence with Hilbert-Schmidt Norms (original HSIC paper)
