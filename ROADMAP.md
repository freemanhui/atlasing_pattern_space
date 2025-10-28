# APS Framework Implementation Roadmap

**Atlasing Pattern Space (APS)** - A framework for structured latent representations  
**Paper**: "Atlasing Pattern Space: A Framework for Structured Latent Representations in LLMs" (v1)

## Core Objective

Implement the complete APS framework as described in the paper:

```
L_APS = L_task + λ_T·L_T + λ_C·L_C + λ_E·L_E
```

Where:
- **L_task**: Task loss (reconstruction, classification, etc.)
- **L_T**: Topology preservation loss (manifold structure)
- **L_C**: Causal invariance loss (stable, generalizable features)
- **L_E**: Energy shaping loss (attractor basins)

## Implementation Phases

### ✅ Phase 001: Energy Basin Visualization (COMPLETE)

**Status**: Complete (70 tests passing)  
**Branch**: `001-energy-basin-viz`  
**Spec**: `specs/001-energy-basin-viz/`

**Completed**:
- ✅ `aps.energy.MemoryEnergy`: Memory-based energy function with attractor basins
- ✅ `aps.viz`: Interactive visualization system for energy landscapes
  - 2D heatmaps with memory pattern markers
  - Interactive point exploration (hover/click)
  - Trajectory visualization (gradient descent paths)
  - Cross-sectional views (1D energy profiles)
  - Export to HTML, PNG, SVG

**Deliverables**:
- Energy module fully functional
- Visualization tools for understanding basin structure
- Examples and documentation complete

---

### ✅ Phase 002: Topology Preservation Module (COMPLETE)

**Status**: Complete (62 tests passing)  
**Branch**: `002-topology`  
**Spec**: `specs/002-topology/plan.md`

**Goal**: Implement topology-preserving loss using continuous k-NN graph approach (Chen et al. 2022)

**Core Formula**:
```
L_T = BCE(A_latent, A_input)
```
Where A_input and A_latent are k-NN adjacency matrices.

**Completed Components**:
- ✅ `aps.topology.graph`: kNN graph construction utilities
  - `knn_indices()`: sklearn-based efficient k-NN computation
  - `adjacency_from_knn()`: convert indices to binary adjacency
  - `knn_graph()`: one-step graph construction with continuous/discrete modes
  - `_continuous_knn_graph()`: differentiable sigmoid-based adjacency
- ✅ `aps.topology.losses`: KNNTopoLoss class
  - BCE loss between input and latent k-NN graphs
  - Continuous (differentiable) k-NN computation
  - Temperature-controlled sigmoid sharpness
- ✅ `aps.topology.model`: TopologicalAutoencoder
  - Combined reconstruction + topology loss
  - Configurable topology weight
  - Simple MLP encoder/decoder architecture

**Implementation Details**:
- Continuous k-NN approach: `A_ij = sigmoid(temp * (d_k - d_ij))`
- Fully differentiable end-to-end
- Efficient sklearn-based k-NN for discrete mode
- PyTorch-native distance computation for continuous mode

**Test Coverage**:
- Graph utilities: 25 tests (shapes, correctness, performance, devices)
- Topology loss: 18 tests (gradients, preservation quality, edge cases)
- Model: 21 tests (training, loss components, integration)
- **Total: 62 tests, 59 passed, 3 skipped (CUDA)**
- ~4s test suite runtime

**Deliverables**:
- Complete topology module ready for use
- Clean API with dataclass configs
- Comprehensive test coverage
- Ready for integration with causality module

---

### 📅 Phase 003: Causal Invariance Module (PLANNED)

**Status**: Planned (after 002 complete)  
**Branch**: `003-causality`  
**Spec**: `specs/003-causality/plan.md`

**Goal**: Implement causal invariance losses (HSIC + IRM) for learning stable, generalizable features

**Core Formulas**:
```
# HSIC Independence:
L_C = HSIC(Z, V)  where V are nuisance variables

# IRM Penalty:
L_C = Σ_e ||∇_w risk_e(w ∘ f)||²  across environments
```

**Components to Implement**:
- `aps.causality.kernels`: RBF and linear kernel functions
- `aps.causality.hsic`: HSIC independence loss
- `aps.causality.irm`: Invariant Risk Minimization loss
- `aps.causality.utils`: Environment handling, augmentation

**Sub-Phases**:
1. Kernel functions (foundation)
2. HSIC loss implementation
3. IRM loss implementation
4. Integration & examples
5. Testing & validation

**Expected Outcomes**:
- HSIC(z, nuisance) < 0.1 after training
- OOD accuracy drop < 10% (vs >30% baseline)
- Colored MNIST: learns shape not color

**Next Steps**: Start after 002-topology is complete

---

### 📅 Phase 004: Integration & Full APS Training (PLANNED)

**Status**: Planned (after 002 & 003 complete)  
**Branch**: `004-integration`  
**Spec**: TBD

**Goal**: Create unified training framework combining all three components (T+C+E)

**Components to Implement**:
- `aps.training.config`: APSConfig dataclass (hyperparameters)
- `aps.training.trainer`: APSTrainer class (coordinates T, C, E)
- `aps.training.callbacks`: Logging, visualization, checkpointing

**Key Features**:
- Unified loss: `L_APS = L_task + λ_T·L_T + λ_C·L_C + λ_E·L_E`
- Curriculum learning (gradually increase regularization)
- Multi-objective optimization
- Integration with `aps.viz` for monitoring

**Expected Outcomes**:
- Full APS training on MNIST, AG News, CIFAR-10
- Ablation studies showing each component's contribution
- Reproduce paper experiments (Section 5)

---

### 📅 Phase 005: Experiments & Validation (PLANNED)

**Status**: Planned (after 004 complete)  
**Branch**: `005-experiments`  
**Spec**: TBD

**Goal**: Reproduce all experiments from paper Section 5

**Experiments**:
1. **Qualitative Visualization** (MNIST, AG News)
   - Clear clusters (E), smooth transitions (T), factor independence (C)
2. **Quantitative Topology** (Swiss roll)
   - Measure trustworthiness, continuity
3. **OOD Generalization** (Colored MNIST)
   - Spurious correlation → test on flipped correlation
4. **Energy Basin Utility** (CIFAR-10)
   - Clustering purity, few-shot classification
5. **Ablation Studies**
   - T-only, C-only, E-only, T+E, C+E, T+C, T+C+E

**Deliverables**:
- Experimental scripts for all paper experiments
- Results validating paper claims
- Visualizations matching paper quality

---

### 📅 Phase 006: Documentation & Release (PLANNED)

**Status**: Planned (after 005 complete)  
**Branch**: `main`  
**Spec**: TBD

**Goal**: Production-ready release with comprehensive documentation

**Tasks**:
1. **API Documentation**:
   - Full docstrings, type hints
   - API reference guide
2. **User Guide**:
   - Quick start tutorial
   - Hyperparameter tuning guide
   - Best practices
3. **Example Scripts**:
   - `examples/train_aps_mnist.py`
   - `examples/train_aps_text.py`
   - `examples/colored_mnist_ood.py`
4. **Paper Alignment**:
   - Map implementation to paper sections
   - Explain design decisions
5. **Performance Optimization**:
   - Profile and optimize bottlenecks
   - Multi-GPU support

**Deliverables**:
- Complete, documented APS framework
- Ready for research community use
- Paper publication materials

---

## Current Status Summary

### Completed ✅
- **Energy (E)**: `aps.energy` module complete
- **Visualization**: `aps.viz` module complete (all 4 user stories)
- **Foundation**: Project structure, testing framework, documentation

### In Progress 🚧
- **Topology (T)**: `specs/002-topology/plan.md` ready for implementation

### Planned 📅
- **Causality (C)**: `specs/003-causality/plan.md` created, awaits 002
- **Integration**: Unified APS trainer, awaits 002 & 003
- **Experiments**: Paper reproduction, awaits 004
- **Release**: Documentation and optimization, awaits 005

## Technical Stack

- **Language**: Python 3.9+
- **Core**: PyTorch (gradients, autograd)
- **Computation**: NumPy, scikit-learn
- **Visualization**: Plotly (interactive), Matplotlib (static)
- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff (linting), black (formatting)

## Performance Targets

| Component | Target | Status |
|-----------|--------|--------|
| Energy landscape (100x100 grid) | < 5s | ✅ Achieved |
| Topology loss (batch=128) | < 500ms | 📅 Phase 002 |
| Causality loss (batch=128) | < 300ms | 📅 Phase 003 |
| Combined APS training | < 2x vanilla AE | 📅 Phase 004 |

## Success Metrics (from Paper Section 5)

| Metric | Target | Phase |
|--------|--------|-------|
| Trustworthiness improvement | +10-20% | 002 |
| Continuity improvement | +10-20% | 002 |
| OOD accuracy drop | < 10% | 003 |
| HSIC(z, nuisance) | < 0.1 | 003 |
| Clustering purity | > 80% | 001 ✅ |
| Full APS vs ablations | Best combined | 004 |

## Repository Structure

```
atlasing_pattern_space/
├── docs/                    # Paper and documentation
│   └── Atlasing Pattern Space (APS) v1.pdf
├── specs/                   # Implementation specifications
│   ├── 001-energy-basin-viz/  # ✅ Complete
│   ├── 002-topology/          # 🚧 In progress
│   └── 003-causality/         # 📅 Planned
├── src/aps/                 # Main package
│   ├── energy/              # ✅ Complete (E component)
│   ├── viz/                 # ✅ Complete (visualization)
│   ├── topology/            # 📅 Phase 002
│   ├── causality/           # 📅 Phase 003
│   └── training/            # 📅 Phase 004
├── tests/                   # Test suite
│   ├── test_energy.py       # ✅ Complete
│   ├── test_viz/            # ✅ Complete (70 tests)
│   ├── test_topology/       # 📅 Phase 002
│   ├── test_causality/      # 📅 Phase 003
│   └── test_training/       # 📅 Phase 004
├── examples/                # Example scripts
│   ├── run_energy_demo.py   # ✅ Complete
│   └── ...                  # 📅 More coming
├── ROADMAP.md               # This file
├── README.md                # Project overview
└── pyproject.toml           # Dependencies

Status: ✅ Complete | 🚧 In Progress | 📅 Planned
```

## Key Design Decisions

### Modularity
Each component (T, C, E) is **independent** but **composable**:
- Use individually: `L = L_task + λ_T * L_T`
- Combine pairwise: `L = L_task + λ_T * L_T + λ_E * L_E`
- Full APS: `L_APS = L_task + λ_T * L_T + λ_C * L_C + λ_E * L_E`

### Paper Alignment
All implementation decisions traced to paper:
- **Topology**: k-NN graph (Chen et al. 2022) - straightforward, differentiable
- **Causality**: HSIC + IRM - well-established, complementary
- **Energy**: Memory-based (Hopfield-style) - explicit attractors

### Testing Philosophy
- Write tests first (TDD approach)
- Unit tests for each function
- Integration tests for workflows
- Performance benchmarks for all components
- Reproduce paper experiments as validation

## Timeline Estimate

| Phase | Duration | Dependencies | Start After |
|-------|----------|--------------|-------------|
| 001 Energy | COMPLETE | None | - |
| 002 Topology | 2-3 weeks | 001 | Now |
| 003 Causality | 2-3 weeks | 002 | 002 complete |
| 004 Integration | 2 weeks | 002, 003 | 003 complete |
| 005 Experiments | 2 weeks | 004 | 004 complete |
| 006 Release | 1 week | 005 | 005 complete |

**Total**: ~10-12 weeks for full implementation

## Getting Started

### For Phase 002 (Topology):
```bash
# Create feature branch
git checkout -b 002-topology

# Review spec
cat specs/002-topology/plan.md

# Start with graph utilities
# Implement: src/aps/topology/graph.py
# Test: tests/test_topology/test_graph.py
```

### For Contributors:
1. Read paper: `docs/Atlasing Pattern Space (APS) v1.pdf`
2. Review phase spec: `specs/00X-phase-name/plan.md`
3. Follow TDD: write tests first
4. Implement incrementally: one sub-phase at a time
5. Validate: run tests, check performance

## References

- **Paper**: Atlasing Pattern Space (APS) v1 (this implementation)
- **Chen et al. 2022**: Local Distance Preserving AE (topology)
- **Arjovsky et al. 2019**: Invariant Risk Minimization (causality)
- **Greenfeld & Shalit 2020**: HSIC for Robustness (causality)
- **Pang et al. 2020**: Latent EBM (energy)
- **Ramsauer et al. 2021**: Hopfield Networks (energy)

## Contact & Contributions

For questions about implementation:
- Review relevant phase spec in `specs/`
- Check existing tests for examples
- Refer to paper for theoretical background

---

**Last Updated**: 2025-01-27  
**Current Phase**: 002-topology (ready to start)  
**Next Milestone**: Complete topology module, start causality module
