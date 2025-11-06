# When Does Causal Regularization Help? 

**A Systematic Study of Boundary Conditions in Spurious Correlation Learning**

[![Code](https://img.shields.io/badge/Code-Python-blue)](src/aps/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **TL;DR**: Autoencoder architectures possess strong implicit causal bias, achieving 82-86% accuracy on 99% spurious correlations *before* explicit regularization. We establish clear boundary conditions for when topology, causality, and energy regularization help—and when they don't.

---

## 🎯 Core Finding: Implicit Causal Bias

**Key Discovery**: Architectural choice (reconstruction) is often more powerful than explicit causal regularization.

```
ColoredMNIST (99% spurious correlation):
  Autoencoder Baseline:  82-86% accuracy
  + Topology (T):        +0-2pp gain
  + Causality (C):       +0-4pp gain
  + Energy (E):          Prevents overfitting, minimal OOD gain

Conclusion: Architecture is primary, explicit regularizers are secondary corrections.
```

---

## 📊 Main Results

### 1. Boundary Conditions for Causal Learning (ColoredMNIST)

| Correlation | Baseline | APS-T | APS-C | APS-Full |
|-------------|----------|-------|-------|----------|
| 99.5%       | 82.69%   | 82.69%| 82.63%| 82.20%   |
| 99%/-99%    | 85.98%   | 84.57%| 84.51%| 86.12%   |

**Key Insights**:
- ✅ **Implicit bias discovered**: Reconstruction forces structural learning
- ⚠️ **Marginal explicit benefits**: Only 0-4pp improvements
- 🎯 **Phase transition at 100%**: All methods fail without causal signal

### 2. Topology Preservation: Domain-Specific

| Domain | Dimensionality | Topology Preservation | Result |
|--------|----------------|----------------------|--------|
| MNIST  | 784D (pixels)  | ✅ 70% improvement   | Success |
| Synthetic | 2D (features) | ❌ 0% preservation   | Complete failure |

**Lesson**: Topology requires high-dimensional data with meaningful distance structure.

### 3. Energy Regularization: Consistent but Marginal

| Dataset | Energy Effect | OOD Accuracy Gain |
|---------|--------------|-------------------|
| MNIST   | Prevents overfitting | Minimal |
| AG News | Negative gen gap (-10.82pp) | +0.11pp (noise) |

**Lesson**: Energy is a capacity control mechanism, not an OOD accuracy booster.

---

## 🧪 The APS Framework (Diagnostic Toolkit)

**Atlasing Pattern Space (APS)** is a modular framework combining three regularizers:

### 1. Topology (T) - Neighborhood Preservation
```python
from aps.topology import KNNTopoLoss

# Preserves k-nearest neighbor relationships
topo_loss = KNNTopoLoss(k=8)
loss = topo_loss(z_latent, x_original)
```

**When it helps**: High-dimensional vision tasks (MNIST)  
**When it fails**: Low-dimensional synthetic data (0% preservation)

### 2. Causality (C) - Invariance Learning
```python
from aps.causality import HSICIndependenceLoss

# Enforces independence from spurious features
causal_loss = HSICIndependenceLoss(sigma=1.0)
loss = causal_loss(z_latent, nuisance_vars)
```

**When it helps**: Strong spurious correlations (>90%), learnable representations  
**When it fails**: Weak domain shift (<5%), frozen embeddings

### 3. Energy (E) - TopologyEnergy (Data-Driven Basins)
```python
from aps.energy import TopologyEnergy

# Creates energy wells from neighborhood density
energy = TopologyEnergy(k=15)
loss = energy(z_latent)  # Lower energy = denser regions
```

**Innovation**: **902% better label alignment** vs memory-based energy (ARI: 0.32 vs 0.03)

**Why**: Derives structure from data rather than arbitrary memory patterns

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/freemanhui/atlasing_pattern_space
cd atlasing_pattern_space

# Install with all dependencies
pip install -e ".[topology,causality,dev]"
```

### Run ColoredMNIST Experiments
```bash
# Full APS (T+C+E)
python scripts/run_colored_mnist.py --experiment aps-full --epochs 50

# Baseline (no regularization)
python scripts/run_colored_mnist.py --experiment baseline --epochs 30

# Topology only
python scripts/run_colored_mnist.py --experiment aps-t --epochs 50

# Causality only  
python scripts/run_colored_mnist.py --experiment aps-c --epochs 50
```

### Generate Paper Figures
```bash
python scripts/generate_paper_figures.py
```

---

## 📁 Repository Structure

```
atlasing_pattern_space/
├── src/aps/                    # Core framework
│   ├── models/                 # APSAutoencoder, APSConvAutoencoder
│   ├── topology/               # KNNTopoLoss
│   ├── causality/              # HSICIndependenceLoss, IRMLoss
│   ├── energy/                 # TopologyEnergy, MemoryEnergy (deprecated)
│   ├── metrics/                # Evaluation metrics
│   └── utils/                  # ColoredMNIST, data utilities
├── scripts/                    # Experiment runners
│   ├── run_colored_mnist.py   # Main ColoredMNIST experiments
│   ├── run_tc_conflict_experiment.py  # T-C conflict analysis
│   └── generate_paper_figures.py      # Figure generation
├── outputs/                    # Experiment results
└── tests/                      # Unit tests
```

---

## 🔑 Key Scientific Contributions

### 1. Implicit Causal Bias Discovery
- Autoencoders achieve 82-86% accuracy before explicit regularization
- Reconstruction forces structural learning, not just memorization
- **Implication**: Start with the right architecture, then regularize

### 2. Boundary Condition Characterization
- **Topology**: Fails on low-dimensional data (0% preservation)
- **Causality**: Needs trainable representations and strong spurious correlations (>90%)
- **Energy**: Prevents overfitting consistently, but doesn't improve OOD accuracy

### 3. Honest Negative Results
- We report complete failures transparently (topology on synthetic data)
- Demonstrate that regularizers are not universally beneficial
- Provide practitioners with decision criteria for when to apply each component

---

## 🎓 Decision Framework

### Component Selection Matrix

| Scenario | T (Topology) | C (Causality) | E (Energy) |
|----------|--------------|---------------|------------|
| **High-dim vision** | ✅ Use | ✅ If strong spurious | ✅ Always |
| **Low-dim synthetic** | ❌ Skip | ✅ Use | ✅ Always |
| **NLP (frozen BERT)** | ❌ Skip | ❌ Skip | ✅ Use |
| **NLP (trainable)** | ❓ Test | ✅ Use | ✅ Always |

### When to Use Each Component

**Topology (T)**:
- ✅ High-dimensional data (>100D)
- ✅ Meaningful distance structure
- ❌ Low-dimensional features (<10D)
- ❌ Weak manifold structure

**Causality (C)**:
- ✅ Strong spurious correlations (>90%)
- ✅ Trainable representations
- ❌ Weak domain shift (<5%)
- ❌ Frozen pre-trained embeddings

**Energy (E)**:
- ✅ Always use for overfitting prevention
- ⚠️ Marginal OOD accuracy gains
- ✅ Works across all modalities

---

## 🔬 TopologyEnergy Innovation

**Problem**: Memory-based energy functions catastrophically fail when combined with topology.

**Solution**: TopologyEnergy derives energy from data structure, not arbitrary patterns.

### Comparison: MemoryEnergy vs TopologyEnergy

| Metric | MemoryEnergy | TopologyEnergy | Improvement |
|--------|--------------|----------------|-------------|
| **Label Alignment (ARI)** | 0.03 | 0.32 | **+902%** |
| **Trustworthiness** | 0.58 | 0.88 | **+51.6%** |
| **Reconstruction Error** | 11.7M (collapsed) | 0.31 | **37M× better** |
| **Parameters** | Learnable memory | None (data-driven) | **0 params** |

**Key Insight**: Energy should emerge from data structure, not impose arbitrary patterns.

---

## 📝 Citation

If you use this code or find our results useful, please cite:

```bibtex
@article{hui2024boundary,
  title={When Does Causal Regularization Help? A Systematic Study of Boundary Conditions in Spurious Correlation Learning},
  author={Hui, Freeman},
  year={2024},
  note={arXiv preprint}
}
```

---

## 📚 Further Reading

- **Experiment results**: `outputs/`
- **Code documentation**: See docstrings in `src/aps/`

---

## 📧 Contact

- **Author**: Freeman Hui
- **GitHub**: [github.com/freemanhui/atlasing_pattern_space](https://github.com/freemanhui/atlasing_pattern_space)

For questions, please open an issue on GitHub.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
