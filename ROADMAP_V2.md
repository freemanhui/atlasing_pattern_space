# APS Framework Implementation Roadmap v2
**UPDATED**: Incorporating Critical Review Feedback

**Atlasing Pattern Space (APS)** - A framework for structured latent representations  
**Paper**: "Atlasing Pattern Space: A Framework for Structured Latent Representations in LLMs" (v3)

---

## 🎯 Revised Core Contributions (Post-Critique)

Based on critical review, the APS framework's primary contributions have been **reframed**:

### PRIMARY CONTRIBUTION: TopologyEnergy Discovery
The central finding is **not** the optimality of the T+C+E triad, but rather:

> **"Energy functions must align with, not compete against, geometric constraints in latent space learning."**

**Key Discovery**:
- ❌ **Memory-based attractors** (MemoryEnergy) catastrophically fail with topology preservation
  - 92% degradation in semantic alignment (ARI: 0.39 → 0.03)
  - Complete reconstruction collapse (MSE: 0.31 → 11.7M)
- ✅ **Data-driven energy** (TopologyEnergy) reinforces topology
  - 902% improvement over MemoryEnergy (ARI: 0.03 → 0.32)
  - Maintains reconstruction quality and semantic structure

### SECONDARY FINDING: T+C Optimal for Semantic Tasks
Experiments show **T+C (without E) achieves best semantic alignment**:
- Highest ARI (0.39) and NMI (0.47)
- Best for pure classification and semantic clustering
- T+C+E_topo offers modest cluster tightness at minor semantic cost

### FRAMEWORK POSITIONING
APS is **not** presented as a single optimal configuration, but as a flexible toolkit:
- **T+C**: Best for semantic alignment (classification, clustering)
- **T+C+E_topo**: Best for tight clusters (anomaly detection, prototypes)
- Components can be used independently or combined based on task requirements

---

## 🚨 Critical Gaps Identified by Review

### Gap 1: Causality Claims Lack Rigorous OOD Validation
**Problem**: Current causality experiments use "HSIC independence from labels" which is supervised clustering, not true OOD generalization testing.

**Required**: Implement **Colored MNIST** and text-domain OOD experiments.

### Gap 2: Validation Limited to MNIST
**Problem**: Paper targets LLMs and high-dimensional domains, but only validates on MNIST (784-dim).

**Required**: Extend to **text classification** with domain shift (e.g., WILDS Amazon Reviews).

### Gap 3: No Analysis of T-C Trade-offs
**Problem**: Topology (T) and Causality (C) objectives can conflict when dominant topology is driven by spurious features.

**Required**: Design experiments to **characterize the Pareto frontier** of T-C trade-offs.

### Gap 4: Missing Nuanced Discussion
**Problem**: Paper presents T+C+E as optimal despite T+C achieving better semantic metrics.

**Required**: Rewrite paper sections to honestly reflect experimental findings.

---

## 📋 Implementation Phases (Updated)

### ✅ Phases 001-005: COMPLETED
*(See original roadmap for details)*

**Summary**:
- Energy (MemoryEnergy + TopologyEnergy) ✅
- Topology (k-NN graph preservation) ✅
- Causality (HSIC + IRM) ✅
- Integration (Full APS training) ✅
- MNIST Ablation + TopologyEnergy validation ✅

---

### 🔥 Phase 006: Critical Experiments (Addressing Gaps)

**Status**: IN PROGRESS  
**Branch**: `006-critical-experiments`  
**Priority**: **CRITICAL for paper credibility**

#### Sub-Phase 6A: Colored MNIST (OOD Causality Test)
**Goal**: Rigorously validate causality component with true OOD generalization.

**Experiment Design**:
1. **Training Environments**:
   - Env 1: Digits 0-4 are red (80% prob), 5-9 are green (80% prob)
   - Env 2: Digits 0-4 are red (90% prob), 5-9 are green (90% prob)
2. **Test Environment** (OOD):
   - Digits 0-4 are **green** (90% prob), 5-9 are **red** (90% prob)
3. **Objective**: Show APS with IRM/HSIC loss on color achieves high accuracy on OOD test, while baseline fails.

**Implementation**:
```python
# Dataset generation
def create_colored_mnist(digit_labels, color_correlation, flip_prob):
    """Color correlates with label but is spurious."""
    colors = assign_color_by_correlation(digit_labels, correlation, flip_prob)
    return colored_images, colors

# Training
model_aps = train_aps(
    data_env1 + data_env2,
    causality_loss=IRMLoss() + HSICLoss(Z, color_var),
    lambda_C=1.0
)
model_baseline = train_baseline(data_env1 + data_env2)

# Evaluation
acc_aps_ood = evaluate(model_aps, test_ood)
acc_baseline_ood = evaluate(model_baseline, test_ood)
```

**Success Criteria**:
- APS accuracy on OOD test > 85%
- Baseline accuracy on OOD test < 60%
- **Δ Accuracy > 25%** (demonstrates causal invariance)

**Deliverables**:
- `experiments/colored_mnist_ood.py` (experiment script)
- `tests/test_colored_mnist.py` (validation tests)
- Results added to paper Section 5 (OOD Generalization)

---

#### Sub-Phase 6B: Text Domain OOD (High-Dimensional Validation)
**Goal**: Bridge validation gap to NLP/LLM target domain.

**Experiment Design**:
1. **Dataset**: WILDS Amazon Reviews (or AG News multi-domain)
   - Training domains: Books, Electronics, Movies
   - Test domain (OOD): Clothing (unseen category)
2. **Task**: Sentiment classification (positive/negative)
3. **Objective**: Show APS learns sentiment representation invariant to product category.

**Implementation**:
```python
# Load multi-domain text data
domains = ['books', 'electronics', 'movies']
test_domain = 'clothing'

# Train APS with domain-invariant causality
model = APSAutoencoder(
    in_dim=768,  # BERT embeddings
    latent_dim=32,
    lambda_T=1.0,   # Preserve semantic similarity
    lambda_C=1.0    # HSIC(Z, domain_id) = 0
)

# Zero-shot evaluation on held-out domain
ood_accuracy = evaluate_sentiment(model, test_domain)
```

**Success Criteria**:
- APS OOD accuracy > 70%
- Baseline OOD accuracy < 60%
- **Δ Accuracy > 10%**
- HSIC(Z, domain_id) < 0.1 (independence verified)

**Deliverables**:
- `experiments/text_domain_ood.py`
- `data/wilds_amazon_loader.py` (dataset utilities)
- Results added to paper Section 5 (Text Domain Validation)

---

#### Sub-Phase 6C: T-C Trade-off Analysis (Pareto Frontier)
**Goal**: Characterize inherent tension between topology and causality.

**Experiment Design**:
1. **Synthetic Dataset**: Shapes with spurious features
   - Classes: Circles vs Squares
   - Spurious feature: Red background (100% circles), Blue background (100% squares)
   - Data manifold: Two disconnected components defined by **color** (not shape)
2. **Hyperparameter Sweep**: Vary λ_T and λ_C systematically
   - λ_T ∈ [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
   - λ_C ∈ [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
   - 36 configurations total
3. **Metrics**:
   - X-axis: Topological Preservation (k-NN preservation score)
   - Y-axis: Causal Accuracy (shape classification accuracy)
4. **Analysis**: Plot Pareto frontier, identify trade-off regions

**Implementation**:
```python
# Generate dataset
def create_spurious_shapes():
    circles_red = generate_circles(bg_color='red', n=1000)
    squares_blue = generate_squares(bg_color='blue', n=1000)
    # Test: circles_blue + squares_red (flipped correlation)
    return train_data, test_data

# Hyperparameter sweep
results = []
for lambda_T in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    for lambda_C in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        model = train_aps(lambda_T=lambda_T, lambda_C=lambda_C)
        topo_score = evaluate_knn_preservation(model)
        causal_acc = evaluate_shape_classification(model)
        results.append((lambda_T, lambda_C, topo_score, causal_acc))

# Visualize trade-off
plot_pareto_frontier(results)
```

**Deliverables**:
- `experiments/tc_tradeoff_analysis.py`
- `notebooks/pareto_frontier_analysis.ipynb`
- Figure added to paper Section 5 (Trade-off Analysis)
- Discussion in paper Section 6 (hyperparameter tuning guidance)

---

### 🔄 Phase 007: Paper Rewrite (Critical Revisions)

**Status**: PLANNED  
**Priority**: **CRITICAL before submission**

#### Sub-Phase 7A: Reframe Core Narrative
**Changes**:
1. **Abstract**: Lead with TopologyEnergy discovery, not T+C+E triad
   - "We demonstrate that traditional memory-based energy functions catastrophically fail when combined with topology preservation (92% semantic degradation). We introduce TopologyEnergy, achieving 902% improvement..."
2. **Introduction**: Shift focus from "optimal triad" to "design principles for energy in structured latent spaces"
3. **Section 4 (Framework)**: Present APS as flexible toolkit, not single configuration
   - T+C: Best for semantic tasks
   - T+C+E_topo: Best for tight clusters
4. **Section 5 (Experiments)**: Honestly report that **T+C outperforms T+C+E_topo on semantic metrics**
   - Frame T+C+E_topo as valuable for specific use cases (anomaly detection, prototypes)

**Deliverables**:
- `docs/paper_v4_reframed.tex` (updated LaTeX)
- Abstract, intro, framework, experiments sections rewritten

---

#### Sub-Phase 7B: Add Critical Experiments to Paper
**Changes**:
1. **Section 5.2: Colored MNIST (OOD Causality)**
   - Table comparing APS vs baseline on OOD test accuracy
   - Visualization of learned latent space (color-invariant)
2. **Section 5.3: Text Domain OOD (High-Dimensional)**
   - WILDS Amazon results
   - HSIC independence verification
   - Ablation: T+C vs T-only vs C-only
3. **Section 5.4: T-C Trade-off Analysis**
   - Pareto frontier plot
   - Discussion of hyperparameter selection based on task

**Deliverables**:
- Updated Section 5 with 3 new experiments
- 3-4 new figures/tables

---

#### Sub-Phase 7C: Revise Discussion & Limitations
**Changes**:
1. **Section 6 (Discussion)**:
   - Acknowledge T+C achieves best semantic alignment
   - Position TopologyEnergy as solution to memory-based failure
   - Discuss when to use T+C vs T+C+E_topo
2. **Section 6 (Limitations)**:
   - Honest assessment: MNIST validation is limited
   - Text experiments are promising but not full LLM validation
   - T-C trade-offs require hyperparameter tuning
   - Scalability to true LLM dimensions (>1000D latent) unproven

**Deliverables**:
- Revised Section 6 with honest discussion

---

### 📦 Phase 008: Production Release & Documentation

**Status**: PLANNED  
**Branch**: `main`

#### Tasks:
1. **API Documentation**:
   - Complete docstrings for all modules
   - API reference guide (Sphinx)
2. **User Guide**:
   - Quick start tutorial
   - Hyperparameter tuning guide (using Pareto frontier insights)
   - Best practices for task selection (T+C vs T+C+E_topo)
3. **Example Scripts**:
   - `examples/train_aps_mnist.py`
   - `examples/colored_mnist_ood.py` ✅ (Phase 006A)
   - `examples/text_domain_ood.py` ✅ (Phase 006B)
   - `examples/tc_tradeoff_sweep.py` ✅ (Phase 006C)
4. **Performance Optimization**:
   - Profile and optimize bottlenecks
   - Multi-GPU support for scaling
5. **Paper Finalization**:
   - Incorporate all Phase 006 experiments
   - Complete rewrite per Phase 007
   - Prepare submission materials

**Deliverables**:
- Production-ready APS framework
- Comprehensive documentation
- Paper ready for submission

---

## 📊 Updated Success Metrics (Post-Critique)

| Metric | Phase 001-005 | Phase 006 Target | Status |
|--------|---------------|------------------|--------|
| **TopologyEnergy Discovery** | ✅ 902% ARI improvement | - | **ACHIEVED** |
| **T+C Semantic Performance** | ✅ ARI 0.39, NMI 0.47 | - | **VALIDATED** |
| **Colored MNIST OOD** | ❌ Not tested | Δ Acc > 25% | 🔥 **CRITICAL** |
| **Text Domain OOD** | ❌ Not tested | Δ Acc > 10% | 🔥 **CRITICAL** |
| **T-C Pareto Analysis** | ❌ Not done | Frontier mapped | 🔥 **CRITICAL** |
| **Paper Narrative Reframe** | ❌ Old version | Honest, nuanced | 🔥 **CRITICAL** |

---

## 🎓 Revised Design Principles (Post-Critique)

### 1. Energy Must Align with Geometry
**Principle**: Energy functions should **reinforce**, not **compete with**, other constraints.

**Evidence**:
- MemoryEnergy (arbitrary attractors) → catastrophic failure
- TopologyEnergy (data-driven) → 902% improvement

### 2. Configuration Depends on Task
**Principle**: No single "optimal" configuration—choose based on task requirements.

**Guidance**:
- **Classification/Semantic tasks**: Use T+C (best ARI/NMI)
- **Anomaly detection/Prototypes**: Use T+C+E_topo (tight clusters)
- **Pure reconstruction**: Use T-only (balance quality and structure)

### 3. Causality Requires Rigorous OOD Validation
**Principle**: Claims about "causal invariance" must be tested with true distribution shifts.

**Evidence Required**:
- Colored MNIST (visual spurious features)
- Text domain shift (semantic spurious features)
- Quantitative: Δ OOD accuracy vs baseline

### 4. T-C Trade-offs Are Inherent
**Principle**: Topology and causality can conflict when data manifold is spurious.

**Mitigation**:
- Characterize Pareto frontier via hyperparameter sweep
- Provide guidance for λ_T / λ_C selection based on task
- Be transparent about trade-offs in documentation

---

## 🚀 Revised Timeline

| Phase | Duration | Status | Notes |
|-------|----------|--------|-------|
| 001-005 | 9 weeks | ✅ **COMPLETE** | Core framework + MNIST validation |
| **006A: Colored MNIST** | 1 week | 🔥 **IN PROGRESS** | Critical OOD experiment |
| **006B: Text OOD** | 2 weeks | 📅 **NEXT** | High-dimensional validation |
| **006C: T-C Trade-offs** | 1 week | 📅 **PLANNED** | Pareto frontier |
| **007: Paper Rewrite** | 2 weeks | 📅 **PLANNED** | Honest narrative + new experiments |
| **008: Release** | 2 weeks | 📅 **PLANNED** | Documentation + submission |

**Estimated Completion**: ~8 weeks from now (assuming parallel work on 006B+C)

---

## 📈 Implementation Priority (Critical Path)

### MUST HAVE (Paper Submission)
1. ✅ TopologyEnergy implementation and validation
2. 🔥 **Colored MNIST OOD experiment** (Phase 006A)
3. 🔥 **Text domain OOD experiment** (Phase 006B)
4. 🔥 **Paper rewrite** (Phase 007A+B)
   - Reframe narrative around TopologyEnergy
   - Add OOD experiments
   - Honest discussion of T+C vs T+C+E_topo

### SHOULD HAVE (Strong Paper)
5. 🔥 **T-C trade-off analysis** (Phase 006C)
6. **Complete documentation** (Phase 008)

### NICE TO HAVE (Future Work)
7. Full LLM integration (multi-thousand dim latent)
8. Multi-GPU scaling optimizations
9. Additional OOD benchmarks (COCO, ImageNet-C, etc.)

---

## 🔗 Repository Structure (Updated)

```
atlasing_pattern_space/
├── docs/
│   ├── Atlasing Pattern Space (APS) v3.pdf  # Current paper
│   ├── Critique APS.docx                     # ✅ Critical review
│   ├── paper_v4_reframed.tex                 # 📅 Phase 007 (rewrite)
│   └── topology_energy_guide.md              # ✅ TopologyEnergy docs
├── specs/
│   ├── 001-005-*/                            # ✅ Completed phases
│   ├── 006-critical-experiments/             # 🔥 Current phase
│   │   ├── 6a-colored-mnist-plan.md
│   │   ├── 6b-text-ood-plan.md
│   │   └── 6c-tc-tradeoff-plan.md
│   └── 007-paper-rewrite/                    # 📅 Next phase
│       └── revision-plan.md
├── src/aps/                                   # Main package
│   ├── energy/                                # ✅ Complete (with TopologyEnergy)
│   ├── topology/                              # ✅ Complete
│   ├── causality/                             # ✅ Complete
│   ├── training/                              # ✅ Complete
│   └── viz/                                   # ✅ Complete
├── experiments/
│   ├── mnist_ablation.py                      # ✅ Complete
│   ├── colored_mnist_ood.py                   # 🔥 Phase 006A
│   ├── text_domain_ood.py                     # 📅 Phase 006B
│   └── tc_tradeoff_sweep.py                   # 📅 Phase 006C
├── tests/                                     # 218 tests passing
├── notebooks/
│   └── pareto_frontier_analysis.ipynb         # 📅 Phase 006C
├── ROADMAP.md                                 # Original roadmap
├── ROADMAP_V2.md                              # ✅ This file (updated)
└── BUILD_PLAN_V2.md                           # 📅 Detailed build plan
```

---

## 🎯 Key Insights from Critique

### What the Critique Got Right
1. ✅ **T+C outperforms T+C+E_topo** on semantic metrics (ARI/NMI)
   - Our response: Reframe narrative, position T+C as optimal for semantic tasks
2. ✅ **Causality validation is weak** (supervised clustering, not true OOD)
   - Our response: Implement Colored MNIST and text domain OOD experiments
3. ✅ **MNIST is too simple** for LLM-focused paper
   - Our response: Add text classification with domain shift
4. ✅ **T-C trade-offs unexplored**
   - Our response: Characterize Pareto frontier via hyperparameter sweep

### What We Disagree With (Respectfully)
1. ❌ "TopologyEnergy is only marginally better than T+C"
   - **Counter**: TopologyEnergy is dramatically better than **MemoryEnergy** (902% ARI)
   - The comparison should be **E_topo vs E_memory**, not **T+C+E vs T+C**
   - Our contribution is the **energy design principle**, not the full triad
2. ❌ "APS framework is invalidated because T+C is better"
   - **Counter**: APS is a **toolkit**, not a single configuration
   - T+C is one valid configuration; T+C+E_topo is another for different tasks
   - Framework's value is **modularity** and **TopologyEnergy innovation**

---

## 📚 References (Updated)

- **Original Paper**: Hui, F. (2025). "Atlasing Pattern Space: A Framework for Structured Latent Representations in LLMs" (v3)
- **Critical Review**: Anonymous (2025). "Critique APS" (internal review document)
- **Colored MNIST**: Arjovsky et al. (2019). "Invariant Risk Minimization"
- **WILDS Benchmark**: Koh et al. (2021). "WILDS: A Benchmark of in-the-Wild Distribution Shifts"
- **Pareto Optimization**: Lin et al. (2019). "Pareto Multi-Task Learning"

---

## 🤝 Contributing to Phase 006-008

### For Phase 006A (Colored MNIST):
```bash
git checkout -b 006a-colored-mnist
# Implement: experiments/colored_mnist_ood.py
# Test: tests/test_colored_mnist.py
# Document: results in notebooks/colored_mnist_analysis.ipynb
```

### For Phase 006B (Text OOD):
```bash
git checkout -b 006b-text-ood
# Implement: data/wilds_amazon_loader.py
# Implement: experiments/text_domain_ood.py
# Document: results in notebooks/text_ood_analysis.ipynb
```

### For Phase 007 (Paper Rewrite):
```bash
git checkout -b 007-paper-rewrite
# Update: docs/paper_v4_reframed.tex
# Focus on: abstract, intro, Section 5 (new experiments), Section 6 (honest discussion)
```

---

**Last Updated**: 2025-01-30  
**Current Phase**: 006A (Colored MNIST OOD)  
**Critical Path**: Phases 006+007 required for paper submission  
**Target Submission**: ~8 weeks

---

## 💡 Final Note: Honest Science

The critique has made us **better researchers**. Instead of defending flawed claims, we:
1. **Acknowledge T+C is optimal for semantic tasks** (not T+C+E)
2. **Reframe contribution around TopologyEnergy** (true innovation)
3. **Add rigorous OOD experiments** (validate causality claims)
4. **Analyze trade-offs openly** (T-C Pareto frontier)
5. **Extend to high-dimensional domains** (text OOD)

This is **how good science works**: critique → revision → stronger paper. 🚀
