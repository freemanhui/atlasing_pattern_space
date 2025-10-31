# Phase 006: Critical Experiments - Quick Reference

**Addressing Critical Review Feedback**

---

## 🎯 What Are We Doing?

The critical review identified **three major gaps** in our experimental validation. Phase 006 addresses all of them with rigorous new experiments.

---

## 📋 The Three Experiments

### 6A: Colored MNIST (OOD Causality) 🔥 WEEK 1

**Why**: Validate that causality component enables OOD generalization  
**What**: MNIST digits with spurious color correlation that flips in test set  
**Success**: APS acc > 85%, Baseline < 60%, Δ > 25%  

**Files**:
- `src/aps/data/colored_mnist.py` (dataset)
- `experiments/colored_mnist_ood.py` (experiment script)
- `notebooks/colored_mnist_analysis.ipynb` (analysis)

**Start Here**:
```bash
git checkout -b 006a-colored-mnist
code specs/006-critical-experiments/BUILD_PLAN.md  # Read Day 1-5 section
code src/aps/data/colored_mnist.py                  # Implement dataset
pytest tests/test_data/test_colored_mnist.py       # Test it
python experiments/colored_mnist_ood.py --epochs 20 # Run experiment
```

---

### 6B: Text Domain OOD (High-Dimensional) 🔥 WEEKS 2-3

**Why**: Bridge validation gap to NLP/LLM target domain  
**What**: Amazon sentiment across domains (Books/Electronics/Movies → Clothing)  
**Success**: APS acc > 70%, Baseline < 60%, Δ > 10%  

**Files**:
- `src/aps/data/wilds_amazon.py` (dataset)
- `experiments/text_domain_ood.py` (experiment script)
- `notebooks/text_ood_analysis.ipynb` (analysis)

**Start Here**:
```bash
git checkout -b 006b-text-ood
code specs/006-critical-experiments/BUILD_PLAN.md  # Read Sub-Phase 6B
pip install transformers datasets wilds            # Install deps
code src/aps/data/wilds_amazon.py                  # Implement dataset
python experiments/text_domain_ood.py --epochs 20  # Run experiment
```

---

### 6C: T-C Trade-off (Pareto Frontier) 🔥 WEEK 4

**Why**: Characterize tension between topology and causality objectives  
**What**: Hyperparameter sweep (36 configs) on synthetic shapes with spurious features  
**Success**: Clear Pareto frontier + hyperparameter guidance  

**Files**:
- `src/aps/data/spurious_shapes.py` (dataset)
- `experiments/tc_tradeoff_sweep.py` (sweep script)
- `notebooks/pareto_frontier_analysis.ipynb` (visualization)

**Start Here**:
```bash
git checkout -b 006c-tc-tradeoff
code specs/006-critical-experiments/BUILD_PLAN.md  # Read Sub-Phase 6C
code src/aps/data/spurious_shapes.py               # Implement dataset
python experiments/tc_tradeoff_sweep.py --epochs 50 # Run sweep (takes ~2hrs)
jupyter notebook notebooks/pareto_frontier_analysis.ipynb # Visualize
```

---

## 📊 Success Metrics

| Experiment | Metric | Target | Impact |
|------------|--------|--------|--------|
| **6A: Colored MNIST** | Δ OOD Accuracy | > 25% | Validates causality claims |
| **6B: Text Domain OOD** | Δ OOD Accuracy | > 10% | Extends to high-dim NLP |
| **6C: T-C Trade-off** | Pareto Frontier | Mapped | Provides tuning guidance |

---

## 🚀 Quick Start (Week 1)

### Day 1: Setup
```bash
cd /Users/freeman.hui/Documents/Coding/atlasing_pattern_space
git checkout -b 006-critical-experiments
pip install transformers datasets wilds scikit-learn pillow
```

### Day 2-3: Implement Colored MNIST Dataset
```bash
# Create dataset file
code src/aps/data/colored_mnist.py

# Write tests (TDD)
code tests/test_data/test_colored_mnist.py
pytest tests/test_data/test_colored_mnist.py -v

# Verify it works
python -c "from aps.data.colored_mnist import ColoredMNIST; d = ColoredMNIST('env1'); print(d[0])"
```

### Day 4: Implement Experiment Script
```bash
code experiments/colored_mnist_ood.py

# Test run (small epochs)
python experiments/colored_mnist_ood.py --epochs 5

# Full run
python experiments/colored_mnist_ood.py --epochs 20
```

### Day 5: Analysis & Documentation
```bash
# Create analysis notebook
jupyter notebook notebooks/colored_mnist_analysis.ipynb

# Document results
code specs/006-critical-experiments/6a-colored-mnist-results.md

# Commit progress
git add .
git commit -m "Phase 006A complete: Colored MNIST OOD experiment"
```

---

## 📚 Key Documents

1. **Detailed Build Plan**: `specs/006-critical-experiments/BUILD_PLAN.md`
   - Complete implementation specifications
   - Day-by-day breakdown
   - Code examples for all experiments

2. **Updated Roadmap**: `ROADMAP_V2.md`
   - Reframed contributions post-critique
   - Phase 006-008 timeline
   - Success criteria

3. **Critique Response**: `CRITIQUE_RESPONSE.md`
   - Executive summary of changes
   - What we agreed/disagreed with
   - Lessons learned

---

## 🎯 Critical Path

**For Paper Submission**, we MUST complete:

1. ✅ TopologyEnergy implementation (done)
2. 🔥 **6A: Colored MNIST** (Week 1)
3. 🔥 **6B: Text Domain OOD** (Weeks 2-3)
4. 🔥 **Paper Rewrite** (Phase 007, Weeks 5-6)

**Should Have** (strengthens paper):
5. **6C: T-C Trade-off** (Week 4)
6. Complete documentation (Phase 008)

---

## 💡 Tips

### For Colored MNIST (6A)
- Start with correlation=0.8 for env1, 0.9 for env2
- Test set should have **flipped** correlation
- Use HSIC loss for color independence
- Use IRM loss for environment invariance

### For Text Domain OOD (6B)
- Precompute BERT embeddings to save time
- If WILDS unavailable, use AG News with synthetic domains
- Focus on sentiment (binary classification)
- HSIC penalizes dependence on domain_id

### For T-C Trade-off (6C)
- Use simple shapes (circles vs squares) with color backgrounds
- Sweep λ_T and λ_C in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
- Plot results as 2D scatter (topo score vs causal accuracy)
- Identify Pareto frontier

---

## 📞 Need Help?

1. **Stuck on implementation?** → Read `BUILD_PLAN.md` (detailed code examples)
2. **Confused about goals?** → Read `CRITIQUE_RESPONSE.md` (why we're doing this)
3. **Need big picture?** → Read `ROADMAP_V2.md` (overall strategy)

---

## 🚦 Status Tracking

| Sub-Phase | Status | Branch | Completion |
|-----------|--------|--------|------------|
| **6A: Colored MNIST** | 🔥 **IN PROGRESS** | `006a-colored-mnist` | 0% |
| **6B: Text Domain OOD** | 📅 **NEXT** | `006b-text-ood` | 0% |
| **6C: T-C Trade-off** | 📅 **PLANNED** | `006c-tc-tradeoff` | 0% |

---

**Last Updated**: 2025-01-30  
**Current Focus**: Sub-Phase 6A (Colored MNIST OOD)  
**Timeline**: 4 weeks for all three experiments  
**Priority**: CRITICAL for paper submission

---

## 🎓 Remember

This phase is about **rigor**. The critique challenged our causality claims and validation scope. Phase 006 addresses every concern with:

1. **True OOD testing** (not just HSIC from labels)
2. **High-dimensional validation** (text, not just MNIST)
3. **Honest trade-off analysis** (Pareto frontier)

By completing these experiments, we transform a promising paper into a **compelling, well-validated contribution**. 🚀
