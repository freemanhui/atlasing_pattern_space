# Phase 006 Progress Tracker

**Current Phase**: 006A - Colored MNIST OOD Experiment  
**Status**: 🔥 IN PROGRESS  
**Started**: 2025-01-30  

---

## ✅ Completed

### Day 1-2: Dataset Implementation (COMPLETE)

**Date**: 2025-01-30

**Completed Tasks**:
- [x] Created branch `006a-colored-mnist`
- [x] Implemented `src/aps/data/colored_mnist.py` (231 lines)
- [x] Added `ColoredMNIST` class with spurious color correlation
- [x] Implemented `create_colored_mnist_loaders()` helper
- [x] Added `get_correlation_stats()` for verification
- [x] Tested dataset implementation successfully

**Verification Results**:
```
Env1: 80.1% correlation (48061/60000 correct matches)
Env2: 90.1% correlation (54084/60000 correct matches)  
Test (OOD): 0.0% correlation (FLIPPED - this is correct!)
```

**Key Features**:
- Training environments have color spuriously correlated with digit groups (0-4 vs 5-9)
- Test environment has **flipped** correlation to test OOD generalization
- Returns dict with `image`, `label`, `color`, `environment` for both HSIC and IRM losses
- RGB images (3, 28, 28) with red/green colors

**Files Created**:
- `src/aps/data/colored_mnist.py` ✅
- `src/aps/data/__init__.py` ✅

**Commit**: `c4410b0` - "Phase 006A: Implement ColoredMNIST dataset"

---

## 🔄 In Progress

### Day 3-4: Experiment Script Implementation

**Current Status**: NOT STARTED  
**Target Completion**: 2025-01-31  

**Tasks**:
- [ ] Implement `experiments/colored_mnist_ood.py`
  - [ ] `train_baseline()` - simple classifier without causality
  - [ ] `train_aps_causal()` - APS with HSIC + IRM losses
  - [ ] `evaluate_ood()` - OOD test accuracy
  - [ ] `main()` - experiment runner with results

- [ ] Test experiment script with small epochs (--epochs 5)
- [ ] Run full experiment (--epochs 20)
- [ ] Verify success criteria:
  - [ ] APS OOD accuracy > 85%
  - [ ] Baseline OOD accuracy < 60%
  - [ ] Δ Accuracy > 25%

**Next Steps**:
1. Review `specs/006-critical-experiments/BUILD_PLAN.md` Day 3-4 section
2. Implement experiment script
3. Run initial tests
4. Full training run
5. Document results

---

## 📅 Upcoming

### Day 5: Analysis & Documentation

**Target Completion**: 2025-02-01

**Tasks**:
- [ ] Create `notebooks/colored_mnist_analysis.ipynb`
  - [ ] Latent space visualizations (colored by digit vs color)
  - [ ] HSIC(Z, color) over training
  - [ ] Accuracy curves (train/test)
  - [ ] Ablation: HSIC-only vs IRM-only vs both

- [ ] Write `specs/006-critical-experiments/6a-colored-mnist-results.md`
  - [ ] Summarize findings
  - [ ] Key figures
  - [ ] Prepare text for paper Section 5.2

- [ ] Commit Phase 006A as complete

---

## 📊 Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| **APS OOD Accuracy** | > 85% | ⏳ Pending |
| **Baseline OOD Accuracy** | < 60% | ⏳ Pending |
| **Δ Accuracy (Improvement)** | > 25% | ⏳ Pending |
| **HSIC(Z, color)** | < 0.1 after training | ⏳ Pending |
| **Visualizations** | Digit-based clustering | ⏳ Pending |

---

## 🚦 Overall Phase 006 Status

| Sub-Phase | Status | Completion | Branch |
|-----------|--------|------------|--------|
| **6A: Colored MNIST** | 🔥 **IN PROGRESS** | 40% | `006a-colored-mnist` |
| **6B: Text Domain OOD** | 📅 **NEXT** | 0% | - |
| **6C: T-C Trade-off** | 📅 **PLANNED** | 0% | - |

**Estimated Completion**: 
- 6A: 2025-02-01 (3 days remaining)
- 6B: 2025-02-14 (10 days)
- 6C: 2025-02-21 (5 days)

---

## 💡 Notes & Learnings

### Dataset Implementation
- **Correlation verification is critical**: We confirmed env1=80%, env2=90%, test=0% (flipped)
- **Test set has 0% correlation** because it's intentionally flipped (opposite of training)
- **Color assignment works correctly**: Uses torch.rand() for stochastic flipping
- **DataLoader ready**: `create_colored_mnist_loaders()` returns train + test_ood

### Next Challenges
1. **Causality losses**: Need to ensure HSIC and IRM are properly integrated
2. **Baseline comparison**: Must be fair (same architecture, just no causality)
3. **Success criteria**: May need to tune λ_C if results don't meet targets initially

---

**Last Updated**: 2025-01-30  
**Next Update**: After Day 3-4 (experiment script complete)
