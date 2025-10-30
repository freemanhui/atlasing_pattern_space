# Phase 006A Results: Colored MNIST OOD Experiment

**Date**: 2025-01-30  
**Status**: ⚠️ INCOMPLETE - Negative Result  
**Conclusion**: MNIST too robust for spurious correlation demonstration

---

## 🎯 Experiment Goal

Validate that APS causality component (HSIC independence) enables OOD generalization by learning digit classification invariant to spurious color features.

**Success Criteria**:
- APS OOD accuracy > 85%
- Baseline OOD accuracy < 60%
- Improvement (Δ) > 25%
- HSIC(Z, color) < 0.1

---

## 📊 Results Summary

### Quantitative Results (20 epochs, batch=128)

| Metric | Baseline | APS (HSIC) | Target | Status |
|--------|----------|------------|--------|--------|
| **OOD Accuracy** | 95.5% | 95.0% | Baseline < 60%, APS > 85% | ⚠️  **Both too high** |
| **Improvement (Δ)** | - | -0.6% | > 25% | ❌ **Negative!** |
| **HSIC(Z, color)** | N/A | 0.00002 | < 0.1 | ✅ **Perfect** |

### Key Findings

1. ✅ **HSIC working perfectly**: 0.00002 << 0.1
   - APS successfully learns Z independent of color
   - Independence is enforced as intended

2. ❌ **Baseline unexpectedly strong**: 95.5% OOD accuracy
   - Baseline should fail on OOD (< 60%) but doesn't
   - Learns robust shape features despite color correlation

3. ❌ **No improvement from causality**: -0.6% (APS slightly worse)
   - Causality regularization doesn't help
   - May hurt by constraining latent space unnecessarily

4. ✅ **Both models achieve high accuracy**: ~95% on flipped correlation
   - Demonstrates MNIST digits are distinguishable by shape alone
   - Color correlation (80%/90%) not strong enough to be spurious

---

## 🔍 Analysis

### Why Did Baseline Succeed?

The baseline classifier learns **shape-based features** that generalize across color distributions:

**Training**:
- Env1: 80% color correlation (digits 0-4 red, 5-9 green)
- Env2: 90% color correlation

**Test (OOD)**:
- 0% color correlation (FLIPPED: digits 0-4 green, 5-9 red)

**Expected**: Baseline should rely on color → fails on OOD  
**Actual**: Baseline ignores color → succeeds on OOD (95.5%)

**Reason**: MNIST digit shapes are **highly discriminative**. Even with color correlation, gradients flow primarily through shape features because they're more informative.

### Why Didn't APS Help?

APS enforces Z ⊥ color via HSIC, but this doesn't help because:

1. **Baseline already ignores color** (implicitly)
2. **HSIC constraint may hurt** by removing any color information (even if not spurious)
3. **32D latent bottleneck** may lose shape information

**Trade-off**: APS gets Z ⊥ color (good) but sacrifices some shape information (bad) → net -0.6%

---

## 🚨 Critical Issue: Experiment Design Flaw

### The Problem

This experiment **fails to demonstrate spurious correlation** because:

1. **Task is too easy**: Shape features alone → 95% accuracy
2. **Spurious feature is weak**: 80%/90% correlation not strong enough
3. **Natural robustness**: MNIST designed to be translation/scale invariant → also color invariant

### What We Actually Demonstrated

✅ **HSIC successfully enforces independence** (technical success)  
❌ **OOD failure scenario doesn't exist for MNIST** (experimental failure)

This is a **valid negative result** but doesn't validate our causality claims for the paper.

---

## 💡 Proposed Solutions

### Option 1: Make Spurious Correlation Stronger

**Changes**:
- Increase correlation to 99%/100% (near-perfect in training)
- Use more saturated colors (harder to ignore)
- Add color to digit stroke (not just background)

**Expected**: Baseline overfits to color → fails OOD

**Risk**: May need many epochs for overfitting to kick in

### Option 2: Use Different Dataset

**Option 2A: Colored MNIST with Noise**
- Add shape noise to make color more informative
- Force baseline to use color as shortcut

**Option 2B: Synthetic Binary Classification**
- Simple shapes (circles vs squares) with spurious color
- Deliberately design strong spurious correlation

**Option 2C: Move to Text Domain OOD (Phase 006B)**
- Text classification with domain shift
- Natural spurious correlation (topic correlates with sentiment)

### Option 3: Accept & Document Negative Result

**Honest reporting**:
- "Colored MNIST too robust for spurious correlation"
- "HSIC successfully enforces independence but baseline already robust"
- "Proceed to text domain for proper OOD validation"

**Advantage**: Shows scientific integrity (reporting negative results)

---

## 🎯 Recommendation

### SHORT TERM: Try Option 1 (Stronger Correlation)

```python
# Modify dataset with 99% correlation
env1 = ColoredMNIST('env1', correlation=0.99)
env2 = ColoredMNIST('env2', correlation=0.995)

# Train longer for overfitting
--epochs 50
```

**If still fails**: Move to Option 2C (Text Domain OOD)

### LONG TERM: Phase 006B (Text Domain OOD)

Text classification with domain shift is:
- **Naturally spurious**: Topic ≠ Sentiment but correlated
- **High-dimensional**: BERT embeddings (768-dim)
- **Better validation** for LLM-focused paper

**Priority**: Text domain more important for paper credibility

---

## 📈 Technical Validation

### What We Successfully Validated

1. ✅ **ColoredMNIST dataset works**:
   - Correlations verified (80%/90%/flipped)
   - DataLoader integration correct
   - 120K train, 10K test samples

2. ✅ **HSIC loss works**:
   - Enforces Z ⊥ color (HSIC = 2e-05)
   - Differentiable, stable training
   - No numerical issues

3. ✅ **Experiment infrastructure solid**:
   - Baseline vs APS comparison
   - Automatic metrics
   - Results saved to JSON

4. ✅ **Code ready for Phase 006B**:
   - Can reuse experiment structure
   - Just swap dataset + adjust architecture

---

## 📝 For Paper Section 5.2

### If We Fix This Experiment

**Include**:
- Colored MNIST as OOD validation
- HSIC successfully enforces independence
- Baseline < 60%, APS > 85%, Δ > 25%

### If We Don't Fix (Current State)

**Option A - Report Negative Result**:
- "Colored MNIST proved too robust for spurious correlation demonstration"
- "Baseline achieved 95.5% OOD accuracy, indicating shape features sufficient"
- "HSIC successfully enforced independence (2e-05) but baseline already robust"
- "Motivated stronger validation with text domain shift (Section 5.3)"

**Option B - Skip This Experiment**:
- Proceed directly to Text Domain OOD (Phase 006B)
- Cite Colored MNIST difficulty in limitations

---

## 🔄 Next Steps

### Immediate (Option 1 - Retry)

1. **Modify correlation**:
   ```python
   # src/aps/data/colored_mnist.py line 232
   correlation_env1=0.99,
   correlation_env2=0.995,
   ```

2. **Train longer**:
   ```bash
   python experiments/colored_mnist_ood.py --epochs 50 --lambda-C 2.0
   ```

3. **Check if baseline fails OOD**:
   - Target: Baseline < 70% OOD
   - If yes: Success! Update paper
   - If no: Move to Phase 006B

### Alternative (Option 2C - Pivot)

1. **Document findings** in this file ✅ Done
2. **Commit progress**: "Phase 006A: Negative result, MNIST too robust"
3. **Start Phase 006B**: Text Domain OOD (critical for paper)

---

## 📊 Data Files

- **Results**: `outputs/colored_mnist_ood/results.json`
- **Models**: `outputs/colored_mnist_ood/baseline_model.pt`, `aps_model.pt`
- **Dataset**: `data/MNIST/` (auto-downloaded)

---

## 💬 Key Takeaways

1. **MNIST is NOT a good OOD testbed for spurious correlation** - too robust
2. **HSIC implementation works perfectly** - enforces independence as intended
3. **Need stronger spurious features** OR **move to different domain**
4. **Text domain (Phase 006B) more critical** for paper validation
5. **Negative results are valuable** - shows we're doing rigorous science

**Decision Point**: Retry with 99% correlation OR move to Phase 006B?

---

**Last Updated**: 2025-01-30  
**Status**: Awaiting decision on next steps
