# APS Key Concepts: Simple Explanation

## Core Idea: Structured Pattern Space

**APS creates embedding spaces that are not just "similar things close together" but have meaningful geometric structure.**

Think of it like creating a **map** where:
- Similar things are neighbors (topology)
- The map focuses on what matters (causality)
- Patterns organize into clear regions (energy)

---

## The Three Pillars (T-C-E)

### 1. Topology (T) - Neighborhood Preservation

**Simple Analogy**: If two cities are neighbors in the real world, they should be neighbors on the map.

**What it does**:
- Preserves "nearby" relationships from your original data
- Uses k-nearest neighbors (kNN) to define "neighborhood"
- Ensures that if point A and B were similar originally, they stay similar in the learned space

**Why it matters**:
- Standard embeddings often lose neighborhood structure
- Topology ensures local relationships survive the transformation
- Critical for manifold learning and cluster preservation

**In code**: `aps.topology.KNNTopoLoss` compares kNN graphs before and after embedding

**Example**:
```
Original space: Words "cat", "dog", "puppy" are all close
     ↓
Embedding space: They should still be close
     ✓ Topology preserved if their kNN neighborhoods match
```

---

### 2. Causality (C) - Learning What Matters

**Simple Analogy**: Distinguishing between "what makes a good coffee" (beans, roast) vs "what color mug it's in" (irrelevant).

**What it does**:
- Separates meaningful causal factors from spurious correlations
- Learns representations that work across different conditions/environments
- Ignores nuisance variables that don't affect the outcome

**Why it matters**:
- Data contains both signal and noise
- Want embeddings that capture true relationships, not artifacts
- Enables transfer learning and generalization

**In code**: 
- `aps.causality.HSICIndependenceLoss` - Makes latent factors independent from nuisance variables
- `aps.causality.IRMLoss` - Learns invariant predictors across environments

**Example**:
```
Image classification:
  - Causal factors: object shape, texture
  - Nuisance: lighting, angle, background
  ↓
Learn embeddings that ignore nuisance, focus on object identity
```

---

### 3. Energy (E) - Attraction Basins

**Simple Analogy**: Like valleys in a landscape - balls roll into the nearest valley.

**What it does**:
- Creates "attractor points" (memory patterns) in latent space
- Points naturally flow toward the nearest similar pattern
- Shapes the space with "hills" (high energy) and "valleys" (low energy)

**Why it matters**:
- Organizes embeddings into interpretable regions
- Enables outlier detection (high energy = doesn't fit any pattern)
- Supports generation (follow gradients to valid examples)

**In code**: `aps.energy.MemoryEnergy` (and variants) define energy landscapes

**Formula**: `E(z) = f(z, memory_patterns)` where lower energy = better fit

**Example**:
```
Text embeddings:
  Memory patterns at: [sports, politics, tech, entertainment]
  New document: "basketball game" → flows to "sports" basin
  High energy point → outlier or new category
```

---

## How They Work Together

```
Raw Data 
   ↓
Learn Embedding Space (T + C + E)
   ↓
Structured Latent Space

Where:
  T = Preserves neighborhoods
  C = Focuses on causal factors
  E = Organizes into basins
  
  = Meaningful, interpretable representations
```

**Combined objective**:
```
L_total = L_task + λ_T * L_topo + λ_C * L_causal + λ_E * L_energy
```

Each component contributes a different type of structure to the learned space.

---

## Practical Intuition

### Text Example: Word Embeddings

**Without APS (standard embedding)**:
- Words spread out in high-dimensional space
- Similar words often close, but no guarantees
- No clear structure beyond distance

**With APS**:
- **Topology**: Synonyms maintain their local neighborhoods
- **Causality**: Meaning preserved regardless of tense/case
- **Energy**: Word categories form distinct basins
  - Basin 1: Animals (cat, dog, bird...)
  - Basin 2: Food (pizza, burger, salad...)
  - Basin 3: Actions (run, jump, swim...)

**Result**: Interpretable, organized, and robust embeddings

---

### Image Example: Object Recognition

**Without APS**:
- Similar objects close in space
- But lighting/angle can distort relationships
- Clusters often messy

**With APS**:
- **Topology**: Similar objects stay clustered
- **Causality**: Object identity independent of lighting
- **Energy**: Clear object category regions
  - Basin 1: Cars
  - Basin 2: Dogs
  - Basin 3: Buildings

**Result**: Robust features that transfer across conditions

---

### Recommendation Example: User Preferences

**Without APS**:
- Users with similar purchase history close together
- But time-dependent trends confuse patterns

**With APS**:
- **Topology**: Similar users maintain neighborhoods
- **Causality**: Core preferences separated from seasonal trends
- **Energy**: User segments as distinct basins
  - Basin 1: Tech enthusiasts
  - Basin 2: Fashion followers
  - Basin 3: Home decorators

**Result**: Stable user representations for personalization

---

## Visual Intuition: Energy Landscapes

The 3D visualizations you created show this concept directly:

```
2D Latent Space (x, y)
    ↓
Energy Surface (height = energy)
    ↓
Valleys = Pattern basins
Peaks = High-uncertainty regions
Red X = Memory patterns (attractor centers)
```

**What you see**:
- **Valleys (dark/low)**: Points here are well-represented
- **Peaks (bright/high)**: Outliers or uncertain regions
- **Slopes**: Gradient flow direction (where points will move)

---

## Why "Atlasing"?

Like making an **atlas** (book of maps):

1. **Maps show relationships** (topology)
   - Distance between cities
   - Connecting roads

2. **Focus on important features** (causality)
   - Major landmarks, not every tree
   - Meaningful boundaries, not artifacts

3. **Organized by regions** (energy)
   - Countries have clear boundaries
   - Cities cluster in habitable zones

**APS creates a structured "atlas" of your data's pattern space.**

---

## Key Innovation vs. Standard Embeddings

| Aspect | Standard Embeddings | APS |
|--------|-------------------|-----|
| **Goal** | Minimize reconstruction error | ✓ Plus geometric structure |
| **Neighborhoods** | Hope they're preserved | ✓ Explicitly enforce (T) |
| **Factors** | Learn any features | ✓ Focus on causal ones (C) |
| **Organization** | Random/smooth space | ✓ Structured basins (E) |
| **Interpretability** | Distance only | ✓ Basin membership, energy |

---

## Energy Variants: Basin Shapes

### MemoryEnergy (Dot Product)
```
  Pattern
     ↓
  ⬇️⬇️⬇️  ← Directional basins
   ⬇️⬇️   (like word vectors)
    ⬇️
```
**Use for**: Text, oriented relationships

### RBFEnergy (Gaussian)
```
    Pattern
      ↓
   ⤵️⤵️⤵️⤵️
  ⤵️  ⬇️  ⤵️  ← Circular basins
   ⤵️⤵️⤵️⤵️   (isotropic)
```
**Use for**: Clustering, spatial data

### MixtureEnergy (Learnable)
```
Pattern 1 (strong)    Pattern 2 (weak)
     ↓                    ↓
  ⬇️⬇️⬇️⬇️              ⤵️⤵️
  ⬇️⬇️⬇️⬇️   vs       ⤵️⤵️
   ⬇️⬇️⬇️             ⤵️
```
**Use for**: Imbalanced categories, hierarchical data

---

## What You Can Do With APS

### 1. **Understand Data Structure**
```python
# Visualize how patterns organize
energy_model = MemoryEnergy(cfg)
landscape = visualizer.compute_landscape()
# See basins, boundaries, outlier regions
```

### 2. **Detect Outliers**
```python
# High energy = doesn't fit any pattern
energy = energy_model.energy(z)
outliers = energy > threshold
```

### 3. **Generate Valid Samples**
```python
# Flow to nearest valid pattern
z = torch.randn(latent_dim)  # Random noise
for _ in range(steps):
    energy.backward()
    z -= lr * z.grad  # Follow gradient
# z now represents valid pattern
```

### 4. **Transfer Knowledge**
```python
# Memory patterns as reusable knowledge
pretrained_patterns = energy_model.mem
new_model.mem.data.copy_(pretrained_patterns)
```

### 5. **Interpretable Classification**
```python
# Basin membership = category
basin_id = energy_model.nearest_pattern(z)
category = basin_labels[basin_id]
```

---

## Initialization Strategies

Memory patterns (basin centers) can be initialized in different ways:

| Method | When to Use | Example |
|--------|------------|---------|
| **Random** | Default baseline | Starting point |
| **Grid** | Visualization, exploration | Demo energy landscapes |
| **Cube** | Maximal separation | Categorical extremes |
| **Sphere** | Normalized data | Word embeddings |
| **K-means** | Real applications | Data-driven initialization |
| **Hierarchical** | Multi-scale data | Coarse + fine patterns |
| **PCA** | Data-driven | Along variance axes |

---

## In One Sentence

**APS creates embedding spaces that preserve relationships (T), focus on meaningful factors (C), and organize around learnable patterns (E).**

The 3D visualizations show this structure directly - you can literally see the organized "valleys" (basins) in pattern space!

---

## What Makes This Different

Most embedding methods:
```
Data → Neural Network → Embeddings
              ↓
         (Hope it works)
```

APS:
```
Data → Neural Network → Embeddings
              ↓
     + Topology constraints (T)
     + Causality objectives (C)
     + Energy shaping (E)
              ↓
     Structured, interpretable space
```

---

## Further Reading

- **Energy Framework**: `docs/ENERGY_FRAMEWORK_OVERVIEW.md`
- **Visualization Guide**: `docs/VISUALIZATION_GUIDE.md`
- **Main Docs**: `WARP.md`
- **Code**: `src/aps/`

The visualizations in `outputs/` show these concepts in action!
