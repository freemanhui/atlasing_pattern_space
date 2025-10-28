# Visualization Guide

This document explains all available energy landscape visualizations in APS.

## Overview

APS provides comprehensive visualizations for understanding energy-based embeddings across both 2D and 3D latent spaces. All visualizations are **interactive HTML files** powered by Plotly.

## Quick Start

```bash
# Generate all 2D visualizations
make dashboard

# Generate 3D surface plots (2D latent + energy as height)
make viz3d

# Generate true 3D latent space visualizations
make embedding3d
```

## 2D Visualizations (2D Latent Space)

### 1. Interactive Dashboard (`interactive_dashboard.html`)

**Command**: `make dashboard` or `python scripts/interactive_dashboard.py`

**What it shows**:
- 4-panel comparison of different beta values (1.0, 3.0, 5.0, 10.0)
- Energy as color (heatmap)
- Memory patterns marked with red X

**Key insights**:
- How beta controls basin sharpness
- Basin boundaries and decision surfaces
- Memory pattern locations

**Interactions**:
- Hover: See exact (x, y, energy) values
- Zoom: Click and drag
- Pan: Use toolbar

---

### 2. Cross-Section Comparison (`cross_section_comparison.html`)

**Command**: `make dashboard` (creates both files)

**What it shows**:
- Energy profile along y=0 line
- 4 curves for different beta values
- Valley depths and barrier heights

**Key insights**:
- Basin depth increases with beta
- Energy barriers between basins
- Basin separation clarity

---

## 3D Surface Visualizations (2D Latent + Energy Height)

### 3. 3D Surface Comparison (`energy_3d_surfaces.html`)

**Command**: `make viz3d` or `python scripts/interactive_dashboard_3d.py`

**What it shows**:
- 4 rotatable 3D surfaces (one per beta value)
- Energy as surface height (z-axis)
- Memory patterns placed at their energy values
- Contour projections on base

**Key insights**:
- **Valleys = attraction basins** (literally)
- **Peaks = barriers** between basins
- Basin steepness visible as slope

**Interactions**:
- Click and drag: Rotate view
- Scroll: Zoom in/out
- Hover: See coordinates + energy

---

### 4. Detailed 3D Surface (`energy_3d_detailed.html`)

**Command**: `make viz3d`

**What it shows**:
- Single high-resolution 3D surface (100x100 grid)
- Labeled memory patterns (M1, M2, ...)
- Contour lines projected on base
- Beta = 5.0 (optimal for visualization)

**Best for**:
- Publication figures
- Detailed exploration
- Teaching/presentations

---

### 5. Wireframe View (`energy_3d_wireframe.html`)

**Command**: `make viz3d`

**What it shows**:
- Wireframe 3D surfaces (emphasizes structure)
- Dropdown to switch between beta values
- Grid lines show curvature

**Key insights**:
- Basin structure without color distraction
- Curvature and gradient directions
- Geometric relationships

---

## True 3D Latent Space Visualizations

### 6. 3D Cube with Energy (`energy_3d_cube.html`)

**Command**: `make embedding3d` or `python scripts/run_3d_embedding.py`

**What it shows**:
- 3D latent space (latent_dim=3)
- 2000 sample points colored by energy
- Memory patterns at cube vertices
- Dashed cube edges for reference

**Key insights**:
- Energy distribution in true 3D space
- How points cluster near memory patterns
- Geometric arrangement of basins

**Interactions**:
- Full 3D rotation
- Color indicates energy level
- Low energy (dark) = near memory patterns

---

### 7. Cross-Sectional Slices (`energy_3d_slices.html`)

**Command**: `make embedding3d`

**What it shows**:
- 5 horizontal slices at different z-levels: -1.0, -0.5, 0.0, 0.5, 1.0
- Energy heatmap on each slice
- Memory patterns shown when on or near slice

**Key insights**:
- How energy structure varies through 3D space
- Basin boundaries at different depths
- Patterns appear/disappear as you move through z-levels

**Best for**:
- Understanding 3D structure via 2D sections
- Comparing energy at different depths

---

### 8. Gradient Descent Trajectories (`energy_3d_trajectory.html`)

**Command**: `make embedding3d`

**What it shows**:
- 20 trajectories starting from random points
- Lines show gradient descent flow
- Color indicates energy along path
- Red diamonds = memory pattern destinations

**Key insights**:
- **Visualizes basin attraction dynamics**
- Points flow "downhill" toward nearest memory
- Trajectory curvature shows gradient field
- Convergence to stable attractors

**Best for**:
- Understanding optimization dynamics
- Visualizing how embeddings evolve
- Teaching gradient-based methods

---

## Interpretation Tips

### Reading Energy Values

| Energy Range | Meaning |
|--------------|---------|
| Low (< 0) | Point is well-represented by memory pattern |
| Medium (0-2) | Point is between basins |
| High (> 2) | Point doesn't match any pattern (outlier) |

### Beta Parameter Effects

| Beta | Basin Shape | Use Case |
|------|-------------|----------|
| 1.0 | Wide, overlapping | Smooth interpolation |
| 3.0 | Moderate separation | Balanced |
| 5.0 | Clear, distinct | Category learning |
| 10.0 | Sharp, deep | Discrete states |

### Color Scales

- **Viridis** (default): Dark = low energy, bright = high energy
- **Plasma** (trajectories): Dark red = low, bright yellow = high

---

## File Locations

All visualizations are saved to `outputs/`:

```
outputs/
├── interactive_dashboard.html          # 2D multi-panel
├── cross_section_comparison.html       # 2D cross-section
├── energy_3d_surfaces.html            # 3D surfaces (4-panel)
├── energy_3d_detailed.html            # 3D detailed single view
├── energy_3d_wireframe.html           # 3D wireframe
├── energy_3d_cube.html                # True 3D scatter
├── energy_3d_slices.html              # True 3D slices
└── energy_3d_trajectory.html          # True 3D trajectories
```

---

## Customization

### Changing Parameters

Edit the script files to customize:

**Beta values**:
```python
beta_values = [1.0, 3.0, 5.0, 10.0]  # Add/remove as needed
```

**Resolution** (quality vs. speed):
```python
resolution = 50  # Higher = smoother but slower
```

**Memory pattern layout**:
```python
# 2D: corners of square
memory_patterns = [[-1,-1], [1,-1], [-1,1], [1,1]]

# 3D: corners of cube
memory_patterns = [[-1,-1,-1], [1,-1,-1], ...]
```

**Number of sample points**:
```python
n_samples = 2000  # For 3D scatter plots
```

---

## Performance Notes

| Visualization | Time | Size | Best For |
|---------------|------|------|----------|
| 2D Dashboard | ~5s | ~2MB | Quick exploration |
| 3D Surfaces | ~10s | ~5MB | Deep understanding |
| 3D Slices | ~8s | ~3MB | Structural analysis |
| 3D Trajectories | ~15s | ~4MB | Dynamics |

All times are approximate on modern hardware.

---

## Troubleshooting

### Plots not showing?

- Check browser console for errors
- Try opening in Chrome/Firefox (best Plotly support)
- File might be too large - reduce resolution

### Memory errors?

- Reduce `resolution` parameter
- Reduce `n_samples` for scatter plots
- Close other applications

### Slow rendering?

- Interactive HTML files can be large
- Consider exporting to PNG for static figures (see export docs)

---

## Next Steps

For exporting to publication formats (PNG, SVG, PDF), see `docs/EXPORT_GUIDE.md` (coming in Phase 7).

For programmatic access to landscapes, see API docs in `src/aps/viz/`.
