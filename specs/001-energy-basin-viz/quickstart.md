# Quickstart Guide: Energy Basin Visualization

## Installation

```bash
# Assuming aps package is already installed
pip install -e ".[visualization]"  # Installs viz dependencies
```

## Basic Usage (User Story 1 - MVP)

### Step 1: Load Your Trained Model

```python
import torch
from aps.energy import MemoryEnergy
from aps.viz import EnergyLandscapeVisualizer

# Load your trained model with MemoryEnergy
model = torch.load("my_trained_model.pt")
energy_module = model.energy  # Assuming energy is a MemoryEnergy instance

# Get latent dimensionality
latent_dim = 2  # Or extract from model config
```

### Step 2: Create Visualizer

```python
# Initialize visualizer
viz = EnergyLandscapeVisualizer(
    energy_module=energy_module,
    latent_dim=latent_dim,
    resolution=100  # 100x100 grid
)
```

### Step 3: Compute and Plot Energy Landscape

```python
# Compute energy landscape
landscape = viz.compute_landscape()

# Create heatmap visualization
fig = viz.plot_heatmap(landscape)
fig.show()  # Display in notebook or save to file
```

**Expected Output**: 2D heatmap with memory patterns marked, basins visible as low-energy regions (darker/cooler colors).

---

## Interactive Exploration (User Story 2)

### Enable Click/Hover Interactions

```python
from aps.viz import VisualizationConfig, InteractionHandler

# Create interactive visualization
config = VisualizationConfig(tooltip_enabled=True)
fig = viz.plot_heatmap(landscape, config=config)

# Setup interaction handler (for Plotly backend)
handler = InteractionHandler()

@fig.on_click
def handle_click(event):
    point_info = handler.on_click(event, landscape, basins)
    print(f"Energy: {point_info.energy:.3f}")
    print(f"Basin ID: {point_info.basin_id}")
    print(f"Nearest pattern: {point_info.nearest_pattern_id}")

fig.show()
```

---

## Adding Trajectories (User Story 3)

### Visualize Gradient Descent Paths

```python
# Add trajectory from a starting point
start_point = (-1.5, 0.5)
trajectory = viz.add_trajectory(landscape, start_point)

# Plot landscape with trajectory overlaid
fig = viz.plot_heatmap(landscape)
fig.add_trajectory(trajectory)  # Overlay trajectory on heatmap
fig.show()

# Add multiple trajectories
for start in [(-1.0, 1.0), (0.5, -0.5), (1.5, 0.0)]:
    traj = viz.add_trajectory(landscape, start)
    fig.add_trajectory(traj)
```

**Expected Output**: Energy landscape with colored trajectories showing gradient flow to basins.

---

## Cross-Sectional Analysis (User Story 4)

### Plot Energy Profile Along a Line

```python
# Define cross-section line
start = (-2.0, 0.0)
end = (2.0, 0.0)

# Generate cross-section
fig_cross, cross_section = viz.plot_cross_section(
    landscape, 
    start, 
    end, 
    num_samples=100
)

fig_cross.show()  # 1D plot of energy vs position
print(f"Basins crossed: {cross_section.basins_crossed}")
```

---

## Advanced: High-Dimensional Latent Spaces

### Using Dimensionality Reduction

```python
# For latent_dim > 2, use PCA or t-SNE
viz_highdim = EnergyLandscapeVisualizer(
    energy_module=energy_module,
    latent_dim=10,  # Original high-dimensional space
    resolution=100,
    projection_method="pca"  # or "tsne"
)

landscape_projected = viz_highdim.compute_landscape()
fig = viz_highdim.plot_heatmap(landscape_projected)

# Note: Visualization is a 2D projection
print(f"Projection method: {landscape_projected.projection_method}")
```

---

## Exporting Visualizations

### Save to Files

```python
# Export interactive HTML
viz.export(landscape, "energy_landscape.html", format="html")

# Export static images
viz.export(landscape, "energy_landscape.png", format="png")
viz.export(landscape, "energy_landscape.svg", format="svg")
```

---

## Configuration Options

### Customizing Appearance

```python
from aps.viz import VisualizationConfig

config = VisualizationConfig(
    color_scheme="coolwarm",  # or "viridis", "plasma", etc.
    color_scale="log",  # Use logarithmic scale for wide energy ranges
    show_memory_markers=True,
    marker_size=15,
    show_grid=True,
    trajectory_color_by_basin=True
)

fig = viz.plot_heatmap(landscape, config=config)
```

---

## Common Use Cases

### 1. Quick Basin Inspection

```python
# Load model, visualize basins, done!
viz = EnergyLandscapeVisualizer(energy_module, latent_dim=2)
landscape = viz.compute_landscape()
basins = viz.identify_basins(landscape)

print(f"Found {len(basins)} basins")
for basin_id, basin in basins.items():
    print(f"  Basin {basin_id}: depth = {basin.depth:.3f}")
    
viz.plot_heatmap(landscape).show()
```

### 2. Compare Basin Depths

```python
# Quantify basin quality
landscape = viz.compute_landscape()
basins = viz.identify_basins(landscape)

depths = [b.depth for b in basins.values()]
print(f"Mean basin depth: {np.mean(depths):.3f}")
print(f"Shallowest basin: {np.min(depths):.3f}")
print(f"Deepest basin: {np.max(depths):.3f}")
```

### 3. Trajectory Analysis

```python
# Test basin coverage from random starting points
n_samples = 20
start_points = np.random.uniform(-2, 2, (n_samples, 2))

trajectories = []
for start in start_points:
    traj = viz.add_trajectory(landscape, tuple(start))
    trajectories.append(traj)

# Count convergence to each basin
from collections import Counter
destinations = [t.destination_basin_id for t in trajectories]
coverage = Counter(destinations)
print(f"Basin coverage: {coverage}")
```

---

## Performance Tips

1. **Grid Resolution**: Start with 50x50 for quick previews, use 100x100 or higher for final analysis
2. **Caching**: Save computed landscapes to disk for reuse
   ```python
   landscape.save("landscape_cache.npz")
   landscape = EnergyLandscape.load("landscape_cache.npz")
   ```
3. **Batch Trajectories**: Compute multiple trajectories in parallel (if implemented)
4. **Adaptive Resolution**: Use coarse grid initially, refine interactively

---

## Troubleshooting

**Energy values span too wide a range?**
- Use logarithmic color scale: `config.color_scale = "log"`

**Too many memory patterns cluttering the view?**
- Disable markers: `config.show_memory_markers = False`
- Or implement clustering (future feature)

**Gradient descent not converging?**
- Increase `max_steps` in `add_trajectory()`
- Decrease `learning_rate` for more stable descent

**Visualization too slow?**
- Reduce `resolution` (e.g., 50x50 instead of 100x100)
- Use pre-computed basins instead of recomputing on each plot
