# Data Model: Energy Basin Visualization

## Core Entities

### EnergyLandscape
Represents the computed energy surface over latent space.

**Attributes**:
- `grid_points`: 2D array of (x, y) coordinates in latent space
- `energy_values`: 2D array of energy values corresponding to grid_points
- `latent_dim`: Original dimensionality of latent space
- `resolution`: Grid density (e.g., 100x100)
- `projection_method`: If latent_dim > 2, the method used (PCA, t-SNE, or None)
- `bounds`: Min/max coordinates for x and y axes

**Relationships**:
- Has many `MemoryPattern` (markers on the landscape)
- Contains multiple `Basin` regions

---

### MemoryPattern
Represents a learned attractor (memory) from the MemoryEnergy module.

**Attributes**:
- `id`: Unique identifier (integer index)
- `position_latent`: Coordinates in original latent space (n-dimensional)
- `position_2d`: Coordinates in visualization space (2D projection if needed)
- `energy`: Energy value at this location (should be local minimum)
- `beta_weight`: Sharpness parameter for this basin

**Relationships**:
- Belongs to one `EnergyLandscape`
- Is center of one `Basin`

---

### Basin
Represents a region of attraction around a memory pattern.

**Attributes**:
- `pattern_id`: ID of associated MemoryPattern (foreign key)
- `depth`: Energy difference between basin center and rim (computed)
- `points`: Set of grid points that belong to this basin
- `boundary`: Approximate boundary contour (optional)

**Relationships**:
- Belongs to one `MemoryPattern`
- Contains multiple `Point` objects

---

### Point
Represents a specific location in latent space with computed properties.

**Attributes**:
- `coordinates_2d`: (x, y) position in visualization
- `coordinates_latent`: Position in original latent space (if applicable)
- `energy`: Computed energy value
- `nearest_pattern_id`: ID of closest MemoryPattern
- `nearest_pattern_distance`: Distance to nearest pattern
- `basin_id`: Which basin this point belongs to (via gradient descent)

**Relationships**:
- May belong to one `Basin`
- References one `MemoryPattern` (nearest)

---

### Trajectory
Represents a gradient descent path from starting point to basin.

**Attributes**:
- `id`: Unique identifier
- `start_point`: Starting coordinates (2D)
- `end_point`: Final coordinates (2D)
- `destination_basin_id`: Basin where trajectory converged
- `path_points`: Sequence of (x, y, energy, gradient_magnitude) tuples
- `num_steps`: Number of gradient descent iterations
- `converged`: Boolean indicating if it reached a basin

**Relationships**:
- Ends in one `Basin`
- Contains many `PathPoint` objects (embedded)

---

### CrossSection
Represents a 1D slice through the energy landscape.

**Attributes**:
- `id`: Unique identifier
- `start_point_2d`: Starting (x, y) coordinates
- `end_point_2d`: Ending (x, y) coordinates  
- `sample_points`: List of (position_along_line, energy) tuples
- `num_samples`: Number of points sampled along the line
- `basins_crossed`: List of basin IDs intersected by this line

**Relationships**:
- Intersects multiple `Basin` objects
- References `MemoryPattern` objects within its path

---

## Visualization State

### VisualizationConfig
Encapsulates user preferences and display settings.

**Attributes**:
- `color_scheme`: Name of color map (e.g., "viridis", "coolwarm")
- `color_scale`: Linear or logarithmic
- `show_memory_markers`: Boolean
- `show_grid`: Boolean
- `grid_resolution`: Current grid density
- `tooltip_enabled`: Boolean for hover interactions
- `trajectory_colors`: Color mapping for trajectories by basin

**Usage**: Passed to rendering functions to control visual appearance.

---

## Data Flow

### Initialization
1. Load trained model with `MemoryEnergy` module
2. Extract memory patterns (positions, beta values)
3. Determine latent dimensionality
4. Apply dimensionality reduction if needed → Create `EnergyLandscape`

### Energy Computation
1. Generate grid of points in 2D visualization space
2. Map grid points back to latent space (inverse projection if needed)
3. Compute energy for each grid point using MemoryEnergy
4. Store as `EnergyLandscape.energy_values`

### Basin Identification
1. For each grid point, run gradient descent
2. Record which `MemoryPattern` it converges to
3. Group points by convergence → Create `Basin` objects
4. Compute basin depth for each basin

### Interactive Features
- **Point Click**: Query `Point` object for energy, nearest pattern, basin ID
- **Trajectory**: Create `Trajectory` object with gradient descent simulation
- **Cross-Section**: Create `CrossSection` by sampling along user-drawn line

---

## Data Persistence (Optional)

For large landscapes or pre-computed results:
- Serialize `EnergyLandscape` to disk (NumPy .npz format)
- Cache basin memberships to avoid recomputation
- Export `Trajectory` and `CrossSection` data for later analysis

File format suggestion:
```
energy_landscape.npz:
  - grid_x: 2D array
  - grid_y: 2D array
  - energy: 2D array
  - memory_patterns: structured array
  - basins: structured array
```
