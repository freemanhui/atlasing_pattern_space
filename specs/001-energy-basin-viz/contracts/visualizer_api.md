# Energy Basin Visualizer API Contract

## Core Interfaces

### EnergyLandscapeVisualizer

Main class for creating and managing energy basin visualizations.

```python
class EnergyLandscapeVisualizer:
    """
    Creates interactive visualizations of energy landscapes in latent space.
    """
    
    def __init__(
        self,
        energy_module: MemoryEnergy,
        latent_dim: int,
        resolution: int = 100,
        projection_method: str = "pca"
    ):
        """
        Initialize visualizer with energy module and configuration.
        
        Args:
            energy_module: Trained MemoryEnergy instance
            latent_dim: Dimensionality of the latent space
            resolution: Grid density for energy computation (default: 100x100)
            projection_method: Method for dim reduction if latent_dim > 2 ("pca", "tsne", or "none")
        
        Raises:
            ValueError: If latent_dim < 2 or resolution < 10
        """
        
    def compute_landscape(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> EnergyLandscape:
        """
        Compute energy values across 2D grid.
        
        Args:
            bounds: Optional (xmin, xmax, ymin, ymax) for visualization region.
                    If None, inferred from memory pattern locations.
        
        Returns:
            EnergyLandscape object containing grid and energy values
            
        Performance: Must complete in < 5 seconds for 100x100 grid (SC-001)
        """
        
    def plot_heatmap(
        self,
        landscape: EnergyLandscape,
        config: Optional[VisualizationConfig] = None,
        **kwargs
    ) -> Figure:
        """
        Create 2D heatmap visualization of energy landscape.
        
        Args:
            landscape: Computed EnergyLandscape object
            config: Optional display configuration
            **kwargs: Additional matplotlib/plotly arguments
        
        Returns:
            Figure object (matplotlib or plotly depending on backend)
            
        Features:
            - Memory pattern markers overlaid
            - Hover tooltips with energy values
            - Configurable color schemes
        """
        
    def identify_basins(
        self,
        landscape: EnergyLandscape,
        max_iter: int = 100,
        tolerance: float = 1e-5
    ) -> Dict[int, Basin]:
        """
        Identify basin regions using gradient descent from grid points.
        
        Args:
            landscape: Computed EnergyLandscape
            max_iter: Maximum gradient descent iterations per point
            tolerance: Convergence threshold for gradient magnitude
        
        Returns:
            Dictionary mapping pattern_id to Basin object
            
        Performance: Should complete within landscape computation time budget (SC-001)
        """
    
    def add_trajectory(
        self,
        landscape: EnergyLandscape,
        start_point: Tuple[float, float],
        learning_rate: float = 0.1,
        max_steps: int = 100
    ) -> Trajectory:
        """
        Simulate and visualize gradient descent trajectory.
        
        Args:
            landscape: Computed EnergyLandscape
            start_point: Starting (x, y) coordinates
            learning_rate: Step size for gradient descent
            max_steps: Maximum iterations
        
        Returns:
            Trajectory object with path and convergence info
            
        Performance: Must complete in < 2 seconds (SC-003)
        """
    
    def plot_cross_section(
        self,
        landscape: EnergyLandscape,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_samples: int = 100
    ) -> Tuple[Figure, CrossSection]:
        """
        Generate 1D cross-sectional energy profile.
        
        Args:
            landscape: Computed EnergyLandscape
            start: Starting (x, y) coordinates
            end: Ending (x, y) coordinates
            num_samples: Number of points to sample along line
        
        Returns:
            (Figure, CrossSection) tuple
            - Figure: 1D plot of energy vs position
            - CrossSection: Data object with samples and basin crossings
        """
    
    def export(
        self,
        landscape: EnergyLandscape,
        filepath: str,
        format: str = "html"
    ) -> None:
        """
        Export visualization to file.
        
        Args:
            landscape: Computed EnergyLandscape with visualizations
            filepath: Output file path
            format: Export format ("html", "png", "svg")
        
        Raises:
            ValueError: If format not supported
            IOError: If file cannot be written
            
        Requirements:
            - HTML exports must preserve interactivity (SC-007)
            - PNG/SVG exports must maintain visual quality (SC-007)
        """
```

---

### VisualizationConfig

Configuration object for display preferences.

```python
@dataclass
class VisualizationConfig:
    """User preferences for visualization appearance."""
    
    color_scheme: str = "viridis"  # Matplotlib/Plotly colormap name
    color_scale: str = "linear"    # "linear" or "log"
    show_memory_markers: bool = True
    marker_size: int = 10
    show_grid: bool = False
    grid_alpha: float = 0.3
    tooltip_enabled: bool = True
    trajectory_color_by_basin: bool = True
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.color_scale not in ["linear", "log"]:
            raise ValueError("color_scale must be 'linear' or 'log'")
        if not (0 <= self.grid_alpha <= 1):
            raise ValueError("grid_alpha must be in [0, 1]")
```

---

### InteractionHandler

Handles user interactions (clicks, hovers) for interactive visualizations.

```python
class InteractionHandler:
    """
    Manages interactive events for energy landscape visualizations.
    """
    
    def on_hover(
        self,
        event: HoverEvent,
        landscape: EnergyLandscape
    ) -> Dict[str, Any]:
        """
        Handle hover event and return tooltip data.
        
        Args:
            event: Hover event with (x, y) coordinates
            landscape: EnergyLandscape for querying
        
        Returns:
            Dictionary with {
                "coordinates": (x, y),
                "energy": float,
                "nearest_pattern_id": int,
                "basin_id": int (if known)
            }
            
        Performance: Must respond in < 100ms (SC-002)
        """
    
    def on_click(
        self,
        event: ClickEvent,
        landscape: EnergyLandscape,
        basins: Dict[int, Basin]
    ) -> PointInfo:
        """
        Handle click event and return detailed point information.
        
        Args:
            event: Click event with (x, y) coordinates
            landscape: EnergyLandscape for querying
            basins: Dictionary of Basin objects
        
        Returns:
            PointInfo object with all computed properties
            
        Performance: Must respond in < 100ms (SC-002)
        """
    
    def on_trajectory_request(
        self,
        event: ClickEvent,
        visualizer: EnergyLandscapeVisualizer
    ) -> Trajectory:
        """
        Handle user request to add trajectory from clicked point.
        
        Args:
            event: Click event for starting point
            visualizer: Visualizer instance to compute trajectory
        
        Returns:
            Trajectory object
        """
```

---

## Data Structures

### EnergyLandscape

```python
@dataclass
class EnergyLandscape:
    """Computed energy surface over 2D grid."""
    
    grid_x: np.ndarray  # 2D array of x coordinates
    grid_y: np.ndarray  # 2D array of y coordinates
    energy_values: np.ndarray  # 2D array of energy values
    memory_patterns: List[MemoryPattern]
    latent_dim: int
    projection_method: Optional[str]
    bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    
    def get_energy_at(self, x: float, y: float) -> float:
        """Interpolate energy value at arbitrary (x, y) point."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for export."""
```

### Basin

```python
@dataclass
class Basin:
    """Region of attraction around a memory pattern."""
    
    pattern_id: int
    center: Tuple[float, float]  # 2D coordinates of pattern
    depth: float  # Energy gap from center to rim
    points: Set[Tuple[int, int]]  # Grid indices belonging to this basin
    boundary: Optional[np.ndarray] = None  # Contour points
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within basin."""
```

### Trajectory

```python
@dataclass
class Trajectory:
    """Gradient descent path from start to basin."""
    
    id: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    path_points: List[Tuple[float, float, float, float]]  # (x, y, energy, grad_mag)
    destination_basin_id: int
    num_steps: int
    converged: bool
    
    def get_path_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract x and y coordinates for plotting."""
```

### CrossSection

```python
@dataclass
class CrossSection:
    """1D energy profile along a line."""
    
    id: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    sample_points: List[Tuple[float, float]]  # (distance_along_line, energy)
    num_samples: int
    basins_crossed: List[int]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
```

---

## Error Handling

All methods should raise appropriate exceptions:

- `ValueError`: Invalid input parameters
- `RuntimeError`: Computation failures (e.g., gradient descent divergence)
- `IOError`: File export/import errors
- `MemoryError`: Grid resolution too high for available memory

---

## Performance Contracts

Aligned with Success Criteria from spec.md:

- `compute_landscape()`: < 5 seconds for 100x100 grid (SC-001)
- `on_hover()`, `on_click()`: < 100ms response time (SC-002)
- `add_trajectory()`: < 2 seconds per trajectory (SC-003)
- Energy accuracy: < 1% error vs MemoryEnergy ground truth (SC-005)
