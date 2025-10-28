"""
Main visualizer class for energy landscape visualization.

This module provides the EnergyLandscapeVisualizer class for creating
interactive visualizations of energy landscapes in latent space.
"""

from typing import Optional, Tuple, List
import torch
import numpy as np

from .data_structures import EnergyLandscape, MemoryPattern, Basin, Trajectory, CrossSection
from .config import VisualizationConfig
from .utils import create_grid, infer_bounds, identify_basin_by_gradient_descent, basin_depth, simulate_trajectory, sample_line, detect_basin_crossings
from .backends.plotly_backend import PlotlyBackend


class EnergyLandscapeVisualizer:
    """
    Creates interactive visualizations of energy landscapes in latent space.
    
    Example:
        >>> from aps.energy import MemoryEnergy
        >>> viz = EnergyLandscapeVisualizer(
        ...     energy_module=energy,
        ...     latent_dim=2,
        ...     resolution=100
        ... )
        >>> landscape = viz.compute_landscape()
        >>> fig = viz.plot_heatmap(landscape)
        >>> fig.show()
    """
    
    def __init__(
        self,
        energy_module: Optional[torch.nn.Module],
        latent_dim: int,
        resolution: int = 100,
        projection_method: str = "pca"
    ):
        """
        Initialize visualizer with energy module and configuration.
        
        Args:
            energy_module: Trained MemoryEnergy instance (or compatible module)
            latent_dim: Dimensionality of the latent space
            resolution: Grid density for energy computation (default: 100x100)
            projection_method: Method for dim reduction if latent_dim > 2 ("pca", "tsne", or "none")
        
        Raises:
            ValueError: If latent_dim < 2 or resolution < 10
        """
        if latent_dim < 2:
            raise ValueError("latent_dim must be >= 2")
        if resolution < 10:
            raise ValueError("resolution must be >= 10")
        
        self.energy_module = energy_module
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.projection_method = projection_method if latent_dim > 2 else None
        self.backend = PlotlyBackend()
    
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
        """
        # Extract memory patterns from energy module
        if hasattr(self.energy_module, 'memory'):
            memory_params = self.energy_module.memory.detach().cpu().numpy()
        elif hasattr(self.energy_module, 'get_memory_positions'):
            memory_params = self.energy_module.get_memory_positions()
        else:
            raise AttributeError("Energy module must have 'memory' parameter or 'get_memory_positions' method")
        
        # Infer bounds if not provided
        if bounds is None:
            if self.latent_dim == 2:
                bounds = infer_bounds(memory_params, margin=0.5)
            else:
                # For high-dim, use projection
                bounds = self._project_and_infer_bounds(memory_params)
        
        # Create grid
        grid_x, grid_y = create_grid(bounds, resolution=self.resolution)
        
        # Compute energy at each grid point
        energy_values = self._compute_energy_grid(grid_x, grid_y, memory_params)
        
        # Create MemoryPattern objects
        memory_patterns = self._create_memory_patterns(memory_params)
        
        return EnergyLandscape(
            grid_x=grid_x,
            grid_y=grid_y,
            energy_values=energy_values,
            memory_patterns=memory_patterns,
            latent_dim=self.latent_dim,
            projection_method=self.projection_method,
            bounds=bounds
        )
    
    def _compute_energy_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        memory_params: np.ndarray
    ) -> np.ndarray:
        """
        Compute energy at each point in the grid.
        
        Args:
            grid_x: 2D array of x coordinates
            grid_y: 2D array of y coordinates
            memory_params: Memory pattern positions
            
        Returns:
            2D array of energy values
        """
        # Flatten grid for batched computation
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        
        # Create coordinate tensor
        if self.latent_dim == 2:
            coords = torch.tensor(
                np.stack([flat_x, flat_y], axis=1),
                dtype=torch.float32
            )
        else:
            # For higher dimensions, project back
            coords = self._inverse_project(flat_x, flat_y)
        
        # Compute energies
        with torch.no_grad():
            if coords.device != next(self.energy_module.parameters()).device:
                coords = coords.to(next(self.energy_module.parameters()).device)
            energies = self.energy_module(coords)
            energies = energies.cpu().numpy()
        
        # Reshape to grid
        return energies.reshape(grid_x.shape)
    
    def _create_memory_patterns(self, memory_params: np.ndarray) -> list:
        """
        Create MemoryPattern objects from memory parameters.
        
        Args:
            memory_params: Array of memory positions
            
        Returns:
            List of MemoryPattern objects
        """
        patterns = []
        for i, pos_latent in enumerate(memory_params):
            # Get 2D position
            if self.latent_dim == 2:
                pos_2d = (float(pos_latent[0]), float(pos_latent[1]))
            else:
                pos_2d = self._project_single(pos_latent)
            
            # Compute energy at this position
            with torch.no_grad():
                pos_tensor = torch.tensor(pos_latent, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                if pos_tensor.device != next(self.energy_module.parameters()).device:
                    pos_tensor = pos_tensor.to(next(self.energy_module.parameters()).device)
                energy = float(self.energy_module(pos_tensor).cpu().item())
            
            patterns.append(MemoryPattern(
                id=i,
                position_latent=pos_latent,
                position_2d=pos_2d,
                energy=energy,
                beta_weight=getattr(self.energy_module, 'beta', 1.0)
            ))
        
        return patterns
    
    def _project_and_infer_bounds(self, memory_params: np.ndarray) -> Tuple[float, float, float, float]:
        """Project high-dimensional memory patterns and infer bounds."""
        # Simplified: just use PCA projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(memory_params)
        return infer_bounds(projected, margin=0.5)
    
    def _project_single(self, point: np.ndarray) -> Tuple[float, float]:
        """Project a single high-dimensional point to 2D."""
        # Placeholder: in full implementation, store PCA/t-SNE transformer
        return (0.0, 0.0)
    
    def _inverse_project(self, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """Inverse project 2D coordinates back to latent space."""
        # Placeholder: in full implementation, use stored transformer
        return torch.zeros((len(x), self.latent_dim), dtype=torch.float32)
    
    def plot_heatmap(
        self,
        landscape: EnergyLandscape,
        config: Optional[VisualizationConfig] = None,
        **kwargs
    ):
        """
        Create 2D heatmap visualization of energy landscape.
        
        Args:
            landscape: Computed EnergyLandscape object
            config: Optional display configuration
            **kwargs: Additional backend-specific arguments
        
        Returns:
            Figure object (Plotly Figure)
        """
        if config is None:
            config = VisualizationConfig()
        
        return self.backend.plot_heatmap(landscape, config, **kwargs)
    
    def identify_basins(
        self,
        landscape: EnergyLandscape,
        num_samples: int = 100,
        learning_rate: float = 0.01,
        max_steps: int = 100
    ) -> list:
        """
        Identify energy basins using gradient descent from random starting points.
        
        Args:
            landscape: Computed EnergyLandscape object
            num_samples: Number of random starting points for gradient descent
            learning_rate: Step size for gradient descent
            max_steps: Maximum iterations per trajectory
            
        Returns:
            List of Basin objects
            
        Example:
            >>> basins = viz.identify_basins(landscape, num_samples=50)
            >>> for basin in basins:
            ...     print(f"Basin {basin.id}: depth={basin.depth:.3f}")
        """
        # Sample random starting points within bounds
        xmin, xmax, ymin, ymax = landscape.bounds
        np.random.seed(42)  # Reproducibility
        
        start_points = []
        for _ in range(num_samples):
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            start_points.append((x, y))
        
        # Run gradient descent from each starting point
        convergence_points = []
        for start in start_points:
            converged, _ = identify_basin_by_gradient_descent(
                start,
                self.energy_module,
                learning_rate=learning_rate,
                max_steps=max_steps
            )
            convergence_points.append(converged)
        
        # Cluster convergence points to identify distinct basins
        from collections import defaultdict
        basin_clusters = defaultdict(list)
        
        for i, point in enumerate(convergence_points):
            # Find nearest memory pattern
            nearest_pattern_id = None
            min_dist = float('inf')
            
            for pattern in landscape.memory_patterns:
                px, py = pattern.position_2d
                dist = np.sqrt((point[0] - px)**2 + (point[1] - py)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_pattern_id = pattern.id
            
            # Assign to basin based on nearest pattern
            if nearest_pattern_id is not None:
                basin_clusters[nearest_pattern_id].append(start_points[i])
        
        # Create Basin objects
        basins = []
        for pattern_id, points in basin_clusters.items():
            if len(points) == 0:
                continue
            
            # Get pattern center
            pattern = landscape.memory_patterns[pattern_id]
            center = pattern.position_2d
            
            # Calculate basin depth
            depth = basin_depth(points, self.energy_module, center)
            
            basins.append(Basin(
                pattern_id=pattern_id,
                center=center,
                depth=depth,
                points=set(points)
            ))
        
        return basins
    
    def add_trajectory(
        self,
        landscape: EnergyLandscape,
        start_point: Tuple[float, float],
        learning_rate: float = 0.05,
        max_steps: int = 100,
        trajectory_id: Optional[int] = None
    ) -> Trajectory:
        """
        Add a gradient descent trajectory from a starting point.
        
        Args:
            landscape: Computed EnergyLandscape object
            start_point: Starting (x, y) coordinates
            learning_rate: Step size for gradient descent (default: 0.05)
            max_steps: Maximum iterations (default: 100)
            trajectory_id: Optional ID for the trajectory
            
        Returns:
            Trajectory object containing path and convergence info
            
        Performance:
            Must complete in < 2 seconds (SC-003)
            
        Example:
            >>> trajectory = viz.add_trajectory(landscape, (0.5, 0.5))
            >>> print(f"Converged in {trajectory.num_steps} steps")
            >>> print(f"Destination basin: {trajectory.destination_basin_id}")
        """
        # Simulate trajectory
        path_points, end_point, num_steps, converged = simulate_trajectory(
            start_point,
            self.energy_module,
            learning_rate=learning_rate,
            max_steps=max_steps
        )
        
        # Identify destination basin
        destination_basin_id = self._find_nearest_pattern_id(end_point, landscape)
        
        # Generate trajectory ID if not provided
        if trajectory_id is None:
            trajectory_id = id(path_points)  # Use memory address as unique ID
        
        return Trajectory(
            id=trajectory_id,
            start_point=start_point,
            end_point=end_point,
            path_points=path_points,
            destination_basin_id=destination_basin_id,
            num_steps=num_steps,
            converged=converged
        )
    
    def _find_nearest_pattern_id(
        self,
        point: Tuple[float, float],
        landscape: EnergyLandscape
    ) -> int:
        """Find the ID of the nearest memory pattern to a point."""
        x, y = point
        min_dist = float('inf')
        nearest_id = 0
        
        for pattern in landscape.memory_patterns:
            px, py = pattern.position_2d
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_id = pattern.id
        
        return nearest_id
    
    def plot_cross_section(
        self,
        landscape: EnergyLandscape,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        num_samples: int = 100,
        basins: Optional[List[Basin]] = None,
        cross_section_id: Optional[int] = None
    ) -> CrossSection:
        """
        Create a 1D cross-sectional view along a line through the landscape.
        
        Args:
            landscape: Computed EnergyLandscape object
            start_point: Starting (x, y) coordinates
            end_point: Ending (x, y) coordinates
            num_samples: Number of points to sample along the line (default: 100)
            basins: Optional list of Basin objects for crossing detection
            cross_section_id: Optional ID for the cross-section
            
        Returns:
            CrossSection object containing distance-energy profile
            
        Example:
            >>> cross_section = viz.plot_cross_section(
            ...     landscape, (-1.0, 0.0), (1.0, 0.0), num_samples=50
            ... )
            >>> df = cross_section.to_dataframe()
        """
        # Sample points along the line
        sample_points_2d = sample_line(start_point, end_point, num_samples)
        
        # Calculate total line length
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        total_length = np.sqrt(dx**2 + dy**2)
        
        # Compute energy at each point
        sample_points = []
        for i, (x, y) in enumerate(sample_points_2d):
            # Distance along line
            if total_length > 0:
                distance = (i / (num_samples - 1)) * total_length if num_samples > 1 else 0.0
            else:
                distance = 0.0
            
            # Get energy at this point
            energy = landscape.get_energy_at(x, y)
            
            sample_points.append((distance, energy))
        
        # Detect basin crossings if basins provided
        basins_crossed = []
        if basins:
            basins_crossed = detect_basin_crossings(sample_points_2d, basins)
        
        # Generate cross-section ID if not provided
        if cross_section_id is None:
            cross_section_id = id(sample_points)  # Use memory address as unique ID
        
        return CrossSection(
            id=cross_section_id,
            start_point=start_point,
            end_point=end_point,
            sample_points=sample_points,
            num_samples=num_samples,
            basins_crossed=basins_crossed
        )
    
    def export(
        self,
        landscape: EnergyLandscape,
        filename: str,
        format: str = 'html',
        config: Optional[VisualizationConfig] = None
    ):
        """
        Export visualization to file.
        
        Args:
            landscape: Computed EnergyLandscape object
            filename: Output filename (with or without extension)
            format: Output format ('html', 'png', or 'svg')
            config: Optional visualization configuration
            
        Example:
            >>> viz.export(landscape, 'energy_landscape', format='html')
            >>> viz.export(landscape, 'figure.png', format='png')
        """
        # Create figure
        fig = self.plot_heatmap(landscape, config=config)
        
        # Ensure filename has correct extension
        if not filename.endswith(f'.{format}'):
            filename = f'{filename}.{format}'
        
        # Export based on format
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename, format='png')
        elif format == 'svg':
            fig.write_image(filename, format='svg')
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'png', or 'svg'.")
        
        print(f"Exported visualization to {filename}")
