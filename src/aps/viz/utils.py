"""
Utility functions for energy landscape visualization.

This module provides helper functions for grid generation, bounds inference,
interpolation, and basin identification.
"""

from typing import Tuple, Optional, List
import numpy as np
import torch


def create_grid(
    bounds: Tuple[float, float, float, float],
    resolution: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D meshgrid for energy computation.
    
    Args:
        bounds: (xmin, xmax, ymin, ymax) for the grid
        resolution: Number of points along each axis
        
    Returns:
        Tuple of (grid_x, grid_y) 2D arrays
        
    Example:
        >>> bounds = (-2.0, 2.0, -2.0, 2.0)
        >>> grid_x, grid_y = create_grid(bounds, resolution=50)
        >>> grid_x.shape
        (50, 50)
    """
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y


def infer_bounds(
    memory_positions: np.ndarray,
    margin: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Infer visualization bounds from memory pattern positions.
    
    Args:
        memory_positions: Array of shape (n_patterns, 2) with (x, y) positions
        margin: Extra space around patterns as fraction of range
        
    Returns:
        Tuple of (xmin, xmax, ymin, ymax)
        
    Example:
        >>> positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> bounds = infer_bounds(positions, margin=0.2)
        >>> bounds
        (-0.2, 1.2, -0.2, 1.2)
    """
    if memory_positions.shape[0] == 0:
        # Default bounds if no patterns
        return (-2.0, 2.0, -2.0, 2.0)
    
    xmin = memory_positions[:, 0].min()
    xmax = memory_positions[:, 0].max()
    ymin = memory_positions[:, 1].min()
    ymax = memory_positions[:, 1].max()
    
    # Add margin
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    # Handle case where all patterns are at same location
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0
    
    xmin -= margin * x_range
    xmax += margin * x_range
    ymin -= margin * y_range
    ymax += margin * y_range
    
    return (xmin, xmax, ymin, ymax)


def interpolate_energy(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    energy_values: np.ndarray,
    x: float,
    y: float
) -> float:
    """
    Interpolate energy value at arbitrary (x, y) point using bilinear interpolation.
    
    Args:
        grid_x: 2D array of x coordinates
        grid_y: 2D array of y coordinates
        energy_values: 2D array of energy values
        x: X coordinate to interpolate at
        y: Y coordinate to interpolate at
        
    Returns:
        Interpolated energy value
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Extract 1D arrays
    x_1d = grid_x[0, :]
    y_1d = grid_y[:, 0]
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (y_1d, x_1d),
        energy_values,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    return float(interpolator((y, x)))


def basin_depth(
    basin_points: List[Tuple[float, float]],
    energy_func,
    basin_center: Optional[Tuple[float, float]] = None
) -> float:
    """
    Calculate basin depth as gap between mean and minimum energy.
    
    Args:
        basin_points: List of (x, y) coordinates in the basin
        energy_func: Function to compute energy at points
        basin_center: Optional pre-computed center point
        
    Returns:
        Basin depth (mean_energy - min_energy)
        
    Example:
        >>> points = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
        >>> depth = basin_depth(points, energy_module)
    """
    if len(basin_points) == 0:
        return 0.0
    
    # Convert to tensor
    points_array = np.array(basin_points)
    points_tensor = torch.tensor(points_array, dtype=torch.float32)
    
    # Compute energies
    with torch.no_grad():
        energies = energy_func(points_tensor).numpy()
    
    # Depth = mean - min
    return float(np.mean(energies) - np.min(energies))


def identify_basin_by_gradient_descent(
    start_point: Tuple[float, float],
    energy_module: torch.nn.Module,
    learning_rate: float = 0.01,
    max_steps: int = 100,
    tolerance: float = 1e-4
) -> Tuple[Tuple[float, float], int]:
    """
    Identify which basin a point belongs to by gradient descent.
    
    Args:
        start_point: Starting (x, y) coordinates
        energy_module: PyTorch module with energy function
        learning_rate: Step size for gradient descent
        max_steps: Maximum iteration steps
        tolerance: Convergence tolerance for position change
        
    Returns:
        Tuple of (converged_point, num_steps)
        
    Example:
        >>> basin_center, steps = identify_basin_by_gradient_descent(
        ...     (0.5, 0.5), energy_module
        ... )
    """
    # Initialize position
    pos = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)
    
    for step in range(max_steps):
        # Compute energy and gradient
        energy = energy_module(pos.unsqueeze(0)).squeeze()
        
        # Compute gradient
        if pos.grad is not None:
            pos.grad.zero_()
        energy.backward()
        
        # Gradient descent step
        with torch.no_grad():
            old_pos = pos.clone()
            pos -= learning_rate * pos.grad
            
            # Check convergence
            delta = torch.norm(pos - old_pos)
            if delta < tolerance:
                break
    
    converged_point = (float(pos[0].item()), float(pos[1].item()))
    return converged_point, step + 1


def simulate_trajectory(
    start_point: Tuple[float, float],
    energy_module: torch.nn.Module,
    learning_rate: float = 0.05,
    max_steps: int = 100,
    tolerance: float = 1e-4
) -> Tuple[List[Tuple[float, float, float, float]], Tuple[float, float], int, bool]:
    """
    Simulate gradient descent trajectory from a starting point.
    
    Args:
        start_point: Starting (x, y) coordinates
        energy_module: PyTorch module with energy function
        learning_rate: Step size for gradient descent
        max_steps: Maximum iteration steps
        tolerance: Convergence tolerance for position change
        
    Returns:
        Tuple of:
        - path_points: List of (x, y, energy, gradient_magnitude) tuples
        - end_point: Final (x, y) coordinates
        - num_steps: Number of steps taken
        - converged: Whether trajectory converged
        
    Example:
        >>> path, end, steps, conv = simulate_trajectory(
        ...     (0.5, 0.5), energy_module
        ... )
    """
    # Initialize position
    pos = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)
    
    path_points = []
    converged = False
    
    for step in range(max_steps):
        # Compute energy and gradient
        energy = energy_module(pos.unsqueeze(0)).squeeze()
        
        # Compute gradient
        if pos.grad is not None:
            pos.grad.zero_()
        energy.backward()
        
        # Record current state
        gradient_magnitude = float(torch.norm(pos.grad).item())
        path_points.append((
            float(pos[0].item()),
            float(pos[1].item()),
            float(energy.item()),
            gradient_magnitude
        ))
        
        # Gradient descent step
        with torch.no_grad():
            old_pos = pos.clone()
            pos -= learning_rate * pos.grad
            
            # Check convergence
            delta = torch.norm(pos - old_pos)
            if delta < tolerance:
                converged = True
                # Add final point
                final_energy = energy_module(pos.unsqueeze(0)).squeeze()
                path_points.append((
                    float(pos[0].item()),
                    float(pos[1].item()),
                    float(final_energy.item()),
                    0.0  # Gradient magnitude at convergence
                ))
                break
    
    end_point = (float(pos[0].item()), float(pos[1].item()))
    return path_points, end_point, step + 1, converged


def sample_line(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_samples: int
) -> List[Tuple[float, float]]:
    """
    Sample points along a line segment.
    
    Args:
        start: Starting (x, y) coordinates
        end: Ending (x, y) coordinates
        num_samples: Number of points to sample
        
    Returns:
        List of (x, y) coordinate tuples
        
    Example:
        >>> points = sample_line((-1.0, 0.0), (1.0, 0.0), 50)
        >>> len(points)
        50
    """
    start_x, start_y = start
    end_x, end_y = end
    
    # Generate linearly spaced points
    t_values = np.linspace(0, 1, num_samples)
    
    points = []
    for t in t_values:
        x = start_x + t * (end_x - start_x)
        y = start_y + t * (end_y - start_y)
        points.append((x, y))
    
    return points


def detect_basin_crossings(
    sample_points_2d: List[Tuple[float, float]],
    basins: List
) -> List[int]:
    """
    Detect which basins are crossed along a line of sample points.
    
    Args:
        sample_points_2d: List of (x, y) coordinates along the line
        basins: List of Basin objects
        
    Returns:
        List of unique basin IDs crossed by the line
        
    Example:
        >>> crossed = detect_basin_crossings(points, basins)
        >>> print(f"Crossed {len(crossed)} basins")
    """
    if not basins:
        return []
    
    crossed_ids = set()
    
    for x, y in sample_points_2d:
        for basin in basins:
            if basin.contains_point(x, y):
                crossed_ids.add(basin.pattern_id)
                break  # Point can only be in one basin
    
    return sorted(list(crossed_ids))
