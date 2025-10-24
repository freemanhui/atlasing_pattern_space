"""
Data structures for energy landscape visualization.

This module defines the core data structures used to represent energy landscapes,
basins, trajectories, and related entities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class MemoryPattern:
    """
    Represents a learned attractor (memory) from the MemoryEnergy module.
    
    Attributes:
        id: Unique identifier (integer index)
        position_latent: Coordinates in original latent space (n-dimensional)
        position_2d: Coordinates in visualization space (2D projection if needed)
        energy: Energy value at this location (should be local minimum)
        beta_weight: Sharpness parameter for this basin
    """
    id: int
    position_latent: np.ndarray
    position_2d: Tuple[float, float]
    energy: float
    beta_weight: float = 1.0


@dataclass
class Basin:
    """
    Represents a region of attraction around a memory pattern.
    
    Attributes:
        pattern_id: ID of associated MemoryPattern
        center: 2D coordinates of pattern
        depth: Energy difference between basin center and rim
        points: Set of grid indices belonging to this basin
        boundary: Approximate boundary contour (optional)
    """
    pattern_id: int
    center: Tuple[float, float]
    depth: float
    points: Set[Tuple[int, int]] = field(default_factory=set)
    boundary: Optional[np.ndarray] = None
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if point is within basin.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is in basin
        """
        # Simple implementation: check if in points set
        # More sophisticated versions could use boundary
        return any(abs(px - x) < 0.01 and abs(py - y) < 0.01 
                   for px, py in self.points)


@dataclass
class Point:
    """
    Represents a specific location in latent space with computed properties.
    
    Attributes:
        coordinates_2d: (x, y) position in visualization
        coordinates_latent: Position in original latent space (if applicable)
        energy: Computed energy value
        nearest_pattern_id: ID of closest MemoryPattern
        nearest_pattern_distance: Distance to nearest pattern
        basin_id: Which basin this point belongs to (via gradient descent)
    """
    coordinates_2d: Tuple[float, float]
    energy: float
    nearest_pattern_id: Optional[int] = None
    nearest_pattern_distance: Optional[float] = None
    basin_id: Optional[int] = None
    coordinates_latent: Optional[np.ndarray] = None


@dataclass
class Trajectory:
    """
    Represents a gradient descent path from start to basin.
    
    Attributes:
        id: Unique identifier
        start_point: Starting coordinates (2D)
        end_point: Final coordinates (2D)
        path_points: Sequence of (x, y, energy, gradient_magnitude) tuples
        destination_basin_id: Basin where trajectory converged
        num_steps: Number of gradient descent iterations
        converged: Boolean indicating if it reached a basin
    """
    id: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    path_points: List[Tuple[float, float, float, float]]
    destination_basin_id: int
    num_steps: int
    converged: bool
    
    def get_path_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract x and y coordinates for plotting.
        
        Returns:
            Tuple of (x_array, y_array)
        """
        if not self.path_points:
            return np.array([]), np.array([])
        x = np.array([p[0] for p in self.path_points])
        y = np.array([p[1] for p in self.path_points])
        return x, y


@dataclass
class CrossSection:
    """
    Represents a 1D slice through the energy landscape.
    
    Attributes:
        id: Unique identifier
        start_point: Starting (x, y) coordinates
        end_point: Ending (x, y) coordinates
        sample_points: List of (distance_along_line, energy) tuples
        num_samples: Number of points sampled along the line
        basins_crossed: List of basin IDs intersected by this line
    """
    id: int
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    sample_points: List[Tuple[float, float]]
    num_samples: int
    basins_crossed: List[int] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with columns: distance, energy
        """
        return pd.DataFrame(
            self.sample_points,
            columns=['distance', 'energy']
        )


@dataclass
class EnergyLandscape:
    """
    Computed energy surface over 2D grid.
    
    Attributes:
        grid_x: 2D array of x coordinates
        grid_y: 2D array of y coordinates
        energy_values: 2D array of energy values
        memory_patterns: List of MemoryPattern objects
        latent_dim: Original dimensionality of latent space
        projection_method: If latent_dim > 2, the method used (PCA, t-SNE, or None)
        bounds: Min/max coordinates for x and y axes (xmin, xmax, ymin, ymax)
    """
    grid_x: np.ndarray
    grid_y: np.ndarray
    energy_values: np.ndarray
    memory_patterns: List[MemoryPattern]
    latent_dim: int
    projection_method: Optional[str]
    bounds: Tuple[float, float, float, float]
    
    def get_energy_at(self, x: float, y: float) -> float:
        """
        Interpolate energy value at arbitrary (x, y) point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Interpolated energy value
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        x_1d = self.grid_x[0, :]
        y_1d = self.grid_y[:, 0]
        interpolator = RegularGridInterpolator(
            (y_1d, x_1d),
            self.energy_values,
            bounds_error=False,
            fill_value=None
        )
        
        return float(interpolator((y, x)))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for export.
        
        Returns:
            Dictionary representation
        """
        return {
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'energy_values': self.energy_values,
            'memory_patterns': [
                {
                    'id': mp.id,
                    'position_2d': mp.position_2d,
                    'energy': mp.energy
                }
                for mp in self.memory_patterns
            ],
            'latent_dim': self.latent_dim,
            'projection_method': self.projection_method,
            'bounds': self.bounds
        }
    
    def save(self, filepath: str) -> None:
        """
        Save landscape to .npz file.
        
        Args:
            filepath: Path to save file
        """
        data = self.to_dict()
        np.savez(
            filepath,
            grid_x=data['grid_x'],
            grid_y=data['grid_y'],
            energy_values=data['energy_values'],
            latent_dim=data['latent_dim'],
            projection_method=data['projection_method'],
            bounds=data['bounds']
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'EnergyLandscape':
        """
        Load landscape from .npz file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded EnergyLandscape object
        """
        data = np.load(filepath, allow_pickle=True)
        return cls(
            grid_x=data['grid_x'],
            grid_y=data['grid_y'],
            energy_values=data['energy_values'],
            memory_patterns=[],  # Not serialized in basic version
            latent_dim=int(data['latent_dim']),
            projection_method=str(data['projection_method']) if data['projection_method'] else None,
            bounds=tuple(data['bounds'])
        )
