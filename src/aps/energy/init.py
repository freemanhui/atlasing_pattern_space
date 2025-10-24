"""
Memory Pattern Initialization Strategies

Provides various initialization methods for memory patterns in energy models.
"""

import torch
import numpy as np
from typing import Tuple


def random_init(n_mem: int, latent_dim: int, scale: float = 0.5) -> torch.Tensor:
    """
    Random Gaussian initialization.
    
    Args:
        n_mem: Number of memory patterns
        latent_dim: Dimensionality of patterns
        scale: Standard deviation of Gaussian
    
    Returns:
        Memory patterns (n_mem, latent_dim)
    """
    return torch.randn(n_mem, latent_dim) * scale


def grid_init(n_mem: int, latent_dim: int, bounds: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """
    Grid initialization (evenly spaced).
    
    Works best for 2D and 3D. For higher dimensions, falls back to random.
    
    Args:
        n_mem: Number of memory patterns
        latent_dim: Dimensionality of patterns
        bounds: (min, max) bounds for grid
    
    Returns:
        Memory patterns (n_mem, latent_dim)
    """
    if latent_dim == 1:
        # 1D: evenly spaced line
        points = torch.linspace(bounds[0], bounds[1], n_mem).unsqueeze(1)
    
    elif latent_dim == 2:
        # 2D: rectangular grid
        side = int(np.ceil(np.sqrt(n_mem)))
        x = torch.linspace(bounds[0], bounds[1], side)
        y = torch.linspace(bounds[0], bounds[1], side)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_mem]
    
    elif latent_dim == 3:
        # 3D: cubic grid
        side = int(np.ceil(n_mem ** (1/3)))
        x = torch.linspace(bounds[0], bounds[1], side)
        y = torch.linspace(bounds[0], bounds[1], side)
        z = torch.linspace(bounds[0], bounds[1], side)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)[:n_mem]
    
    else:
        # Higher dimensions: use random init
        print(f"Warning: grid_init not optimal for {latent_dim}D, using random")
        points = random_init(n_mem, latent_dim)
    
    return points


def cube_corners_init(latent_dim: int) -> torch.Tensor:
    """
    Initialize patterns at hypercube corners.
    
    Creates 2^latent_dim patterns at {-1, 1}^latent_dim vertices.
    
    Args:
        latent_dim: Dimensionality (2 or 3 recommended)
    
    Returns:
        Memory patterns (2^latent_dim, latent_dim)
    """
    n_mem = 2 ** latent_dim
    
    if latent_dim > 10:
        raise ValueError(f"cube_corners_init creates {n_mem} patterns for {latent_dim}D, too many!")
    
    # Generate all binary combinations
    indices = torch.arange(n_mem)
    binary = ((indices.unsqueeze(1) >> torch.arange(latent_dim)) & 1).float()
    
    # Map {0, 1} to {-1, 1}
    points = binary * 2 - 1
    
    return points


def sphere_init(n_mem: int, latent_dim: int, radius: float = 1.0) -> torch.Tensor:
    """
    Initialize patterns on hypersphere surface.
    
    Uses Fibonacci sphere algorithm for uniform distribution.
    
    Args:
        n_mem: Number of memory patterns
        latent_dim: Dimensionality
        radius: Sphere radius
    
    Returns:
        Memory patterns (n_mem, latent_dim)
    """
    if latent_dim == 2:
        # 2D: evenly spaced angles
        angles = torch.linspace(0, 2 * np.pi, n_mem + 1)[:-1]
        points = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1)
    
    elif latent_dim == 3:
        # 3D: Fibonacci sphere
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n_mem):
            y = 1 - (i / float(n_mem - 1)) * 2  # y from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)
            
            theta = phi * i
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            points.append([x, y, z])
        
        points = torch.tensor(points, dtype=torch.float32) * radius
    
    else:
        # Higher dimensions: sample from Gaussian and normalize
        points = torch.randn(n_mem, latent_dim)
        points = points / torch.norm(points, dim=1, keepdim=True) * radius
    
    return points


def kmeans_init(data: torch.Tensor, n_mem: int, max_iter: int = 100) -> torch.Tensor:
    """
    K-means clustering to initialize patterns from data.
    
    Args:
        data: Data points (n_samples, latent_dim)
        n_mem: Number of clusters/patterns
        max_iter: Maximum K-means iterations
    
    Returns:
        Memory patterns (n_mem, latent_dim) = cluster centers
    """
    n_samples, latent_dim = data.shape
    
    # Initialize centroids randomly from data
    indices = torch.randperm(n_samples)[:n_mem]
    centroids = data[indices].clone()
    
    for _ in range(max_iter):
        # Assign points to nearest centroid
        dists = torch.cdist(data, centroids)
        assignments = torch.argmin(dists, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_mem):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = data[mask].mean(dim=0)
            else:
                # Empty cluster: reinitialize
                new_centroids[k] = data[torch.randint(n_samples, (1,))]
        
        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        
        centroids = new_centroids
    
    return centroids


def hierarchical_init(
    n_mem: int,
    latent_dim: int,
    n_levels: int = 2,
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Hierarchical initialization with nested grids.
    
    Creates patterns at multiple scales for multi-resolution basins.
    
    Args:
        n_mem: Number of memory patterns
        latent_dim: Dimensionality
        n_levels: Number of hierarchy levels
        bounds: (min, max) bounds
    
    Returns:
        Memory patterns (n_mem, latent_dim)
    """
    patterns = []
    per_level = n_mem // n_levels
    
    for level in range(n_levels):
        # Scale decreases with level
        scale = (bounds[1] - bounds[0]) / (2 ** level)
        level_bounds = (-scale, scale)
        
        level_patterns = grid_init(per_level, latent_dim, level_bounds)
        patterns.append(level_patterns)
    
    # Handle remainder
    remainder = n_mem - per_level * n_levels
    if remainder > 0:
        patterns.append(random_init(remainder, latent_dim, scale=0.1))
    
    return torch.cat(patterns, dim=0)


def pca_init(data: torch.Tensor, n_mem: int) -> torch.Tensor:
    """
    Initialize along principal components of data.
    
    Places patterns along the major axes of variation.
    
    Args:
        data: Data points (n_samples, latent_dim)
        n_mem: Number of patterns
    
    Returns:
        Memory patterns (n_mem, latent_dim)
    """
    # Center data
    centered = data - data.mean(dim=0)
    
    # Compute PCA
    U, S, Vh = torch.pca_lowrank(centered, q=min(data.shape))
    
    # Place patterns along principal components
    patterns = []
    latent_dim = data.shape[1]
    per_component = n_mem // latent_dim
    
    for i in range(latent_dim):
        # Spread along i-th component
        U[:, i]
        scales = torch.linspace(-S[i].item(), S[i].item(), per_component)
        
        for scale in scales:
            pattern = torch.zeros(latent_dim)
            pattern[i] = scale
            patterns.append(pattern)
    
    patterns = torch.stack(patterns[:n_mem])
    
    return patterns


# Registry of initialization methods
INIT_REGISTRY = {
    'random': random_init,
    'grid': grid_init,
    'cube': cube_corners_init,
    'sphere': sphere_init,
    'kmeans': kmeans_init,
    'hierarchical': hierarchical_init,
    'pca': pca_init
}


def get_initializer(method: str):
    """
    Get initialization function by name.
    
    Args:
        method: Name of initialization method
    
    Returns:
        Initialization function
    
    Raises:
        ValueError if method not found
    """
    if method not in INIT_REGISTRY:
        available = ', '.join(INIT_REGISTRY.keys())
        raise ValueError(f"Unknown init method '{method}'. Available: {available}")
    
    return INIT_REGISTRY[method]
