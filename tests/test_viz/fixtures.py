"""
Test fixtures for energy visualization tests.

Provides mock MemoryEnergy modules and synthetic data for testing.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest


class MockMemoryEnergy(nn.Module):
    """
    Mock MemoryEnergy module for testing visualization.
    
    Creates a simple energy function with known memory patterns.
    """
    
    def __init__(self, n_mem: int = 4, latent_dim: int = 2, beta: float = 5.0):
        super().__init__()
        self.n_mem = n_mem
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Create memory patterns in a grid pattern
        if latent_dim == 2:
            positions = [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0]
            ][:n_mem]
        else:
            # Random positions for higher dimensions
            positions = np.random.randn(n_mem, latent_dim).tolist()
        
        self.memory = nn.Parameter(
            torch.tensor(positions, dtype=torch.float32),
            requires_grad=True
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for given latent coordinates.
        
        Args:
            z: Tensor of shape (batch, latent_dim) or (latent_dim,)
            
        Returns:
            Energy values of shape (batch,) or scalar
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Distance to each memory pattern
        # z: (batch, latent_dim), memory: (n_mem, latent_dim)
        dists = torch.cdist(z, self.memory)  # (batch, n_mem)
        
        # Energy: -log(sum(exp(beta * similarity)))
        # similarity = -squared distance
        similarities = -dists ** 2
        energy = -torch.logsumexp(self.beta * similarities, dim=-1)
        
        return energy.squeeze()
    
    def get_memory_positions(self) -> np.ndarray:
        """Get memory pattern positions as numpy array."""
        return self.memory.detach().cpu().numpy()


@pytest.fixture
def mock_energy_2d():
    """Fixture providing a 2D mock energy module."""
    return MockMemoryEnergy(n_mem=4, latent_dim=2, beta=5.0)


@pytest.fixture
def mock_energy_highdim():
    """Fixture providing a high-dimensional mock energy module."""
    return MockMemoryEnergy(n_mem=8, latent_dim=10, beta=3.0)


@pytest.fixture
def sample_grid_data():
    """Fixture providing a sample grid for testing."""
    from aps.viz.utils import create_grid
    bounds = (-2.0, 2.0, -2.0, 2.0)
    return create_grid(bounds, resolution=50)


@pytest.fixture
def sample_landscape(mock_energy_2d, sample_grid_data):
    """Fixture providing a complete sample EnergyLandscape."""
    from aps.viz.data_structures import EnergyLandscape, MemoryPattern
    
    grid_x, grid_y = sample_grid_data
    
    # Compute energies
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    coords = torch.tensor(
        np.stack([flat_x, flat_y], axis=1),
        dtype=torch.float32
    )
    
    with torch.no_grad():
        energies = mock_energy_2d(coords).numpy()
    
    energy_values = energies.reshape(grid_x.shape)
    
    # Create memory patterns
    mem_pos = mock_energy_2d.get_memory_positions()
    memory_patterns = [
        MemoryPattern(
            id=i,
            position_latent=mem_pos[i],
            position_2d=(float(mem_pos[i, 0]), float(mem_pos[i, 1])),
            energy=float(mock_energy_2d(torch.tensor(mem_pos[i], dtype=torch.float32)).item()),
            beta_weight=5.0
        )
        for i in range(len(mem_pos))
    ]
    
    return EnergyLandscape(
        grid_x=grid_x,
        grid_y=grid_y,
        energy_values=energy_values,
        memory_patterns=memory_patterns,
        latent_dim=2,
        projection_method=None,
        bounds=(-2.0, 2.0, -2.0, 2.0)
    )
