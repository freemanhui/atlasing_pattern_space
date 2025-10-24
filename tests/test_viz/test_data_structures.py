"""
Unit tests for data structure serialization and methods.
"""

import numpy as np
import pytest
import tempfile
import os
from aps.viz.data_structures import (
    EnergyLandscape,
    MemoryPattern,
    Basin,
    Trajectory,
    CrossSection
)


def test_memory_pattern_creation():
    """Test MemoryPattern dataclass creation."""
    pattern = MemoryPattern(
        id=0,
        position_latent=np.array([1.0, 2.0]),
        position_2d=(1.0, 2.0),
        energy=-5.0,
        beta_weight=3.0
    )
    assert pattern.id == 0
    assert pattern.energy == -5.0
    assert pattern.position_2d == (1.0, 2.0)


def test_basin_contains_point():
    """Test Basin.contains_point() method."""
    basin = Basin(
        pattern_id=0,
        center=(0.0, 0.0),
        depth=2.0,
        points={(0.0, 0.0), (0.1, 0.1)}
    )
    assert basin.contains_point(0.0, 0.0)
    assert not basin.contains_point(5.0, 5.0)


def test_trajectory_get_path_xy():
    """Test Trajectory.get_path_xy() method."""
    path_points = [
        (0.0, 0.0, 1.0, 0.5),
        (0.1, 0.1, 0.8, 0.4),
        (0.2, 0.2, 0.6, 0.3)
    ]
    traj = Trajectory(
        id=0,
        start_point=(0.0, 0.0),
        end_point=(0.2, 0.2),
        path_points=path_points,
        destination_basin_id=1,
        num_steps=3,
        converged=True
    )
    
    x, y = traj.get_path_xy()
    assert len(x) == 3
    assert len(y) == 3
    assert x[0] == 0.0
    assert y[2] == 0.2


def test_trajectory_empty_path():
    """Test Trajectory.get_path_xy() with empty path."""
    traj = Trajectory(
        id=0,
        start_point=(0.0, 0.0),
        end_point=(0.0, 0.0),
        path_points=[],
        destination_basin_id=0,
        num_steps=0,
        converged=False
    )
    
    x, y = traj.get_path_xy()
    assert len(x) == 0
    assert len(y) == 0


def test_cross_section_to_dataframe():
    """Test CrossSection.to_dataframe() method."""
    sample_points = [
        (0.0, 1.0),
        (0.5, 0.8),
        (1.0, 1.2)
    ]
    cross_section = CrossSection(
        id=0,
        start_point=(0.0, 0.0),
        end_point=(1.0, 1.0),
        sample_points=sample_points,
        num_samples=3,
        basins_crossed=[0, 1]
    )
    
    df = cross_section.to_dataframe()
    assert df.shape == (3, 2)
    assert 'distance' in df.columns
    assert 'energy' in df.columns
    assert df['distance'].iloc[0] == 0.0
    assert df['energy'].iloc[1] == 0.8


def test_energy_landscape_get_energy_at():
    """Test EnergyLandscape.get_energy_at() interpolation."""
    # Create simple 3x3 grid
    x = np.array([[-1.0, 0.0, 1.0],
                  [-1.0, 0.0, 1.0],
                  [-1.0, 0.0, 1.0]])
    y = np.array([[-1.0, -1.0, -1.0],
                  [0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0]])
    energy = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
    
    landscape = EnergyLandscape(
        grid_x=x,
        grid_y=y,
        energy_values=energy,
        memory_patterns=[],
        latent_dim=2,
        projection_method=None,
        bounds=(-1.0, 1.0, -1.0, 1.0)
    )
    
    # Test exact grid point
    assert landscape.get_energy_at(0.0, 0.0) == pytest.approx(5.0)
    
    # Test interpolation
    result = landscape.get_energy_at(0.5, 0.0)
    assert 5.0 < result < 6.0


def test_energy_landscape_serialization():
    """Test EnergyLandscape save/load functionality."""
    # Create landscape
    grid_x = np.array([[0.0, 1.0], [0.0, 1.0]])
    grid_y = np.array([[0.0, 0.0], [1.0, 1.0]])
    energy = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    pattern = MemoryPattern(
        id=0,
        position_latent=np.array([0.5, 0.5]),
        position_2d=(0.5, 0.5),
        energy=-1.0
    )
    
    landscape = EnergyLandscape(
        grid_x=grid_x,
        grid_y=grid_y,
        energy_values=energy,
        memory_patterns=[pattern],
        latent_dim=2,
        projection_method="pca",
        bounds=(0.0, 1.0, 0.0, 1.0)
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        filepath = f.name
    
    try:
        landscape.save(filepath)
        
        # Load and verify
        loaded = EnergyLandscape.load(filepath)
        assert loaded.latent_dim == 2
        assert loaded.projection_method == "pca"
        assert loaded.bounds == (0.0, 1.0, 0.0, 1.0)
        assert np.array_equal(loaded.grid_x, grid_x)
        assert np.array_equal(loaded.energy_values, energy)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_energy_landscape_to_dict():
    """Test EnergyLandscape.to_dict() method."""
    pattern = MemoryPattern(
        id=0,
        position_latent=np.array([0.0, 0.0]),
        position_2d=(0.0, 0.0),
        energy=-5.0
    )
    
    landscape = EnergyLandscape(
        grid_x=np.zeros((2, 2)),
        grid_y=np.zeros((2, 2)),
        energy_values=np.ones((2, 2)),
        memory_patterns=[pattern],
        latent_dim=2,
        projection_method=None,
        bounds=(-1.0, 1.0, -1.0, 1.0)
    )
    
    data = landscape.to_dict()
    assert 'grid_x' in data
    assert 'memory_patterns' in data
    assert len(data['memory_patterns']) == 1
    assert data['memory_patterns'][0]['id'] == 0
