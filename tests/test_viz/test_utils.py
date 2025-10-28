"""
Tests for utility functions (grid generation, bounds inference).
"""

import numpy as np
import pytest
from aps.viz.utils import create_grid, infer_bounds, interpolate_energy


def test_create_grid_shape():
    """T019: Test grid generation creates correct shape."""
    bounds = (-2.0, 2.0, -2.0, 2.0)
    grid_x, grid_y = create_grid(bounds, resolution=50)
    
    assert grid_x.shape == (50, 50)
    assert grid_y.shape == (50, 50)


def test_create_grid_bounds():
    """T019: Test grid respects specified bounds."""
    bounds = (-1.0, 3.0, -2.0, 1.0)
    grid_x, grid_y = create_grid(bounds, resolution=10)
    
    assert grid_x.min() == pytest.approx(-1.0)
    assert grid_x.max() == pytest.approx(3.0)
    assert grid_y.min() == pytest.approx(-2.0)
    assert grid_y.max() == pytest.approx(1.0)


def test_infer_bounds_with_patterns():
    """T019: Test bounds inference from memory positions."""
    positions = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, -1.0]
    ])
    
    bounds = infer_bounds(positions, margin=0.5)
    xmin, xmax, ymin, ymax = bounds
    
    # Should include all patterns plus margin
    assert xmin < -1.0
    assert xmax > 1.0
    assert ymin < -1.0
    assert ymax > 1.0


def test_infer_bounds_empty_patterns():
    """T019: Test bounds inference with no patterns."""
    positions = np.array([]).reshape(0, 2)
    bounds = infer_bounds(positions)
    
    # Should return default bounds
    assert bounds == (-2.0, 2.0, -2.0, 2.0)


def test_infer_bounds_single_pattern():
    """T019: Test bounds inference with single pattern."""
    positions = np.array([[0.5, 0.5]])
    bounds = infer_bounds(positions, margin=0.2)
    
    # Should create reasonable bounds around single point
    xmin, xmax, ymin, ymax = bounds
    assert xmin < 0.5 < xmax
    assert ymin < 0.5 < ymax


def test_interpolate_energy():
    """Test energy interpolation at arbitrary points."""
    # Simple 3x3 grid
    grid_x = np.array([[-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0]])
    grid_y = np.array([[-1.0, -1.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0]])
    energy = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
    
    # Test exact grid point
    result = interpolate_energy(grid_x, grid_y, energy, 0.0, 0.0)
    assert result == pytest.approx(5.0)
    
    # Test interpolation between grid points
    result = interpolate_energy(grid_x, grid_y, energy, 0.5, 0.0)
    assert 5.0 < result < 6.0
