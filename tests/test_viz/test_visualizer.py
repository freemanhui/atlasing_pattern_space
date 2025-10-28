"""
Tests for EnergyLandscapeVisualizer (User Story 1 MVP).

Tests written FIRST following TDD approach.
"""

import time
import numpy as np
import pytest


def test_visualizer_initialization(mock_energy_2d):
    """T017: Contract test for EnergyLandscapeVisualizer initialization."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    viz = EnergyLandscapeVisualizer(
        energy_module=mock_energy_2d,
        latent_dim=2,
        resolution=50
    )
    
    assert viz.energy_module is mock_energy_2d
    assert viz.latent_dim == 2
    assert viz.resolution == 50
    assert viz.projection_method is None  # 2D doesn't need projection


def test_visualizer_invalid_params(mock_energy_2d):
    """T017: Test initialization with invalid parameters."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    with pytest.raises(ValueError):
        EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=1)
    
    with pytest.raises(ValueError):
        EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2, resolution=5)


def test_compute_landscape(mock_energy_2d):
    """T018: Contract test for compute_landscape() method."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2, resolution=50)
    landscape = viz.compute_landscape()
    
    assert landscape is not None
    assert landscape.grid_x.shape == (50, 50)
    assert landscape.grid_y.shape == (50, 50)
    assert landscape.energy_values.shape == (50, 50)
    assert len(landscape.memory_patterns) == 4  # mock has 4 patterns
    assert landscape.latent_dim == 2
    assert landscape.projection_method is None  # 2D doesn't need projection


def test_compute_landscape_with_custom_bounds(mock_energy_2d):
    """T018: Test compute_landscape with custom bounds."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
    landscape = viz.compute_landscape(bounds=(-1.0, 1.0, -1.0, 1.0))
    
    assert landscape.bounds == (-1.0, 1.0, -1.0, 1.0)
    xmin, xmax = landscape.grid_x.min(), landscape.grid_x.max()
    assert abs(xmin - (-1.0)) < 0.1
    assert abs(xmax - 1.0) < 0.1


def test_energy_computation_accuracy(mock_energy_2d):
    """T020: Unit test for energy computation accuracy."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2, resolution=50)
    landscape = viz.compute_landscape()
    
    # Test that energy is computed for all grid points
    assert landscape.energy_values.shape == (50, 50)
    assert not np.isnan(landscape.energy_values).any()
    assert not np.isinf(landscape.energy_values).any()
    
    # Test that memory patterns are local minima (have lower energy than average)
    avg_energy = landscape.energy_values.mean()
    for mp in landscape.memory_patterns:
        # Memory patterns should have lower energy than average
        assert mp.energy < avg_energy, f"Pattern {mp.id} energy {mp.energy} >= average {avg_energy}"


def test_performance_100x100_grid(mock_energy_2d):
    """T021: Performance test for 100x100 grid (<5 seconds)."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    
    viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2, resolution=100)
    
    start_time = time.time()
    landscape = viz.compute_landscape()
    elapsed = time.time() - start_time
    
    assert elapsed < 5.0, f"Landscape computation took {elapsed:.2f}s > 5s (SC-001 violation)"
    assert landscape.grid_x.shape == (100, 100)


def test_plot_heatmap_returns_figure(sample_landscape):
    """Test that plot_heatmap returns a Figure object."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    from aps.viz.config import VisualizationConfig
    
    viz = EnergyLandscapeVisualizer(None, latent_dim=2)  # energy_module not needed for plotting
    config = VisualizationConfig()
    
    fig = viz.plot_heatmap(sample_landscape, config=config)
    
    # Check that it returns a Plotly figure
    assert fig is not None
    assert hasattr(fig, 'data')  # Plotly figures have 'data' attribute


def test_plot_heatmap_with_memory_markers(sample_landscape):
    """Test that memory patterns are marked on heatmap."""
    from aps.viz.visualizer import EnergyLandscapeVisualizer
    from aps.viz.config import VisualizationConfig
    
    viz = EnergyLandscapeVisualizer(None, latent_dim=2)
    config = VisualizationConfig(show_memory_markers=True)
    
    fig = viz.plot_heatmap(sample_landscape, config=config)
    
    # Check that markers are added (Plotly figure should have scatter trace)
    assert len(fig.data) >= 2  # At least heatmap + markers
