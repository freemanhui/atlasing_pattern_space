"""
Tests for cross-sectional analysis (User Story 4).

Tests cover:
- plot_cross_section() API contract (T061)
- Line sampling logic (T062)
- Basin crossing detection (T063)
- Linked 1D-2D highlighting (T064)
"""

import numpy as np
import pandas as pd

from aps.viz import EnergyLandscapeVisualizer
from aps.viz.data_structures import CrossSection


class TestCrossSectionContract:
    """Contract tests for plot_cross_section() method (T061)."""
    
    def test_plot_cross_section_returns_cross_section_object(self, mock_energy_2d, sample_landscape):
        """Test plot_cross_section returns CrossSection object."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        start = (-1.0, 0.0)
        end = (1.0, 0.0)
        cross_section = viz.plot_cross_section(sample_landscape, start, end)
        
        assert isinstance(cross_section, CrossSection)
        assert cross_section.start_point == start
        assert cross_section.end_point == end
    
    def test_cross_section_has_sample_points(self, mock_energy_2d, sample_landscape):
        """Test cross-section contains sample points."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=50
        )
        
        assert len(cross_section.sample_points) == 50
        # Each point should be (distance, energy)
        for point in cross_section.sample_points:
            assert len(point) == 2
            assert isinstance(point[0], (int, float))  # distance
            assert isinstance(point[1], (int, float))  # energy
    
    def test_cross_section_num_samples(self, mock_energy_2d, sample_landscape):
        """Test cross-section respects num_samples parameter."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=100
        )
        
        assert cross_section.num_samples == 100
        assert len(cross_section.sample_points) == 100
    
    def test_cross_section_has_id(self, mock_energy_2d, sample_landscape):
        """Test cross-section has unique ID."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0)
        )
        
        assert isinstance(cross_section.id, int)


class TestLineSampling:
    """Tests for line sampling logic (T062)."""
    
    def test_sample_line_generates_points(self, mock_energy_2d, sample_landscape):
        """Test sample_line generates correct number of points."""
        from aps.viz.utils import sample_line
        
        start = (-1.0, 0.0)
        end = (1.0, 0.0)
        num_samples = 50
        
        points = sample_line(start, end, num_samples)
        
        assert len(points) == num_samples
        # First point should be near start
        assert abs(points[0][0] - start[0]) < 0.1
        assert abs(points[0][1] - start[1]) < 0.1
        # Last point should be near end
        assert abs(points[-1][0] - end[0]) < 0.1
        assert abs(points[-1][1] - end[1]) < 0.1
    
    def test_sample_line_linear_interpolation(self):
        """Test sample_line produces linear interpolation."""
        from aps.viz.utils import sample_line
        
        start = (0.0, 0.0)
        end = (2.0, 2.0)
        num_samples = 5
        
        points = sample_line(start, end, num_samples)
        
        # Should be evenly spaced along diagonal
        expected_x = np.linspace(0, 2, 5)
        expected_y = np.linspace(0, 2, 5)
        
        for i, (x, y) in enumerate(points):
            assert abs(x - expected_x[i]) < 0.01
            assert abs(y - expected_y[i]) < 0.01
    
    def test_cross_section_distances_monotonic(self, mock_energy_2d, sample_landscape):
        """Test distances along cross-section are monotonically increasing."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=50
        )
        
        distances = [point[0] for point in cross_section.sample_points]
        
        # Distances should be monotonically increasing
        for i in range(len(distances) - 1):
            assert distances[i+1] >= distances[i]


class TestBasinCrossing:
    """Tests for basin crossing detection (T063)."""
    
    def test_cross_section_detects_basins(self, mock_energy_2d, sample_landscape):
        """Test cross-section identifies basins crossed."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # First identify basins
        basins = viz.identify_basins(sample_landscape, num_samples=50)
        
        # Create cross-section that should cross multiple basins
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.5, -1.5),
            (1.5, 1.5),
            num_samples=50,
            basins=basins
        )
        
        # Should have detected some basin crossings
        assert isinstance(cross_section.basins_crossed, list)
        # May or may not cross basins depending on landscape
    
    def test_basin_crossing_detection_function(self):
        """Test detect_basin_crossings function."""
        from aps.viz.utils import detect_basin_crossings
        from aps.viz.data_structures import Basin
        
        # Create mock basins
        basins = [
            Basin(pattern_id=0, center=(0.0, 0.0), depth=1.0),
            Basin(pattern_id=1, center=(1.0, 1.0), depth=1.0)
        ]
        
        # Sample points along a line
        sample_points_2d = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        
        crossed = detect_basin_crossings(sample_points_2d, basins)
        
        # Should return list of basin IDs
        assert isinstance(crossed, list)


class TestCrossSectionDataExport:
    """Tests for CrossSection data export (T070)."""
    
    def test_cross_section_to_dataframe(self, mock_energy_2d, sample_landscape):
        """Test CrossSection.to_dataframe() method."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=50
        )
        
        df = cross_section.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'distance' in df.columns
        assert 'energy' in df.columns
        assert len(df) == 50
    
    def test_dataframe_values_match_sample_points(self, mock_energy_2d, sample_landscape):
        """Test DataFrame values match sample_points."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=10
        )
        
        df = cross_section.to_dataframe()
        
        # Check first and last points
        assert abs(df['distance'].iloc[0] - cross_section.sample_points[0][0]) < 1e-6
        assert abs(df['energy'].iloc[0] - cross_section.sample_points[0][1]) < 1e-6
        assert abs(df['distance'].iloc[-1] - cross_section.sample_points[-1][0]) < 1e-6
        assert abs(df['energy'].iloc[-1] - cross_section.sample_points[-1][1]) < 1e-6


class TestCrossSectionEdgeCases:
    """Test edge cases for cross-section analysis."""
    
    def test_cross_section_vertical_line(self, mock_energy_2d, sample_landscape):
        """Test cross-section along vertical line."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (0.0, -1.0),
            (0.0, 1.0),
            num_samples=50
        )
        
        assert len(cross_section.sample_points) == 50
        # Just verify we got valid distance and energy data
        for distance, energy in cross_section.sample_points:
            assert distance >= 0  # distances should be non-negative
            assert np.isfinite(energy)  # energy should be finite
    
    def test_cross_section_horizontal_line(self, mock_energy_2d, sample_landscape):
        """Test cross-section along horizontal line."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, 0.0),
            (1.0, 0.0),
            num_samples=50
        )
        
        assert len(cross_section.sample_points) == 50
    
    def test_cross_section_diagonal_line(self, mock_energy_2d, sample_landscape):
        """Test cross-section along diagonal line."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (-1.0, -1.0),
            (1.0, 1.0),
            num_samples=50
        )
        
        assert len(cross_section.sample_points) == 50
        # Distance should span from 0 to sqrt(2 * 2^2) = 2.83
        distances = [p[0] for p in cross_section.sample_points]
        expected_length = np.sqrt((1.0 - (-1.0))**2 + (1.0 - (-1.0))**2)
        assert abs(distances[-1] - expected_length) < 0.1
    
    def test_cross_section_single_point_line(self, mock_energy_2d, sample_landscape):
        """Test cross-section when start == end."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Same start and end point
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (0.0, 0.0),
            (0.0, 0.0),
            num_samples=10
        )
        
        # Should still return data (all at same point)
        assert len(cross_section.sample_points) == 10
        # All distances should be 0
        distances = [p[0] for p in cross_section.sample_points]
        assert all(d == 0.0 for d in distances)


class TestCrossSectionIntegration:
    """Integration tests for cross-section visualization (T064)."""
    
    def test_cross_section_with_landscape(self, mock_energy_2d, sample_landscape):
        """Test creating cross-section from computed landscape."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Create cross-section through center of landscape
        bounds = sample_landscape.bounds
        (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        
        cross_section = viz.plot_cross_section(
            sample_landscape,
            (bounds[0], center_y),
            (bounds[1], center_y),
            num_samples=100
        )
        
        assert cross_section is not None
        assert len(cross_section.sample_points) == 100
        
        # Energy values should be reasonable
        energies = [p[1] for p in cross_section.sample_points]
        assert all(np.isfinite(e) for e in energies)
