"""
Tests for interactive event handling (User Story 2).

Tests cover:
- InteractionHandler API contracts (T033)
- Hover response performance < 100ms (T034)
- Click response performance < 100ms (T035)
- Basin depth calculation (T036)
"""

import time
import pytest
import numpy as np

from aps.viz.interactions import InteractionHandler, PointInfo, ComparisonResult
from aps.viz.data_structures import Basin


class TestInteractionHandlerContract:
    """Contract tests for InteractionHandler (T033)."""
    
    def test_init(self):
        """Test InteractionHandler initialization."""
        handler = InteractionHandler()
        assert handler is not None
        assert not handler._compare_mode
        assert len(handler._clicked_points) == 0
    
    def test_on_hover_returns_point_info(self, sample_landscape):
        """Test on_hover returns PointInfo with required fields."""
        handler = InteractionHandler()
        point_info = handler.on_hover(0.0, 0.0, sample_landscape)
        
        assert isinstance(point_info, PointInfo)
        assert point_info.position == (0.0, 0.0)
        assert isinstance(point_info.energy, (int, float))
        # basin_id, nearest_pattern_id may be None
    
    def test_on_hover_with_basins(self, sample_landscape):
        """Test on_hover with basin information."""
        handler = InteractionHandler()
        
        # Create a mock basin
        basins = [
            Basin(
                pattern_id=0,
                center=(0.0, 0.0),
                depth=1.0,
                points={(0.0, 0.0)}
            )
        ]
        
        point_info = handler.on_hover(0.0, 0.0, sample_landscape, basins)
        assert point_info.basin_id is not None
    
    def test_on_click_returns_point_info(self, sample_landscape):
        """Test on_click returns PointInfo."""
        handler = InteractionHandler()
        point_info = handler.on_click(0.5, 0.5, sample_landscape)
        
        assert isinstance(point_info, PointInfo)
        assert point_info.position == (0.5, 0.5)
        assert len(handler._clicked_points) == 1
    
    def test_on_click_replaces_point_in_normal_mode(self, sample_landscape):
        """Test clicking multiple points replaces in normal mode."""
        handler = InteractionHandler()
        
        handler.on_click(0.0, 0.0, sample_landscape)
        assert len(handler._clicked_points) == 1
        
        handler.on_click(1.0, 1.0, sample_landscape)
        assert len(handler._clicked_points) == 1
        assert handler._clicked_points[0].position == (1.0, 1.0)


class TestCompareMode:
    """Tests for compare mode functionality (T045)."""
    
    def test_enable_compare_mode(self, sample_landscape):
        """Test enabling compare mode."""
        handler = InteractionHandler()
        handler.enable_compare_mode()
        assert handler._compare_mode is True
    
    def test_disable_compare_mode(self, sample_landscape):
        """Test disabling compare mode clears points."""
        handler = InteractionHandler()
        handler.enable_compare_mode()
        handler.on_click(0.0, 0.0, sample_landscape)
        
        handler.disable_compare_mode()
        assert handler._compare_mode is False
        assert len(handler._clicked_points) == 0
    
    def test_compare_mode_accumulates_points(self, sample_landscape):
        """Test compare mode accumulates clicked points."""
        handler = InteractionHandler()
        handler.enable_compare_mode()
        
        handler.on_click(0.0, 0.0, sample_landscape)
        handler.on_click(1.0, 1.0, sample_landscape)
        handler.on_click(-1.0, -1.0, sample_landscape)
        
        assert len(handler._clicked_points) == 3
    
    def test_get_comparison_with_no_points(self):
        """Test get_comparison returns None with < 2 points."""
        handler = InteractionHandler()
        assert handler.get_comparison() is None
    
    def test_get_comparison_with_one_point(self, sample_landscape):
        """Test get_comparison returns None with 1 point."""
        handler = InteractionHandler()
        handler.on_click(0.0, 0.0, sample_landscape)
        assert handler.get_comparison() is None
    
    def test_get_comparison_with_multiple_points(self, sample_landscape):
        """Test get_comparison returns ComparisonResult with 2+ points."""
        handler = InteractionHandler()
        handler.enable_compare_mode()
        
        handler.on_click(0.0, 0.0, sample_landscape)
        handler.on_click(1.0, 1.0, sample_landscape)
        
        result = handler.get_comparison()
        assert isinstance(result, ComparisonResult)
        assert len(result.points) == 2
        assert isinstance(result.energy_range, tuple)
        assert len(result.energy_range) == 2
        assert isinstance(result.basin_diversity, int)
    
    def test_clear_clicks(self, sample_landscape):
        """Test clear_clicks removes all clicked points."""
        handler = InteractionHandler()
        handler.enable_compare_mode()
        handler.on_click(0.0, 0.0, sample_landscape)
        handler.on_click(1.0, 1.0, sample_landscape)
        
        handler.clear_clicks()
        assert len(handler._clicked_points) == 0


class TestPerformance:
    """Performance tests for interaction response times."""
    
    def test_hover_performance(self, sample_landscape):
        """Test on_hover completes in < 100ms (SC-002, T034)."""
        handler = InteractionHandler()
        
        # Warm-up
        handler.on_hover(0.0, 0.0, sample_landscape)
        
        # Time multiple hover events
        num_tests = 10
        start = time.time()
        
        for i in range(num_tests):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            handler.on_hover(x, y, sample_landscape)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_tests
        
        # SC-002: < 100ms per interaction
        assert avg_time < 0.1, f"Average hover time {avg_time*1000:.1f}ms exceeds 100ms"
    
    def test_click_performance(self, sample_landscape):
        """Test on_click completes in < 100ms (SC-002, T035)."""
        handler = InteractionHandler()
        
        # Warm-up
        handler.on_click(0.0, 0.0, sample_landscape)
        
        # Time multiple click events
        num_tests = 10
        start = time.time()
        
        for i in range(num_tests):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            handler.on_click(x, y, sample_landscape)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_tests
        
        # SC-002: < 100ms per interaction
        assert avg_time < 0.1, f"Average click time {avg_time*1000:.1f}ms exceeds 100ms"
    
    def test_hover_with_basins_performance(self, sample_landscape):
        """Test on_hover with basin detection stays under 100ms."""
        handler = InteractionHandler()
        
        # Create mock basins
        basins = [
            Basin(
                pattern_id=i,
                center=(i*0.5, i*0.5),
                depth=1.0,
                points={(i*0.5, i*0.5)}
            )
            for i in range(4)
        ]
        
        # Time multiple hover events with basin detection
        num_tests = 10
        start = time.time()
        
        for i in range(num_tests):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            handler.on_hover(x, y, sample_landscape, basins)
        
        elapsed = time.time() - start
        avg_time = elapsed / num_tests
        
        assert avg_time < 0.1, f"Average hover time with basins {avg_time*1000:.1f}ms exceeds 100ms"


class TestNearestPattern:
    """Tests for finding nearest memory pattern."""
    
    def test_find_nearest_pattern_empty(self, mock_energy_2d):
        """Test finding nearest pattern with no patterns."""
        from aps.viz import EnergyLandscapeVisualizer
        
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        landscape = viz.compute_landscape()
        landscape.memory_patterns = []  # Remove patterns
        
        handler = InteractionHandler()
        pattern_id, distance = handler._find_nearest_pattern(0.0, 0.0, landscape)
        
        assert pattern_id is None
        assert distance is None
    
    def test_find_nearest_pattern(self, sample_landscape):
        """Test finding nearest memory pattern."""
        handler = InteractionHandler()
        
        # Find nearest pattern to (0.0, 0.0)
        pattern_id, distance = handler._find_nearest_pattern(0.0, 0.0, sample_landscape)
        
        assert pattern_id is not None
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_point_info_includes_nearest_pattern(self, sample_landscape):
        """Test PointInfo includes nearest pattern information."""
        handler = InteractionHandler()
        point_info = handler.on_hover(0.0, 0.0, sample_landscape)
        
        assert point_info.nearest_pattern_id is not None
        assert point_info.distance_to_pattern is not None
        assert point_info.distance_to_pattern >= 0
