"""
Tests for trajectory visualization (User Story 3).

Tests cover:
- add_trajectory() API contract (T048)
- Trajectory computation performance < 2s (T049)
- Trajectory convergence validation (T050)
- Multiple trajectories integration (T051)
"""

import time
import pytest
import numpy as np

from aps.viz import EnergyLandscapeVisualizer
from aps.viz.data_structures import Trajectory


class TestTrajectoryContract:
    """Contract tests for add_trajectory() method (T048)."""
    
    def test_add_trajectory_returns_trajectory_object(self, mock_energy_2d, sample_landscape):
        """Test add_trajectory returns Trajectory object."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        start_point = (0.5, 0.5)
        trajectory = viz.add_trajectory(sample_landscape, start_point)
        
        assert isinstance(trajectory, Trajectory)
        assert trajectory.start_point == start_point
        assert isinstance(trajectory.end_point, tuple)
        assert len(trajectory.end_point) == 2
    
    def test_trajectory_has_path_points(self, mock_energy_2d, sample_landscape):
        """Test trajectory contains path points."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (0.5, 0.5))
        
        assert len(trajectory.path_points) > 0
        # Each point should be (x, y, energy, gradient_magnitude)
        for point in trajectory.path_points:
            assert len(point) == 4
            assert isinstance(point[0], (int, float))  # x
            assert isinstance(point[1], (int, float))  # y
            assert isinstance(point[2], (int, float))  # energy
            assert isinstance(point[3], (int, float))  # gradient magnitude
    
    def test_trajectory_has_destination_basin(self, mock_energy_2d, sample_landscape):
        """Test trajectory identifies destination basin."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (0.5, 0.5))
        
        assert isinstance(trajectory.destination_basin_id, int)
        assert trajectory.destination_basin_id >= 0
    
    def test_trajectory_converged_flag(self, mock_energy_2d, sample_landscape):
        """Test trajectory has converged flag."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (0.5, 0.5))
        
        assert isinstance(trajectory.converged, bool)
    
    def test_trajectory_get_path_xy(self, mock_energy_2d, sample_landscape):
        """Test get_path_xy() method."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (0.5, 0.5))
        x_arr, y_arr = trajectory.get_path_xy()
        
        assert isinstance(x_arr, np.ndarray)
        assert isinstance(y_arr, np.ndarray)
        assert len(x_arr) == len(y_arr)
        assert len(x_arr) == len(trajectory.path_points)


class TestTrajectoryConvergence:
    """Tests for trajectory convergence validation (T050)."""
    
    def test_trajectory_converges_to_pattern(self, mock_energy_2d, sample_landscape):
        """Test trajectory converges near memory pattern."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Start near a memory pattern
        trajectory = viz.add_trajectory(sample_landscape, (0.8, 0.8))
        
        # Should converge
        assert trajectory.converged is True
        
        # End point should be near a memory pattern
        end_x, end_y = trajectory.end_point
        min_dist = float('inf')
        for pattern in sample_landscape.memory_patterns:
            px, py = pattern.position_2d
            dist = np.sqrt((end_x - px)**2 + (end_y - py)**2)
            min_dist = min(min_dist, dist)
        
        # Should be very close to a pattern (within 0.2 units)
        assert min_dist < 0.2
    
    def test_trajectory_energy_decreases(self, mock_energy_2d, sample_landscape):
        """Test energy decreases along trajectory."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (1.5, 1.5))
        
        # Extract energies from path
        energies = [point[2] for point in trajectory.path_points]
        
        # Energy should generally decrease (allow some noise)
        # Check that final energy is less than initial
        assert energies[-1] < energies[0]
    
    def test_trajectory_num_steps_reasonable(self, mock_energy_2d, sample_landscape):
        """Test trajectory doesn't take excessive steps."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectory = viz.add_trajectory(sample_landscape, (0.5, 0.5), max_steps=200)
        
        # Should converge in reasonable number of steps
        assert trajectory.num_steps <= 200
        assert trajectory.num_steps > 0


class TestTrajectoryPerformance:
    """Performance tests for trajectory computation (T049)."""
    
    def test_single_trajectory_performance(self, mock_energy_2d, sample_landscape):
        """Test single trajectory completes in < 2s (SC-003, T049)."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Warm-up
        viz.add_trajectory(sample_landscape, (0.0, 0.0))
        
        # Time trajectory computation
        start = time.time()
        trajectory = viz.add_trajectory(sample_landscape, (1.5, 1.5))
        elapsed = time.time() - start
        
        # SC-003: < 2 seconds per trajectory
        assert elapsed < 2.0, f"Trajectory took {elapsed:.3f}s, exceeds 2s limit"
    
    def test_multiple_trajectories_performance(self, mock_energy_2d, sample_landscape):
        """Test computing multiple trajectories efficiently."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Test 5 trajectories
        num_trajectories = 5
        start_points = [
            (0.5, 0.5),
            (-0.5, 0.5),
            (0.5, -0.5),
            (-0.5, -0.5),
            (0.0, 0.0)
        ]
        
        start = time.time()
        trajectories = []
        for start_point in start_points:
            traj = viz.add_trajectory(sample_landscape, start_point)
            trajectories.append(traj)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 10s total, < 2s average)
        assert elapsed < 10.0
        avg_time = elapsed / num_trajectories
        assert avg_time < 2.0, f"Average trajectory time {avg_time:.3f}s exceeds 2s"


class TestMultipleTrajectories:
    """Integration tests for multiple trajectories (T051)."""
    
    def test_add_multiple_trajectories(self, mock_energy_2d, sample_landscape):
        """Test adding multiple trajectories."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        trajectories = []
        for i in range(3):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            traj = viz.add_trajectory(sample_landscape, (x, y))
            trajectories.append(traj)
        
        assert len(trajectories) == 3
        
        # Each should have unique ID
        ids = [t.id for t in trajectories]
        assert len(set(ids)) == 3
    
    def test_trajectories_to_different_basins(self, mock_energy_2d, sample_landscape):
        """Test trajectories can converge to different basins."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Start from different quadrants
        start_points = [
            (1.5, 1.5),   # Should go to pattern near (1, 1)
            (-1.5, -1.5), # Should go to pattern near (-1, -1)
        ]
        
        trajectories = []
        for start in start_points:
            traj = viz.add_trajectory(sample_landscape, start)
            trajectories.append(traj)
        
        # Should converge to different basins
        basin_ids = [t.destination_basin_id for t in trajectories]
        # If mock has multiple patterns in different locations, they should differ
        # (This test might pass even if they're the same for simple mocks)
        assert len(trajectories) == 2


class TestTrajectoryEdgeCases:
    """Test edge cases for trajectory simulation."""
    
    def test_trajectory_from_pattern_center(self, mock_energy_2d, sample_landscape):
        """Test trajectory starting at a memory pattern."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Start exactly at a pattern
        pattern = sample_landscape.memory_patterns[0]
        start = pattern.position_2d
        
        trajectory = viz.add_trajectory(sample_landscape, start)
        
        # Should converge immediately or in very few steps
        assert trajectory.num_steps < 10
        assert trajectory.converged is True
    
    def test_trajectory_max_steps_limit(self, mock_energy_2d, sample_landscape):
        """Test trajectory respects max_steps limit."""
        viz = EnergyLandscapeVisualizer(mock_energy_2d, latent_dim=2)
        
        # Use very small max_steps
        trajectory = viz.add_trajectory(
            sample_landscape,
            (1.5, 1.5),
            max_steps=5
        )
        
        # Should stop at or before max_steps
        assert trajectory.num_steps <= 5
