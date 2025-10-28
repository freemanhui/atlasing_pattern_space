"""
Tests for topology graph construction utilities.
"""

import pytest
import torch
import numpy as np
from aps.topology.graph import knn_indices, adjacency_from_knn, knn_graph


class TestKNNIndices:
    """Tests for knn_indices function."""
    
    def test_knn_indices_shape(self):
        """Test that knn_indices returns correct shape."""
        X = torch.randn(100, 50)
        k = 8
        indices = knn_indices(X, k)
        
        assert indices.shape == (100, k)
        assert indices.dtype == torch.long
    
    def test_knn_indices_correctness_simple(self):
        """Test knn_indices on simple known data."""
        # Create 3 points in 2D where neighbors are obvious
        X = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        indices = knn_indices(X, k=2)
        
        # Point 0: nearest neighbors should be points 1 and 2
        # Point 1: nearest neighbors should be points 0 and 2
        # Point 2: nearest neighbors should be points 0 and 1
        assert indices.shape == (3, 2)
        
        # All points should have the other 2 as neighbors
        assert set(indices[0].tolist()) == {1, 2}
        assert set(indices[1].tolist()) == {0, 2}
        assert set(indices[2].tolist()) == {0, 1}
    
    def test_knn_indices_excludes_self(self):
        """Test that knn_indices excludes the point itself."""
        X = torch.randn(50, 10)
        k = 5
        indices = knn_indices(X, k)
        
        # No point should have itself as a neighbor
        for i in range(50):
            assert i not in indices[i].tolist()
    
    def test_knn_indices_k_too_large(self):
        """Test that ValueError is raised when k >= n_samples."""
        X = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="k must be < n_samples"):
            knn_indices(X, k=10)
        
        with pytest.raises(ValueError, match="k must be < n_samples"):
            knn_indices(X, k=15)
    
    def test_knn_indices_k_too_small(self):
        """Test that ValueError is raised when k < 1."""
        X = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="k must be >= 1"):
            knn_indices(X, k=0)
        
        with pytest.raises(ValueError, match="k must be >= 1"):
            knn_indices(X, k=-1)
    
    def test_knn_indices_device_preservation(self):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            X_cuda = torch.randn(50, 10).cuda()
            indices = knn_indices(X_cuda, k=5)
            assert indices.device.type == 'cuda'
        
        X_cpu = torch.randn(50, 10)
        indices = knn_indices(X_cpu, k=5)
        assert indices.device.type == 'cpu'
    
    def test_knn_indices_type_error(self):
        """Test that TypeError is raised for non-tensor input."""
        X_np = np.random.randn(10, 5)
        
        with pytest.raises(TypeError, match="X must be a torch.Tensor"):
            knn_indices(X_np, k=3)


class TestAdjacencyFromKNN:
    """Tests for adjacency_from_knn function."""
    
    def test_adjacency_shape(self):
        """Test that adjacency matrix has correct shape."""
        indices = torch.randint(0, 100, (100, 8))
        adj = adjacency_from_knn(indices)
        
        assert adj.shape == (100, 100)
        assert adj.dtype == torch.float32
    
    def test_adjacency_binary(self):
        """Test that adjacency matrix is binary (0s and 1s)."""
        indices = torch.randint(0, 50, (50, 5))
        adj = adjacency_from_knn(indices)
        
        unique_values = torch.unique(adj)
        assert len(unique_values) <= 2
        assert torch.all((adj == 0) | (adj == 1))
    
    def test_adjacency_k_ones_per_row(self):
        """Test that each row has exactly k ones."""
        n_samples = 50
        k = 8
        indices = torch.randint(0, n_samples, (n_samples, k))
        adj = adjacency_from_knn(indices, n_samples=n_samples)
        
        row_sums = adj.sum(dim=1)
        # Each row should have exactly k ones (allowing for duplicates in random indices)
        assert torch.all(row_sums <= k)
    
    def test_adjacency_correctness_simple(self):
        """Test adjacency matrix on simple known indices."""
        # 3 samples, k=2
        indices = torch.tensor([
            [1, 2],  # Sample 0's neighbors are 1 and 2
            [0, 2],  # Sample 1's neighbors are 0 and 2
            [0, 1]   # Sample 2's neighbors are 0 and 1
        ])
        
        adj = adjacency_from_knn(indices)
        
        assert adj.shape == (3, 3)
        
        # Check row 0
        assert adj[0, 0] == 0  # Not self-neighbor
        assert adj[0, 1] == 1  # 1 is neighbor
        assert adj[0, 2] == 1  # 2 is neighbor
        
        # Check row 1
        assert adj[1, 0] == 1
        assert adj[1, 1] == 0
        assert adj[1, 2] == 1
        
        # Check row 2
        assert adj[2, 0] == 1
        assert adj[2, 1] == 1
        assert adj[2, 2] == 0
    
    def test_adjacency_n_samples_parameter(self):
        """Test explicit n_samples parameter."""
        indices = torch.tensor([[1, 2], [0, 2]])  # 2 samples, max index is 2
        
        # Infer n_samples from max(indices.shape[0], max(indices)+1)
        adj1 = adjacency_from_knn(indices)
        assert adj1.shape == (3, 3)  # Inferred as 3 because max index is 2
        
        # Explicit n_samples larger than inferred
        adj2 = adjacency_from_knn(indices, n_samples=5)
        assert adj2.shape == (5, 5)
        
        # First 3x3 block should match adj1
        assert torch.allclose(adj1, adj2[:3, :3])
        
        # Remaining rows/cols should be all zeros
        assert torch.all(adj2[3:, :] == 0)
        assert torch.all(adj2[:, 3:] == 0)
    
    def test_adjacency_n_samples_too_small(self):
        """Test ValueError when n_samples < indices.shape[0]."""
        indices = torch.tensor([[1, 2], [0, 2], [0, 1]])  # 3 samples
        
        with pytest.raises(ValueError, match="n_samples .* must be >="):
            adjacency_from_knn(indices, n_samples=2)
    
    def test_adjacency_type_error(self):
        """Test TypeError for non-tensor input."""
        indices_np = np.array([[1, 2], [0, 2]])
        
        with pytest.raises(TypeError, match="indices must be a torch.Tensor"):
            adjacency_from_knn(indices_np)
    
    def test_adjacency_wrong_ndim(self):
        """Test ValueError for wrong number of dimensions."""
        indices_1d = torch.tensor([1, 2, 3])
        
        with pytest.raises(ValueError, match="indices must be 2D"):
            adjacency_from_knn(indices_1d)


class TestKNNGraph:
    """Tests for knn_graph convenience function."""
    
    def test_knn_graph_shape(self):
        """Test that knn_graph returns correct shape."""
        X = torch.randn(100, 50)
        k = 8
        adj = knn_graph(X, k)
        
        assert adj.shape == (100, 100)
        assert adj.dtype == torch.float32
    
    def test_knn_graph_k_ones_per_row(self):
        """Test that each row has exactly k ones (discrete mode)."""
        X = torch.randn(100, 50)
        k = 8
        adj = knn_graph(X, k, continuous=False)
        
        row_sums = adj.sum(dim=1)
        # Each row should have exactly k ones
        assert torch.allclose(row_sums, torch.full_like(row_sums, float(k)))
    
    def test_knn_graph_symmetric_structure(self):
        """Test knn_graph on simple symmetric data (discrete mode)."""
        # Create data where distances are symmetric
        X = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0]  # Far point
        ])
        
        adj = knn_graph(X, k=2, continuous=False)
        
        # First 3 points should have each other as neighbors (not point 3)
        for i in range(3):
            neighbors = torch.nonzero(adj[i]).squeeze().tolist()
            if not isinstance(neighbors, list):
                neighbors = [neighbors]
            # Point 3 (far point) should not be a neighbor of first 3 points
            assert 3 not in neighbors
    
    def test_knn_graph_integration(self):
        """Integration test: knn_graph (discrete) should equal knn_indices + adjacency_from_knn."""
        X = torch.randn(50, 20)
        k = 5
        
        # Method 1: Use knn_graph with discrete mode
        adj1 = knn_graph(X, k, continuous=False)
        
        # Method 2: Use separate functions
        indices = knn_indices(X, k)
        adj2 = adjacency_from_knn(indices, n_samples=X.shape[0])
        
        # Should be identical
        assert torch.allclose(adj1, adj2)
    
    def test_knn_graph_continuous_mode(self):
        """Test continuous (differentiable) mode of knn_graph."""
        X = torch.randn(50, 20, requires_grad=True)
        k = 5
        
        adj = knn_graph(X, k, continuous=True)
        
        # Shape should be correct
        assert adj.shape == (50, 50)
        
        # Values should be in [0, 1] (sigmoid outputs)
        assert torch.all((adj >= 0) & (adj <= 1))
        
        # Should be differentiable
        loss = adj.sum()
        loss.backward()
        assert X.grad is not None
        
        # Diagonal should be zero (no self-loops)
        assert torch.allclose(torch.diag(adj), torch.zeros(50))
        
        # Each row should have approximately k high values (soft k-NN)
        # With sigmoid, values near 1 indicate neighbors
        high_values = (adj > 0.5).sum(dim=1).float()
        # Allow some variation due to soft boundaries
        assert high_values.mean().item() > k * 0.5  # At least half of k on average


class TestPerformance:
    """Performance tests for graph construction."""
    
    def test_knn_indices_performance(self):
        """Benchmark knn_indices on typical batch size."""
        import time
        
        X = torch.randn(128, 50)  # Typical batch
        k = 8
        
        start = time.time()
        indices = knn_indices(X, k)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second for batch of 128)
        assert elapsed < 1.0, f"knn_indices took {elapsed:.3f}s, expected < 1.0s"
    
    def test_adjacency_performance(self):
        """Benchmark adjacency_from_knn on typical batch size."""
        import time
        
        indices = torch.randint(0, 128, (128, 8))
        
        start = time.time()
        adj = adjacency_from_knn(indices)
        elapsed = time.time() - start
        
        # Should be very fast (< 0.1 second)
        assert elapsed < 0.1, f"adjacency_from_knn took {elapsed:.3f}s, expected < 0.1s"
    
    def test_knn_graph_performance(self):
        """Benchmark full knn_graph on typical batch size."""
        import time
        
        X = torch.randn(128, 50)
        k = 8
        
        start = time.time()
        adj = knn_graph(X, k, continuous=False)  # Use discrete for consistency with original tests
        elapsed = time.time() - start
        
        # Combined should still be fast (< 1 second)
        assert elapsed < 1.0, f"knn_graph took {elapsed:.3f}s, expected < 1.0s"
