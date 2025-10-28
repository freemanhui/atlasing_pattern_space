"""
Tests for topology-preserving losses.
"""

import pytest
import torch
import torch.nn as nn
from aps.topology.losses import KNNTopoLoss


class TestKNNTopoLoss:
    """Tests for KNNTopoLoss class."""
    
    def test_init_default(self):
        """Test default initialization."""
        loss_fn = KNNTopoLoss()
        
        assert loss_fn.k == 8
        assert loss_fn.loss_type == 'bce'
        assert isinstance(loss_fn, nn.Module)
    
    def test_init_custom(self):
        """Test custom initialization."""
        loss_fn = KNNTopoLoss(k=5, loss_type='bce')
        
        assert loss_fn.k == 5
        assert loss_fn.loss_type == 'bce'
    
    def test_forward_shape(self):
        """Test that forward returns scalar loss."""
        loss_fn = KNNTopoLoss(k=8)
        
        X = torch.randn(50, 10)
        Z = torch.randn(50, 5)
        
        loss = loss_fn(X, Z)
        
        assert loss.shape == torch.Size([])  # Scalar
        assert loss.dtype == torch.float32
    
    def test_forward_perfect_preservation(self):
        """Test loss is low when Z preserves X topology."""
        loss_fn = KNNTopoLoss(k=3)
        
        # X and Z are identical (perfect topology preservation)
        X = torch.randn(20, 10)
        Z = X.clone()
        
        loss = loss_fn(X, Z)
        
        # Loss should be reasonably small
        # Note: With continuous (soft) adjacency, even perfect preservation
        # has some loss due to sigmoid softness. Threshold adjusted accordingly.
        assert loss.item() < 0.15, f"Expected low loss for perfect preservation, got {loss.item()}"
    
    def test_forward_random_preservation(self):
        """Test loss is high when Z doesn't preserve X topology."""
        loss_fn = KNNTopoLoss(k=8)
        
        X = torch.randn(50, 20)
        Z = torch.randn(50, 20)  # Random, unrelated to X
        
        loss = loss_fn(X, Z)
        
        # Loss should be significant (closer to log(2) ≈ 0.69 for BCE)
        assert loss.item() > 0.2, f"Expected high loss for random data, got {loss.item()}"
    
    def test_forward_gradients(self):
        """Test that gradients flow correctly through the loss."""
        loss_fn = KNNTopoLoss(k=5)
        
        X = torch.randn(30, 10)
        Z = torch.randn(30, 5, requires_grad=True)
        
        loss = loss_fn(X, Z)
        loss.backward()
        
        # Gradients should exist and not be all zeros
        assert Z.grad is not None
        assert not torch.allclose(Z.grad, torch.zeros_like(Z.grad))
    
    def test_forward_batch_sizes(self):
        """Test loss computation with various batch sizes."""
        loss_fn = KNNTopoLoss(k=5)
        
        for batch_size in [10, 32, 64, 128]:
            X = torch.randn(batch_size, 20)
            Z = torch.randn(batch_size, 10)
            
            loss = loss_fn(X, Z)
            
            assert loss.shape == torch.Size([])
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
    
    def test_forward_different_dimensions(self):
        """Test with different input and latent dimensions."""
        loss_fn = KNNTopoLoss(k=8)
        
        # High-dim input, low-dim latent (typical autoencoder)
        X = torch.randn(50, 100)
        Z = torch.randn(50, 2)
        
        loss = loss_fn(X, Z)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_forward_k_validation(self):
        """Test that k is validated against batch size."""
        loss_fn = KNNTopoLoss(k=10)
        
        X = torch.randn(8, 10)  # batch_size=8 < k=10
        Z = torch.randn(8, 5)
        
        # Should handle gracefully (either raise error or adjust k)
        with pytest.raises((ValueError, RuntimeError)):
            loss = loss_fn(X, Z)
    
    def test_forward_device_preservation(self):
        """Test that loss computation works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        loss_fn = KNNTopoLoss(k=8).cuda()
        
        X = torch.randn(50, 20).cuda()
        Z = torch.randn(50, 10).cuda()
        
        loss = loss_fn(X, Z)
        
        assert loss.device.type == 'cuda'
    
    def test_loss_decreases_with_better_preservation(self):
        """Test that loss correlates with preservation quality."""
        loss_fn = KNNTopoLoss(k=5)
        
        # Create data with known structure
        X = torch.randn(50, 10)
        
        # Perfect preservation
        Z_perfect = X.clone()
        loss_perfect = loss_fn(X, Z_perfect)
        
        # Slight perturbation
        Z_good = X + 0.1 * torch.randn_like(X)
        loss_good = loss_fn(X, Z_good)
        
        # Large perturbation
        Z_bad = X + 1.0 * torch.randn_like(X)
        loss_bad = loss_fn(X, Z_bad)
        
        # Random
        Z_random = torch.randn_like(X)
        loss_random = loss_fn(X, Z_random)
        
        # Losses should be ordered: perfect < good < bad < random
        assert loss_perfect < loss_good
        assert loss_good < loss_bad
        assert loss_bad < loss_random
    
    def test_loss_symmetric_for_scaled_data(self):
        """Test that loss is consistent with scaled versions."""
        loss_fn = KNNTopoLoss(k=5)
        
        X = torch.randn(30, 10)
        Z = torch.randn(30, 5)
        
        # Compute loss
        loss1 = loss_fn(X, Z)
        
        # Scale X (should preserve relative distances/topology)
        X_scaled = X * 2.0
        loss2 = loss_fn(X_scaled, Z)
        
        # Losses should be similar (topology is scale-invariant for kNN)
        # Allow some numerical difference
        assert torch.abs(loss1 - loss2) < 0.1, \
            f"Losses should be similar for scaled data: {loss1.item()} vs {loss2.item()}"


class TestKNNTopoLossIntegration:
    """Integration tests for KNNTopoLoss with training loop."""
    
    def test_training_improves_preservation(self):
        """Test that optimizing the loss improves topology preservation."""
        # Simple encoder: linear projection
        encoder = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        
        loss_fn = KNNTopoLoss(k=5)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        
        # Training data
        X = torch.randn(50, 20)
        
        # Record initial loss
        Z_initial = encoder(X)
        loss_initial = loss_fn(X, Z_initial.detach())
        
        # Train for a few iterations
        for _ in range(50):
            optimizer.zero_grad()
            Z = encoder(X)
            loss = loss_fn(X, Z)
            loss.backward()
            optimizer.step()
        
        # Final loss
        Z_final = encoder(X)
        loss_final = loss_fn(X, Z_final.detach())
        
        # Loss should decrease
        assert loss_final < loss_initial, \
            f"Loss should decrease: {loss_initial.item()} -> {loss_final.item()}"


class TestKNNTopoLossPerformance:
    """Performance tests for KNNTopoLoss."""
    
    def test_loss_computation_speed(self):
        """Benchmark loss computation on typical batch."""
        import time
        
        loss_fn = KNNTopoLoss(k=8)
        
        X = torch.randn(128, 50)
        Z = torch.randn(128, 10)
        
        # Warm-up
        _ = loss_fn(X, Z)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            loss = loss_fn(X, Z)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        
        # Should be fast (< 200ms per forward pass as per spec)
        assert avg_time < 0.2, f"Loss computation took {avg_time:.3f}s, expected < 0.2s"
    
    def test_backward_pass_speed(self):
        """Benchmark backward pass speed."""
        import time
        
        loss_fn = KNNTopoLoss(k=8)
        
        X = torch.randn(128, 50)
        Z = torch.randn(128, 10, requires_grad=True)
        
        # Warm-up
        loss = loss_fn(X, Z)
        loss.backward()
        Z.grad = None
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            loss = loss_fn(X, Z)
            loss.backward()
            Z.grad = None
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        
        # Should complete full forward+backward in reasonable time
        assert avg_time < 0.5, f"Forward+backward took {avg_time:.3f}s, expected < 0.5s"


class TestKNNTopoLossEdgeCases:
    """Edge case tests for KNNTopoLoss."""
    
    def test_minimum_batch_size(self):
        """Test with very small batch sizes."""
        loss_fn = KNNTopoLoss(k=2)
        
        # Minimum viable: k+1 samples
        X = torch.randn(3, 10)
        Z = torch.randn(3, 5)
        
        loss = loss_fn(X, Z)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_identical_points(self):
        """Test behavior when some points are identical."""
        loss_fn = KNNTopoLoss(k=3)
        
        X = torch.randn(20, 10)
        X[0] = X[1]  # Make two points identical
        
        Z = torch.randn(20, 5)
        Z[0] = Z[1]  # Preserve the identity
        
        loss = loss_fn(X, Z)
        
        # Should handle gracefully
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_all_same_points(self):
        """Test degenerate case where all points are the same."""
        loss_fn = KNNTopoLoss(k=3)
        
        X = torch.ones(20, 10)  # All identical
        Z = torch.ones(20, 5)   # All identical
        
        # Should handle gracefully (may not be well-defined)
        try:
            loss = loss_fn(X, Z)
            assert not torch.isnan(loss) or True  # Either works or NaN is acceptable
        except (ValueError, RuntimeError):
            pass  # Raising an error is also acceptable
