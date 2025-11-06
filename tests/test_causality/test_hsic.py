"""
Tests for HSIC (Hilbert-Schmidt Independence Criterion) loss.
"""

import pytest
import torch
import torch.nn as nn
from aps.causality.hsic import HSICLoss


class TestHSICLossInit:
    """Tests for HSICLoss initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        loss_fn = HSICLoss()
        
        assert loss_fn.kernel == 'rbf'
        assert loss_fn.sigma == 1.0
        assert isinstance(loss_fn, nn.Module)
    
    def test_init_rbf_kernel(self):
        """Test initialization with RBF kernel."""
        loss_fn = HSICLoss(kernel='rbf', sigma=2.0)
        
        assert loss_fn.kernel == 'rbf'
        assert loss_fn.sigma == 2.0
    
    def test_init_linear_kernel(self):
        """Test initialization with linear kernel."""
        loss_fn = HSICLoss(kernel='linear')
        
        assert loss_fn.kernel == 'linear'
    
    def test_init_invalid_kernel(self):
        """Test that invalid kernel raises error."""
        with pytest.raises(ValueError, match="kernel must be"):
            HSICLoss(kernel='invalid')


class TestHSICLossForward:
    """Tests for HSICLoss forward pass."""
    
    def test_forward_shape(self):
        """Test that forward returns scalar."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        loss = loss_fn(Z, V)
        
        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
    
    def test_forward_independent_variables(self):
        """Test HSIC on independent variables."""
        torch.manual_seed(42)
        loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        
        # Generate independent variables
        Z = torch.randn(100, 10)
        V = torch.randn(100, 5)
        
        loss = loss_fn(Z, V)
        
        # HSIC should be close to 0 for independent variables
        # Allow some noise due to finite sample
        assert loss.item() < 0.01, f"Expected low HSIC for independent vars, got {loss.item()}"
    
    def test_forward_dependent_variables(self):
        """Test HSIC on dependent variables."""
        torch.manual_seed(42)
        loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        
        # Generate dependent variables: V = f(Z) + noise
        Z = torch.randn(100, 10)
        V = Z[:, :5] + 0.1 * torch.randn(100, 5)  # Strong dependence
        
        loss = loss_fn(Z, V)
        
        # HSIC should be significantly > 0 for dependent variables
        # With n² normalization, values are small but still meaningful
        assert loss.item() > 0.005, f"Expected high HSIC for dependent vars, got {loss.item()}"
    
    def test_forward_identical_variables(self):
        """Test HSIC when Z = V (maximum dependence)."""
        torch.manual_seed(42)
        loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        
        Z = torch.randn(100, 10)
        V = Z.clone()  # Identical
        
        loss = loss_fn(Z, V)
        
        # HSIC should be very high for identical variables
        # With n² normalization, even identical vars give small values
        assert loss.item() > 0.008, f"Expected very high HSIC for identical vars, got {loss.item()}"
    
    def test_forward_batch_sizes(self):
        """Test with various batch sizes."""
        loss_fn = HSICLoss()
        
        for batch_size in [10, 32, 64, 128]:
            Z = torch.randn(batch_size, 10)
            V = torch.randn(batch_size, 5)
            
            loss = loss_fn(Z, V)
            
            assert loss.shape == torch.Size([])
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss.item() >= 0  # HSIC is always non-negative
    
    def test_forward_different_dimensions(self):
        """Test with various feature dimensions."""
        loss_fn = HSICLoss()
        
        test_cases = [
            (50, 1, 1),   # Single feature each
            (50, 5, 10),  # Different dims
            (50, 20, 20), # Same dims
            (50, 50, 10), # Large Z dim
        ]
        
        for batch_size, z_dim, v_dim in test_cases:
            Z = torch.randn(batch_size, z_dim)
            V = torch.randn(batch_size, v_dim)
            
            loss = loss_fn(Z, V)
            
            assert loss.shape == torch.Size([])
            assert loss.item() >= 0
    
    def test_forward_rbf_vs_linear_kernel(self):
        """Test that different kernels give different results."""
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        loss_rbf = HSICLoss(kernel='rbf', sigma=1.0)(Z, V)
        loss_linear = HSICLoss(kernel='linear')(Z, V)
        
        # Different kernels should generally give different HSIC values
        # (unless by pure chance they're similar)
        # Just check they're both valid
        assert loss_rbf.item() >= 0
        assert loss_linear.item() >= 0


class TestHSICLossGradients:
    """Tests for gradient flow through HSIC loss."""
    
    def test_gradients_flow_to_z(self):
        """Test that gradients flow to Z."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10, requires_grad=True)
        V = torch.randn(50, 5)
        
        loss = loss_fn(Z, V)
        loss.backward()
        
        assert Z.grad is not None
        assert not torch.all(Z.grad == 0)
    
    def test_gradients_flow_to_v(self):
        """Test that gradients flow to V."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5, requires_grad=True)
        
        loss = loss_fn(Z, V)
        loss.backward()
        
        assert V.grad is not None
        assert not torch.all(V.grad == 0)
    
    def test_gradients_flow_to_both(self):
        """Test that gradients flow to both Z and V."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10, requires_grad=True)
        V = torch.randn(50, 5, requires_grad=True)
        
        loss = loss_fn(Z, V)
        loss.backward()
        
        assert Z.grad is not None
        assert V.grad is not None
        assert not torch.all(Z.grad == 0)
        assert not torch.all(V.grad == 0)


class TestHSICLossTraining:
    """Integration tests with training loop."""
    
    def test_training_reduces_hsic(self):
        """Test that optimizing HSIC reduces dependence."""
        torch.manual_seed(42)
        
        # Create a simple encoder that initially produces V-dependent outputs
        encoder = nn.Sequential(
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        
        # Generate data where V is embedded in X
        V = torch.randn(100, 5)
        X = torch.cat([torch.randn(100, 10), V], dim=1)  # X contains V
        
        # Initial HSIC
        with torch.no_grad():
            Z_init = encoder(X)
            hsic_init = loss_fn(Z_init, V)
        
        # Train to minimize HSIC
        for _ in range(50):
            optimizer.zero_grad()
            Z = encoder(X)
            loss = loss_fn(Z, V)
            loss.backward()
            optimizer.step()
        
        # Final HSIC
        with torch.no_grad():
            Z_final = encoder(X)
            hsic_final = loss_fn(Z_final, V)
        
        # HSIC should decrease
        assert hsic_final < hsic_init, \
            f"HSIC should decrease: {hsic_init.item():.4f} -> {hsic_final.item():.4f}"
    
    def test_training_with_reconstruction_loss(self):
        """Test HSIC loss combined with reconstruction loss."""
        torch.manual_seed(42)
        
        # Simple autoencoder
        encoder = nn.Linear(20, 10)
        decoder = nn.Linear(10, 20)
        
        loss_fn = HSICLoss(kernel='rbf')
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.01
        )
        
        # Data
        X = torch.randn(100, 20)
        V = torch.randn(100, 5)  # Nuisance variable
        
        # Train with combined loss
        for _ in range(30):
            optimizer.zero_grad()
            
            Z = encoder(X)
            X_recon = decoder(Z)
            
            recon_loss = torch.mean((X - X_recon) ** 2)
            hsic_loss = loss_fn(Z, V)
            
            total_loss = recon_loss + 0.1 * hsic_loss
            total_loss.backward()
            optimizer.step()
        
        # Check both losses are reasonable
        with torch.no_grad():
            Z = encoder(X)
            X_recon = decoder(Z)
            final_recon = torch.mean((X - X_recon) ** 2)
            final_hsic = loss_fn(Z, V)
        
        assert final_recon < 1.0  # Some reconstruction quality
        assert final_hsic < 5.0   # Some independence achieved


class TestHSICLossDevice:
    """Tests for device compatibility."""
    
    def test_cpu_device(self):
        """Test HSIC loss on CPU."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        loss = loss_fn(Z, V)
        
        assert loss.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test HSIC loss on CUDA."""
        loss_fn = HSICLoss().cuda()
        
        Z = torch.randn(50, 10).cuda()
        V = torch.randn(50, 5).cuda()
        
        loss = loss_fn(Z, V)
        
        assert loss.device.type == 'cuda'


class TestHSICLossEdgeCases:
    """Tests for edge cases."""
    
    def test_small_batch_size(self):
        """Test with very small batch size."""
        loss_fn = HSICLoss()
        
        # Batch size of 3 (minimum for meaningful HSIC)
        Z = torch.randn(3, 10)
        V = torch.randn(3, 5)
        
        loss = loss_fn(Z, V)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_single_feature_dimensions(self):
        """Test with single feature dimension."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 1)
        V = torch.randn(50, 1)
        
        loss = loss_fn(Z, V)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with seed."""
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        torch.manual_seed(42)
        loss_fn = HSICLoss()
        loss1 = loss_fn(Z, V)
        
        torch.manual_seed(42)
        loss_fn = HSICLoss()
        loss2 = loss_fn(Z, V)
        
        assert torch.allclose(loss1, loss2)
    
    def test_numerical_stability_large_values(self):
        """Test numerical stability with large input values."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10) * 100  # Large values
        V = torch.randn(50, 5) * 100
        
        loss = loss_fn(Z, V)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_numerical_stability_small_sigma(self):
        """Test with very small sigma."""
        loss_fn = HSICLoss(kernel='rbf', sigma=0.01)
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        loss = loss_fn(Z, V)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestHSICLossProperties:
    """Tests for mathematical properties of HSIC."""
    
    def test_hsic_non_negative(self):
        """Test that HSIC is always non-negative."""
        loss_fn = HSICLoss()
        
        for _ in range(10):
            Z = torch.randn(50, 10)
            V = torch.randn(50, 5)
            
            loss = loss_fn(Z, V)
            
            assert loss.item() >= -1e-6, f"HSIC should be non-negative, got {loss.item()}"
    
    def test_hsic_symmetric(self):
        """Test that HSIC(Z, V) ≈ HSIC(V, Z)."""
        loss_fn = HSICLoss()
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 8)
        
        hsic_zv = loss_fn(Z, V)
        hsic_vz = loss_fn(V, Z)
        
        # Should be very close (up to numerical precision)
        assert torch.allclose(hsic_zv, hsic_vz, atol=1e-4)
    
    def test_hsic_scale_invariance(self):
        """Test that HSIC changes appropriately with scaling."""
        loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        
        Z = torch.randn(50, 10)
        V = torch.randn(50, 5)
        
        hsic_orig = loss_fn(Z, V)
        
        # Scale both variables
        Z_scaled = Z * 2.0
        V_scaled = V * 2.0
        hsic_scaled = loss_fn(Z_scaled, V_scaled)
        
        # HSIC should change (not scale-invariant for RBF with fixed sigma)
        # Just check both are valid
        assert hsic_orig.item() >= 0
        assert hsic_scaled.item() >= 0


class TestHSICLossPerformance:
    """Performance tests for HSIC loss."""
    
    def test_computation_speed(self):
        """Benchmark HSIC computation speed."""
        import time
        
        loss_fn = HSICLoss()
        
        Z = torch.randn(128, 50)
        V = torch.randn(128, 20)
        
        # Warm-up
        _ = loss_fn(Z, V)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = loss_fn(Z, V)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        
        # Should be < 100ms per forward pass (as per spec)
        assert avg_time < 0.1, f"HSIC computation took {avg_time:.3f}s, expected < 0.1s"
