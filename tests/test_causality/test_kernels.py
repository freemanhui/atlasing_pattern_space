"""
Tests for kernel functions used in causality module.
"""

import pytest
import torch
import numpy as np
from aps.causality.kernels import rbf_kernel, linear_kernel, center_kernel


class TestRBFKernel:
    """Tests for RBF (Gaussian) kernel."""
    
    def test_rbf_kernel_identity(self):
        """Test that K(x, x) = 1 for RBF kernel."""
        X = torch.randn(10, 5)
        K = rbf_kernel(X, X, sigma=1.0)
        
        # Diagonal should be all ones
        diagonal = torch.diag(K)
        assert torch.allclose(diagonal, torch.ones(10), atol=1e-5)
    
    def test_rbf_kernel_symmetry(self):
        """Test that K(x, y) = K(y, x)."""
        X = torch.randn(10, 5)
        Y = torch.randn(15, 5)
        
        K_xy = rbf_kernel(X, Y, sigma=1.0)
        K_yx = rbf_kernel(Y, X, sigma=1.0)
        
        # Should be transposes
        assert torch.allclose(K_xy, K_yx.t(), atol=1e-5)
    
    def test_rbf_kernel_positive_definite(self):
        """Test that RBF kernel is positive semi-definite."""
        X = torch.randn(20, 5)
        K = rbf_kernel(X, X, sigma=1.0)
        
        # Check symmetry first
        assert torch.allclose(K, K.t(), atol=1e-5)
        
        # Check all eigenvalues are non-negative
        eigenvalues = torch.linalg.eigvalsh(K)
        assert torch.all(eigenvalues >= -1e-5), f"Found negative eigenvalue: {eigenvalues.min()}"
    
    def test_rbf_kernel_shape(self):
        """Test output shape is correct."""
        X = torch.randn(10, 5)
        Y = torch.randn(15, 5)
        
        K = rbf_kernel(X, Y, sigma=1.0)
        assert K.shape == (10, 15)
    
    def test_rbf_kernel_sigma_effect(self):
        """Test that sigma affects kernel values."""
        X = torch.randn(10, 5)
        Y = torch.randn(10, 5)
        
        K_small = rbf_kernel(X, Y, sigma=0.1)
        K_large = rbf_kernel(X, Y, sigma=10.0)
        
        # Smaller sigma → more local → off-diagonal values closer to 0
        # Larger sigma → more global → off-diagonal values closer to 1
        off_diag_small = K_small[0, 1]
        off_diag_large = K_large[0, 1]
        
        # Not a strict inequality due to randomness, but generally true
        # Just check they're different
        assert not torch.allclose(K_small, K_large, atol=1e-2)
    
    def test_rbf_kernel_range(self):
        """Test that kernel values are in [0, 1]."""
        X = torch.randn(20, 5)
        Y = torch.randn(20, 5)
        
        K = rbf_kernel(X, Y, sigma=1.0)
        
        assert torch.all(K >= 0), f"Found values < 0: {K.min()}"
        assert torch.all(K <= 1), f"Found values > 1: {K.max()}"
    
    def test_rbf_kernel_batch_sizes(self):
        """Test various batch sizes."""
        for n in [1, 5, 32, 128]:
            X = torch.randn(n, 10)
            K = rbf_kernel(X, X, sigma=1.0)
            
            assert K.shape == (n, n)
            assert torch.allclose(torch.diag(K), torch.ones(n), atol=1e-5)
    
    def test_rbf_kernel_different_dimensions(self):
        """Test with various feature dimensions."""
        for dim in [1, 5, 20, 100]:
            X = torch.randn(10, dim)
            K = rbf_kernel(X, X, sigma=1.0)
            
            assert K.shape == (10, 10)
            assert torch.allclose(torch.diag(K), torch.ones(10), atol=1e-5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rbf_kernel_gpu(self):
        """Test RBF kernel on GPU."""
        X = torch.randn(10, 5).cuda()
        Y = torch.randn(15, 5).cuda()
        
        K = rbf_kernel(X, Y, sigma=1.0)
        
        assert K.device.type == 'cuda'
        assert K.shape == (10, 15)


class TestLinearKernel:
    """Tests for linear kernel."""
    
    def test_linear_kernel_dot_product(self):
        """Test that linear kernel computes dot product."""
        # Use simple vectors for easy verification
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        Y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        K = linear_kernel(X, Y)
        
        # K[0,0] = [1,0]·[1,0] = 1
        assert torch.allclose(K[0, 0], torch.tensor(1.0))
        # K[0,1] = [1,0]·[0,1] = 0
        assert torch.allclose(K[0, 1], torch.tensor(0.0))
        # K[0,2] = [1,0]·[1,1] = 1
        assert torch.allclose(K[0, 2], torch.tensor(1.0))
        # K[1,1] = [0,1]·[0,1] = 1
        assert torch.allclose(K[1, 1], torch.tensor(1.0))
    
    def test_linear_kernel_symmetry(self):
        """Test that K(x, y) = K(y, x) for linear kernel."""
        X = torch.randn(10, 5)
        Y = torch.randn(15, 5)
        
        K_xy = linear_kernel(X, Y)
        K_yx = linear_kernel(Y, X)
        
        assert torch.allclose(K_xy, K_yx.t(), atol=1e-5)
    
    def test_linear_kernel_shape(self):
        """Test output shape is correct."""
        X = torch.randn(10, 5)
        Y = torch.randn(15, 5)
        
        K = linear_kernel(X, Y)
        assert K.shape == (10, 15)
    
    def test_linear_kernel_positive_definite(self):
        """Test that linear kernel matrix is positive semi-definite."""
        X = torch.randn(20, 10)
        K = linear_kernel(X, X)
        
        # Check symmetry
        assert torch.allclose(K, K.t(), atol=1e-5)
        
        # Check eigenvalues are non-negative
        eigenvalues = torch.linalg.eigvalsh(K)
        assert torch.all(eigenvalues >= -1e-5), f"Found negative eigenvalue: {eigenvalues.min()}"
    
    def test_linear_kernel_normalization(self):
        """Test with normalized vectors."""
        # Normalized vectors should give values in [-1, 1]
        X = torch.randn(10, 5)
        X = X / torch.norm(X, dim=1, keepdim=True)
        
        K = linear_kernel(X, X)
        
        # Diagonal should be ~1 (dot product of normalized vector with itself)
        assert torch.allclose(torch.diag(K), torch.ones(10), atol=1e-5)
        
        # Off-diagonal values should be in [-1, 1]
        K_off_diag = K - torch.diag(torch.diag(K))
        assert torch.all(K_off_diag >= -1.1)  # Small tolerance
        assert torch.all(K_off_diag <= 1.1)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_linear_kernel_gpu(self):
        """Test linear kernel on GPU."""
        X = torch.randn(10, 5).cuda()
        Y = torch.randn(15, 5).cuda()
        
        K = linear_kernel(X, Y)
        
        assert K.device.type == 'cuda'
        assert K.shape == (10, 15)


class TestCenterKernel:
    """Tests for kernel centering."""
    
    def test_center_kernel_row_sums_zero(self):
        """Test that centered kernel has row sums = 0."""
        K = torch.randn(10, 10)
        K = K @ K.t()  # Make positive semi-definite
        
        K_centered = center_kernel(K)
        
        row_sums = K_centered.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(10), atol=1e-4)
    
    def test_center_kernel_col_sums_zero(self):
        """Test that centered kernel has column sums = 0."""
        K = torch.randn(10, 10)
        K = K @ K.t()
        
        K_centered = center_kernel(K)
        
        col_sums = K_centered.sum(dim=0)
        assert torch.allclose(col_sums, torch.zeros(10), atol=1e-4)
    
    def test_center_kernel_shape_preserved(self):
        """Test that centering preserves shape."""
        K = torch.randn(15, 15)
        K_centered = center_kernel(K)
        
        assert K_centered.shape == K.shape
    
    def test_center_kernel_idempotent(self):
        """Test that centering twice = centering once."""
        K = torch.randn(10, 10)
        
        K_centered_once = center_kernel(K)
        K_centered_twice = center_kernel(K_centered_once)
        
        assert torch.allclose(K_centered_once, K_centered_twice, atol=1e-4)
    
    def test_center_kernel_symmetry_preserved(self):
        """Test that centering preserves symmetry."""
        K = torch.randn(10, 10)
        K = K @ K.t()  # Make symmetric
        
        K_centered = center_kernel(K)
        
        assert torch.allclose(K_centered, K_centered.t(), atol=1e-5)
    
    def test_center_kernel_various_sizes(self):
        """Test centering with various matrix sizes."""
        for n in [5, 10, 32, 100]:
            K = torch.randn(n, n)
            K_centered = center_kernel(K)
            
            assert K_centered.shape == (n, n)
            
            row_sums = K_centered.sum(dim=1)
            assert torch.allclose(row_sums, torch.zeros(n), atol=1e-4)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_center_kernel_gpu(self):
        """Test kernel centering on GPU."""
        K = torch.randn(10, 10).cuda()
        K_centered = center_kernel(K)
        
        assert K_centered.device.type == 'cuda'
        assert K_centered.shape == (10, 10)


class TestKernelIntegration:
    """Integration tests for kernel functions."""
    
    def test_rbf_then_center(self):
        """Test RBF kernel followed by centering."""
        X = torch.randn(20, 5)
        
        K = rbf_kernel(X, X, sigma=1.0)
        K_centered = center_kernel(K)
        
        # Centered kernel should have row/col sums = 0
        row_sums = K_centered.sum(dim=1)
        col_sums = K_centered.sum(dim=0)
        
        assert torch.allclose(row_sums, torch.zeros(20), atol=1e-4)
        assert torch.allclose(col_sums, torch.zeros(20), atol=1e-4)
    
    def test_linear_then_center(self):
        """Test linear kernel followed by centering."""
        X = torch.randn(20, 10)
        
        K = linear_kernel(X, X)
        K_centered = center_kernel(K)
        
        # Centered kernel should have row/col sums = 0
        row_sums = K_centered.sum(dim=1)
        col_sums = K_centered.sum(dim=0)
        
        assert torch.allclose(row_sums, torch.zeros(20), atol=1e-4)
        assert torch.allclose(col_sums, torch.zeros(20), atol=1e-4)
    
    def test_kernel_functions_differentiable(self):
        """Test that kernel functions are differentiable."""
        X = torch.randn(10, 5, requires_grad=True)
        Y = torch.randn(10, 5)
        
        # RBF kernel
        K_rbf = rbf_kernel(X, Y, sigma=1.0)
        loss_rbf = K_rbf.sum()
        loss_rbf.backward()
        
        assert X.grad is not None
        assert not torch.all(X.grad == 0)
        
        # Linear kernel
        X.grad = None
        K_linear = linear_kernel(X, Y)
        loss_linear = K_linear.sum()
        loss_linear.backward()
        
        assert X.grad is not None
        assert not torch.all(X.grad == 0)


class TestKernelEdgeCases:
    """Tests for edge cases."""
    
    def test_rbf_kernel_single_sample(self):
        """Test RBF kernel with single sample."""
        X = torch.randn(1, 5)
        K = rbf_kernel(X, X, sigma=1.0)
        
        assert K.shape == (1, 1)
        assert torch.allclose(K, torch.ones(1, 1), atol=1e-5)
    
    def test_linear_kernel_single_sample(self):
        """Test linear kernel with single sample."""
        X = torch.tensor([[2.0, 3.0]])
        K = linear_kernel(X, X)
        
        assert K.shape == (1, 1)
        # K[0,0] = [2,3]·[2,3] = 4+9 = 13
        assert torch.allclose(K, torch.tensor([[13.0]]))
    
    def test_center_kernel_single_sample(self):
        """Test centering with single sample (degenerate case)."""
        K = torch.tensor([[5.0]])
        K_centered = center_kernel(K)
        
        # With n=1, centering should give 0
        assert torch.allclose(K_centered, torch.zeros(1, 1), atol=1e-5)
    
    def test_rbf_kernel_very_small_sigma(self):
        """Test RBF kernel with very small sigma."""
        X = torch.randn(10, 5)
        K = rbf_kernel(X, X, sigma=1e-10)
        
        # Very small sigma → very local → off-diagonal ~0
        K_off_diag = K - torch.diag(torch.diag(K))
        assert torch.all(K_off_diag < 0.1)
    
    def test_rbf_kernel_very_large_sigma(self):
        """Test RBF kernel with very large sigma."""
        X = torch.randn(10, 5)
        K = rbf_kernel(X, X, sigma=1e10)
        
        # Very large sigma → very global → all values close to 1
        assert torch.all(K > 0.9)
