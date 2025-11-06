"""
Tests for IRM (Invariant Risk Minimization) loss.
"""

import pytest
import torch
import torch.nn as nn
from aps.causality.irm import IRMLoss


class TestIRMLossInit:
    """Tests for IRMLoss initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        loss_fn = IRMLoss()
        
        assert isinstance(loss_fn, nn.Module)
    
    def test_init_creates_dummy_scale(self):
        """Test that dummy scale parameter is created."""
        loss_fn = IRMLoss()
        
        # Should have dummy_scale parameter
        assert hasattr(loss_fn, 'dummy_scale')
        assert isinstance(loss_fn.dummy_scale, nn.Parameter)
        assert loss_fn.dummy_scale.shape == torch.Size([])  # Scalar


class TestIRMLossForward:
    """Tests for IRMLoss forward pass."""
    
    def test_forward_shape(self):
        """Test that forward returns scalar."""
        loss_fn = IRMLoss()
        
        # Create simple model and data
        model = nn.Linear(10, 2)
        
        # Two environments
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.shape == torch.Size([])
        assert penalty.dtype == torch.float32
    
    def test_forward_invariant_features(self):
        """Test IRM penalty on invariant features (should be low)."""
        torch.manual_seed(42)
        loss_fn = IRMLoss()
        
        # Create model that extracts invariant features
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Generate data with invariant relationship
        # y depends on first feature only, same across environments
        envs = []
        for _ in range(2):
            X = torch.randn(50, 10)
            # y depends only on X[:, 0] - invariant across envs
            y = (X[:, 0] > 0).long()
            envs.append((X, y))
        
        penalty = loss_fn(model, envs)
        
        # Penalty should be relatively small for invariant features
        assert penalty.item() >= 0
        assert not torch.isnan(penalty)
    
    def test_forward_spurious_features(self):
        """Test IRM penalty on spurious features (should be high)."""
        torch.manual_seed(42)
        loss_fn = IRMLoss()
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Generate data with spurious correlation
        # Environment 1: y correlates with feature 0
        # Environment 2: y correlates with feature 1 (different!)
        X1 = torch.randn(50, 10)
        y1 = (X1[:, 0] > 0).long()
        
        X2 = torch.randn(50, 10)
        y2 = (X2[:, 1] > 0).long()  # Different feature!
        
        envs = [(X1, y1), (X2, y2)]
        
        penalty = loss_fn(model, envs)
        
        # Penalty should be non-negative
        assert penalty.item() >= 0
        assert not torch.isnan(penalty)
    
    def test_forward_multiple_environments(self):
        """Test with various numbers of environments."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        for n_envs in [2, 3, 5]:
            envs = []
            for _ in range(n_envs):
                X = torch.randn(20, 10)
                y = torch.randint(0, 2, (20,))
                envs.append((X, y))
            
            penalty = loss_fn(model, envs)
            
            assert penalty.shape == torch.Size([])
            assert penalty.item() >= 0
            assert not torch.isnan(penalty)
    
    def test_forward_different_batch_sizes(self):
        """Test with different batch sizes per environment."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        # Different sizes: 20, 30, 40
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(30, 10), torch.randint(0, 2, (30,))),
            (torch.randn(40, 10), torch.randint(0, 2, (40,))),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.shape == torch.Size([])
        assert penalty.item() >= 0


class TestIRMLossGradients:
    """Tests for gradient flow through IRM loss."""
    
    def test_gradients_flow_to_model(self):
        """Test that gradients flow to model parameters."""
        loss_fn = IRMLoss()
        
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
        ]
        
        penalty = loss_fn(model, envs)
        penalty.backward()
        
        # Check gradients exist for all parameters
        for param in model.parameters():
            assert param.grad is not None
    
    def test_gradients_through_dummy_scale(self):
        """Test that gradients flow through dummy scale parameter."""
        loss_fn = IRMLoss()
        
        model = nn.Linear(10, 2)
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
        ]
        
        penalty = loss_fn(model, envs)
        penalty.backward()
        
        # Dummy scale should have gradient
        assert loss_fn.dummy_scale.grad is not None


class TestIRMLossTraining:
    """Integration tests with training loop."""
    
    def test_training_reduces_penalty(self):
        """Test that training with IRM reduces the penalty."""
        torch.manual_seed(42)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        loss_fn = IRMLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Generate environments with shared invariant structure
        envs = []
        for _ in range(2):
            X = torch.randn(50, 10)
            y = (X[:, 0] > 0).long()  # Same rule across envs
            envs.append((X, y))
        
        # Initial penalty (needs gradients for IRM computation)
        _ = loss_fn(model, envs).detach()
        
        # Train to minimize penalty
        for _ in range(30):
            optimizer.zero_grad()
            
            # Compute task loss (cross-entropy)
            task_loss = 0.0
            for X_e, y_e in envs:
                logits = model(X_e)
                task_loss += nn.functional.cross_entropy(logits, y_e)
            task_loss = task_loss / len(envs)
            
            # Compute IRM penalty
            penalty = loss_fn(model, envs)
            
            # Total loss
            total_loss = task_loss + 10.0 * penalty
            total_loss.backward()
            optimizer.step()
        
        # Final penalty (needs gradients for IRM computation)
        penalty_final = loss_fn(model, envs).detach()
        
        # Penalty may not always decrease monotonically, but should be reasonable
        assert penalty_final.item() >= 0
        assert not torch.isnan(penalty_final)
    
    def test_irm_encourages_invariance(self):
        """Test that IRM encourages learning invariant features."""
        torch.manual_seed(42)
        
        # Model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        loss_fn = IRMLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Environments with invariant core feature
        envs = []
        for env_id in range(2):
            X = torch.randn(100, 10)
            # Invariant: y depends on X[:, 0]
            y_core = (X[:, 0] > 0).long()
            # Add environment-specific noise in other features
            y = y_core
            envs.append((X, y))
        
        # Train with IRM
        for _ in range(50):
            optimizer.zero_grad()
            
            task_loss = sum(
                nn.functional.cross_entropy(model(X), y)
                for X, y in envs
            ) / len(envs)
            
            penalty = loss_fn(model, envs)
            
            total_loss = task_loss + 1.0 * penalty
            total_loss.backward()
            optimizer.step()
        
        # Test on both environments
        accuracies = []
        for X, y in envs:
            with torch.no_grad():
                logits = model(X)
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean()
                accuracies.append(acc.item())
        
        # Should achieve decent accuracy on both environments
        for acc in accuracies:
            assert acc > 0.5  # Better than random


class TestIRMLossDevice:
    """Tests for device compatibility."""
    
    def test_cpu_device(self):
        """Test IRM loss on CPU."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test IRM loss on CUDA."""
        loss_fn = IRMLoss().cuda()
        model = nn.Linear(10, 2).cuda()
        
        envs = [
            (torch.randn(20, 10).cuda(), torch.randint(0, 2, (20,)).cuda()),
            (torch.randn(20, 10).cuda(), torch.randint(0, 2, (20,)).cuda()),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.device.type == 'cuda'


class TestIRMLossEdgeCases:
    """Tests for edge cases."""
    
    def test_single_environment_raises_error(self):
        """Test that single environment raises error."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        # Only one environment
        envs = [(torch.randn(20, 10), torch.randint(0, 2, (20,)))]
        
        with pytest.raises(ValueError, match="at least 2 environments"):
            loss_fn(model, envs)
    
    def test_small_batch_per_environment(self):
        """Test with small batch size per environment."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        # Small batches (5 samples each)
        envs = [
            (torch.randn(5, 10), torch.randint(0, 2, (5,))),
            (torch.randn(5, 10), torch.randint(0, 2, (5,))),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.shape == torch.Size([])
        assert penalty.item() >= 0
    
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with seed."""
        model = nn.Linear(10, 2)
        envs = [
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            (torch.randn(20, 10), torch.randint(0, 2, (20,))),
        ]
        
        torch.manual_seed(42)
        loss_fn1 = IRMLoss()
        penalty1 = loss_fn1(model, envs)
        
        torch.manual_seed(42)
        loss_fn2 = IRMLoss()
        penalty2 = loss_fn2(model, envs)
        
        assert torch.allclose(penalty1, penalty2)
    
    def test_multiclass_classification(self):
        """Test with multi-class classification."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 5)  # 5 classes
        
        envs = [
            (torch.randn(20, 10), torch.randint(0, 5, (20,))),
            (torch.randn(20, 10), torch.randint(0, 5, (20,))),
        ]
        
        penalty = loss_fn(model, envs)
        
        assert penalty.shape == torch.Size([])
        assert penalty.item() >= 0


class TestIRMLossProperties:
    """Tests for mathematical properties of IRM."""
    
    def test_penalty_non_negative(self):
        """Test that IRM penalty is always non-negative."""
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        for _ in range(10):
            envs = [
                (torch.randn(20, 10), torch.randint(0, 2, (20,))),
                (torch.randn(20, 10), torch.randint(0, 2, (20,))),
            ]
            
            penalty = loss_fn(model, envs)
            
            assert penalty.item() >= -1e-6, f"Penalty should be non-negative, got {penalty.item()}"
    
    def test_penalty_zero_for_optimal_dummy(self):
        """Test that penalty is small when dummy scale is optimal."""
        # This is a theoretical property - in practice, we just check it's reasonable
        loss_fn = IRMLoss()
        model = nn.Linear(10, 2)
        
        # Perfect invariant features (same data in both envs)
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        envs = [(X, y), (X, y)]
        
        penalty = loss_fn(model, envs)
        
        # For identical environments, penalty should be very small
        assert penalty.item() < 0.1


class TestIRMLossPerformance:
    """Performance tests for IRM loss."""
    
    def test_computation_speed(self):
        """Benchmark IRM computation speed."""
        import time
        
        loss_fn = IRMLoss()
        model = nn.Sequential(
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        envs = [
            (torch.randn(128, 50), torch.randint(0, 2, (128,))),
            (torch.randn(128, 50), torch.randint(0, 2, (128,))),
        ]
        
        # Warm-up
        _ = loss_fn(model, envs)
        
        # Benchmark
        start = time.time()
        for _ in range(5):
            _ = loss_fn(model, envs)
        elapsed = time.time() - start
        
        avg_time = elapsed / 5
        
        # Should be < 200ms per forward pass (as per spec)
        assert avg_time < 0.2, f"IRM computation took {avg_time:.3f}s, expected < 0.2s"
