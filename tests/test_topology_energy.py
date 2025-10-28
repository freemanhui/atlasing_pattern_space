"""
Tests for topology-aware energy function.
"""

import pytest
import torch
from aps.energy import TopologyEnergy, TopologyEnergyConfig, AdaptiveTopologyEnergy


def test_topology_energy_basic():
    """Test basic topology energy initialization and computation."""
    cfg = TopologyEnergyConfig(latent_dim=2, k=5, mode='agreement')
    energy_fn = TopologyEnergy(cfg)
    
    # Create synthetic data
    X_original = torch.randn(50, 10)
    z_latent = torch.randn(50, 2)
    
    # Set target adjacency
    energy_fn.set_target_adjacency(X_original)
    assert energy_fn.target_adjacency is not None
    assert energy_fn.target_adjacency.shape == (50, 50)
    
    # Compute energy
    energy = energy_fn.energy(z_latent)
    assert isinstance(energy, torch.Tensor)
    assert energy.shape == ()  # Scalar


def test_topology_energy_requires_target():
    """Test that energy computation fails without target adjacency."""
    cfg = TopologyEnergyConfig(latent_dim=2, k=5)
    energy_fn = TopologyEnergy(cfg)
    
    z_latent = torch.randn(50, 2)
    
    with pytest.raises(RuntimeError, match="Target adjacency not set"):
        energy_fn.energy(z_latent)


def test_topology_energy_modes():
    """Test different energy computation modes."""
    X_original = torch.randn(30, 10)
    z_latent = torch.randn(30, 2)
    
    modes = ['agreement', 'disagreement', 'jaccard']
    
    for mode in modes:
        cfg = TopologyEnergyConfig(latent_dim=2, k=5, mode=mode)
        energy_fn = TopologyEnergy(cfg)
        energy_fn.set_target_adjacency(X_original)
        
        energy = energy_fn.energy(z_latent)
        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()
        assert not torch.isnan(energy)
        assert not torch.isinf(energy)


def test_topology_energy_continuous_vs_discrete():
    """Test continuous vs discrete kNN graph modes."""
    X_original = torch.randn(30, 10)
    z_latent = torch.randn(30, 2, requires_grad=True)
    
    for continuous in [True, False]:
        cfg = TopologyEnergyConfig(latent_dim=2, k=5, continuous=continuous)
        energy_fn = TopologyEnergy(cfg)
        energy_fn.set_target_adjacency(X_original)
        
        energy = energy_fn.energy(z_latent)
        
        if continuous:
            # Should be differentiable
            assert energy.requires_grad
            energy.backward()
            assert z_latent.grad is not None
            z_latent.grad.zero_()
        else:
            # Discrete mode may not be differentiable
            pass


def test_topology_energy_agreement_behavior():
    """Test that agreement mode rewards preserved topology."""
    # Create perfectly preserved topology
    X = torch.randn(20, 5)
    
    cfg = TopologyEnergyConfig(latent_dim=5, k=3, mode='agreement', continuous=False)
    energy_fn = TopologyEnergy(cfg)
    energy_fn.set_target_adjacency(X)
    
    # Same data should have very low (negative) energy
    energy_same = energy_fn.energy(X)
    
    # Random latent should have higher energy
    z_random = torch.randn(20, 5)
    energy_random = energy_fn.energy(z_random)
    
    # Agreement mode: more negative = better
    assert energy_same < energy_random


def test_topology_energy_scale():
    """Test energy scaling parameter."""
    X_original = torch.randn(30, 10)
    z_latent = torch.randn(30, 2)
    
    cfg1 = TopologyEnergyConfig(latent_dim=2, k=5, scale=1.0)
    cfg2 = TopologyEnergyConfig(latent_dim=2, k=5, scale=2.0)
    
    energy_fn1 = TopologyEnergy(cfg1)
    energy_fn2 = TopologyEnergy(cfg2)
    
    energy_fn1.set_target_adjacency(X_original)
    energy_fn2.set_target_adjacency(X_original)
    
    energy1 = energy_fn1.energy(z_latent)
    energy2 = energy_fn2.energy(z_latent)
    
    # Energy2 should be 2x energy1
    assert torch.allclose(energy2, 2.0 * energy1, rtol=1e-5)


def test_adaptive_topology_energy():
    """Test adaptive topology energy with weight updates."""
    X_original = torch.randn(30, 10)
    z_latent = torch.randn(30, 2)
    
    cfg = TopologyEnergyConfig(latent_dim=2, k=5, mode='agreement')
    energy_fn = AdaptiveTopologyEnergy(cfg, min_weight=0.5, max_weight=2.0)
    energy_fn.set_target_adjacency(X_original)
    
    # Initial weight should be 1.0
    assert energy_fn.adaptive_weight == 1.0
    
    # Update adaptive weight
    energy_fn.update_adaptive_weight(z_latent)
    
    # Weight should be within bounds
    assert 0.5 <= energy_fn.adaptive_weight <= 2.0
    
    # Compute energy with adaptive weight
    energy = energy_fn.energy(z_latent)
    assert isinstance(energy, torch.Tensor)
    assert energy.shape == ()


def test_topology_energy_gradient_flow():
    """Test that gradients flow through topology energy."""
    X_original = torch.randn(20, 5)
    z_latent = torch.randn(20, 2, requires_grad=True)
    
    cfg = TopologyEnergyConfig(latent_dim=2, k=3, continuous=True)
    energy_fn = TopologyEnergy(cfg)
    energy_fn.set_target_adjacency(X_original)
    
    energy = energy_fn.energy(z_latent)
    energy.backward()
    
    # Gradients should exist and be finite
    assert z_latent.grad is not None
    assert torch.isfinite(z_latent.grad).all()
    assert (z_latent.grad.abs() > 0).any()  # Some non-zero gradients


def test_topology_energy_batch_consistency():
    """Test energy consistency across multiple calls."""
    X_original = torch.randn(30, 10)
    z_latent = torch.randn(30, 2)
    
    cfg = TopologyEnergyConfig(latent_dim=2, k=5, mode='agreement')
    energy_fn = TopologyEnergy(cfg)
    energy_fn.set_target_adjacency(X_original)
    
    # Multiple calls should give same result
    energy1 = energy_fn.energy(z_latent)
    energy2 = energy_fn.energy(z_latent)
    
    assert torch.allclose(energy1, energy2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
