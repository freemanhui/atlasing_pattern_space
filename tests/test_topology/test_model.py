"""
Tests for TopologicalAutoencoder model.
"""

import pytest
import torch
import torch.nn as nn
from aps.topology.model import TopologicalAutoencoder, TopoAEConfig


class TestTopoAEConfig:
    """Tests for TopoAEConfig dataclass."""
    
    def test_config_default(self):
        """Test default configuration."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        
        assert config.in_dim == 50
        assert config.latent_dim == 2
        assert config.hidden == 64
        assert config.lr == 1e-3
        assert config.topo_k == 8
        assert config.topo_weight == 1.0
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = TopoAEConfig(
            in_dim=100,
            latent_dim=10,
            hidden=128,
            lr=0.001,
            topo_k=5,
            topo_weight=0.5
        )
        
        assert config.in_dim == 100
        assert config.latent_dim == 10
        assert config.hidden == 128
        assert config.lr == 0.001
        assert config.topo_k == 5
        assert config.topo_weight == 0.5


class TestTopologicalAutoencoder:
    """Tests for TopologicalAutoencoder class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        assert isinstance(model, nn.Module)
        assert model.config == config
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'topo_loss')
    
    def test_encoder_decoder_architecture(self):
        """Test encoder and decoder have correct architecture."""
        config = TopoAEConfig(in_dim=50, latent_dim=2, hidden=64)
        model = TopologicalAutoencoder(config)
        
        # Test encoder forward pass
        x = torch.randn(10, 50)
        z = model.encoder(x)
        assert z.shape == (10, 2)
        
        # Test decoder forward pass
        x_recon = model.decoder(z)
        assert x_recon.shape == (10, 50)
    
    def test_forward_returns_recon_and_latent(self):
        """Test forward pass returns reconstruction and latent."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        x_recon, z = model(x)
        
        assert x_recon.shape == (32, 50)
        assert z.shape == (32, 2)
    
    def test_encode_method(self):
        """Test encode method returns latent representation."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        z = model.encode(x)
        
        assert z.shape == (32, 2)
    
    def test_compute_loss_components(self):
        """Test compute_loss returns dict with loss components."""
        config = TopoAEConfig(in_dim=50, latent_dim=2, topo_weight=1.0)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        loss_dict = model.compute_loss(x)
        
        # Should have reconstruction, topology, and total loss
        assert 'recon_loss' in loss_dict
        assert 'topo_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        # All should be scalars
        for key, value in loss_dict.items():
            assert value.shape == torch.Size([])
            assert not torch.isnan(value)
            assert not torch.isinf(value)
    
    def test_compute_loss_total_equals_weighted_sum(self):
        """Test total loss equals weighted sum of components."""
        config = TopoAEConfig(in_dim=50, latent_dim=2, topo_weight=0.5)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        loss_dict = model.compute_loss(x)
        
        expected_total = loss_dict['recon_loss'] + 0.5 * loss_dict['topo_loss']
        
        assert torch.allclose(loss_dict['total_loss'], expected_total)
    
    def test_compute_loss_zero_topo_weight(self):
        """Test topology loss can be disabled with weight=0."""
        config = TopoAEConfig(in_dim=50, latent_dim=2, topo_weight=0.0)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        loss_dict = model.compute_loss(x)
        
        # Total should equal recon loss only
        assert torch.allclose(loss_dict['total_loss'], loss_dict['recon_loss'])
    
    def test_forward_gradients_flow(self):
        """Test gradients flow through the model."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        loss_dict = model.compute_loss(x)
        
        loss_dict['total_loss'].backward()
        
        # Check encoder has gradients
        for param in model.encoder.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
        
        # Check decoder has gradients
        for param in model.decoder.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
    
    def test_different_dimensions(self):
        """Test model works with various dimensions."""
        test_cases = [
            (10, 2),
            (50, 5),
            (100, 10),
            (200, 20),
        ]
        
        for in_dim, latent_dim in test_cases:
            config = TopoAEConfig(in_dim=in_dim, latent_dim=latent_dim)
            model = TopologicalAutoencoder(config)
            
            x = torch.randn(16, in_dim)
            x_recon, z = model(x)
            
            assert x_recon.shape == (16, in_dim)
            assert z.shape == (16, latent_dim)
    
    def test_batch_size_validation(self):
        """Test model handles batch size < k gracefully."""
        config = TopoAEConfig(in_dim=50, latent_dim=2, topo_k=10)
        model = TopologicalAutoencoder(config)
        
        # Batch size 8 < k=10
        x = torch.randn(8, 50)
        
        # Should raise error in compute_loss
        with pytest.raises((ValueError, RuntimeError)):
            model.compute_loss(x)
        
        # But forward should work (no topology loss computed)
        x_recon, z = model(x)
        assert x_recon.shape == (8, 50)
        assert z.shape == (8, 2)


class TestTopologicalAutoencoderTraining:
    """Integration tests for training TopologicalAutoencoder."""
    
    def test_training_reduces_loss(self):
        """Test that training reduces both reconstruction and topology loss."""
        config = TopoAEConfig(in_dim=20, latent_dim=2, lr=0.01, topo_weight=1.0)
        model = TopologicalAutoencoder(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        # Generate training data
        x = torch.randn(50, 20)
        
        # Record initial losses
        with torch.no_grad():
            initial_losses = model.compute_loss(x)
        
        # Train for a few steps
        for _ in range(20):
            optimizer.zero_grad()
            loss_dict = model.compute_loss(x)
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        # Record final losses
        with torch.no_grad():
            final_losses = model.compute_loss(x)
        
        # Losses should decrease
        assert final_losses['recon_loss'] < initial_losses['recon_loss']
        assert final_losses['topo_loss'] < initial_losses['topo_loss']
        assert final_losses['total_loss'] < initial_losses['total_loss']
    
    def test_training_improves_reconstruction(self):
        """Test that training improves reconstruction quality."""
        config = TopoAEConfig(in_dim=20, latent_dim=5, lr=0.01)
        model = TopologicalAutoencoder(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        x = torch.randn(50, 20)
        
        # Initial reconstruction error
        with torch.no_grad():
            x_recon_init, _ = model(x)
            mse_init = torch.mean((x - x_recon_init) ** 2)
        
        # Train
        for _ in range(50):
            optimizer.zero_grad()
            loss_dict = model.compute_loss(x)
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        # Final reconstruction error
        with torch.no_grad():
            x_recon_final, _ = model(x)
            mse_final = torch.mean((x - x_recon_final) ** 2)
        
        # MSE should decrease
        assert mse_final < mse_init
    
    def test_topology_weight_affects_preservation(self):
        """Test that higher topology weight improves preservation."""
        # Train two models: one with low topo weight, one with high
        configs = [
            TopoAEConfig(in_dim=20, latent_dim=2, topo_weight=0.1),
            TopoAEConfig(in_dim=20, latent_dim=2, topo_weight=5.0),
        ]
        
        x = torch.randn(50, 20)
        
        final_topo_losses = []
        
        for config in configs:
            model = TopologicalAutoencoder(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Train
            for _ in range(30):
                optimizer.zero_grad()
                loss_dict = model.compute_loss(x)
                loss_dict['total_loss'].backward()
                optimizer.step()
            
            # Record final topology loss
            with torch.no_grad():
                final_losses = model.compute_loss(x)
                final_topo_losses.append(final_losses['topo_loss'].item())
        
        # Model with higher topo weight should have lower final topo loss
        assert final_topo_losses[1] < final_topo_losses[0]


class TestTopologicalAutoencoderDevice:
    """Tests for device compatibility."""
    
    def test_cpu_device(self):
        """Test model works on CPU."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(16, 50)
        x_recon, z = model(x)
        
        assert x_recon.device.type == 'cpu'
        assert z.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test model works on CUDA."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config).cuda()
        
        x = torch.randn(16, 50).cuda()
        x_recon, z = model(x)
        
        assert x_recon.device.type == 'cuda'
        assert z.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_computation_on_cuda(self):
        """Test loss computation works on CUDA."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config).cuda()
        
        x = torch.randn(32, 50).cuda()
        loss_dict = model.compute_loss(x)
        
        for value in loss_dict.values():
            assert value.device.type == 'cuda'


class TestTopologicalAutoencoderEdgeCases:
    """Tests for edge cases."""
    
    def test_single_sample_forward(self):
        """Test forward pass works with single sample (batch_size=1)."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(1, 50)
        
        # Forward should work
        x_recon, z = model(x)
        assert x_recon.shape == (1, 50)
        assert z.shape == (1, 2)
        
        # compute_loss should fail (batch_size < topo_k)
        with pytest.raises((ValueError, RuntimeError)):
            model.compute_loss(x)
    
    def test_large_latent_dim(self):
        """Test model works when latent_dim > in_dim (unusual but valid)."""
        config = TopoAEConfig(in_dim=10, latent_dim=20)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(16, 10)
        x_recon, z = model(x)
        
        assert x_recon.shape == (16, 10)
        assert z.shape == (16, 20)
    
    def test_model_eval_mode(self):
        """Test model can switch between train and eval modes."""
        config = TopoAEConfig(in_dim=50, latent_dim=2)
        model = TopologicalAutoencoder(config)
        
        x = torch.randn(32, 50)
        
        # Train mode
        model.train()
        assert model.training
        loss_dict_train = model.compute_loss(x)
        
        # Eval mode
        model.eval()
        assert not model.training
        with torch.no_grad():
            loss_dict_eval = model.compute_loss(x)
        
        # Both should work
        assert 'total_loss' in loss_dict_train
        assert 'total_loss' in loss_dict_eval
