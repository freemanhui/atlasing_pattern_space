"""
Tests for APSAutoencoder - unified model with T+C+E losses.
"""

import pytest
import torch
import torch.nn as nn
from aps.models.aps_autoencoder import APSAutoencoder, APSConfig


class TestAPSConfig:
    """Tests for APSConfig dataclass."""
    
    def test_config_minimal(self):
        """Test config with minimal required fields."""
        config = APSConfig(in_dim=784, latent_dim=2)
        
        assert config.in_dim == 784
        assert config.latent_dim == 2
        assert config.hidden_dims == [64]  # default
        assert config.lambda_T == 1.0
        assert config.lambda_C == 1.0
        assert config.lambda_E == 1.0
    
    def test_config_custom(self):
        """Test config with custom values."""
        config = APSConfig(
            in_dim=100,
            latent_dim=10,
            hidden_dims=[128, 64],
            lambda_T=0.5,
            lambda_C=2.0,
            lambda_E=0.1,
            topo_k=5,
        )
        
        assert config.hidden_dims == [128, 64]
        assert config.lambda_T == 0.5
        assert config.lambda_C == 2.0
        assert config.lambda_E == 0.1
        assert config.topo_k == 5
    
    def test_config_ablation_topology_only(self):
        """Test config for topology-only ablation."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=1.0,
            lambda_C=0.0,
            lambda_E=0.0,
        )
        
        assert config.lambda_T == 1.0
        assert config.lambda_C == 0.0
        assert config.lambda_E == 0.0


class TestAPSAutoencoderInit:
    """Tests for APSAutoencoder initialization."""
    
    def test_init_with_all_losses(self):
        """Test initialization with all losses enabled."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'topo_loss')
        assert hasattr(model, 'hsic_loss')
        assert hasattr(model, 'energy_fn')
    
    def test_init_topology_only(self):
        """Test initialization with only topology loss."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=1.0,
            lambda_C=0.0,
            lambda_E=0.0,
        )
        model = APSAutoencoder(config)
        
        assert hasattr(model, 'topo_loss')
        # Other losses should still exist but won't be used (lambda=0)
    
    def test_encoder_decoder_architecture(self):
        """Test encoder and decoder are built correctly."""
        config = APSConfig(
            in_dim=784,
            latent_dim=10,
            hidden_dims=[128, 64],
        )
        model = APSAutoencoder(config)
        
        # Test encoder forward
        x = torch.randn(16, 784)
        z = model.encoder(x)
        assert z.shape == (16, 10)
        
        # Test decoder forward
        x_recon = model.decoder(z)
        assert x_recon.shape == (16, 784)


class TestAPSAutoencoderForward:
    """Tests for APSAutoencoder forward pass."""
    
    def test_forward_shape(self):
        """Test forward pass returns correct shapes."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        x_recon, z = model(x)
        
        assert x_recon.shape == (32, 50)
        assert z.shape == (32, 2)
    
    def test_encode_method(self):
        """Test encode method."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        z = model.encode(x)
        
        assert z.shape == (32, 2)
    
    def test_forward_different_batch_sizes(self):
        """Test with various batch sizes."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        for batch_size in [1, 16, 32, 128]:
            x = torch.randn(batch_size, 50)
            x_recon, z = model(x)
            
            assert x_recon.shape == (batch_size, 50)
            assert z.shape == (batch_size, 2)


class TestAPSAutoencoderLoss:
    """Tests for APSAutoencoder loss computation."""
    
    def test_compute_loss_all_components(self):
        """Test loss computation with all components."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=1.0,
            lambda_C=1.0,
            lambda_E=1.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        
        # Check all losses present
        assert 'recon' in losses
        assert 'topo' in losses
        assert 'energy' in losses
        assert 'total' in losses
        
        # All should be scalars
        for key, value in losses.items():
            assert value.shape == torch.Size([])
            assert not torch.isnan(value)
            assert not torch.isinf(value)
    
    def test_compute_loss_ablation_T_only(self):
        """Test loss with only topology enabled."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=1.0,
            lambda_C=0.0,
            lambda_E=0.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        
        assert 'recon' in losses
        assert 'topo' in losses
        assert 'total' in losses
        
        # Total should equal recon + T
        expected_total = losses['recon'] + config.lambda_T * losses['topo']
        assert torch.allclose(losses['total'], expected_total)
    
    def test_compute_loss_ablation_C_only(self):
        """Test loss with only causality enabled."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=0.0,
            lambda_C=1.0,
            lambda_E=0.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        nuisance = torch.randn(32, 5)
        losses = model.compute_loss(x, nuisance=nuisance)
        
        assert 'recon' in losses
        assert 'hsic' in losses
        assert 'total' in losses
    
    def test_compute_loss_ablation_E_only(self):
        """Test loss with only energy enabled."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=0.0,
            lambda_C=0.0,
            lambda_E=1.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        
        assert 'recon' in losses
        assert 'energy' in losses
        assert 'total' in losses
    
    def test_compute_loss_baseline(self):
        """Test loss with all regularizations disabled (baseline AE)."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=0.0,
            lambda_C=0.0,
            lambda_E=0.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        
        # Total should equal recon only
        assert torch.allclose(losses['total'], losses['recon'])
    
    def test_compute_loss_with_nuisance(self):
        """Test loss computation with nuisance variable (HSIC)."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_C=1.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        nuisance = torch.randn(32, 5)
        
        losses = model.compute_loss(x, nuisance=nuisance)
        
        assert 'hsic' in losses
        assert losses['hsic'].item() >= 0


class TestAPSAutoencoderGradients:
    """Tests for gradient flow."""
    
    def test_gradients_flow_to_encoder(self):
        """Test gradients flow to encoder parameters."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        losses['total'].backward()
        
        # Check encoder has gradients
        for param in model.encoder.parameters():
            assert param.grad is not None
    
    def test_gradients_flow_to_decoder(self):
        """Test gradients flow to decoder parameters."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        losses = model.compute_loss(x)
        losses['total'].backward()
        
        # Check decoder has gradients
        for param in model.decoder.parameters():
            assert param.grad is not None
    
    def test_gradients_with_all_losses(self):
        """Test gradients flow with all losses enabled."""
        config = APSConfig(
            in_dim=50,
            latent_dim=2,
            lambda_T=1.0,
            lambda_C=1.0,
            lambda_E=1.0,
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(32, 50)
        nuisance = torch.randn(32, 5)
        
        losses = model.compute_loss(x, nuisance=nuisance)
        losses['total'].backward()
        
        # All model parameters should have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAPSAutoencoderTraining:
    """Integration tests with training loop."""
    
    def test_training_reduces_losses(self):
        """Test that training reduces all losses."""
        torch.manual_seed(42)
        
        config = APSConfig(
            in_dim=20,
            latent_dim=2,
            lambda_T=0.5,
            lambda_C=0.0,  # Skip HSIC for simplicity
            lambda_E=0.1,
        )
        model = APSAutoencoder(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Generate data
        x = torch.randn(50, 20)
        
        # Initial losses
        with torch.no_grad():
            initial_losses = model.compute_loss(x)
        
        # Train
        for _ in range(30):
            optimizer.zero_grad()
            losses = model.compute_loss(x)
            losses['total'].backward()
            optimizer.step()
        
        # Final losses
        with torch.no_grad():
            final_losses = model.compute_loss(x)
        
        # Losses should decrease
        assert final_losses['recon'] < initial_losses['recon']
        assert final_losses['total'] < initial_losses['total']
    
    def test_training_with_hsic(self):
        """Test training with HSIC loss."""
        torch.manual_seed(42)
        
        config = APSConfig(
            in_dim=20,
            latent_dim=5,
            lambda_T=0.0,
            lambda_C=1.0,
            lambda_E=0.0,
        )
        model = APSAutoencoder(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Data with nuisance
        x = torch.randn(50, 20)
        nuisance = torch.randn(50, 3)
        
        # Train
        for _ in range(20):
            optimizer.zero_grad()
            losses = model.compute_loss(x, nuisance=nuisance)
            losses['total'].backward()
            optimizer.step()
        
        # Final HSIC should be reasonable
        with torch.no_grad():
            final_losses = model.compute_loss(x, nuisance=nuisance)
        
        assert final_losses['hsic'].item() >= 0
        assert not torch.isnan(final_losses['hsic'])


class TestAPSAutoencoderDevice:
    """Tests for device compatibility."""
    
    def test_cpu_device(self):
        """Test model works on CPU."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(16, 50)
        x_recon, z = model(x)
        
        assert x_recon.device.type == 'cpu'
        assert z.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test model works on CUDA."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config).cuda()
        
        x = torch.randn(16, 50).cuda()
        x_recon, z = model(x)
        
        assert x_recon.device.type == 'cuda'
        assert z.device.type == 'cuda'


class TestAPSAutoencoderEdgeCases:
    """Tests for edge cases."""
    
    def test_single_sample(self):
        """Test with single sample (batch_size=1)."""
        config = APSConfig(in_dim=50, latent_dim=2)
        model = APSAutoencoder(config)
        
        x = torch.randn(1, 50)
        
        # Forward should work
        x_recon, z = model(x)
        assert x_recon.shape == (1, 50)
        assert z.shape == (1, 2)
        
        # Loss computation might fail due to topology (batch < k)
        # This is expected behavior
    
    def test_large_latent_dim(self):
        """Test with large latent dimension."""
        config = APSConfig(in_dim=50, latent_dim=100)
        model = APSAutoencoder(config)
        
        x = torch.randn(16, 50)
        x_recon, z = model(x)
        
        assert z.shape == (16, 100)
    
    def test_multiple_hidden_layers(self):
        """Test with multiple hidden layers."""
        config = APSConfig(
            in_dim=784,
            latent_dim=10,
            hidden_dims=[256, 128, 64],
        )
        model = APSAutoencoder(config)
        
        x = torch.randn(16, 784)
        x_recon, z = model(x)
        
        assert x_recon.shape == (16, 784)
        assert z.shape == (16, 10)
