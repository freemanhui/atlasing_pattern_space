"""
Topological Autoencoder model.

Combines standard autoencoder architecture with topology-preserving loss.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import KNNTopoLoss


@dataclass
class TopoAEConfig:
    """
    Configuration for TopologicalAutoencoder.
    
    Args:
        in_dim: Input dimension
        latent_dim: Latent space dimension
        hidden: Hidden layer size (default: 64)
        lr: Learning rate (default: 1e-3)
        topo_k: k for k-NN topology loss (default: 8)
        topo_weight: Weight for topology loss (default: 1.0)
    
    Example:
        >>> config = TopoAEConfig(in_dim=50, latent_dim=2, topo_weight=0.5)
        >>> model = TopologicalAutoencoder(config)
    """
    in_dim: int
    latent_dim: int
    hidden: int = 64
    lr: float = 1e-3
    topo_k: int = 8
    topo_weight: float = 1.0


class TopologicalAutoencoder(nn.Module):
    """
    Autoencoder with topology-preserving loss.
    
    Combines reconstruction loss (MSE) with k-NN topology preservation loss.
    The total loss is: L = L_recon + λ_T * L_topo
    
    Architecture:
        - Encoder: in_dim → hidden → latent_dim
        - Decoder: latent_dim → hidden → in_dim
        - Both use ReLU activations
    
    Args:
        config: TopoAEConfig with model hyperparameters
    
    Example:
        >>> config = TopoAEConfig(in_dim=50, latent_dim=2)
        >>> model = TopologicalAutoencoder(config)
        >>> x = torch.randn(32, 50)
        >>> 
        >>> # Forward pass
        >>> x_recon, z = model(x)
        >>> 
        >>> # Training
        >>> loss_dict = model.compute_loss(x)
        >>> loss_dict['total_loss'].backward()
    
    References:
        - APS Paper Section 4: Combined objective with topology term
        - Chen et al. (2022) for continuous k-NN approach
    """
    
    def __init__(self, config: TopoAEConfig):
        """Initialize TopologicalAutoencoder."""
        super().__init__()
        
        self.config = config
        
        # Encoder: in_dim → hidden → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.latent_dim)
        )
        
        # Decoder: latent_dim → hidden → in_dim
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.in_dim)
        )
        
        # Topology preservation loss
        self.topo_loss = KNNTopoLoss(k=config.topo_k)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
        
        Returns:
            Tuple of (reconstruction, latent):
                - reconstruction: shape (batch_size, in_dim)
                - latent: shape (batch_size, latent_dim)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
        
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with breakdown.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
        
        Returns:
            Dictionary with loss components:
                - 'recon_loss': Reconstruction loss (MSE)
                - 'topo_loss': Topology preservation loss
                - 'total_loss': Weighted sum of losses
        
        Raises:
            ValueError: If batch_size <= topo_k
        
        Example:
            >>> config = TopoAEConfig(in_dim=50, latent_dim=2, topo_weight=0.5)
            >>> model = TopologicalAutoencoder(config)
            >>> x = torch.randn(32, 50)
            >>> losses = model.compute_loss(x)
            >>> print(f"Total: {losses['total_loss']:.4f}")
            >>> print(f"Recon: {losses['recon_loss']:.4f}")
            >>> print(f"Topo: {losses['topo_loss']:.4f}")
        """
        # Forward pass
        x_recon, z = self(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Topology preservation loss
        topo_loss = self.topo_loss(x, z)
        
        # Combined loss
        total_loss = recon_loss + self.config.topo_weight * topo_loss
        
        return {
            'recon_loss': recon_loss,
            'topo_loss': topo_loss,
            'total_loss': total_loss
        }
