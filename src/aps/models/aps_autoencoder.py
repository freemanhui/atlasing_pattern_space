"""
APSAutoencoder: Unified autoencoder with Topology, Causality, and Energy losses.

This model combines:
- Reconstruction loss (MSE)
- Topology preservation (kNN adjacency)
- Causality (HSIC independence)
- Energy landscape shaping (memory-based attractors)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..topology.knn_topo_loss import KNNTopoLoss, adjacency_from_knn, knn_indices
from ..causality.hsic import HSICLoss
from ..energy.energy import MemoryEnergy, MemoryEnergyConfig


@dataclass
class APSConfig:
    """
    Configuration for APSAutoencoder.
    
    Architecture:
        in_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions (encoder/decoder are symmetric)
    
    Loss weights:
        lambda_T: Weight for topology loss
        lambda_C: Weight for causality (HSIC) loss
        lambda_E: Weight for energy loss
    
    Topology parameters:
        topo_k: Number of neighbors for kNN graph
    
    Causality parameters:
        hsic_sigma: RBF kernel bandwidth (default: 1.0)
    
    Energy parameters:
        n_mem: Number of memory patterns
        beta: Energy sharpness parameter (higher = sharper basins)
        alpha: L2 regularization on latent space
    """
    # Architecture
    in_dim: int
    latent_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    
    # Loss weights
    lambda_T: float = 1.0
    lambda_C: float = 1.0
    lambda_E: float = 1.0
    
    # Topology
    topo_k: int = 8
    
    # Causality
    hsic_sigma: float = 1.0
    
    # Energy
    n_mem: int = 8
    beta: float = 5.0
    alpha: float = 0.0


class APSAutoencoder(nn.Module):
    """
    Unified autoencoder with Topology, Causality, and Energy losses.
    
    Loss function:
        L_total = L_recon + λ_T * L_topo + λ_C * L_causal + λ_E * L_energy
    
    Where:
        - L_recon: MSE reconstruction loss
        - L_topo: kNN adjacency preservation (BCE)
        - L_causal: HSIC independence loss (latent Z vs nuisance N)
        - L_energy: Memory-based energy shaping
    
    Example:
        >>> config = APSConfig(in_dim=784, latent_dim=2, lambda_T=0.5, lambda_E=0.1)
        >>> model = APSAutoencoder(config)
        >>> x = torch.randn(32, 784)
        >>> losses = model.compute_loss(x)
        >>> losses['total'].backward()
    """
    
    def __init__(self, config: APSConfig):
        super().__init__()
        self.config = config
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Loss components
        self.topo_loss = KNNTopoLoss(k=config.topo_k)
        self.hsic_loss = HSICLoss(kernel='rbf', sigma=config.hsic_sigma)
        
        # Create energy config
        energy_cfg = MemoryEnergyConfig(
            latent_dim=config.latent_dim,
            n_mem=config.n_mem,
            beta=config.beta,
            alpha=config.alpha
        )
        self.energy_fn = MemoryEnergy(energy_cfg)
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder network."""
        layers = []
        in_dim = self.config.in_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network (symmetric to encoder)."""
        layers = []
        in_dim = self.config.latent_dim
        
        # Reverse hidden layers
        for hidden_dim in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, self.config.in_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_dim)
        
        Returns:
            x_recon: Reconstruction (batch_size, in_dim)
            z: Latent representation (batch_size, latent_dim)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        nuisance: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            nuisance: Optional nuisance variable for HSIC (batch_size, nuisance_dim)
        
        Returns:
            Dictionary with loss components:
                - 'recon': Reconstruction loss
                - 'topo': Topology loss (if lambda_T > 0)
                - 'hsic': HSIC loss (if lambda_C > 0 and nuisance provided)
                - 'energy': Energy loss (if lambda_E > 0)
                - 'total': Combined loss
        """
        # Forward pass
        x_recon, z = self.forward(x)
        
        # Reconstruction loss (always computed)
        loss_recon = F.mse_loss(x_recon, x)
        
        # Initialize losses dict
        losses = {'recon': loss_recon}
        total_loss = loss_recon
        
        # Topology loss
        if self.config.lambda_T > 0:
            # Compute target topology from original space
            x_knn_idx = knn_indices(x, k=self.config.topo_k)
            x_adj = adjacency_from_knn(x_knn_idx, n=x.shape[0])
            
            # Compute loss (comparing latent topology to target)
            loss_topo = self.topo_loss(z, x_adj)
            losses['topo'] = loss_topo
            total_loss = total_loss + self.config.lambda_T * loss_topo
        
        # Causality loss (HSIC)
        if self.config.lambda_C > 0 and nuisance is not None:
            loss_hsic = self.hsic_loss(z, nuisance)
            losses['hsic'] = loss_hsic
            total_loss = total_loss + self.config.lambda_C * loss_hsic
        
        # Energy loss
        if self.config.lambda_E > 0:
            energy = self.energy_fn.energy(z).mean()
            losses['energy'] = energy
            total_loss = total_loss + self.config.lambda_E * energy
        
        losses['total'] = total_loss
        return losses
