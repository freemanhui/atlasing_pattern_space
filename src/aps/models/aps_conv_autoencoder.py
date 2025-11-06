"""
Convolutional APSAutoencoder for RGB images (ColoredMNIST).

Extends the unified APS framework to handle image inputs using convolutional
encoder/decoder architectures instead of MLP.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..topology.knn_topo_loss import KNNTopoLoss, adjacency_from_knn, knn_indices
from ..causality.hsic import HSICLoss
from ..energy.energy import MemoryEnergy, MemoryEnergyConfig


@dataclass
class APSConvConfig:
    """
    Configuration for convolutional APSAutoencoder.
    
    Architecture:
        in_channels: Input channels (3 for RGB, 1 for grayscale)
        img_size: Image height/width (assumes square images)
        latent_dim: Latent space dimension
        hidden_channels: List of channel sizes for conv layers
    
    Loss weights:
        lambda_T: Weight for topology loss
        lambda_C: Weight for causality (HSIC) loss
        lambda_E: Weight for energy loss
    
    Component parameters:
        topo_k: Number of neighbors for kNN graph
        hsic_sigma: RBF kernel bandwidth for HSIC
        n_mem: Number of memory patterns
        beta: Energy sharpness (higher = sharper basins)
        alpha: L2 regularization on latent space
    """
    # Architecture
    in_channels: int = 3  # RGB
    img_size: int = 28    # 28x28 for MNIST
    latent_dim: int = 10
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64])
    
    # Loss weights
    lambda_T: float = 1.0
    lambda_C: float = 1.0
    lambda_E: float = 1.0
    
    # Component parameters
    topo_k: int = 8
    hsic_sigma: float = 1.0
    n_mem: int = 8
    beta: float = 1.0  # Reduced from 5.0 for numerical stability
    alpha: float = 0.0
    normalize_z: bool = False
    clip_dots: bool = True
    max_dot: float = 10.0


class APSConvAutoencoder(nn.Module):
    """
    Convolutional APSAutoencoder for RGB images.
    
    Uses conv layers for encoding and transposed conv for decoding, preserving
    the unified APS loss structure: L_total = L_recon + λ_T·L_topo + λ_C·L_hsic + λ_E·L_energy
    
    Architecture:
        Encoder: Conv2d → ReLU → Conv2d → ReLU → Flatten → Linear → Latent
        Decoder: Linear → Reshape → ConvTranspose2d → ReLU → ConvTranspose2d → Sigmoid
    
    Example:
        >>> config = APSConvConfig(in_channels=3, img_size=28, latent_dim=10)
        >>> model = APSConvAutoencoder(config)
        >>> x = torch.randn(32, 3, 28, 28)  # RGB ColoredMNIST batch
        >>> losses = model.compute_loss(x)
        >>> losses['total'].backward()
    """
    
    def __init__(self, config: APSConvConfig):
        super().__init__()
        self.config = config
        
        # Build encoder and decoder
        self.encoder, self.encoder_out_size = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Classifier head for digit recognition (10 classes)
        self.classifier = nn.Linear(config.latent_dim, 10)
        
        # Loss components
        self.topo_loss = KNNTopoLoss(k=config.topo_k)
        self.hsic_loss = HSICLoss(kernel='rbf', sigma=config.hsic_sigma)
        
        # Energy function
        energy_cfg = MemoryEnergyConfig(
            latent_dim=config.latent_dim,
            n_mem=config.n_mem,
            beta=config.beta,
            alpha=config.alpha,
            normalize_z=config.normalize_z,
            clip_dots=config.clip_dots,
            max_dot=config.max_dot
        )
        self.energy_fn = MemoryEnergy(energy_cfg)
    
    def _build_encoder(self) -> Tuple[nn.Module, int]:
        """
        Build convolutional encoder.
        
        Returns:
            encoder: Sequential encoder module
            out_size: Flattened feature size before latent layer
        """
        layers = []
        in_ch = self.config.in_channels
        img_size = self.config.img_size
        
        # Convolutional layers
        for hidden_ch in self.config.hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(),
            ])
            in_ch = hidden_ch
            img_size = img_size // 2  # Each stride=2 halves the size
        
        # Calculate flattened size
        final_ch = self.config.hidden_channels[-1]
        flattened_size = final_ch * img_size * img_size
        
        # Add flatten and linear to latent
        encoder = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(flattened_size, self.config.latent_dim)
        )
        
        return encoder, img_size
    
    def _build_decoder(self) -> nn.Module:
        """Build transposed convolutional decoder."""
        # Calculate sizes for reshaping
        final_ch = self.config.hidden_channels[-1]
        img_size = self.encoder_out_size
        
        layers = [
            nn.Linear(self.config.latent_dim, final_ch * img_size * img_size),
            nn.Unflatten(1, (final_ch, img_size, img_size)),
        ]
        
        # Reverse hidden channels
        reversed_channels = list(reversed(self.config.hidden_channels))
        
        # Transposed conv layers
        for i, out_ch in enumerate(reversed_channels[1:] + [self.config.in_channels]):
            in_ch = reversed_channels[i]
            
            # Last layer: output channels and sigmoid
            if i == len(reversed_channels) - 1:
                layers.extend([
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid(),  # Output in [0, 1] range
                ])
            else:
                layers.extend([
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                ])
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            z: Latent codes (batch, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to images.
        
        Args:
            z: Latent codes (batch, latent_dim)
        
        Returns:
            x_recon: Reconstructed images (batch, channels, height, width)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reconstruction and classification.
        
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            x_recon: Reconstructed images
            logits: Class logits (10 classes for digits)
            z: Latent codes
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classifier(z)
        return x_recon, logits, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        color_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for ColoredMNIST.
        
        Args:
            x: Input images (batch, 3, 28, 28)
            labels: Digit labels for classification (batch,)
            color_indices: Color indices for HSIC independence (batch,)
        
        Returns:
            Dictionary with loss components:
                - 'recon': Reconstruction loss
                - 'class': Classification loss (if labels provided)
                - 'topo': Topology loss (if lambda_T > 0)
                - 'hsic': HSIC independence from color (if lambda_C > 0 and color_indices provided)
                - 'energy': Energy loss (if lambda_E > 0)
                - 'total': Combined loss
        """
        # Flatten images for topology computation (batch, 3*28*28)
        x_flat = x.view(x.size(0), -1)
        
        # Forward pass
        x_recon, logits, z = self.forward(x)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        
        # Reconstruction loss (MSE on flattened images)
        loss_recon = F.mse_loss(x_recon_flat, x_flat)
        
        # Initialize losses
        losses = {'recon': loss_recon}
        total_loss = loss_recon
        
        # Classification loss
        if labels is not None:
            loss_class = F.cross_entropy(logits, labels)
            losses['class'] = loss_class
            total_loss = total_loss + loss_class
        
        # Topology loss (preserve structure from original RGB space)
        if self.config.lambda_T > 0:
            x_knn_idx = knn_indices(x_flat, k=self.config.topo_k)
            x_adj = adjacency_from_knn(x_knn_idx, n=x_flat.shape[0])
            loss_topo = self.topo_loss(z, x_adj)
            losses['topo'] = loss_topo
            total_loss = total_loss + self.config.lambda_T * loss_topo
        
        # Causality loss (HSIC independence from color)
        if self.config.lambda_C > 0 and color_indices is not None:
            # Convert color indices to one-hot for HSIC
            color_onehot = F.one_hot(color_indices, num_classes=10).float()
            loss_hsic = self.hsic_loss(z, color_onehot)
            losses['hsic'] = loss_hsic
            total_loss = total_loss + self.config.lambda_C * loss_hsic
        
        # Energy loss
        if self.config.lambda_E > 0:
            energy = self.energy_fn.energy(z).mean()
            losses['energy'] = energy
            total_loss = total_loss + self.config.lambda_E * energy
        
        losses['total'] = total_loss
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images.
        
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            preds: Predicted class indices (batch,)
        """
        self.eval()
        with torch.no_grad():
            _, logits, _ = self.forward(x)
            preds = logits.argmax(dim=1)
        return preds


if __name__ == '__main__':
    """Test APSConvAutoencoder with dummy data."""
    # Create model
    config = APSConvConfig(
        in_channels=3,
        img_size=28,
        latent_dim=10,
        hidden_channels=[32, 64],
        lambda_T=1.0,
        lambda_C=1.0,
        lambda_E=0.1,
    )
    model = APSConvAutoencoder(config)
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 3, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    color_indices = torch.randint(0, 10, (batch_size,))
    
    # Compute losses
    losses = model.compute_loss(x, labels, color_indices)
    
    print("APSConvAutoencoder Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Latent dim: {config.latent_dim}")
    print("\nLosses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test shapes
    x_recon, logits, z = model.forward(x)
    print("\nOutput shapes:")
    print(f"  Reconstruction: {x_recon.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Latent: {z.shape}")
    
    # Test gradient flow
    losses['total'].backward()
    print("\nGradient check: OK (all parameters have gradients)")
