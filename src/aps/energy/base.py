"""
Base Energy Module

Provides abstract base class for energy-based models in APS.
All energy variants should inherit from BaseEnergy.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEnergy(nn.Module, ABC):
    """
    Abstract base class for energy functions in APS.
    
    Energy functions define attraction basins in latent space.
    Lower energy = stronger attraction to learned patterns.
    
    Subclasses must implement:
    - energy(z): Compute energy values for latent points
    - _init_patterns(): Initialize memory patterns
    
    Standard interface provides:
    - loss(): Average energy for training
    - forward(): Negative energy (score)
    - memory property: Access to memory patterns
    """
    
    def __init__(self, latent_dim: int, n_mem: int):
        """
        Initialize base energy module.
        
        Args:
            latent_dim: Dimensionality of latent space
            n_mem: Number of memory patterns (attractors)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mem = n_mem
        
        # Memory patterns (subclass must initialize)
        self.mem: nn.Parameter = None
    
    @abstractmethod
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for latent points.
        
        Args:
            z: Latent points (batch_size, latent_dim)
        
        Returns:
            Energy values (batch_size,)
            Lower energy = point is near memory patterns
            Higher energy = point is far from all patterns
        """
        pass
    
    @abstractmethod
    def _init_patterns(self, init_method: str = 'random', **kwargs):
        """
        Initialize memory patterns.
        
        Args:
            init_method: Initialization strategy ('random', 'grid', 'kmeans', etc.)
            **kwargs: Additional arguments for initialization
        """
        pass
    
    @property
    def memory(self) -> torch.Tensor:
        """
        Access memory patterns.
        
        Returns:
            Memory patterns (n_mem, latent_dim)
        """
        return self.mem
    
    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute average energy loss for training.
        
        Args:
            z: Latent points (batch_size, latent_dim)
        
        Returns:
            Scalar loss (mean energy)
        """
        return torch.mean(self.energy(z))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute negative energy (score).
        
        Args:
            z: Latent points (batch_size, latent_dim)
        
        Returns:
            Scores (batch_size,)
            Higher score = closer to memory patterns
        """
        return -self.energy(z)
    
    def nearest_pattern(self, z: torch.Tensor) -> torch.Tensor:
        """
        Find nearest memory pattern for each point.
        
        Args:
            z: Latent points (batch_size, latent_dim)
        
        Returns:
            Indices of nearest patterns (batch_size,)
        """
        # Compute distances to all patterns
        dists = torch.cdist(z, self.mem)  # (batch_size, n_mem)
        return torch.argmin(dists, dim=1)
    
    def basin_assignment(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Soft basin assignment probabilities.
        
        Args:
            z: Latent points (batch_size, latent_dim)
            temperature: Softmax temperature (lower = harder assignment)
        
        Returns:
            Assignment probabilities (batch_size, n_mem)
        """
        # Negative distance as logits
        dists = torch.cdist(z, self.mem)
        logits = -dists / temperature
        return torch.softmax(logits, dim=1)
    
    def gradient_at(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy gradient at points.
        
        Args:
            z: Latent points (batch_size, latent_dim)
        
        Returns:
            Gradients (batch_size, latent_dim)
        """
        z = z.clone().requires_grad_(True)
        energy = self.energy(z)
        grad, = torch.autograd.grad(
            energy.sum(), z, create_graph=False
        )
        return grad
    
    def hessian_at(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy Hessian at a single point.
        
        Args:
            z: Single latent point (latent_dim,)
        
        Returns:
            Hessian matrix (latent_dim, latent_dim)
        """
        z = z.clone().requires_grad_(True)
        energy = self.energy(z.unsqueeze(0)).squeeze()
        
        # Compute Hessian via double differentiation
        hessian = torch.zeros(self.latent_dim, self.latent_dim)
        grad = torch.autograd.grad(energy, z, create_graph=True)[0]
        
        for i in range(self.latent_dim):
            hessian[i] = torch.autograd.grad(
                grad[i], z, retain_graph=True
            )[0]
        
        return hessian
    
    def local_curvature(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute local curvature (trace of Hessian).
        
        Positive curvature = convex basin (attractor)
        Negative curvature = saddle point
        
        Args:
            z: Single latent point (latent_dim,)
        
        Returns:
            Scalar curvature value
        """
        hess = self.hessian_at(z)
        return torch.trace(hess)
    
    def info_dict(self) -> dict:
        """
        Return information about the energy model.
        
        Returns:
            Dictionary with model info
        """
        return {
            'type': self.__class__.__name__,
            'latent_dim': self.latent_dim,
            'n_mem': self.n_mem,
            'memory_norm': torch.norm(self.mem, dim=1).mean().item(),
            'n_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(latent_dim={self.latent_dim}, n_mem={self.n_mem})"
