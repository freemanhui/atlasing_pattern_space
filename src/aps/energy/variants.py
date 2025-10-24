"""
Energy Variants

Additional energy function implementations beyond the basic MemoryEnergy.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from .base import BaseEnergy
from .init import get_initializer


@dataclass
class RBFEnergyConfig:
    latent_dim: int = 2
    n_mem: int = 8
    sigma: float = 1.0  # RBF width (per-pattern if list)
    init_method: str = 'random'
    learnable_sigma: bool = False  # Learn sigma during training


class RBFEnergy(BaseEnergy):
    """
    RBF (Radial Basis Function) energy with Gaussian basins.
    
    Energy is negative log of sum of Gaussians centered at memory patterns.
    E(z) = -log(Σ exp(-||z - m_i||² / (2σ_i²)))
    
    Provides smooth, isotropic basins around each pattern.
    """
    
    def __init__(self, cfg: RBFEnergyConfig):
        super().__init__(latent_dim=cfg.latent_dim, n_mem=cfg.n_mem)
        self.cfg = cfg
        self._init_patterns(cfg.init_method)
        
        # Initialize sigma (width parameter)
        if isinstance(cfg.sigma, (list, tuple)):
            sigma_init = torch.tensor(cfg.sigma, dtype=torch.float32)
        else:
            sigma_init = torch.full((cfg.n_mem,), cfg.sigma, dtype=torch.float32)
        
        if cfg.learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(sigma_init))
        else:
            self.register_buffer('log_sigma', torch.log(sigma_init))
    
    def _init_patterns(self, init_method: str = 'random', **kwargs):
        init_fn = get_initializer(init_method)
        
        if init_method == 'cube':
            patterns = init_fn(self.latent_dim)
        else:
            patterns = init_fn(self.n_mem, self.latent_dim, **kwargs)
        
        self.mem = nn.Parameter(patterns)
    
    @property
    def sigma(self) -> torch.Tensor:
        """Get current sigma values."""
        return torch.exp(self.log_sigma)
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute RBF energy."""
        # Squared distances to each pattern: (batch, n_mem)
        dists_sq = torch.cdist(z, self.mem).pow(2)
        
        # Gaussian kernel
        sigma_sq = self.sigma.pow(2)
        gaussians = torch.exp(-dists_sq / (2 * sigma_sq.unsqueeze(0)))
        
        # Energy is negative log-sum
        energy = -torch.log(gaussians.sum(dim=1) + 1e-8)
        
        return energy


@dataclass
class MixtureEnergyConfig:
    latent_dim: int = 2
    n_mem: int = 8
    init_method: str = 'random'
    learnable_weights: bool = True  # Learn mixture weights
    learnable_sharpness: bool = True  # Learn per-pattern beta
    init_beta: float = 5.0
    init_weight: float = 1.0


class MixtureEnergy(BaseEnergy):
    """
    Mixture model energy with learnable per-pattern parameters.
    
    Each memory pattern has:
    - weight (w_i): Importance of pattern
    - beta (β_i): Sharpness of basin
    
    E(z) = -log(Σ w_i exp(β_i · z·m_i))
    
    More flexible than MemoryEnergy with global parameters.
    """
    
    def __init__(self, cfg: MixtureEnergyConfig):
        super().__init__(latent_dim=cfg.latent_dim, n_mem=cfg.n_mem)
        self.cfg = cfg
        self._init_patterns(cfg.init_method)
        
        # Initialize weights (in log space for stability)
        log_weights = torch.log(torch.full((cfg.n_mem,), cfg.init_weight))
        if cfg.learnable_weights:
            self.log_weights = nn.Parameter(log_weights)
        else:
            self.register_buffer('log_weights', log_weights)
        
        # Initialize per-pattern sharpness
        log_beta = torch.log(torch.full((cfg.n_mem,), cfg.init_beta))
        if cfg.learnable_sharpness:
            self.log_beta = nn.Parameter(log_beta)
        else:
            self.register_buffer('log_beta', log_beta)
    
    def _init_patterns(self, init_method: str = 'random', **kwargs):
        init_fn = get_initializer(init_method)
        
        if init_method == 'cube':
            patterns = init_fn(self.latent_dim)
        else:
            patterns = init_fn(self.n_mem, self.latent_dim, **kwargs)
        
        self.mem = nn.Parameter(patterns)
    
    @property
    def weights(self) -> torch.Tensor:
        """Get normalized mixture weights."""
        return torch.softmax(self.log_weights, dim=0)
    
    @property
    def beta(self) -> torch.Tensor:
        """Get per-pattern sharpness values."""
        return torch.exp(self.log_beta)
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute mixture energy."""
        # Dot products: (batch, n_mem)
        dots = z @ self.mem.t()
        
        # Weighted and sharpened scores
        beta = self.beta.unsqueeze(0)  # (1, n_mem)
        weights = self.weights.unsqueeze(0)  # (1, n_mem)
        
        scores = weights * torch.exp(beta * dots)
        
        # Energy is negative log-sum
        energy = -torch.log(scores.sum(dim=1) + 1e-8)
        
        return energy
    
    def info_dict(self) -> dict:
        """Extended info with per-pattern parameters."""
        base_info = super().info_dict()
        base_info.update({
            'weights': self.weights.detach().cpu().tolist(),
            'beta_values': self.beta.detach().cpu().tolist(),
            'weight_entropy': -torch.sum(self.weights * torch.log(self.weights + 1e-8)).item()
        })
        return base_info
