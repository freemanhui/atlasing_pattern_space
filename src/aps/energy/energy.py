from dataclasses import dataclass
import torch
import torch.nn as nn
from .base import BaseEnergy
from .init import get_initializer

@dataclass
class MemoryEnergyConfig:
    latent_dim: int = 2
    n_mem: int = 8
    beta: float = 5.0   # sharpness
    alpha: float = 0.0  # L2 on z
    init_method: str = 'random'  # initialization strategy
    init_scale: float = 0.5  # for random init

class MemoryEnergy(BaseEnergy):
    """Memory-based log-sum-exp energy; lower near memory patterns => attractors."""
    def __init__(self, cfg: MemoryEnergyConfig):
        super().__init__(latent_dim=cfg.latent_dim, n_mem=cfg.n_mem)
        self.cfg = cfg
        self._init_patterns(cfg.init_method, scale=cfg.init_scale)
    
    def _init_patterns(self, init_method: str = 'random', **kwargs):
        """Initialize memory patterns using specified method."""
        init_fn = get_initializer(init_method)
        
        if init_method == 'cube':
            patterns = init_fn(self.latent_dim)
        elif init_method in ['kmeans', 'pca']:
            # These require data, use random fallback
            patterns = get_initializer('random')(self.n_mem, self.latent_dim, scale=kwargs.get('scale', 0.5))
        elif init_method == 'grid':
            patterns = init_fn(self.n_mem, self.latent_dim, bounds=kwargs.get('bounds', (-1.0, 1.0)))
        elif init_method == 'sphere':
            patterns = init_fn(self.n_mem, self.latent_dim, radius=kwargs.get('radius', 1.0))
        elif init_method == 'random':
            patterns = init_fn(self.n_mem, self.latent_dim, scale=kwargs.get('scale', 0.5))
        else:
            patterns = init_fn(self.n_mem, self.latent_dim)
        
        self.mem = nn.Parameter(patterns)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log-sum-exp energy."""
        dots = z @ self.mem.t()       # (N, n_mem)
        lse  = torch.logsumexp(self.cfg.beta * dots, dim=1)
        e    = 0.5 * self.cfg.alpha * torch.sum(z**2, dim=1) - lse
        return e
