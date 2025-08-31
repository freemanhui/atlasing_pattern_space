from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MemoryEnergyConfig:
    latent_dim: int = 2
    n_mem: int = 8
    beta: float = 5.0   # sharpness
    alpha: float = 0.0  # L2 on z

class MemoryEnergy(nn.Module):
    """Memory-based log-sum-exp energy; lower near memory patterns => attractors."""
    def __init__(self, cfg: MemoryEnergyConfig):
        super().__init__()
        self.cfg = cfg
        self.mem = nn.Parameter(torch.randn(cfg.n_mem, cfg.latent_dim) * 0.5)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        dots = z @ self.mem.t()       # (N, n_mem)
        lse  = torch.logsumexp(self.cfg.beta * dots, dim=1)
        e    = 0.5 * self.cfg.alpha * torch.sum(z**2, dim=1) - lse
        return e

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.energy(z))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return -self.energy(z)  # higher score near memories
