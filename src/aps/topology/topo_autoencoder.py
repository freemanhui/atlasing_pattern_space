from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from .knn_topo_loss import KNNTopoLoss

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, x): return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, out_dim, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z): return self.net(z)

@dataclass
class TopoAEConfig:
    in_dim: int
    latent_dim: int = 2
    hidden: int = 64
    lr: float = 1e-3
    topo_k: int = 8
    topo_weight: float = 1.0

class TopologicalAutoencoder(nn.Module):
    def __init__(self, cfg: TopoAEConfig):
        super().__init__()
        self.enc = MLPEncoder(cfg.in_dim, cfg.latent_dim, cfg.hidden)
        self.dec = MLPDecoder(cfg.in_dim, cfg.latent_dim, cfg.hidden)
        self.mse = nn.MSELoss()
        self.topo = KNNTopoLoss(k=cfg.topo_k)
        self.cfg = cfg

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return z, xhat

    def fit(self, X: torch.Tensor, target_adj: torch.Tensor, epochs: int = 100):
        opt = optim.Adam(self.parameters(), lr=self.cfg.lr)
        for ep in range(epochs):
            z, xhat = self(X)
            loss_recon = self.mse(xhat, X)
            loss_topo = self.topo(z, target_adj)
            loss = loss_recon + self.cfg.topo_weight * loss_topo
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (ep+1) % max(1, epochs//5) == 0:
                print(f"[TopoAE] {ep+1}/{epochs} recon={loss_recon.item():.4f} topo={loss_topo.item():.4f}")
        return self
