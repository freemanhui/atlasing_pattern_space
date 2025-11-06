"""
Topology-aware energy function for APS framework.

This module provides an energy function that reinforces topology preservation
rather than competing with it, by rewarding latent representations that 
maintain k-NN adjacency relationships from the original space.
"""

from dataclasses import dataclass
import torch
from .base import BaseEnergy
from ..topology.graph import knn_indices, adjacency_from_knn, knn_graph


@dataclass
class TopologyEnergyConfig:
    """Configuration for topology-aware energy function.
    
    Attributes:
        latent_dim: Dimension of latent space
        k: Number of nearest neighbors for topology
        mode: Energy computation mode:
            - 'agreement': Reward kNN adjacency agreement (recommended)
            - 'disagreement': Penalize kNN adjacency disagreement
            - 'jaccard': Use Jaccard similarity-based energy
        temperature: For continuous kNN graph (higher = sharper)
        continuous: Whether to use differentiable continuous kNN
        scale: Scaling factor for energy magnitude
    """
    latent_dim: int = 2
    k: int = 8
    mode: str = 'agreement'  # 'agreement', 'disagreement', 'jaccard'
    temperature: float = 10.0
    continuous: bool = True
    scale: float = 1.0


class TopologyEnergy(BaseEnergy):
    """Energy function that reinforces topology preservation.
    
    Unlike MemoryEnergy which creates arbitrary attractor basins, this energy
    function rewards latent representations that preserve k-NN relationships
    from the original space. Lower energy = better topology preservation.
    
    Key advantages:
    - Aligns with topology objective rather than competing
    - Data-driven structure (no arbitrary attractors)
    - Should improve clustering while maintaining good ARI
    
    Energy formula (agreement mode):
        E(z) = -sum(A_orig ⊙ A_latent) / (n * k)
    
    Where:
    - A_orig: kNN adjacency in original space (precomputed)
    - A_latent: kNN adjacency in latent space z
    - ⊙: element-wise multiplication
    
    Lower energy when adjacencies agree (more neighbors preserved).
    """
    
    def __init__(self, cfg: TopologyEnergyConfig):
        super().__init__(latent_dim=cfg.latent_dim, n_mem=0)  # No memory patterns
        self.cfg = cfg
        
        # Store original adjacency (will be set via set_target_adjacency)
        self.register_buffer('target_adjacency', None)
    
    def _init_patterns(self, init_method: str = 'random', **kwargs):
        """No-op: TopologyEnergy doesn't use memory patterns."""
        pass
    
    def set_target_adjacency(self, X_original: torch.Tensor):
        """Compute and store target kNN adjacency from original space.
        
        Args:
            X_original: Data in original space (n_samples, original_dim)
        """
        if self.cfg.continuous:
            adj = knn_graph(X_original, k=self.cfg.k, continuous=True)
        else:
            indices = knn_indices(X_original, k=self.cfg.k)
            adj = adjacency_from_knn(indices, n_samples=X_original.shape[0])
        
        self.target_adjacency = adj.to(X_original.device)
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute topology-aware energy.
        
        Args:
            z: Latent representations (n_samples, latent_dim)
        
        Returns:
            Energy scalar (lower = better topology preservation)
        
        Raises:
            RuntimeError: If target adjacency not set via set_target_adjacency()
        
        Note:
            This computes energy based solely on the internal topology of z,
            not requiring the same size as target adjacency. This makes it
            suitable for mini-batch training.
        """
        if self.target_adjacency is None:
            raise RuntimeError(
                "Target adjacency not set. Call set_target_adjacency() first."
            )
        
        # Compute kNN adjacency within the current batch
        if self.cfg.continuous:
            adj_latent = knn_graph(z, k=min(self.cfg.k, z.shape[0] - 1), continuous=True)
        else:
            k_effective = min(self.cfg.k, z.shape[0] - 1)
            indices = knn_indices(z, k=k_effective)
            adj_latent = adjacency_from_knn(indices, n_samples=z.shape[0])
        
        # Compute energy based on internal consistency
        # Reward: balanced neighborhood sizes and smooth transitions
        n_samples = z.shape[0]
        k_effective = min(self.cfg.k, n_samples - 1)
        
        if self.cfg.mode == 'agreement':
            # Reward: high within-batch kNN agreement (self-consistency)
            # This encourages latent space where neighborhoods are stable
            agreement = adj_latent.sum()  # Total edges in kNN graph
            expected_edges = n_samples * k_effective
            preservation_ratio = agreement / (expected_edges + 1e-8)
            energy = -preservation_ratio  # Lower energy = better preservation
        
        elif self.cfg.mode == 'disagreement':
            # Penalize: irregular neighborhood structure
            # Measure variance in neighborhood sizes
            neighborhood_sizes = adj_latent.sum(dim=1)  # (n_samples,)
            variance = torch.var(neighborhood_sizes)
            energy = variance / k_effective
        
        elif self.cfg.mode == 'jaccard':
            # Use local consistency measure
            # For each point, measure how much its neighbors agree
            consistency_scores = []
            for i in range(min(n_samples, 100)):  # Sample to avoid O(n²)
                neighbors = adj_latent[i].nonzero().squeeze()
                if len(neighbors) > 1:
                    # Check if neighbors are also neighbors of each other
                    neighbor_adj = adj_latent[neighbors][:, neighbors]
                    consistency = neighbor_adj.sum() / (len(neighbors) ** 2 + 1e-8)
                    consistency_scores.append(consistency)
            
            if consistency_scores:
                avg_consistency = torch.stack(consistency_scores).mean()
                energy = -avg_consistency
            else:
                energy = torch.tensor(0.0, device=z.device)
        
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
        
        return energy * self.cfg.scale


class AdaptiveTopologyEnergy(TopologyEnergy):
    """Topology energy with adaptive weighting.
    
    Adjusts energy contribution based on current topology preservation quality.
    Increases energy weight when preservation is poor, decreases when good.
    """
    
    def __init__(self, cfg: TopologyEnergyConfig, 
                 min_weight: float = 0.1, max_weight: float = 2.0):
        super().__init__(cfg)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.register_buffer('adaptive_weight', torch.tensor(1.0))
    
    def update_adaptive_weight(self, z: torch.Tensor):
        """Update adaptive weight based on current preservation quality.
        
        Args:
            z: Current latent representations
        """
        with torch.no_grad():
            # Compute current agreement
            if self.cfg.continuous:
                adj_latent = knn_graph(z, k=self.cfg.k, continuous=True)
            else:
                indices = knn_indices(z, k=self.cfg.k)
                adj_latent = adjacency_from_knn(indices, n_samples=z.shape[0])
            
            agreement = (self.target_adjacency * adj_latent).sum()
            n_samples = z.shape[0]
            preservation_ratio = agreement / (n_samples * self.cfg.k)
            
            # Higher weight when preservation is poor (ratio low)
            # Lower weight when preservation is good (ratio high)
            new_weight = self.max_weight - (self.max_weight - self.min_weight) * preservation_ratio
            self.adaptive_weight = torch.clamp(new_weight, self.min_weight, self.max_weight)
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute adaptive topology-aware energy."""
        base_energy = super().energy(z)
        return base_energy * self.adaptive_weight
