"""
Topology-preserving losses for latent representations.

This module provides loss functions that encourage latent representations
to preserve the topological structure of the input space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import knn_graph


class KNNTopoLoss(nn.Module):
    """
    k-Nearest Neighbor Topology Preservation Loss.
    
    Compares k-NN graphs in input space and latent space using binary
    cross-entropy loss. Encourages the latent representation to preserve
    neighborhood relationships from the input space.
    
    Based on: Chen et al. (2022) "Local Distance Preserving Auto-encoders 
    using Continuous k-Nearest Neighbours Graphs"
    
    Args:
        k: Number of nearest neighbors (default: 8)
        loss_type: Type of loss to use, 'bce' for binary cross-entropy (default: 'bce')
    
    Example:
        >>> loss_fn = KNNTopoLoss(k=8)
        >>> X = torch.randn(50, 20)  # Input features
        >>> Z = torch.randn(50, 10)  # Latent representations
        >>> loss = loss_fn(X, Z)
        >>> loss.backward()
    
    References:
        - Paper Section 4.1: L_T = BCE(A_latent, A_input)
        - Chen et al. 2022: arXiv:2206.05909
    """
    
    def __init__(self, k: int = 8, loss_type: str = 'bce'):
        """
        Initialize topology loss.
        
        Args:
            k: Number of nearest neighbors
            loss_type: 'bce' for binary cross-entropy (only option currently)
        
        Raises:
            ValueError: If k < 1 or loss_type is invalid
        """
        super().__init__()
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        if loss_type not in ['bce']:
            raise ValueError(f"loss_type must be 'bce', got {loss_type}")
        
        self.k = k
        self.loss_type = loss_type
    
    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute topology preservation loss.
        
        Args:
            X: Input features, shape (batch_size, input_dim)
            Z: Latent representations, shape (batch_size, latent_dim)
        
        Returns:
            Scalar loss value (lower is better)
        
        Raises:
            ValueError: If batch_size < k + 1
        
        Note:
            The loss compares k-NN adjacency matrices:
            L_T = BCE(A_Z, A_X)
            where A_X[i,j] = 1 if j is a k-NN of i in X, else 0
        """
        if X.shape[0] != Z.shape[0]:
            raise ValueError(
                f"Batch sizes must match: X has {X.shape[0]}, Z has {Z.shape[0]}"
            )
        
        batch_size = X.shape[0]
        
        if batch_size <= self.k:
            raise ValueError(
                f"Batch size ({batch_size}) must be > k ({self.k})"
            )
        
        # Compute k-NN graphs
        # A_X: adjacency matrix in input space
        # A_Z: adjacency matrix in latent space
        A_X = knn_graph(X, self.k)  # (batch, batch)
        A_Z = knn_graph(Z, self.k)  # (batch, batch)
        
        # Compute BCE loss
        # We want A_Z to match A_X (latent neighbors = input neighbors)
        loss = F.binary_cross_entropy(A_Z, A_X, reduction='mean')
        
        return loss
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'k={self.k}, loss_type={self.loss_type}'
