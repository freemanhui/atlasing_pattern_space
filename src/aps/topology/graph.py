"""
Graph construction utilities for topology preservation.

This module provides functions to compute k-nearest neighbor graphs
and convert them to adjacency matrices for topology-preserving losses.
"""

from typing import Tuple
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_indices(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest neighbor indices for each point in X.
    
    Args:
        X: Input tensor of shape (n_samples, n_features)
        k: Number of nearest neighbors (excluding self)
    
    Returns:
        Tensor of shape (n_samples, k) containing indices of k-nearest neighbors
        
    Raises:
        ValueError: If k >= n_samples or k < 1
    
    Example:
        >>> X = torch.randn(100, 50)
        >>> indices = knn_indices(X, k=8)
        >>> indices.shape
        torch.Size([100, 8])
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    
    n_samples = X.shape[0]
    
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= n_samples:
        raise ValueError(f"k must be < n_samples ({n_samples}), got {k}")
    
    # Convert to numpy for sklearn (more efficient for kNN)
    X_np = X.detach().cpu().numpy()
    
    # Use sklearn's efficient kNN implementation
    # n_neighbors = k + 1 because it includes the point itself
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
    nbrs.fit(X_np)
    
    # Get k+1 neighbors (includes self), then exclude self
    distances, indices = nbrs.kneighbors(X_np)
    
    # Exclude self (first column is always the point itself with distance 0)
    indices_no_self = indices[:, 1:]  # Shape: (n_samples, k)
    
    # Convert back to torch tensor on same device as input
    indices_tensor = torch.from_numpy(indices_no_self).long().to(X.device)
    
    return indices_tensor


def adjacency_from_knn(indices: torch.Tensor, n_samples: int = None) -> torch.Tensor:
    """
    Convert k-NN indices to binary adjacency matrix.
    
    Args:
        indices: k-NN indices tensor of shape (n_samples, k)
        n_samples: Number of samples (inferred from indices if None)
    
    Returns:
        Binary adjacency matrix of shape (n_samples, n_samples)
        A[i, j] = 1 if j is a k-NN of i, else 0
        
    Example:
        >>> indices = torch.tensor([[1, 2], [0, 2], [0, 1]])  # 3 samples, k=2
        >>> adj = adjacency_from_knn(indices)
        >>> adj.shape
        torch.Size([3, 3])
        >>> adj[0, 1]  # Sample 0 has sample 1 as neighbor
        tensor(1.)
    """
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {indices.shape}")
    
    n_samples_inferred = indices.shape[0]
    max_index = indices.max().item() if indices.numel() > 0 else -1
    
    if n_samples is None:
        # Infer from max(indices.shape[0], max(indices) + 1)
        n_samples = max(n_samples_inferred, max_index + 1 if max_index >= 0 else n_samples_inferred)
    else:
        if n_samples < n_samples_inferred:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= indices.shape[0] ({n_samples_inferred})"
            )
        if max_index >= n_samples:
            raise ValueError(
                f"indices contain values >= n_samples ({n_samples}), max index is {max_index}"
            )
    
    k = indices.shape[1]
    
    # Create adjacency matrix
    adjacency = torch.zeros(n_samples, n_samples, dtype=torch.float32, device=indices.device)
    
    # Set A[i, j] = 1 for all j in kNN(i)
    row_indices = torch.arange(n_samples_inferred, device=indices.device).unsqueeze(1).expand(-1, k)
    adjacency[row_indices, indices] = 1.0
    
    return adjacency


def knn_graph(X: torch.Tensor, k: int, continuous: bool = True) -> torch.Tensor:
    """
    Compute k-NN graph (adjacency matrix) in one step.
    
    Convenience function that combines knn_indices and adjacency_from_knn.
    Can compute either discrete (binary) or continuous (differentiable) adjacency.
    
    Args:
        X: Input tensor of shape (n_samples, n_features)
        k: Number of nearest neighbors (excluding self)
        continuous: If True, use continuous relaxation (differentiable).
                   If False, use hard binary adjacency (non-differentiable).
    
    Returns:
        Adjacency matrix of shape (n_samples, n_samples)
        - If continuous=True: soft adjacency in [0, 1] using sigmoid activation
        - If continuous=False: binary adjacency {0, 1}
        
    Example:
        >>> X = torch.randn(100, 50, requires_grad=True)
        >>> adj = knn_graph(X, k=8, continuous=True)
        >>> adj.shape
        torch.Size([100, 100])
        >>> adj.requires_grad
        True
        
    Note:
        The continuous version uses the approach from Chen et al. (2022):
        A_ij = sigmoid(temperature * (d_k - d_ij))
        where d_k is the distance to the k-th nearest neighbor.
    """
    if continuous:
        return _continuous_knn_graph(X, k)
    else:
        indices = knn_indices(X, k)
        adjacency = adjacency_from_knn(indices, n_samples=X.shape[0])
        return adjacency


def _continuous_knn_graph(X: torch.Tensor, k: int, temperature: float = 10.0) -> torch.Tensor:
    """
    Compute continuous (differentiable) k-NN adjacency matrix.
    
    Uses sigmoid activation based on pairwise distances:
    A_ij = sigmoid(temperature * (d_k[i] - d_ij))
    
    where d_ij is the distance from point i to point j,
    and d_k[i] is the distance from point i to its k-th nearest neighbor.
    
    This creates a soft, differentiable version of the k-NN graph where:
    - Points closer than the k-th neighbor have adjacency close to 1
    - Points farther than the k-th neighbor have adjacency close to 0
    - The transition is smooth (differentiable)
    
    Args:
        X: Input tensor of shape (n_samples, n_features)
        k: Number of nearest neighbors
        temperature: Sharpness of sigmoid (higher = sharper transition)
    
    Returns:
        Continuous adjacency matrix of shape (n_samples, n_samples)
        Values in [0, 1], differentiable w.r.t. X
        
    References:
        Chen et al. (2022) "Local Distance Preserving Auto-encoders 
        using Continuous k-Nearest Neighbours Graphs"
    """
    n_samples = X.shape[0]
    
    # Compute pairwise distances: (n_samples, n_samples)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    XX = torch.sum(X * X, dim=1, keepdim=True)  # (n, 1)
    distances = XX + XX.t() - 2 * torch.mm(X, X.t())  # (n, n)
    distances = torch.clamp(distances, min=0.0)  # Numerical stability
    distances = torch.sqrt(distances + 1e-8)  # Add epsilon for numerical stability
    
    # For each point, find the distance to its k-th nearest neighbor
    # Sort distances along each row (ascending)
    sorted_dists, _ = torch.sort(distances, dim=1)
    # Index k (0-indexed) gives the k-th nearest neighbor (excluding self at index 0)
    d_k = sorted_dists[:, k]  # Shape: (n_samples,)
    
    # Compute continuous adjacency using sigmoid
    # A_ij = sigmoid(temperature * (d_k[i] - d_ij))
    # If d_ij < d_k[i] (closer than k-th neighbor), sigmoid input is positive → A_ij ≈ 1
    # If d_ij > d_k[i] (farther than k-th neighbor), sigmoid input is negative → A_ij ≈ 0
    d_k_expanded = d_k.unsqueeze(1)  # (n_samples, 1)
    adjacency = torch.sigmoid(temperature * (d_k_expanded - distances))
    
    # Zero out self-loops (diagonal)
    adjacency = adjacency * (1 - torch.eye(n_samples, device=X.device))
    
    return adjacency
