"""
Kernel functions for causality module.

Provides RBF and linear kernels used in HSIC independence testing.
"""

import torch


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel matrix between X and Y.
    
    The RBF kernel is defined as:
        K(x, y) = exp(-||x - y||² / (2σ²))
    
    Args:
        X: Input tensor of shape (n_samples_X, n_features)
        Y: Input tensor of shape (n_samples_Y, n_features)
        sigma: Bandwidth parameter (default: 1.0)
    
    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y)
        
    Properties:
        - K(x, x) = 1.0 (self-similarity)
        - K(x, y) ∈ [0, 1] (similarity measure)
        - Symmetric: K(x, y) = K(y, x)
        - Positive semi-definite
    
    Example:
        >>> X = torch.randn(100, 50)
        >>> Y = torch.randn(80, 50)
        >>> K = rbf_kernel(X, Y, sigma=1.0)
        >>> K.shape
        torch.Size([100, 80])
        >>> K[0, 0]  # Self-similarity (if X == Y)
        tensor(1.0000)
    
    Note:
        Smaller sigma → more local (kernel values decay faster)
        Larger sigma → more global (kernel values decay slower)
    """
    # Compute pairwise squared Euclidean distances
    # ||x - y||² = ||x||² + ||y||² - 2x·y
    XX = torch.sum(X * X, dim=1, keepdim=True)  # (n_X, 1)
    YY = torch.sum(Y * Y, dim=1, keepdim=True)  # (n_Y, 1)
    XY = torch.mm(X, Y.t())  # (n_X, n_Y)
    
    # Squared distances: (n_X, n_Y)
    distances_sq = XX + YY.t() - 2 * XY
    
    # Numerical stability: clamp negative values (from floating point errors)
    distances_sq = torch.clamp(distances_sq, min=0.0)
    
    # Compute RBF kernel: exp(-d²/(2σ²))
    gamma = 1.0 / (2.0 * sigma ** 2)
    K = torch.exp(-gamma * distances_sq)
    
    return K


def linear_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute linear kernel matrix between X and Y.
    
    The linear kernel is simply the dot product:
        K(x, y) = x · y
    
    Args:
        X: Input tensor of shape (n_samples_X, n_features)
        Y: Input tensor of shape (n_samples_Y, n_features)
    
    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y)
        
    Properties:
        - K(x, y) = x · y (dot product)
        - Symmetric: K(x, y) = K(y, x)
        - Positive semi-definite
        - For normalized vectors: K(x, y) ∈ [-1, 1]
    
    Example:
        >>> X = torch.randn(100, 50)
        >>> Y = torch.randn(80, 50)
        >>> K = linear_kernel(X, Y)
        >>> K.shape
        torch.Size([100, 80])
    
    Note:
        Linear kernel is equivalent to computing in original feature space
        (no transformation to higher-dimensional space).
    """
    return torch.mm(X, Y.t())


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """
    Center a kernel matrix using the centering matrix H.
    
    Centered kernel: K_centered = H @ K @ H
    where H = I - (1/n) * 1·1ᵀ is the centering matrix.
    
    This removes the mean from the kernel features, which is necessary
    for HSIC computation.
    
    Args:
        K: Kernel matrix of shape (n_samples, n_samples)
    
    Returns:
        Centered kernel matrix of shape (n_samples, n_samples)
        
    Properties:
        - Row sums = 0
        - Column sums = 0
        - Preserves symmetry
        - Idempotent: center(center(K)) = center(K)
    
    Example:
        >>> K = rbf_kernel(X, X, sigma=1.0)
        >>> K_centered = center_kernel(K)
        >>> K_centered.sum(dim=0)  # Column sums
        tensor([0., 0., ..., 0.])
        >>> K_centered.sum(dim=1)  # Row sums
        tensor([0., 0., ..., 0.])
    
    Note:
        Centering is essential for HSIC to measure dependence
        relative to the mean, not absolute values.
    """
    # Create centering matrix: H = I - (1/n) * 1·1ᵀ
    # Instead of explicitly creating H, we use efficient computation:
    # H @ K @ H = K - K_row_mean - K_col_mean + K_mean
    
    # Compute row and column means
    K_row_mean = K.mean(dim=1, keepdim=True)  # (n, 1)
    K_col_mean = K.mean(dim=0, keepdim=True)  # (1, n)
    K_mean = K.mean()  # scalar
    
    # Center the kernel
    K_centered = K - K_row_mean - K_col_mean + K_mean
    
    return K_centered
