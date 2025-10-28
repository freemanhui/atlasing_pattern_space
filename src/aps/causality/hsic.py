"""
HSIC (Hilbert-Schmidt Independence Criterion) loss.

Implements independence testing between latent representations and nuisance variables.
"""

import torch
import torch.nn as nn
from .kernels import rbf_kernel, linear_kernel, center_kernel


class HSICLoss(nn.Module):
    """
    HSIC (Hilbert-Schmidt Independence Criterion) Loss.
    
    Measures statistical dependence between two random variables using
    kernel embeddings. Minimizing HSIC encourages independence.
    
    HSIC = 0 ⟺ Variables are independent
    HSIC > 0 indicates dependence (higher = stronger dependence)
    
    Based on: Gretton et al. (2005) "Measuring Statistical Dependence 
    with Hilbert-Schmidt Norms"
    
    Args:
        kernel: Kernel type, 'rbf' or 'linear' (default: 'rbf')
        sigma: Bandwidth for RBF kernel (default: 1.0)
    
    Example:
        >>> loss_fn = HSICLoss(kernel='rbf', sigma=1.0)
        >>> 
        >>> # Training loop (Colored MNIST)
        >>> for batch_x, batch_color in dataloader:
        >>>     z = encoder(batch_x)
        >>>     x_recon = decoder(z)
        >>>     
        >>>     recon_loss = F.mse_loss(x_recon, batch_x)
        >>>     hsic_loss = loss_fn(z, batch_color.unsqueeze(1))
        >>>     
        >>>     # Minimize HSIC to enforce independence
        >>>     loss = recon_loss + lambda_C * hsic_loss
        >>>     loss.backward()
    
    References:
        - Gretton et al. 2005: Original HSIC paper
        - Greenfeld & Shalit 2020: HSIC for robust learning (ICML)
        - APS Paper Section 4.1: L_C = HSIC(Z, V)
    """
    
    def __init__(self, kernel: str = 'rbf', sigma: float = 1.0):
        """
        Initialize HSIC loss.
        
        Args:
            kernel: 'rbf' or 'linear'
            sigma: Bandwidth for RBF kernel (ignored for linear kernel)
        
        Raises:
            ValueError: If kernel is not 'rbf' or 'linear'
        """
        super().__init__()
        
        if kernel not in ['rbf', 'linear']:
            raise ValueError(f"kernel must be 'rbf' or 'linear', got {kernel}")
        
        self.kernel = kernel
        self.sigma = sigma
    
    def forward(self, Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute HSIC between latent Z and nuisance V.
        
        Args:
            Z: Latent representations, shape (batch_size, latent_dim)
            V: Nuisance variables, shape (batch_size, nuisance_dim)
        
        Returns:
            HSIC value (scalar tensor, non-negative)
            Lower values indicate more independence
        
        Algorithm:
            1. Compute kernel matrices K_Z and K_V
            2. Center both matrices
            3. Return trace(K_Z_centered @ K_V_centered) / (n-1)²
        
        Note:
            Uses unbiased HSIC estimator with (n-1)² normalization.
        """
        n = Z.shape[0]
        
        # Compute kernel matrices
        if self.kernel == 'rbf':
            K_Z = rbf_kernel(Z, Z, sigma=self.sigma)
            K_V = rbf_kernel(V, V, sigma=self.sigma)
        else:  # linear
            K_Z = linear_kernel(Z, Z)
            K_V = linear_kernel(V, V)
        
        # Center kernel matrices
        K_Z_c = center_kernel(K_Z)
        K_V_c = center_kernel(K_V)
        
        # Compute HSIC (unbiased estimator)
        # HSIC = (1/n²) * trace(K_Z_c @ K_V_c)
        # This is the standard biased estimator that scales better
        # Efficient trace computation: trace(A @ B) = sum(A * B^T)
        hsic = torch.sum(K_Z_c * K_V_c) / (n ** 2)
        
        return hsic
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'kernel={self.kernel}, sigma={self.sigma}'
