"""
IRM (Invariant Risk Minimization) loss.

Implements environment-invariant learning by penalizing gradient variance across environments.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class IRMLoss(nn.Module):
    """
    IRM (Invariant Risk Minimization) Loss.
    
    Encourages learning representations that work invariantly across different
    environments by penalizing gradient variance of per-environment risks.
    
    The IRM penalty measures how much the optimal classifier weight varies
    across environments. If a representation captures true causal features,
    the same classifier should work well in all environments.
    
    Based on: Arjovsky et al. (2019) "Invariant Risk Minimization"
    
    Example:
        >>> loss_fn = IRMLoss()
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        >>> 
        >>> # Create environments (different data distributions)
        >>> envs = [
        >>>     (torch.randn(50, 10), torch.randint(0, 2, (50,))),  # Env 1
        >>>     (torch.randn(50, 10), torch.randint(0, 2, (50,))),  # Env 2
        >>> ]
        >>> 
        >>> # Training loop
        >>> for X_e, y_e in envs:
        >>>     logits = model(X_e)
        >>>     task_loss += F.cross_entropy(logits, y_e)
        >>> 
        >>> irm_penalty = loss_fn(model, envs)
        >>> total_loss = task_loss + lambda_irm * irm_penalty
        >>> total_loss.backward()
    
    References:
        - Arjovsky et al. 2019: "Invariant Risk Minimization" (arXiv:1907.02893)
        - APS Paper Section 4.1: Multi-environment IRM loss
    """
    
    def __init__(self):
        """Initialize IRM loss with dummy scale parameter."""
        super().__init__()
        
        # Dummy scale parameter for computing gradients
        # Initialized to 1.0 (identity scaling)
        self.dummy_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self, 
        model: nn.Module, 
        envs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute IRM penalty across environments.
        
        Args:
            model: Model that produces logits (e.g., classifier)
            envs: List of (X, y) tuples, one per environment
                  X: Input features (batch_size, input_dim)
                  y: Labels (batch_size,) as class indices
        
        Returns:
            IRM penalty (scalar tensor, non-negative)
            Higher penalty indicates environment-specific features
        
        Algorithm:
            For each environment e:
                1. Compute logits: logits_e = model(X_e)
                2. Scale by dummy: scaled_logits = dummy_scale * logits_e
                3. Compute risk: risk_e = CrossEntropy(scaled_logits, y_e)
                4. Compute gradient: grad_e = ∇_{dummy_scale} risk_e
            
            Penalty = Σ_e ||grad_e||²
        
        Raises:
            ValueError: If fewer than 2 environments provided
        
        Note:
            The dummy scale trick allows computing gradients that
            indicate how much the optimal classifier varies across envs.
        """
        if len(envs) < 2:
            raise ValueError(f"IRM requires at least 2 environments, got {len(envs)}")
        
        penalties = []
        
        for X_e, y_e in envs:
            # Forward pass through model
            logits_e = model(X_e)
            
            # Scale logits by dummy parameter
            scaled_logits = self.dummy_scale * logits_e
            
            # Compute per-environment risk (cross-entropy)
            risk_e = nn.functional.cross_entropy(scaled_logits, y_e)
            
            # Compute gradient of risk w.r.t. dummy_scale
            # create_graph=True to allow backprop through this gradient
            grad = torch.autograd.grad(
                risk_e,
                self.dummy_scale,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Squared gradient norm
            penalty_e = grad ** 2
            penalties.append(penalty_e)
        
        # Sum penalties across environments
        total_penalty = sum(penalties) / len(envs)
        
        return total_penalty
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'dummy_scale={self.dummy_scale.item():.4f}'
