import torch
import torch.nn as nn

def _hsic(K, L):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - (1.0/n) * torch.ones((n,n), device=K.device)
    HKH = H @ K @ H
    HLH = H @ L @ H
    return torch.trace(HKH @ HLH) / ((n-1)**2 + 1e-8)

def rbf_kernel(x, sigma=None):
    x2 = (x**2).sum(dim=1, keepdim=True)
    d2 = x2 + x2.t() - 2 * x @ x.t()
    if sigma is None:
        med = torch.median(d2.detach())
        sigma = torch.sqrt(0.5 * med + 1e-8)
    return torch.exp(-d2 / (2*sigma**2 + 1e-8))

class HSICIndependenceLoss(nn.Module):
    """Penalize dependence between Z and N (nuisance) to encourage causal factors."""
    def forward(self, Z: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        return _hsic(rbf_kernel(Z), rbf_kernel(N))

class IRMLoss(nn.Module):
    """IRM surrogate: penalize gradient of risk wrt scaling across environments."""
    def __init__(self, criterion=None):
        super().__init__()
        self.crit = nn.MSELoss() if criterion is None else criterion

    def forward(self, preds, ys) -> torch.Tensor:
        penalty = 0.0
        for yhat, y in zip(preds, ys):
            scale = torch.tensor(1.0, requires_grad=True, device=yhat.device)
            loss_e = self.crit(yhat * scale, y)
            grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
            penalty = penalty + torch.sum(grad**2)
        return penalty / max(1, len(preds))
