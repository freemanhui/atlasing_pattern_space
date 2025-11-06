import torch
import torch.nn as nn

def knn_indices(arr: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(arr, arr, p=2.0)
    d.fill_diagonal_(float('inf'))
    k = min(k, arr.shape[0] - 1)  # Ensure k is not out of range
    idx = torch.topk(d, k=k, largest=False).indices  # (N,k)
    return idx

def adjacency_from_knn(idx: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros((n, n), dtype=torch.float32, device=idx.device)
    rows = torch.arange(n, device=idx.device).unsqueeze(1).repeat(1, idx.shape[1])
    A[rows, idx] = 1.0
    A[idx, rows] = 1.0
    return A

class KNNTopoLoss(nn.Module):
    """Offline-friendly surrogate topology loss via kNN adjacency matching."""
    def __init__(self, reduction='mean', k: int = 8):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction)
        self.k = k

    def forward(self, Z: torch.Tensor, target_adj: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(Z, k=self.k)
        A = adjacency_from_knn(idx, n=Z.shape[0])
        A = torch.maximum(A, A.t())
        return self.bce(A, target_adj)
