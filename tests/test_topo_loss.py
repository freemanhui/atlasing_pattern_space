import torch
from aps.topology.knn_topo_loss import KNNTopoLoss, adjacency_from_knn, knn_indices

def test_knn_topo_loss_shapes():
    X = torch.randn(20, 8)
    Z = torch.randn(20, 2)
    idx = knn_indices(X, k=5)
    A = adjacency_from_knn(idx, n=20)
    loss = KNNTopoLoss(k=5)(Z, A)
    assert loss.shape == ()
