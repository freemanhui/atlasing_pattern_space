"""Topology preservation module for APS framework.

This module provides utilities and losses for preserving topological structure
in latent representations using k-NN graph-based approaches.
"""

from .graph import knn_indices, adjacency_from_knn, knn_graph

__all__ = [
    'knn_indices',
    'adjacency_from_knn',
    'knn_graph',
]

from .topo_autoencoder import TopologicalAutoencoder, TopoAEConfig
from .knn_topo_loss import KNNTopoLoss, knn_indices, adjacency_from_knn

__all__ = ['TopologicalAutoencoder', 'TopoAEConfig', 'KNNTopoLoss', 'knn_indices', 'adjacency_from_knn']
