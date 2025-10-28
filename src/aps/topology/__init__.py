"""Topology preservation module for APS framework.

This module provides utilities and losses for preserving topological structure
in latent representations using k-NN graph-based approaches.
"""

from .graph import knn_indices, adjacency_from_knn, knn_graph
from .losses import KNNTopoLoss
from .model import TopologicalAutoencoder, TopoAEConfig

__all__ = [
    'knn_indices',
    'adjacency_from_knn',
    'knn_graph',
    'KNNTopoLoss',
    'TopologicalAutoencoder',
    'TopoAEConfig',
]
