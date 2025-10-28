"""
Utilities for APS experiments.
"""

from .datasets import (
    get_mnist_dataloaders,
    get_colored_mnist_dataloaders,
    ColoredMNIST,
    create_few_shot_split,
    get_embeddings_from_model,
)

from .metrics import (
    compute_topology_metrics,
    compute_clustering_metrics,
    compute_hsic,
    compute_reconstruction_error,
    few_shot_accuracy,
    evaluate_model_comprehensive,
)

__all__ = [
    # Datasets
    'get_mnist_dataloaders',
    'get_colored_mnist_dataloaders',
    'ColoredMNIST',
    'create_few_shot_split',
    'get_embeddings_from_model',
    # Metrics
    'compute_topology_metrics',
    'compute_clustering_metrics',
    'compute_hsic',
    'compute_reconstruction_error',
    'few_shot_accuracy',
    'evaluate_model_comprehensive',
]
