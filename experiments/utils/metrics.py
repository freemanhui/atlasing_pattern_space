"""
Evaluation metrics for APS experiments.

Includes:
- Topology metrics (trustworthiness, continuity, kNN preservation)
- Clustering metrics (ARI, NMI, silhouette)
- Independence metrics (HSIC)
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from typing import Tuple, Optional
from aps.metrics.topo import trustworthiness, knn_preservation


def compute_topology_metrics(
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Compute topology preservation metrics.
    
    Args:
        X_orig: Original space data (N, D_orig)
        X_emb: Embedded space data (N, D_emb)
        k: Number of neighbors for metrics
    
    Returns:
        Dictionary with:
            - trustworthiness: [0, 1], higher is better
            - continuity: [0, 1], higher is better
            - knn_preservation: [0, 1], higher is better
    """
    # Use existing APS metrics
    trust = trustworthiness(X_orig, X_emb, k=k)
    
    # Continuity is trustworthiness with swapped arguments
    cont = trustworthiness(X_emb, X_orig, k=k)
    
    # kNN preservation (Jaccard similarity)
    knn_pres = knn_preservation(X_orig, X_emb, k=k)
    
    return {
        'trustworthiness': float(trust),
        'continuity': float(cont),
        'knn_preservation': float(knn_pres),
    }


def compute_clustering_metrics(
    X: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute clustering quality metrics.
    
    Args:
        X: Data points (N, D)
        labels_true: Ground truth labels (N,)
        labels_pred: Predicted cluster labels (N,). If None, uses k-means
    
    Returns:
        Dictionary with:
            - ari: Adjusted Rand Index [-1, 1], higher is better
            - nmi: Normalized Mutual Information [0, 1], higher is better
            - silhouette: Silhouette score [-1, 1], higher is better
    """
    from sklearn.cluster import KMeans
    
    # If no predictions, run k-means
    if labels_pred is None:
        n_clusters = len(np.unique(labels_true))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_pred = kmeans.fit_predict(X)
    
    # Compute metrics
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Silhouette requires at least 2 clusters
    if len(np.unique(labels_pred)) > 1:
        sil = silhouette_score(X, labels_pred)
    else:
        sil = 0.0
    
    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'silhouette': float(sil),
    }


def compute_hsic(
    Z: np.ndarray,
    V: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion).
    
    Measures statistical dependence between Z and V.
    Lower values indicate more independence.
    
    Args:
        Z: First variable (N, D1)
        V: Second variable (N, D2)
        sigma: RBF kernel bandwidth
    
    Returns:
        HSIC value (non-negative, lower = more independent)
    """
    from aps.causality.hsic import HSICLoss
    from aps.causality.kernels import rbf_kernel, center_kernel
    
    # Convert to torch tensors
    Z_torch = torch.from_numpy(Z).float()
    V_torch = torch.from_numpy(V).float()
    
    # Compute kernel matrices
    K_Z = rbf_kernel(Z_torch, Z_torch, sigma=sigma)
    K_V = rbf_kernel(V_torch, V_torch, sigma=sigma)
    
    # Center kernels
    K_Z_c = center_kernel(K_Z)
    K_V_c = center_kernel(K_V)
    
    # Compute HSIC
    n = Z.shape[0]
    hsic = torch.sum(K_Z_c * K_V_c) / (n ** 2)
    
    return float(hsic.item())


def compute_reconstruction_error(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    metric: str = 'mse',
) -> float:
    """
    Compute reconstruction error.
    
    Args:
        X_orig: Original data (N, D)
        X_recon: Reconstructed data (N, D)
        metric: 'mse' or 'mae'
    
    Returns:
        Reconstruction error
    """
    if metric == 'mse':
        return float(np.mean((X_orig - X_recon) ** 2))
    elif metric == 'mae':
        return float(np.mean(np.abs(X_orig - X_recon)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def few_shot_accuracy(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    method: str = 'nearest',
) -> float:
    """
    Compute few-shot classification accuracy.
    
    Args:
        support_embeddings: Support set embeddings (K*N_way, D)
        support_labels: Support set labels (K*N_way,)
        query_embeddings: Query set embeddings (N_query, D)
        query_labels: Query set labels (N_query,)
        method: 'nearest' for nearest neighbor, 'prototype' for prototypical
    
    Returns:
        Accuracy on query set
    """
    if method == 'nearest':
        # Simple nearest neighbor
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(support_embeddings)
        
        distances, indices = nn.kneighbors(query_embeddings)
        predictions = support_labels[indices.flatten()]
        
    elif method == 'prototype':
        # Prototypical networks
        unique_labels = np.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            prototype = support_embeddings[mask].mean(axis=0)
            prototypes.append(prototype)
        
        prototypes = np.stack(prototypes)
        
        # Predict based on nearest prototype
        distances = np.linalg.norm(
            query_embeddings[:, None, :] - prototypes[None, :, :],
            axis=2
        )
        predictions = unique_labels[distances.argmin(axis=1)]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    accuracy = (predictions == query_labels).mean()
    return float(accuracy)


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    X_orig: np.ndarray,
    X_emb: np.ndarray,
    labels: np.ndarray,
    X_recon: Optional[np.ndarray] = None,
    nuisance: Optional[np.ndarray] = None,
) -> dict:
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model: Trained model
        X_orig: Original data
        X_emb: Embedded data
        labels: True labels
        X_recon: Reconstructed data (optional)
        nuisance: Nuisance variables for HSIC (optional)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Topology metrics
    topo_metrics = compute_topology_metrics(X_orig, X_emb, k=10)
    metrics.update({f'topo/{k}': v for k, v in topo_metrics.items()})
    
    # Clustering metrics
    cluster_metrics = compute_clustering_metrics(X_emb, labels)
    metrics.update({f'cluster/{k}': v for k, v in cluster_metrics.items()})
    
    # Reconstruction error
    if X_recon is not None:
        recon_error = compute_reconstruction_error(X_orig, X_recon)
        metrics['reconstruction/mse'] = recon_error
    
    # Independence (HSIC)
    if nuisance is not None:
        hsic = compute_hsic(X_emb, nuisance)
        metrics['causality/hsic'] = hsic
    
    return metrics
