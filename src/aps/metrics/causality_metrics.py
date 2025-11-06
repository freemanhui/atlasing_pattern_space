"""
Causality-Specific Evaluation Metrics

Metrics for assessing causal invariance learning, particularly for ColoredMNIST
experiments where we want to verify that models learn to ignore spurious features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr, spearmanr


def color_label_correlation(
    latent_codes: torch.Tensor,
    color_indices: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'pearson'
) -> Dict[str, float]:
    """
    Measure correlation between latent codes and spurious color feature vs causal label.
    
    A good causal model should have low correlation with color (spurious) and high
    correlation with labels (causal).
    
    Args:
        latent_codes: Latent embeddings (N, latent_dim)
        color_indices: Color assigned to each sample (N,)
        labels: True digit labels (N,)
        method: 'pearson' or 'spearman'
        
    Returns:
        Dictionary with:
            - 'color_corr_mean': Mean absolute correlation with color across latent dims
            - 'color_corr_max': Max absolute correlation with color
            - 'label_corr_mean': Mean absolute correlation with label across latent dims
            - 'label_corr_max': Max absolute correlation with label
            - 'causal_ratio': label_corr_mean / color_corr_mean (higher is better)
    """
    latent_np = latent_codes.cpu().numpy()
    color_np = color_indices.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    latent_dim = latent_codes.shape[1]
    color_corrs = []
    label_corrs = []
    
    corr_fn = pearsonr if method == 'pearson' else spearmanr
    
    # Compute correlation for each latent dimension
    for d in range(latent_dim):
        z_d = latent_np[:, d]
        
        # Correlation with spurious color
        color_corr, _ = corr_fn(z_d, color_np)
        color_corrs.append(abs(color_corr))
        
        # Correlation with causal label
        label_corr, _ = corr_fn(z_d, labels_np)
        label_corrs.append(abs(label_corr))
    
    color_corr_mean = np.mean(color_corrs)
    color_corr_max = np.max(color_corrs)
    label_corr_mean = np.mean(label_corrs)
    label_corr_max = np.max(label_corrs)
    
    # Ratio: higher means model relies more on causal feature
    causal_ratio = label_corr_mean / (color_corr_mean + 1e-8)
    
    return {
        'color_corr_mean': float(color_corr_mean),
        'color_corr_max': float(color_corr_max),
        'label_corr_mean': float(label_corr_mean),
        'label_corr_max': float(label_corr_max),
        'causal_ratio': float(causal_ratio),
    }


def environment_invariance_score(
    model: nn.Module,
    env_loaders: list,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure how invariant model predictions are across different environments.
    
    For ColoredMNIST, environments differ in color-label correlation. A causally
    invariant model should have similar accuracy across all environments.
    
    Args:
        model: Trained model with classifier head
        env_loaders: List of dataloaders for different environments
        device: Device to run inference on
        
    Returns:
        Dictionary with:
            - 'env_accuracies': List of per-environment accuracies
            - 'acc_mean': Mean accuracy across environments
            - 'acc_std': Standard deviation of accuracies (lower is more invariant)
            - 'acc_min': Worst-case environment accuracy
            - 'invariance_score': 1 - (acc_std / acc_mean) (0-1, higher is better)
    """
    model.eval()
    env_accuracies = []
    
    with torch.no_grad():
        for env_idx, loader in enumerate(env_loaders):
            correct = 0
            total = 0
            
            for batch in loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Get predictions (assuming model returns tuple: (recon, logits, latent))
                outputs = model(images)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    logits = outputs[1]  # (recon, logits, latent)
                elif isinstance(outputs, tuple):
                    logits = outputs[0]  # Fallback
                else:
                    logits = outputs
                
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            acc = correct / total
            env_accuracies.append(acc)
    
    acc_mean = np.mean(env_accuracies)
    acc_std = np.std(env_accuracies)
    acc_min = np.min(env_accuracies)
    
    # Invariance score: normalized inverse std (higher = more invariant)
    invariance_score = 1.0 - (acc_std / (acc_mean + 1e-8))
    
    return {
        'env_accuracies': env_accuracies,
        'acc_mean': float(acc_mean),
        'acc_std': float(acc_std),
        'acc_min': float(acc_min),
        'invariance_score': float(invariance_score),
    }


def spurious_feature_reliance(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure how much model relies on spurious color feature vs causal shape feature.
    
    Strategy: For each sample, we check if the model's prediction aligns more with
    the color (spurious) or the actual digit shape (causal).
    
    Args:
        model: Trained model
        test_loader: Test dataloader (should have low color-label correlation)
        device: Device to run inference on
        
    Returns:
        Dictionary with:
            - 'accuracy': Overall classification accuracy
            - 'color_aligned_correct': Accuracy when color matches prediction
            - 'color_misaligned_correct': Accuracy when color doesn't match prediction
            - 'reliance_gap': color_aligned - color_misaligned (should be near 0)
    """
    model.eval()
    
    total = 0
    correct = 0
    color_aligned_correct = 0
    color_aligned_total = 0
    color_misaligned_correct = 0
    color_misaligned_total = 0
    
    with torch.no_grad():
        for images, labels, color_indices in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            color_indices = color_indices.to(device)
            
            # Get predictions (tuple: (recon, logits, latent))
            outputs = model(images)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                logits = outputs[1]  # (recon, logits, latent)
            elif isinstance(outputs, tuple):
                logits = outputs[0]  # Fallback
            else:
                logits = outputs
            
            preds = logits.argmax(dim=1)
            
            # Overall accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Check color alignment
            color_matches_pred = (color_indices == preds)
            
            # Accuracy when color aligns with prediction
            aligned_mask = color_matches_pred
            if aligned_mask.sum() > 0:
                color_aligned_correct += ((preds == labels) & aligned_mask).sum().item()
                color_aligned_total += aligned_mask.sum().item()
            
            # Accuracy when color doesn't align with prediction
            misaligned_mask = ~color_matches_pred
            if misaligned_mask.sum() > 0:
                color_misaligned_correct += ((preds == labels) & misaligned_mask).sum().item()
                color_misaligned_total += misaligned_mask.sum().item()
    
    accuracy = correct / total
    color_aligned_acc = color_aligned_correct / max(color_aligned_total, 1)
    color_misaligned_acc = color_misaligned_correct / max(color_misaligned_total, 1)
    
    # Reliance gap: should be near 0 for causal model
    # Positive gap means model performs better when color aligns (bad - spurious reliance)
    reliance_gap = color_aligned_acc - color_misaligned_acc
    
    return {
        'accuracy': float(accuracy),
        'color_aligned_correct': float(color_aligned_acc),
        'color_misaligned_correct': float(color_misaligned_acc),
        'reliance_gap': float(reliance_gap),
    }


def compute_causal_metrics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    env_loaders: Optional[list] = None,
    device: str = 'cpu'
) -> Dict[str, any]:
    """
    Comprehensive causal evaluation for ColoredMNIST experiments.
    
    Args:
        model: Trained model with encoder and classifier
        test_loader: Test dataloader (low color correlation)
        env_loaders: Optional list of environment loaders for invariance test
        device: Device to run inference on
        
    Returns:
        Dictionary containing all causal metrics
    """
    results = {}
    
    # 1. Extract latent codes and compute correlations
    model.eval()
    all_latents = []
    all_colors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, color_indices in test_loader:
            images = images.to(device)
            
            # Get latent codes
            if hasattr(model, 'encode'):
                latents = model.encode(images)
            elif hasattr(model, 'encoder'):
                latents = model.encoder(images)
            else:
                # Try forward pass and extract latents from tuple
                outputs = model(images)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    latents = outputs[2]  # (recon, logits, latent)
                else:
                    raise ValueError("Cannot extract latent codes from model")
            
            all_latents.append(latents.cpu())
            all_colors.append(color_indices)
            all_labels.append(labels)
    
    all_latents = torch.cat(all_latents, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute correlations
    corr_metrics = color_label_correlation(all_latents, all_colors, all_labels)
    results.update(corr_metrics)
    
    # 2. Spurious feature reliance
    reliance_metrics = spurious_feature_reliance(model, test_loader, device)
    results.update(reliance_metrics)
    
    # 3. Environment invariance (if multiple environments provided)
    if env_loaders is not None:
        inv_metrics = environment_invariance_score(model, env_loaders, device)
        results.update(inv_metrics)
    
    return results


if __name__ == '__main__':
    """
    Demo: Test causality metrics with synthetic data.
    """
    # Create synthetic latent codes and features
    n_samples = 1000
    latent_dim = 10
    
    # Good causal model: latents correlate with label, not color
    latents_good = torch.randn(n_samples, latent_dim)
    labels = torch.randint(0, 10, (n_samples,))
    colors_random = torch.randint(0, 10, (n_samples,))
    
    # Add strong label signal to latents
    for i in range(10):
        mask = labels == i
        latents_good[mask, 0] += i * 0.5
    
    print("Good causal model (latents depend on label, not color):")
    metrics_good = color_label_correlation(latents_good, colors_random, labels)
    for k, v in metrics_good.items():
        print(f"  {k}: {v:.4f}")
    
    # Bad spurious model: latents correlate with color
    latents_bad = torch.randn(n_samples, latent_dim)
    colors_corr = labels.clone()  # Color perfectly predicts label
    
    # Add strong color signal to latents
    for i in range(10):
        mask = colors_corr == i
        latents_bad[mask, 0] += i * 0.5
    
    print("\nBad spurious model (latents depend on color):")
    metrics_bad = color_label_correlation(latents_bad, colors_corr, labels)
    for k, v in metrics_bad.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nCausal ratio comparison:")
    print(f"  Good model: {metrics_good['causal_ratio']:.4f}")
    print(f"  Bad model: {metrics_bad['causal_ratio']:.4f}")
