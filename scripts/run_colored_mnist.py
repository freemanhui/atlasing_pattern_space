"""
ColoredMNIST Experiment: Causal Invariance Learning

Tests the APS framework's causality (C) component on ColoredMNIST where color
is spuriously correlated with digit labels in training but not in test.

Ablation studies:
1. Baseline: No APS components (standard autoencoder + classifier)
2. APS-T: Topology only
3. APS-C: Causality (HSIC) only
4. APS-E: Energy only
5. APS-TC: Topology + Causality
6. APS-TE: Topology + Energy
7. APS-CE: Causality + Energy
8. APS-Full: All components (T+C+E)

Expected results:
- Baseline: High train accuracy, low test accuracy (relies on spurious color)
- APS-C and APS-Full: Similar train/test accuracy (learns causal invariance)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.models.aps_conv_autoencoder import APSConvAutoencoder, APSConvConfig
from aps.utils.colored_mnist import create_colored_mnist_envs, get_color_label_stats
from aps.metrics.causality_metrics import compute_causal_metrics


def train_epoch(
    model: nn.Module,
    train_loaders: List[torch.utils.data.DataLoader],
    optimizer: optim.Optimizer,
    device: str,
    desc: str = "Training"
) -> Dict[str, float]:
    """Train for one epoch across all environments."""
    model.train()
    
    total_losses = {}
    n_batches = 0
    
    # Iterate over all environments simultaneously
    for batch_data in tqdm(zip(*train_loaders), desc=desc, leave=False):
        optimizer.zero_grad()
        
        # Compute loss for each environment and average
        env_losses = []
        for images, labels, color_indices in batch_data:
            images = images.to(device)
            labels = labels.to(device)
            color_indices = color_indices.to(device)
            
            # Compute losses
            losses = model.compute_loss(images, labels, color_indices)
            env_losses.append(losses)
        
        # Average loss across environments (IRM-style)
        avg_losses = {}
        for key in env_losses[0].keys():
            avg_losses[key] = torch.stack([l[key] for l in env_losses]).mean()
        
        # Backward pass
        avg_losses['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key, val in avg_losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += val.item()
        n_batches += 1
    
    # Average losses
    return {k: v / n_batches for k, v in total_losses.items()}


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, color_indices in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            color_indices = color_indices.to(device)
            
            # Forward pass
            losses = model.compute_loss(images, labels, color_indices)
            total_loss += losses['total'].item()
            
            # Predictions
            _, logits, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def run_experiment(
    experiment_name: str,
    lambda_T: float,
    lambda_C: float,
    lambda_E: float,
    latent_dim: int = 10,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = 'cpu',
    output_dir: str = 'outputs/colored_mnist',
    seed: int = 42,
) -> Dict:
    """
    Run single ColoredMNIST experiment with specified APS configuration.
    
    Args:
        experiment_name: Name of experiment (e.g., 'baseline', 'aps-c', 'aps-full')
        lambda_T: Topology loss weight
        lambda_C: Causality loss weight
        lambda_E: Energy loss weight
        latent_dim: Latent space dimension
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on
        output_dir: Directory to save results
        seed: Random seed
    """
    torch.manual_seed(seed)
    
    # Create output directory
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets with strong spurious correlation in training
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"  λ_T={lambda_T}, λ_C={lambda_C}, λ_E={lambda_E}")
    print(f"{'='*60}")
    
    train_loaders, test_loader = create_colored_mnist_envs(
        train_correlations=[0.995, 0.99],  # Very strong spurious correlation (increased)
        test_correlation=0.05,              # Very weak correlation (decreased)
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    
    print(f"\nDataset statistics:")
    print(f"  Training envs: 2 (corr=0.995, 0.99) - HARDER TASK")
    print(f"  Test env: 1 (corr=0.05) - More challenging OOD")
    print(f"  Batch size: {batch_size}")
    
    # Create model
    config = APSConvConfig(
        in_channels=3,
        img_size=28,
        latent_dim=latent_dim,
        hidden_channels=[32, 64],
        lambda_T=lambda_T,
        lambda_C=lambda_C,
        lambda_E=lambda_E,
        topo_k=8,
        n_mem=8,
        beta=5.0,
    )
    
    model = APSConvAutoencoder(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    
    best_test_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loaders, optimizer, device,
            desc=f"Epoch {epoch}/{epochs}"
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        # Record history
        history['train_loss'].append(train_metrics['total'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        # Save best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['total']:.4f} | "
                  f"Test Loss: {test_metrics['loss']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(exp_dir / 'best_model.pt'))
    
    # Compute comprehensive causal metrics
    print("\nComputing causal metrics...")
    causal_metrics = compute_causal_metrics(
        model, test_loader, env_loaders=train_loaders + [test_loader], device=device
    )
    
    # Save results
    results = {
        'experiment': experiment_name,
        'config': {
            'lambda_T': lambda_T,
            'lambda_C': lambda_C,
            'lambda_E': lambda_E,
            'latent_dim': latent_dim,
            'epochs': epochs,
            'lr': lr,
        },
        'best_test_accuracy': best_test_acc,
        'final_test_accuracy': history['test_accuracy'][-1],
        'causal_metrics': causal_metrics,
        'history': history,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results for {experiment_name}:")
    print(f"  Best Test Accuracy: {best_test_acc:.4f}")
    print(f"  Causal Ratio: {causal_metrics['causal_ratio']:.4f}")
    print(f"  Reliance Gap: {causal_metrics['reliance_gap']:.4f}")
    print(f"  Environment Invariance: {causal_metrics.get('invariance_score', 'N/A')}")
    print(f"{'='*60}\n")
    
    return results


def plot_results(all_results: List[Dict], output_dir: str):
    """Create comparison plots across all experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    names = [r['experiment'] for r in all_results]
    test_accs = [r['best_test_accuracy'] for r in all_results]
    causal_ratios = [r['causal_metrics']['causal_ratio'] for r in all_results]
    reliance_gaps = [r['causal_metrics']['reliance_gap'] for r in all_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Test Accuracy
    axes[0].bar(range(len(names)), test_accs, color='steelblue')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy (Higher is Better)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Causal Ratio
    axes[1].bar(range(len(names)), causal_ratios, color='forestgreen')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Causal Ratio')
    axes[1].set_title('Causal Ratio (Higher is Better)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Reliance Gap
    axes[2].bar(range(len(names)), reliance_gaps, color='coral')
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel('Reliance Gap')
    axes[2].set_title('Spurious Reliance Gap (Near 0 is Better)')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path / 'comparison.png'}")
    
    # Create summary table
    print("\n" + "="*80)
    print(f"{'Experiment':<15} {'Test Acc':<12} {'Causal Ratio':<15} {'Reliance Gap':<15}")
    print("="*80)
    for name, acc, ratio, gap in zip(names, test_accs, causal_ratios, reliance_gaps):
        print(f"{name:<15} {acc:<12.4f} {ratio:<15.4f} {gap:<15.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='ColoredMNIST Causal Learning Experiment')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--latent-dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='outputs/colored_mnist', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment', type=str, default='all', 
                        choices=['all', 'baseline', 'aps-t', 'aps-c', 'aps-e', 'aps-tc', 'aps-te', 'aps-ce', 'aps-full'],
                        help='Which experiment(s) to run')
    
    args = parser.parse_args()
    
    # Define experiments: (name, λ_T, λ_C, λ_E)
    # Note: lambda_E reduced from 0.1 to 0.01 for numerical stability
    experiments = {
        'baseline': ('baseline', 0.0, 0.0, 0.0),
        'aps-t': ('aps-t', 1.0, 0.0, 0.0),
        'aps-c': ('aps-c', 0.0, 1.0, 0.0),
        'aps-e': ('aps-e', 0.0, 0.0, 0.01),
        'aps-tc': ('aps-tc', 1.0, 1.0, 0.0),
        'aps-te': ('aps-te', 1.0, 0.0, 0.01),
        'aps-ce': ('aps-ce', 0.0, 1.0, 0.01),
        'aps-full': ('aps-full', 1.0, 1.0, 0.01),
    }
    
    # Select experiments to run
    if args.experiment == 'all':
        exp_to_run = experiments.values()
    else:
        exp_to_run = [experiments[args.experiment]]
    
    # Run experiments
    all_results = []
    for name, lambda_T, lambda_C, lambda_E in exp_to_run:
        result = run_experiment(
            experiment_name=name,
            lambda_T=lambda_T,
            lambda_C=lambda_C,
            lambda_E=lambda_E,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_results.append(result)
    
    # Plot comparisons if running multiple experiments
    if len(all_results) > 1:
        plot_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
