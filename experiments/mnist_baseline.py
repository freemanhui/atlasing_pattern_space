#!/usr/bin/env python3
"""
Baseline MNIST Experiment - Phase 5.2

Train baseline and T-only models on MNIST to:
1. Validate the experimental pipeline
2. Establish baseline performance
3. Demonstrate topology preservation benefits

Usage:
    python experiments/mnist_baseline.py --config baseline
    python experiments/mnist_baseline.py --config t-only
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from aps.models import APSAutoencoder, APSConfig
from aps.training import Trainer, TrainingConfig, OptimizerConfig
from utils import (
    get_mnist_dataloaders,
    get_embeddings_from_model,
    evaluate_model_comprehensive,
)


# Experiment configurations
CONFIGS = {
    'baseline': {
        'name': 'baseline',
        'description': 'Pure autoencoder (no regularization)',
        'lambda_T': 0.0,
        'lambda_C': 0.0,
        'lambda_E': 0.0,
    },
    't-only': {
        'name': 't-only',
        'description': 'Topology preservation only',
        'lambda_T': 1.0,
        'lambda_C': 0.0,
        'lambda_E': 0.0,
    },
    'e-only': {
        'name': 'e-only',
        'description': 'Energy basins only',
        'lambda_T': 0.0,
        'lambda_C': 0.0,
        'lambda_E': 0.5,
    },
    't+e': {
        'name': 't+e',
        'description': 'Topology + Energy',
        'lambda_T': 1.0,
        'lambda_C': 0.0,
        'lambda_E': 0.5,
    },
}


def create_model(config_name: str, latent_dim: int = 2) -> APSAutoencoder:
    """Create model from experiment configuration."""
    exp_config = CONFIGS[config_name]
    
    model_config = APSConfig(
        in_dim=784,  # MNIST flattened
        latent_dim=latent_dim,
        hidden_dims=[256, 128],
        lambda_T=exp_config['lambda_T'],
        lambda_C=exp_config['lambda_C'],
        lambda_E=exp_config['lambda_E'],
        topo_k=10,
        n_mem=10,
        beta=5.0,
    )
    
    return APSAutoencoder(model_config)


def train_model(
    model: APSAutoencoder,
    train_loader,
    val_loader,
    experiment_name: str,
    epochs: int = 50,
    device: str = 'cpu',
) -> Trainer:
    """Train the model."""
    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=128,
        optimizer=OptimizerConfig(name='adam', lr=1e-3),
        device=device,
        experiment_name=experiment_name,
        output_dir='./experiments/results/checkpoints',
        log_freq=50,
        save_freq=10,
        use_tensorboard=False,
        use_wandb=False,
    )
    
    trainer = Trainer(model, train_config)
    trainer.train(train_loader, val_loader)
    
    return trainer


def evaluate_model(
    model: APSAutoencoder,
    test_loader,
    device: str = 'cpu',
) -> dict:
    """Comprehensive model evaluation."""
    # Extract embeddings
    embeddings, labels = get_embeddings_from_model(model, test_loader, device)
    
    # Get original data and reconstructions
    model.eval()
    model.to(device)
    
    all_orig = []
    all_recon = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                data, _ = batch
            else:
                data, _, _ = batch
            
            data = data.to(device)
            recon, _ = model(data)
            
            all_orig.append(data.cpu().numpy())
            all_recon.append(recon.cpu().numpy())
    
    X_orig = np.concatenate(all_orig, axis=0)
    X_recon = np.concatenate(all_recon, axis=0)
    
    # Compute all metrics
    metrics = evaluate_model_comprehensive(
        model=model,
        X_orig=X_orig,
        X_emb=embeddings,
        labels=labels,
        X_recon=X_recon,
    )
    
    return metrics


def plot_2d_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
):
    """Plot 2D embedding colored by class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=20,
    )
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit Class')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")


def compare_metrics(results: dict, save_path: Path):
    """Create comparison table of metrics."""
    # Extract metrics
    configs = list(results.keys())
    
    # Prepare data
    metrics_to_compare = [
        'topo/trustworthiness',
        'topo/continuity',
        'topo/knn_preservation',
        'cluster/ari',
        'cluster/nmi',
        'cluster/silhouette',
        'reconstruction/mse',
    ]
    
    # Create table
    table_lines = []
    table_lines.append("Metric Comparison")
    table_lines.append("=" * 80)
    
    # Header
    header = f"{'Metric':<30} " + " ".join([f"{c:>12}" for c in configs])
    table_lines.append(header)
    table_lines.append("-" * 80)
    
    # Rows
    for metric in metrics_to_compare:
        if all(metric in results[c] for c in configs):
            row = f"{metric:<30}"
            for config in configs:
                value = results[config][metric]
                row += f" {value:>12.4f}"
            table_lines.append(row)
    
    table_lines.append("=" * 80)
    
    # Calculate improvements
    if 'baseline' in configs and 't-only' in configs:
        table_lines.append("\nTopology Improvement (T-only vs Baseline):")
        for metric in ['topo/trustworthiness', 'topo/continuity']:
            if metric in results['baseline']:
                baseline_val = results['baseline'][metric]
                tonly_val = results['t-only'][metric]
                improvement = ((tonly_val - baseline_val) / baseline_val) * 100
                table_lines.append(f"  {metric}: {improvement:+.2f}%")
    
    table_str = "\n".join(table_lines)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(table_str)
    
    # Also print
    print("\n" + table_str)


def main():
    parser = argparse.ArgumentParser(description='MNIST Baseline Experiment')
    parser.add_argument('--config', type=str, required=True,
                       choices=list(CONFIGS.keys()),
                       help='Experiment configuration')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='Latent space dimension')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate (skip training)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    results_dir = Path('./experiments/results')
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    checkpoints_dir = results_dir / 'checkpoints'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment info
    exp_config = CONFIGS[args.config]
    experiment_name = f"mnist_{exp_config['name']}"
    
    print("="*60)
    print("MNIST Baseline Experiment")
    print("="*60)
    print(f"Configuration: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"Lambda_T: {exp_config['lambda_T']}")
    print(f"Lambda_E: {exp_config['lambda_E']}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir='./data',
        batch_size=128,
        val_split=0.1,
        flatten=True,
        seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args.config, latent_dim=args.latent_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train or load
    checkpoint_path = checkpoints_dir / experiment_name / 'checkpoints' / 'final_model.pt'
    
    if not args.eval_only:
        print("\nTraining model...")
        trainer = train_model(
            model,
            train_loader,
            val_loader,
            experiment_name,
            epochs=args.epochs,
            device=args.device,
        )
    elif checkpoint_path.exists():
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run without --eval-only to train first")
        return
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device=args.device)
    
    # Print metrics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for key, value in sorted(metrics.items()):
        print(f"{key:<30}: {value:.4f}")
    
    # Save metrics
    metrics_path = tables_dir / f"{experiment_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    
    # Visualize embeddings (only for 2D)
    if args.latent_dim == 2:
        print("\nGenerating 2D embedding plot...")
        embeddings, labels = get_embeddings_from_model(
            model, test_loader, device=args.device
        )
        
        plot_path = figures_dir / f"{experiment_name}_embedding.png"
        plot_2d_embedding(
            embeddings,
            labels,
            title=f"MNIST Embeddings - {exp_config['description']}",
            save_path=plot_path,
        )
    
    print("\n" + "="*60)
    print(f"Experiment complete: {experiment_name}")
    print("="*60)


if __name__ == '__main__':
    main()
