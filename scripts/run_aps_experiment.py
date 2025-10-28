#!/usr/bin/env python3
"""
Run APS experiments with ablation studies.

This script provides a unified interface for training APS models with different
configurations, supporting ablation studies for T, C, and E components.

Examples:
    # Full model (T+C+E)
    python scripts/run_aps_experiment.py --experiment full --epochs 100
    
    # Topology only
    python scripts/run_aps_experiment.py --experiment T-only --epochs 100
    
    # Baseline (no regularization)
    python scripts/run_aps_experiment.py --experiment baseline --epochs 100
    
    # Custom configuration
    python scripts/run_aps_experiment.py --latent 10 --lambda-T 0.5 --lambda-E 0.1
"""

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from aps.models import APSAutoencoder, APSConfig
from aps.training import Trainer, TrainingConfig, OptimizerConfig, SchedulerConfig
from aps.utils.data import toy_corpus, cooc_ppmi, svd_embed


def get_ablation_config(experiment: str, base_config: APSConfig) -> APSConfig:
    """
    Get model configuration for ablation study.
    
    Args:
        experiment: Experiment name ('full', 'T-only', 'C-only', 'E-only', 'baseline')
        base_config: Base configuration to modify
    
    Returns:
        Modified configuration for the ablation
    """
    if experiment == 'full':
        # All components enabled
        base_config.lambda_T = 1.0
        base_config.lambda_C = 1.0
        base_config.lambda_E = 1.0
    elif experiment == 'T-only':
        # Only topology
        base_config.lambda_T = 1.0
        base_config.lambda_C = 0.0
        base_config.lambda_E = 0.0
    elif experiment == 'C-only':
        # Only causality
        base_config.lambda_T = 0.0
        base_config.lambda_C = 1.0
        base_config.lambda_E = 0.0
    elif experiment == 'E-only':
        # Only energy
        base_config.lambda_T = 0.0
        base_config.lambda_C = 0.0
        base_config.lambda_E = 1.0
    elif experiment == 'T+E':
        # Topology + Energy
        base_config.lambda_T = 1.0
        base_config.lambda_C = 0.0
        base_config.lambda_E = 1.0
    elif experiment == 'baseline':
        # No regularization (pure autoencoder)
        base_config.lambda_T = 0.0
        base_config.lambda_C = 0.0
        base_config.lambda_E = 0.0
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    return base_config


def create_toy_dataset(n_samples: int = 1000, in_dim: int = 50):
    """
    Create a toy dataset from text corpus.
    
    Args:
        n_samples: Ignored (for compatibility - uses fixed toy corpus)
        in_dim: Input dimension
    
    Returns:
        Train and validation data loaders
    """
    # Generate toy corpus (fixed dataset)
    corpus = toy_corpus()
    
    # Compute PPMI co-occurrence
    cooc, vocab = cooc_ppmi(corpus)
    
    # Create SVD embeddings
    embeddings = svd_embed(cooc, d=in_dim)
    
    # Convert to tensor
    X = torch.from_numpy(embeddings).float()
    
    return X


def main():
    parser = argparse.ArgumentParser(description='Run APS experiments')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default='full',
                       choices=['full', 'T-only', 'C-only', 'E-only', 'T+E', 'baseline'],
                       help='Ablation experiment to run')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name (default: based on experiment type)')
    
    # Model configuration
    parser.add_argument('--in-dim', type=int, default=50,
                       help='Input dimension')
    parser.add_argument('--latent', type=int, default=2,
                       help='Latent dimension')
    parser.add_argument('--hidden', type=int, nargs='+', default=[64],
                       help='Hidden layer dimensions')
    
    # Loss weights (override ablation defaults)
    parser.add_argument('--lambda-T', type=float, default=None,
                       help='Topology loss weight')
    parser.add_argument('--lambda-C', type=float, default=None,
                       help='Causality loss weight')
    parser.add_argument('--lambda-E', type=float, default=None,
                       help='Energy loss weight')
    
    # Component-specific parameters
    parser.add_argument('--topo-k', type=int, default=8,
                       help='k for kNN topology')
    parser.add_argument('--n-mem', type=int, default=8,
                       help='Number of memory patterns')
    parser.add_argument('--beta', type=float, default=5.0,
                       help='Energy sharpness parameter')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['step', 'cosine', 'exponential'],
                       help='Learning rate scheduler')
    parser.add_argument('--grad-clip', type=float, default=None,
                       help='Gradient clipping value')
    
    # Data configuration
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples in toy dataset')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction')
    
    # Logging and checkpointing
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Checkpoint save frequency (epochs)')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Logging frequency (steps)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable tensorboard logging')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='aps',
                       help='Wandb project name')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create model configuration
    model_config = APSConfig(
        in_dim=args.in_dim,
        latent_dim=args.latent,
        hidden_dims=args.hidden,
        topo_k=args.topo_k,
        n_mem=args.n_mem,
        beta=args.beta,
    )
    
    # Apply ablation configuration
    model_config = get_ablation_config(args.experiment, model_config)
    
    # Override with command-line arguments if provided
    if args.lambda_T is not None:
        model_config.lambda_T = args.lambda_T
    if args.lambda_C is not None:
        model_config.lambda_C = args.lambda_C
    if args.lambda_E is not None:
        model_config.lambda_E = args.lambda_E
    
    # Create model
    model = APSAutoencoder(model_config)
    
    # Create training configuration
    experiment_name = args.experiment_name or f"aps_{args.experiment}"
    
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        optimizer=OptimizerConfig(
            name=args.optimizer,
            lr=args.lr,
        ),
        scheduler=SchedulerConfig(name=args.scheduler),
        device=args.device,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        patience=args.patience,
    )
    
    # Create dataset
    print("Creating dataset...")
    X = create_toy_dataset(n_samples=args.n_samples, in_dim=args.in_dim)
    
    # Split into train/val
    n_val = int(len(X) * args.val_split)
    n_train = len(X) - n_val
    train_data, val_data = random_split(X, [n_train, n_val])
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(train_data.dataset[train_data.indices]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data.dataset[val_data.indices]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Print configuration
    print("\n" + "="*50)
    print(f"Experiment: {args.experiment}")
    print(f"Model: latent_dim={args.latent}, hidden={args.hidden}")
    print(f"Losses: T={model_config.lambda_T}, C={model_config.lambda_C}, E={model_config.lambda_E}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Device: {args.device}")
    print("="*50 + "\n")
    
    # Create trainer and train
    trainer = Trainer(model, train_config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
