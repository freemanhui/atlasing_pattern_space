"""T-C Conflict Experiment: Topology vs Causality Trade-offs

Runs hyperparameter sweep over λ_T and λ_C to characterize trade-offs
between topology preservation and causal invariance.

Usage:
    python scripts/run_tc_conflict_experiment.py --config all    # Run full sweep
    python scripts/run_tc_conflict_experiment.py --lambda-T 1.0 --lambda-C 1.0  # Single config
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.data.colored_clusters import create_colored_clusters_loaders
from aps.models import APSConfig, APSAutoencoder
from aps.metrics import knn_preservation
from sklearn.metrics import silhouette_score


@dataclass
class TCConflictConfig:
    """Configuration for T-C conflict experiment."""
    
    # Experiment settings
    experiment_name: str = 'tc_conflict'
    lambda_T: float = 1.0
    lambda_C: float = 1.0
    seed: int = 42
    device: str = (
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    
    # Data settings
    train_samples: int = 5000
    val_samples: int = 1000
    test_samples: int = 2000
    train_correlation: float = 0.9  # High spurious correlation
    test_correlation: float = 0.5   # Random correlation
    batch_size: int = 64
    
    # Model architecture
    in_dim: int = 4  # [shape_x, shape_y, color_red, color_blue]
    latent_dim: int = 2  # 2D for visualization
    hidden_dims: list = None  # [32, 16]
    
    # Component hyperparameters
    topo_k: int = 8
    hsic_sigma: float = 1.0
    
    # Training settings
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Output
    output_dir: str = './outputs/tc_conflict'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16]


class TCConflictModel(nn.Module):
    """APS model with classifier for T-C conflict experiment."""
    
    def __init__(self, aps_config: APSConfig):
        super().__init__()
        self.aps = APSAutoencoder(aps_config)
        
        # Classifier on latent space
        self.classifier = nn.Sequential(
            nn.Linear(aps_config.latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, X):
        z, x_recon = self.aps(X)
        logits = self.classifier(z)
        return z, x_recon, logits


def train_epoch(
    model: TCConflictModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        X = batch['X'].to(device)
        y = batch['y'].to(device)
        X_color = X[:, 2:4]  # Color features (one-hot)
        
        # Forward
        z, x_recon, logits = model(X)
        
        # Classification loss
        cls_loss = nn.CrossEntropyLoss()(logits, y)
        
        # APS losses (reconstruction + T + C)
        # Note: C should penalize dependence on X_color
        aps_losses = model.aps.compute_loss(X, nuisance=X_color)
        aps_loss = aps_losses['total']
        
        # Total loss
        loss = cls_loss + aps_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'aps': f'{aps_loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate_metrics(
    model: TCConflictModel,
    loader: DataLoader,
    device: str
) -> dict:
    """Evaluate all metrics for T-C conflict analysis."""
    model.eval()
    
    # Collect all data
    all_X = []
    all_X_shape = []
    all_z = []
    all_y = []
    all_color = []
    all_pred = []
    
    for batch in loader:
        X = batch['X'].to(device)
        X_shape = batch['X_shape']  # Keep on CPU for metrics
        y = batch['y']
        color = batch['color']
        
        # Forward
        z, _, logits = model(X)
        pred = logits.argmax(dim=1)
        
        # Store
        all_X.append(X.cpu())
        all_X_shape.append(X_shape)
        all_z.append(z.cpu())
        all_y.append(y)
        all_color.append(color)
        all_pred.append(pred.cpu())
    
    # Concatenate
    X = torch.cat(all_X, dim=0)
    X_shape = torch.cat(all_X_shape, dim=0)
    z = torch.cat(all_z, dim=0)
    y = torch.cat(all_y, dim=0)
    color = torch.cat(all_color, dim=0)
    pred = torch.cat(all_pred, dim=0)
    
    # Compute metrics
    metrics = {}
    
    # 1. Causal Test Accuracy
    metrics['causal_accuracy'] = (pred == y).float().mean().item()
    
    # 2. Topological Preservation (shape features → latent)
    try:
        metrics['topo_preservation'] = knn_preservation(
            X_shape.numpy(), z.numpy(), k=8
        ).item()
    except Exception:
        metrics['topo_preservation'] = 0.0
    
    # 3. Color Reliance Gap
    # Accuracy when color matches vs doesn't match
    color_matches = (color == y)
    if color_matches.sum() > 0 and (~color_matches).sum() > 0:
        acc_match = (pred[color_matches] == y[color_matches]).float().mean().item()
        acc_mismatch = (pred[~color_matches] == y[~color_matches]).float().mean().item()
        metrics['color_reliance'] = acc_match - acc_mismatch
    else:
        metrics['color_reliance'] = 0.0
    
    # 4. Shape Cluster Quality (silhouette score)
    if len(np.unique(y.numpy())) > 1:
        try:
            metrics['cluster_quality'] = silhouette_score(
                z.numpy(), y.numpy()
            )
        except Exception:
            metrics['cluster_quality'] = 0.0
    else:
        metrics['cluster_quality'] = 0.0
    
    return metrics


def run_single_experiment(config: TCConflictConfig):
    """Run a single (λ_T, λ_C) configuration."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: λ_T={config.lambda_T}, λ_C={config.lambda_C}")
    print(f"{'='*80}")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir) / f'T{config.lambda_T}_C{config.lambda_C}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    print("\n[1/4] Loading Colored Clusters data...")
    loaders = create_colored_clusters_loaders(
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        test_samples=config.test_samples,
        train_correlation=config.train_correlation,
        test_correlation=config.test_correlation,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("\n[2/4] Creating APS model...")
    aps_config = APSConfig(
        in_dim=config.in_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        lambda_T=config.lambda_T,
        lambda_C=config.lambda_C,
        lambda_E=0.0,  # No energy for T-C conflict study
        topo_k=config.topo_k,
        hsic_sigma=config.hsic_sigma
    )
    
    model = TCConflictModel(aps_config).to(config.device)
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print("\n[3/4] Training...")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_metrics': []
    }
    
    best_test_acc = 0
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config.device, epoch
        )
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == config.epochs:
            test_metrics = evaluate_metrics(model, test_loader, config.device)
            
            print(f"  Epoch {epoch}:")
            print(f"    Train Acc: {100*train_metrics['accuracy']:.1f}%")
            print(f"    Test Acc: {100*test_metrics['causal_accuracy']:.1f}%")
            print(f"    Topo Pres: {test_metrics['topo_preservation']:.3f}")
            print(f"    Color Rel: {test_metrics['color_reliance']:.3f}")
            print(f"    Cluster Q: {test_metrics['cluster_quality']:.3f}")
            
            history['test_metrics'].append({
                'epoch': epoch,
                **test_metrics
            })
            
            # Save best model
            if test_metrics['causal_accuracy'] > best_test_acc:
                best_test_acc = test_metrics['causal_accuracy']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': asdict(config),
                    'metrics': test_metrics
                }, output_dir / 'best_model.pt')
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
    
    # Final evaluation
    print("\n[4/4] Final evaluation...")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model'])
    
    # Evaluate
    final_metrics = evaluate_metrics(model, test_loader, config.device)
    
    print("\nFinal Metrics:")
    print(f"  Causal Accuracy: {100*final_metrics['causal_accuracy']:.2f}%")
    print(f"  Topo Preservation: {final_metrics['topo_preservation']:.3f}")
    print(f"  Color Reliance: {final_metrics['color_reliance']:.3f}")
    print(f"  Cluster Quality: {final_metrics['cluster_quality']:.3f}")
    
    # Save history (convert numpy types to Python types)
    history_json = {
        'train_loss': [float(x) for x in history['train_loss']],
        'train_acc': [float(x) for x in history['train_acc']],
        'test_metrics': [
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
             for k, v in m.items()}
            for m in history['test_metrics']
        ]
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_json, f, indent=2)
    
    # Save final metrics (convert numpy types to Python types)
    final_metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in final_metrics.items()}
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics_json, f, indent=2)
    
    # Save latent embeddings for visualization
    model.eval()
    with torch.no_grad():
        test_dataset = loaders['test_dataset']
        X_test = test_dataset.X.to(config.device)
        z_test, _, _ = model(X_test)
        
        torch.save({
            'z': z_test.cpu(),
            'y': test_dataset.y,
            'color': test_dataset.color,
            'X_shape': test_dataset.X_shape
        }, output_dir / 'latent_embeddings.pt')
    
    print(f"\nResults saved to {output_dir}")
    
    return final_metrics


def run_hyperparameter_sweep():
    """Run full hyperparameter sweep over λ_T and λ_C."""
    
    # Define grid
    lambda_T_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    lambda_C_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Running hyperparameter sweep:")
    print(f"  λ_T: {lambda_T_values}")
    print(f"  λ_C: {lambda_C_values}")
    print(f"  Total: {len(lambda_T_values) * len(lambda_C_values)} configurations")
    
    results = {}
    
    for lambda_T in lambda_T_values:
        for lambda_C in lambda_C_values:
            config = TCConflictConfig(
                experiment_name=f'tc_T{lambda_T}_C{lambda_C}',
                lambda_T=lambda_T,
                lambda_C=lambda_C
            )
            
            metrics = run_single_experiment(config)
            results[(lambda_T, lambda_C)] = metrics
    
    # Save aggregate results
    output_dir = Path('./outputs/tc_conflict')
    with open(output_dir / 'sweep_results.json', 'w') as f:
        # Convert tuple keys to strings and numpy types to Python types
        results_json = {
            f'T{lt}_C{lc}': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in metrics.items()}
            for (lt, lc), metrics in results.items()
        }
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*80)
    print("Hyperparameter sweep complete!")
    print(f"Results saved to {output_dir / 'sweep_results.json'}")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='T-C Conflict Experiment')
    parser.add_argument('--config', type=str, default='single',
                       choices=['single', 'all'],
                       help='Run single config or full sweep')
    parser.add_argument('--lambda-T', type=float, default=1.0,
                       help='Topology weight')
    parser.add_argument('--lambda-C', type=float, default=1.0,
                       help='Causality weight')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.config == 'all':
        # Run full hyperparameter sweep
        run_hyperparameter_sweep()
    else:
        # Run single configuration
        config = TCConflictConfig(
            lambda_T=args.lambda_T,
            lambda_C=args.lambda_C,
            epochs=args.epochs,
            seed=args.seed
        )
        run_single_experiment(config)


if __name__ == '__main__':
    main()
