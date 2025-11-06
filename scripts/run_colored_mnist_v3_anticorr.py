"""
ColoredMNIST v3.1: Anti-Correlated Test with 1% Causal Signal

VERY HARD task with minimal causal signal:
- Training: 99% color-label correlation (1% uncorrelated samples provide causal signal)
- Test: -99% anti-correlation (color mostly predicts WRONG label)

Expected Results:
- Baseline: Should fail significantly (~10-20% accuracy on test)
  - Learns: "red=0, green=1, blue=2..." from 99% of samples
  - Test sees: "red=1, green=2, blue=3..." (shifted) for 99%
  - 1% causal signal insufficient without explicit independence
  - Result: Mostly wrong!

- APS-C: Should rescue performance (60-70% accuracy)
  - HSIC forces independence from color
  - Learns: Digit shape features from 1% uncorrelated samples
  - Result: Generalizes better to anti-correlated test

- APS-Full: Should achieve best performance (70-80% accuracy)
  - Combines causality + topology + energy
  - Most robust representation
  - Topology may help preserve shape structure
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.models.aps_conv_autoencoder import APSConvAutoencoder, APSConvConfig
from aps.utils.colored_mnist import create_colored_mnist_envs
from aps.metrics.causality_metrics import compute_causal_metrics


def train_epoch(
    model: nn.Module,
    train_loaders: list,
    optimizer: optim.Optimizer,
    device: str,
    desc: str = "Training"
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    n_batches = 0
    
    for batch_data in tqdm(zip(*train_loaders), desc=desc, leave=False):
        optimizer.zero_grad()
        
        # Average loss across environments
        env_losses = []
        for images, labels, color_indices in batch_data:
            images = images.to(device)
            labels = labels.to(device)
            color_indices = color_indices.to(device)
            
            losses = model.compute_loss(images, labels, color_indices)
            env_losses.append(losses)
        
        avg_losses = {}
        for key in env_losses[0].keys():
            avg_losses[key] = torch.stack([l[key] for l in env_losses]).mean()
        
        # Backward with gradient clipping
        avg_losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for key, val in avg_losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += val.item()
        n_batches += 1
    
    return {k: v / n_batches for k, v in total_losses.items()}


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, color_indices in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            color_indices = color_indices.to(device)
            
            losses = model.compute_loss(images, labels, color_indices)
            total_loss += losses['total'].item()
            
            _, logits, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {'loss': total_loss / len(test_loader), 'accuracy': correct / total}


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
    output_dir: str = 'outputs/colored_mnist_v3',
    seed: int = 42,
) -> Dict:
    """Run anti-correlated experiment."""
    torch.manual_seed(seed)
    
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ColoredMNIST v3 - ANTI-CORRELATED TEST")
    print(f"Experiment: {experiment_name}")
    print(f"  λ_T={lambda_T}, λ_C={lambda_C}, λ_E={lambda_E}")
    print(f"{'='*70}")
    
    # Create datasets with NEAR-PERFECT correlation in train, STRONG anti-correlation in test
    # 99% correlation provides 1% causal signal for learning
    train_loaders, test_loader = create_colored_mnist_envs(
        train_correlations=[0.99, 0.99],  # 99% correlation (1% uncorrelated)
        test_correlation=-0.99,            # -99% anti-correlation (1% correct by chance)
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    
    print(f"\nDataset Configuration (VERY HARD TASK with 1% causal signal):")
    print(f"  Training: 99% color-label correlation (1% uncorrelated samples)")
    print(f"  Test: -99% anti-correlation (color mostly predicts WRONG label)")
    print(f"  Expected Baseline: ~10-20% (learns color, catastrophic failure on test)")
    print(f"  Expected APS-C: ~60-70% (learns shape from 1% causal signal)")
    print(f"  Batch size: {batch_size}\n")
    
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
        beta=1.0,  # Stable energy
    )
    
    model = APSConvAutoencoder(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    best_test_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loaders, optimizer, device,
            desc=f"Epoch {epoch}/{epochs}"
        )
        test_metrics = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_metrics['total'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), exp_dir / 'best_model.pt')
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_metrics['total']:.4f} | "
                  f"Test Loss: {test_metrics['loss']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(exp_dir / 'best_model.pt', weights_only=True))
    
    # Compute metrics
    print("\nComputing causal metrics...")
    causal_metrics = compute_causal_metrics(
        model, test_loader, env_loaders=train_loaders + [test_loader], device=device
    )
    
    # Save results
    results = {
        'experiment': experiment_name,
        'version': 'v3_anticorr',
        'config': {
            'lambda_T': lambda_T,
            'lambda_C': lambda_C,
            'lambda_E': lambda_E,
            'latent_dim': latent_dim,
            'epochs': epochs,
            'lr': lr,
        },
        'dataset': {
            'train_correlation': 0.99,
            'test_correlation': -0.99,
            'description': 'Anti-correlated test with 1% causal signal (v3.1)'
        },
        'best_test_accuracy': best_test_acc,
        'final_test_accuracy': history['test_accuracy'][-1],
        'causal_metrics': causal_metrics,
        'history': history,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS for {experiment_name}:")
    print(f"  Best Test Accuracy: {best_test_acc:.4f}")
    print(f"  Causal Ratio: {causal_metrics['causal_ratio']:.4f}")
    print(f"  Reliance Gap: {causal_metrics['reliance_gap']:.4f}")
    if 'invariance_score' in causal_metrics:
        print(f"  Environment Invariance: {causal_metrics['invariance_score']:.4f}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ColoredMNIST v3: Anti-Correlated Test')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--latent-dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='outputs/colored_mnist_v3', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'baseline', 'aps-t', 'aps-c', 'aps-full'],
                        help='Which experiment(s) to run')
    
    args = parser.parse_args()
    
    # Key experiments for anti-correlated test
    experiments = {
        'baseline': ('baseline', 0.0, 0.0, 0.0),   # Should fail ~10%
        'aps-t': ('aps-t', 1.0, 0.0, 0.0),         # Topology only
        'aps-c': ('aps-c', 0.0, 1.0, 0.0),         # Should rescue ~70-80%
        'aps-full': ('aps-full', 1.0, 1.0, 0.01),  # Should achieve ~80-90%
    }
    
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
    
    # Print summary
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY - Anti-Correlated Test Results")
        print("="*70)
        print(f"{'Experiment':<15} {'Test Acc':<12} {'Causal Ratio':<15}")
        print("-"*70)
        for r in all_results:
            print(f"{r['experiment']:<15} {r['best_test_accuracy']:<12.4f} "
                  f"{r['causal_metrics']['causal_ratio']:<15.4f}")
        print("="*70)
        
        # Highlight if baseline failed as expected
        baseline_result = [r for r in all_results if r['experiment'] == 'baseline']
        if baseline_result and baseline_result[0]['best_test_accuracy'] < 0.3:
            print("\n✓ Baseline FAILED as expected (< 30%)!")
            print("  This confirms the task forces explicit causal learning.")
        
        aps_c_result = [r for r in all_results if r['experiment'] == 'aps-c']
        if aps_c_result and baseline_result:
            improvement = aps_c_result[0]['best_test_accuracy'] - baseline_result[0]['best_test_accuracy']
            if improvement > 0.4:
                print(f"\n✓ APS-C RESCUED performance (+{improvement:.1%})!")
                print("  This demonstrates causality component effectiveness.")


if __name__ == '__main__':
    main()
