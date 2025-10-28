#!/usr/bin/env python3
"""
MNIST TopologyEnergy Experiments

Compare TopologyEnergy vs MemoryEnergy in T+C+E configuration.
Tests whether data-driven energy improves over memory-based attractors.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aps.topology import KNNTopoLoss
from aps.causality import HSICLoss
from aps.energy import MemoryEnergy, MemoryEnergyConfig, TopologyEnergy, TopologyEnergyConfig
from utils.metrics import (
    compute_topology_metrics,
    compute_clustering_metrics,
    compute_reconstruction_error,
)


class APSModel(nn.Module):
    """Full APS model with configurable energy type."""
    
    def __init__(
        self,
        in_dim: int = 784,
        latent_dim: int = 2,
        hidden: int = 128,
        topo_k: int = 15,
        topo_weight: float = 1.0,
        causal_weight: float = 0.5,
        energy_weight: float = 0.3,
        energy_type: str = 'topology',  # 'memory' or 'topology'
        n_mem: int = 10,
        beta: float = 5.0,
    ):
        super().__init__()
        
        # Core autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )
        
        # Topology
        self.topo_loss_fn = KNNTopoLoss(k=topo_k)
        self.topo_weight = topo_weight
        
        # Causality
        self.hsic_loss_fn = HSICLoss()
        self.causal_weight = causal_weight
        
        # Energy
        self.energy_type = energy_type
        if energy_type == 'memory':
            cfg = MemoryEnergyConfig(
                latent_dim=latent_dim, n_mem=n_mem, beta=beta, alpha=0.0
            )
            self.energy_fn = MemoryEnergy(cfg)
        elif energy_type == 'topology':
            cfg = TopologyEnergyConfig(
                latent_dim=latent_dim,
                k=topo_k,
                mode='agreement',
                continuous=True,
                scale=1.0
            )
            self.energy_fn = TopologyEnergy(cfg)
        else:
            raise ValueError(f"Unknown energy_type: {energy_type}")
        
        self.energy_weight = energy_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        labels: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        losses = {}
        
        # Reconstruction
        recon_loss = nn.functional.mse_loss(x_recon, x)
        losses["recon"] = recon_loss
        
        # Topology
        topo_loss = self.topo_loss_fn(x_orig, z)
        losses["topo"] = topo_loss
        
        # Causality
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
        hsic_loss = self.hsic_loss_fn(z, labels_onehot)
        losses["causal"] = hsic_loss
        
        # Energy
        energy_vals = self.energy_fn(z)
        energy_loss = energy_vals.mean()
        losses["energy"] = energy_loss
        
        # Total
        total_loss = (
            recon_loss +
            self.topo_weight * topo_loss +
            self.causal_weight * hsic_loss +
            self.energy_weight * energy_loss
        )
        losses["total"] = total_loss
        
        return losses


def get_mnist_loaders(batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST train and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def initialize_topology_energy(model: APSModel, train_loader: DataLoader, device: str):
    """Initialize TopologyEnergy target adjacency."""
    if model.energy_type != 'topology':
        return
    
    print("\n  Initializing TopologyEnergy target adjacency...")
    
    # Collect subset of training data
    X_samples = []
    max_samples = 5000
    n_collected = 0
    
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            x = data.view(data.size(0), -1)
            X_samples.append(x)
            n_collected += x.shape[0]
            if n_collected >= max_samples:
                break
    
    X_train = torch.cat(X_samples, dim=0)[:max_samples]
    model.energy_fn.set_target_adjacency(X_train)
    print(f"  Target adjacency set from {X_train.shape[0]} training samples\n")


def train_epoch(
    model: APSModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {
        "recon": 0.0,
        "topo": 0.0,
        "causal": 0.0,
        "energy": 0.0,
        "total": 0.0,
    }
    n_batches = 0
    
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        x_orig = data.view(data.size(0), -1)
        x = x_orig.clone()
        
        optimizer.zero_grad()
        
        x_recon, z = model(x)
        losses = model.compute_loss(x, x_recon, z, labels, x_orig)
        
        losses["total"].backward()
        optimizer.step()
        
        # Accumulate
        for key in epoch_losses:
            if key in losses:
                epoch_losses[key] += losses[key].item()
        n_batches += 1
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    return epoch_losses


def train_model(
    model: APSModel,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> Dict[str, List[float]]:
    """Train model and track history."""
    initialize_topology_energy(model, train_loader, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_topo": [],
        "train_causal": [],
        "train_energy": [],
    }
    
    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device)
        
        history["train_loss"].append(train_losses["total"])
        history["train_recon"].append(train_losses["recon"])
        history["train_topo"].append(train_losses["topo"])
        history["train_causal"].append(train_losses["causal"])
        history["train_energy"].append(train_losses["energy"])
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_losses['total']:.4f} | "
                  f"Recon: {train_losses['recon']:.4f} | "
                  f"Topo: {train_losses['topo']:.4f} | "
                  f"Energy: {train_losses['energy']:.4f}")
    
    return history


def evaluate_model(
    model: APSModel,
    test_loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate model comprehensively."""
    model.eval()
    
    # Collect embeddings and labels
    embeddings_list = []
    labels_list = []
    recon_list = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            x = data.view(data.size(0), -1)
            x_recon, z = model(x)
            embeddings_list.append(z.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            recon_list.append(x_recon.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels_array = np.concatenate(labels_list)
    reconstructions = np.vstack(recon_list)
    
    # Original data
    test_data_tensor = test_loader.dataset.data.float() / 255.0
    test_data_flat = test_data_tensor.view(-1, 784).numpy()
    
    # Compute metrics
    topo_metrics = compute_topology_metrics(test_data_flat, embeddings, k=15)
    cluster_metrics = compute_clustering_metrics(embeddings, labels_array)
    recon_error = compute_reconstruction_error(test_data_flat, reconstructions)
    
    metrics = {
        "reconstruction_error": recon_error,
        **topo_metrics,
        **cluster_metrics,
    }
    
    return metrics, embeddings, labels_array


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, title: str, save_path: Path):
    """Plot 2D embeddings."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Digit')
    plt.title(title)
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MNIST TopologyEnergy vs MemoryEnergy')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=2, help='Latent dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--energy-weight', type=float, default=0.3, 
                       help='Energy loss weight (lower for TopologyEnergy)')
    args = parser.parse_args()
    
    # Setup
    out_dir = Path("outputs/topo_energy_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    # Configurations to compare
    configs = [
        {
            'name': 't_c_e_memory',
            'energy_type': 'memory',
            'energy_weight': args.energy_weight * 3.33,  # Use higher weight for memory
        },
        {
            'name': 't_c_e_topo',
            'energy_type': 'topology',
            'energy_weight': args.energy_weight,  # Lower weight for topology
        },
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"  Energy Type: {config['energy_type']}")
        print(f"  Energy Weight: {config['energy_weight']:.3f}")
        print(f"{'='*60}")
        
        # Create model
        model = APSModel(
            in_dim=784,
            latent_dim=args.latent_dim,
            hidden=128,
            topo_k=15,
            topo_weight=1.0,
            causal_weight=0.5,
            energy_weight=config['energy_weight'],
            energy_type=config['energy_type'],
            n_mem=10,
            beta=5.0,
        ).to(args.device)
        
        # Train
        history = train_model(
            model,
            train_loader,
            args.epochs,
            args.lr,
            args.device,
        )
        
        # Evaluate
        print(f"\nEvaluating {config['name']}...")
        metrics, embeddings, labels_array = evaluate_model(model, test_loader, args.device)
        
        # Print metrics
        print(f"\nMetrics for {config['name']}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Save
        results = {
            'config': config,
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'metrics': metrics,
        }
        all_results[config['name']] = results
        
        # Save individual results
        results_path = metrics_dir / f"{config['name']}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot embeddings
        if args.latent_dim == 2:
            plot_path = plots_dir / f"{config['name']}_embedding.png"
            plot_embeddings(embeddings, labels_array, config['name'], plot_path)
            print(f"  Saved plot: {plot_path}")
        
        # Save checkpoint
        ckpt_path = ckpt_dir / f"{config['name']}.pt"
        torch.save(model.state_dict(), ckpt_path)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    metric_names = list(all_results[configs[0]['name']]['metrics'].keys())
    
    for metric_name in metric_names:
        print(f"\n{metric_name}:")
        for config in configs:
            value = all_results[config['name']]['metrics'][metric_name]
            print(f"  {config['name']:20s}: {value:.4f}")
        
        # Compute improvement
        if len(configs) == 2:
            baseline_val = all_results[configs[0]['name']]['metrics'][metric_name]
            new_val = all_results[configs[1]['name']]['metrics'][metric_name]
            
            if baseline_val != 0:
                pct_change = ((new_val - baseline_val) / abs(baseline_val)) * 100
                direction = "↑" if pct_change > 0 else "↓"
                print(f"  Change: {direction} {abs(pct_change):.1f}%")
    
    # Save comparison
    comparison_path = out_dir / "comparison_summary.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
