#!/usr/bin/env python3
"""
MNIST Ablation Study - Phase 5.3

Systematic ablation study of APS components on MNIST:
- Baseline (reconstruction only)
- T-only (topology)
- C-only (causality via HSIC)
- E-only (energy)
- T+C, T+E, C+E (pairwise)
- T+C+E (full model)

Each configuration is trained for full epochs and evaluated on:
- Reconstruction quality
- Topology preservation (trustworthiness, continuity, kNN preservation)
- Clustering quality (ARI, NMI, silhouette)
- Independence (HSIC between latent factors)
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


def setup_dirs() -> Tuple[Path, Path, Path]:
    """Create output directories."""
    out_dir = Path("outputs/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    return out_dir, metrics_dir, plots_dir, ckpt_dir


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


class AblationModel(nn.Module):
    """
    Unified model for ablation study.
    
    Supports selective activation of:
    - Topology loss (T)
    - Causality/independence loss (C)
    - Energy landscape shaping (E)
    """
    
    def __init__(
        self,
        in_dim: int = 784,
        latent_dim: int = 2,
        hidden: int = 128,
        use_topo: bool = False,
        use_causal: bool = False,
        use_energy: bool = False,
        topo_k: int = 15,
        topo_weight: float = 1.0,
        causal_weight: float = 1.0,
        energy_weight: float = 0.1,
        energy_type: str = 'memory',  # 'memory' or 'topology'
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
        
        # Component flags
        self.use_topo = use_topo
        self.use_causal = use_causal
        self.use_energy = use_energy
        
        # Topology
        if use_topo:
            self.topo_loss_fn = KNNTopoLoss(k=topo_k)
            self.topo_weight = topo_weight
        
        # Causality (HSIC independence)
        if use_causal:
            self.hsic_loss_fn = HSICLoss()
            self.causal_weight = causal_weight
        
        # Energy
        if use_energy:
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
        """Forward pass: encode -> decode."""
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
        """
        Compute total loss with selected components.
        
        Args:
            x: Flattened input (batch_size, 784)
            x_recon: Reconstructed output
            z: Latent representation
            labels: Class labels (used as nuisance for causality)
            x_orig: Original input space for topology
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss (always present)
        recon_loss = nn.functional.mse_loss(x_recon, x)
        losses["recon"] = recon_loss
        total_loss = recon_loss
        
        # Topology loss
        if self.use_topo:
            topo_loss = self.topo_loss_fn(x_orig, z)
            losses["topo"] = topo_loss
            total_loss = total_loss + self.topo_weight * topo_loss
        
        # Causality loss (independence from labels)
        if self.use_causal:
            # Use one-hot labels as nuisance variable
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            hsic_loss = self.hsic_loss_fn(z, labels_onehot)
            losses["causal"] = hsic_loss
            total_loss = total_loss + self.causal_weight * hsic_loss
        
        # Energy loss
        if self.use_energy:
            energy_vals = self.energy_fn(z)
            energy_loss = energy_vals.mean()  # Minimize average energy
            losses["energy"] = energy_loss
            total_loss = total_loss + self.energy_weight * energy_loss
        
        losses["total"] = total_loss
        return losses


def train_epoch(
    model: AblationModel,
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
    
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        x_orig = data.view(data.size(0), -1)  # Flatten for topology
        x = x_orig.clone()
        
        optimizer.zero_grad()
        
        x_recon, z = model(x)
        losses = model.compute_loss(x, x_recon, z, labels, x_orig)
        
        losses["total"].backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            if key in losses:
                epoch_losses[key] += losses[key].item()
        n_batches += 1
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    return epoch_losses


def initialize_topology_energy(model: AblationModel, train_loader: DataLoader, device: str):
    """Initialize TopologyEnergy target adjacency from training data."""
    if not (model.use_energy and model.energy_type == 'topology'):
        return
    
    print("  Initializing TopologyEnergy target adjacency...")
    
    # Collect a subset of training data for target adjacency
    # Use first few batches (up to 5000 samples) to keep memory manageable
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
    print(f"  Target adjacency set from {X_train.shape[0]} training samples")


def train_model(
    config_name: str,
    model: AblationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    ckpt_dir: Path,
) -> Dict[str, List[float]]:
    """Train model and track metrics."""
    # Initialize TopologyEnergy if needed
    initialize_topology_energy(model, train_loader, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_topo": [],
        "train_causal": [],
        "train_energy": [],
    }
    
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device)
        
        # Log
        history["train_loss"].append(train_losses["total"])
        history["train_recon"].append(train_losses["recon"])
        history["train_topo"].append(train_losses.get("topo", 0.0))
        history["train_causal"].append(train_losses.get("causal", 0.0))
        history["train_energy"].append(train_losses.get("energy", 0.0))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses['total']:.4f}")
    
    # Save checkpoint
    ckpt_path = ckpt_dir / f"{config_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
    
    return history


def run_ablation_study(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 2,
    device: str = "cpu",
):
    """Run full ablation study."""
    
    # Setup
    out_dir, metrics_dir, plots_dir, ckpt_dir = setup_dirs()
    train_loader, test_loader = get_mnist_loaders(batch_size)
    
    # Configuration matrix
    configs = [
        # Name, use_topo, use_causal, use_energy
        ("baseline", False, False, False),
        ("t_only", True, False, False),
        ("c_only", False, True, False),
        ("e_only", False, False, True),
        ("t_c", True, True, False),
        ("t_e", True, False, True),
        ("c_e", False, True, True),
        ("t_c_e", True, True, True),
    ]
    
    all_results = {}
    
    for config_name, use_topo, use_causal, use_energy in configs:
        print(f"\n{'#'*60}")
        print(f"Configuration: {config_name}")
        print(f"  Topology: {use_topo}")
        print(f"  Causality: {use_causal}")
        print(f"  Energy: {use_energy}")
        print(f"{'#'*60}")
        
        # Create model
        model = AblationModel(
            in_dim=784,
            latent_dim=latent_dim,
            hidden=128,
            use_topo=use_topo,
            use_causal=use_causal,
            use_energy=use_energy,
            topo_k=15,
            topo_weight=1.0,
            causal_weight=0.1,
            energy_weight=0.1,
            n_mem=10,
            beta=5.0,
        ).to(device)
        
        # Train
        history = train_model(
            config_name,
            model,
            train_loader,
            test_loader,
            epochs,
            lr,
            device,
            ckpt_dir,
        )
        
        # Evaluate
        print(f"\nEvaluating {config_name}...")
        model.eval()
        
        # Collect embeddings and labels
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                x = data.view(data.size(0), -1)
                _, z = model(x)
                embeddings_list.append(z.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        labels_array = np.concatenate(labels_list)
        
        # Compute metrics
        test_data_tensor = test_loader.dataset.data.float() / 255.0
        test_data_flat = test_data_tensor.view(-1, 784).numpy()
        
        # Compute reconstruction error
        recon_list = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                x = data.view(data.size(0), -1)
                x_recon, _ = model(x)
                recon_list.append(x_recon.cpu().numpy())
        reconstructions = np.vstack(recon_list)
        
        # Calculate individual metrics
        from utils.metrics import (
            compute_topology_metrics,
            compute_clustering_metrics,
            compute_reconstruction_error,
        )
        
        # Topology metrics
        topo_metrics = compute_topology_metrics(test_data_flat, embeddings, k=15)
        
        # Clustering metrics  
        cluster_metrics = compute_clustering_metrics(embeddings, labels_array)
        
        # Reconstruction error
        recon_error = compute_reconstruction_error(test_data_flat, reconstructions)
        
        # Combine all metrics
        metrics = {
            "reconstruction_error": recon_error,
            **topo_metrics,
            **cluster_metrics,
        }
        
        # Store results
        all_results[config_name] = {
            "config": {
                "use_topo": use_topo,
                "use_causal": use_causal,
                "use_energy": use_energy,
            },
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "metrics": metrics,
        }
        
        # Save individual config metrics
        metrics_path = metrics_dir / f"{config_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_results[config_name], f, indent=2)
        
        # Plot embedding
        if latent_dim == 2:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels_array,
                cmap="tab10",
                s=1,
                alpha=0.6,
            )
            plt.colorbar(scatter)
            plt.title(f"{config_name} - MNIST Embedding")
            plt.xlabel("Latent Dim 0")
            plt.ylabel("Latent Dim 1")
            plt.tight_layout()
            
            plot_path = plots_dir / f"{config_name}_embedding.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved: {plot_path}")
    
    # Save summary
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation study complete!")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*60}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<12} {'Recon↓':<10} {'Trust↑':<10} {'Cont↑':<10} {'ARI↑':<10} {'Sil↑':<10}")
    print("-"*80)
    
    for config_name in configs:
        config_name = config_name[0]
        m = all_results[config_name]["metrics"]
        print(
            f"{config_name:<12} "
            f"{m['reconstruction_error']:<10.4f} "
            f"{m['trustworthiness']:<10.4f} "
            f"{m['continuity']:<10.4f} "
            f"{m['ari']:<10.4f} "
            f"{m['silhouette']:<10.4f}"
        )
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="MNIST Ablation Study - Phase 5.3")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent", type=int, default=2, help="Latent dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "cpu" and torch.backends.mps.is_available():
        args.device = "mps"
    
    print(f"Using device: {args.device}")
    
    run_ablation_study(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent,
        device=args.device,
    )


if __name__ == "__main__":
    main()
