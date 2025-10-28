#!/usr/bin/env python3
"""
MNIST OOD Robustness - Phase 5.4

Tests out-of-distribution generalization of learned embeddings:
1. Rotated MNIST (15°, 30°, 45°, 60°)
2. Noisy MNIST (Gaussian noise at multiple σ levels)
3. FashionMNIST transfer (zero-shot embedding quality)

For each OOD scenario, we evaluate:
- Reconstruction error (how well model handles corruption)
- Topology preservation (does embedding structure degrade?)
- Clustering quality (are classes still separable?)
- kNN accuracy (nearest neighbor classification in latent space)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

from aps.topology import KNNTopoLoss
from aps.causality import HSICIndependenceLoss
from aps.energy import MemoryEnergy, MemoryEnergyConfig
from utils.metrics import (
    knn_preservation,
    trustworthiness,
    continuity,
    clustering_metrics,
)


def setup_dirs() -> Tuple[Path, Path, Path]:
    """Create output directories."""
    out_dir = Path("outputs/ood")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    return out_dir, metrics_dir, plots_dir


class AblationModel(nn.Module):
    """Same model architecture as ablation study for consistency."""
    
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
        n_mem: int = 10,
        beta: float = 5.0,
    ):
        super().__init__()
        
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
        
        self.use_topo = use_topo
        self.use_causal = use_causal
        self.use_energy = use_energy
        
        if use_topo:
            self.topo_loss_fn = KNNTopoLoss(k=topo_k)
            self.topo_weight = topo_weight
        
        if use_causal:
            self.hsic_loss_fn = HSICIndependenceLoss()
            self.causal_weight = causal_weight
        
        if use_energy:
            cfg = MemoryEnergyConfig(
                latent_dim=latent_dim, n_mem=n_mem, beta=beta, alpha=0.0
            )
            self.energy_fn = MemoryEnergy(cfg)
            self.energy_weight = energy_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def load_model(ckpt_path: Path, config: Dict, device: str) -> AblationModel:
    """Load trained model from checkpoint."""
    model = AblationModel(
        in_dim=784,
        latent_dim=config.get("latent_dim", 2),
        hidden=config.get("hidden", 128),
        use_topo=config.get("use_topo", False),
        use_causal=config.get("use_causal", False),
        use_energy=config.get("use_energy", False),
    ).to(device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def get_mnist_test_loader(batch_size: int = 256) -> DataLoader:
    """Load standard MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return test_loader


def get_rotated_mnist(angle: float, batch_size: int = 256) -> DataLoader:
    """Create rotated MNIST test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(degrees=(angle, angle)),
    ])
    
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return test_loader


def get_noisy_mnist(noise_std: float, batch_size: int = 256) -> DataLoader:
    """Create noisy MNIST test set."""
    # Load raw data
    test_dataset = datasets.MNIST(root="./data", train=False, download=True)
    data = test_dataset.data.float() / 255.0
    labels = test_dataset.targets
    
    # Add Gaussian noise
    noise = torch.randn_like(data) * noise_std
    noisy_data = torch.clamp(data + noise, 0.0, 1.0)
    
    # Normalize
    noisy_data = (noisy_data - 0.1307) / 0.3081
    noisy_data = noisy_data.unsqueeze(1)  # Add channel dim
    
    # Create dataset
    noisy_dataset = TensorDataset(noisy_data, labels)
    noisy_loader = DataLoader(
        noisy_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return noisy_loader


def get_fashion_mnist_loader(batch_size: int = 256) -> DataLoader:
    """Load FashionMNIST test set (for transfer eval)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use MNIST normalization
    ])
    
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return test_loader


def embed_data(
    model: AblationModel,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed data through model.
    
    Returns:
        embeddings: (N, latent_dim)
        labels: (N,)
        reconstructions: (N, 784)
    """
    embeddings_list = []
    labels_list = []
    recons_list = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            x = data.view(data.size(0), -1)
            
            x_recon, z = model(x)
            
            embeddings_list.append(z.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            recons_list.append(x_recon.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels_array = np.concatenate(labels_list)
    reconstructions = np.vstack(recons_list)
    
    return embeddings, labels_array, reconstructions


def evaluate_ood(
    model: AblationModel,
    loader: DataLoader,
    original_data: np.ndarray,
    device: str,
    k: int = 15,
) -> Dict[str, float]:
    """
    Evaluate model on OOD data.
    
    Args:
        model: Trained model
        loader: OOD data loader
        original_data: Original MNIST test data for topology baseline
        device: Device
        k: k for kNN metrics
    
    Returns:
        Dictionary of metrics
    """
    # Embed OOD data
    embeddings, labels, reconstructions = embed_data(model, loader, device)
    
    # Get original data from loader
    ood_data_list = []
    for data, _ in loader:
        ood_data_list.append(data.view(data.size(0), -1).numpy())
    ood_data = np.vstack(ood_data_list)
    
    # Reconstruction error
    recon_error = np.mean((ood_data - reconstructions) ** 2)
    
    # Topology metrics (compare to original space structure)
    trust = trustworthiness(original_data, embeddings, k=k)
    cont = continuity(original_data, embeddings, k=k)
    knn_pres = knn_preservation(original_data, embeddings, k=k)
    
    # Clustering metrics
    clustering = clustering_metrics(embeddings, labels)
    
    # kNN accuracy in latent space
    # Use first 5000 samples for training, rest for testing
    n_train = 5000
    X_train, y_train = embeddings[:n_train], labels[:n_train]
    X_test, y_test = embeddings[n_train:], labels[n_train:]
    
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    knn_acc = knn_clf.score(X_test, y_test)
    
    return {
        "reconstruction_error": float(recon_error),
        "trustworthiness": float(trust),
        "continuity": float(cont),
        "knn_preservation": float(knn_pres),
        "ari": float(clustering["ari"]),
        "nmi": float(clustering["nmi"]),
        "silhouette": float(clustering["silhouette"]),
        "knn_accuracy": float(knn_acc),
    }


def plot_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
):
    """Plot multiple embeddings in a grid."""
    n_plots = len(embeddings_dict)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, embeddings) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
        labels = labels_dict[name]
        
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap="tab10",
            s=1,
            alpha=0.5,
        )
        ax.set_title(name)
        ax.set_xlabel("Latent Dim 0")
        ax.set_ylabel("Latent Dim 1")
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")
    
    plt.colorbar(scatter, ax=axes[-1])
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_ood_experiments(
    checkpoint_dir: Path,
    config_name: str = "t_c_e",
    device: str = "cpu",
    latent_dim: int = 2,
):
    """Run full OOD robustness experiments."""
    
    out_dir, metrics_dir, plots_dir = setup_dirs()
    
    # Load model config
    config = {
        "latent_dim": latent_dim,
        "hidden": 128,
    }
    
    # Map config name to component flags
    config_map = {
        "baseline": {"use_topo": False, "use_causal": False, "use_energy": False},
        "t_only": {"use_topo": True, "use_causal": False, "use_energy": False},
        "c_only": {"use_topo": False, "use_causal": True, "use_energy": False},
        "e_only": {"use_topo": False, "use_causal": False, "use_energy": True},
        "t_c": {"use_topo": True, "use_causal": True, "use_energy": False},
        "t_e": {"use_topo": True, "use_causal": False, "use_energy": True},
        "c_e": {"use_topo": False, "use_causal": True, "use_energy": True},
        "t_c_e": {"use_topo": True, "use_causal": True, "use_energy": True},
    }
    
    config.update(config_map[config_name])
    
    # Load checkpoint
    ckpt_path = checkpoint_dir / f"{config_name}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, config, device)
    
    # Get original MNIST test data (for baseline)
    print("\n" + "="*60)
    print("Loading original MNIST test set (baseline)...")
    print("="*60)
    mnist_loader = get_mnist_test_loader()
    
    # Get original data for topology baseline
    test_dataset = datasets.MNIST(root="./data", train=False, download=True)
    original_data = test_dataset.data.float() / 255.0
    original_data = original_data.view(-1, 784).numpy()
    
    # Embed original
    mnist_emb, mnist_labels, _ = embed_data(model, mnist_loader, device)
    
    results = {}
    embeddings_dict = {"Original": mnist_emb}
    labels_dict = {"Original": mnist_labels}
    
    # 1. Rotated MNIST
    print("\n" + "="*60)
    print("Testing Rotated MNIST...")
    print("="*60)
    
    rotation_results = {}
    for angle in [15, 30, 45, 60]:
        print(f"\nRotation: {angle}°")
        rot_loader = get_rotated_mnist(angle)
        metrics = evaluate_ood(model, rot_loader, original_data, device)
        rotation_results[f"rot_{angle}"] = metrics
        
        # Store embeddings for visualization
        rot_emb, rot_labels, _ = embed_data(model, rot_loader, device)
        embeddings_dict[f"Rot {angle}°"] = rot_emb
        labels_dict[f"Rot {angle}°"] = rot_labels
        
        print(f"  Recon Error: {metrics['reconstruction_error']:.4f}")
        print(f"  Trustworthiness: {metrics['trustworthiness']:.4f}")
        print(f"  kNN Accuracy: {metrics['knn_accuracy']:.4f}")
    
    results["rotation"] = rotation_results
    
    # 2. Noisy MNIST
    print("\n" + "="*60)
    print("Testing Noisy MNIST...")
    print("="*60)
    
    noise_results = {}
    for noise_std in [0.1, 0.2, 0.3, 0.5]:
        print(f"\nNoise σ: {noise_std}")
        noisy_loader = get_noisy_mnist(noise_std)
        metrics = evaluate_ood(model, noisy_loader, original_data, device)
        noise_results[f"noise_{noise_std}"] = metrics
        
        # Store embeddings for visualization
        noisy_emb, noisy_labels, _ = embed_data(model, noisy_loader, device)
        embeddings_dict[f"Noise σ={noise_std}"] = noisy_emb
        labels_dict[f"Noise σ={noise_std}"] = noisy_labels
        
        print(f"  Recon Error: {metrics['reconstruction_error']:.4f}")
        print(f"  Trustworthiness: {metrics['trustworthiness']:.4f}")
        print(f"  kNN Accuracy: {metrics['knn_accuracy']:.4f}")
    
    results["noise"] = noise_results
    
    # 3. FashionMNIST transfer
    print("\n" + "="*60)
    print("Testing FashionMNIST transfer...")
    print("="*60)
    
    fashion_loader = get_fashion_mnist_loader()
    fashion_metrics = evaluate_ood(model, fashion_loader, original_data, device)
    results["fashion_mnist"] = fashion_metrics
    
    # Store embeddings for visualization
    fashion_emb, fashion_labels, _ = embed_data(model, fashion_loader, device)
    embeddings_dict["FashionMNIST"] = fashion_emb
    labels_dict["FashionMNIST"] = fashion_labels
    
    print(f"  Recon Error: {fashion_metrics['reconstruction_error']:.4f}")
    print(f"  Trustworthiness: {fashion_metrics['trustworthiness']:.4f}")
    print(f"  kNN Accuracy: {fashion_metrics['knn_accuracy']:.4f}")
    
    # Save results
    results_path = metrics_dir / f"{config_name}_ood_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved: {results_path}")
    print(f"{'='*60}")
    
    # Plot all embeddings
    if latent_dim == 2:
        plot_path = plots_dir / f"{config_name}_ood_embeddings.png"
        plot_embeddings(
            embeddings_dict,
            labels_dict,
            plot_path,
            f"{config_name} - OOD Robustness",
        )
        print(f"Plots saved: {plot_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"OOD ROBUSTNESS SUMMARY - {config_name}")
    print("="*80)
    print(f"{'Scenario':<20} {'Recon↓':<10} {'Trust↑':<10} {'kNN-Acc↑':<10}")
    print("-"*80)
    
    # Rotation
    for angle in [15, 30, 45, 60]:
        m = rotation_results[f"rot_{angle}"]
        print(
            f"{'Rotation ' + str(angle) + '°':<20} "
            f"{m['reconstruction_error']:<10.4f} "
            f"{m['trustworthiness']:<10.4f} "
            f"{m['knn_accuracy']:<10.4f}"
        )
    
    # Noise
    for noise_std in [0.1, 0.2, 0.3, 0.5]:
        m = noise_results[f"noise_{noise_std}"]
        print(
            f"{'Noise σ=' + str(noise_std):<20} "
            f"{m['reconstruction_error']:<10.4f} "
            f"{m['trustworthiness']:<10.4f} "
            f"{m['knn_accuracy']:<10.4f}"
        )
    
    # Fashion
    m = fashion_metrics
    print(
        f"{'FashionMNIST':<20} "
        f"{m['reconstruction_error']:<10.4f} "
        f"{m['trustworthiness']:<10.4f} "
        f"{m['knn_accuracy']:<10.4f}"
    )
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="MNIST OOD Robustness - Phase 5.4")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/ablation/checkpoints",
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="t_c_e",
        choices=["baseline", "t_only", "c_only", "e_only", "t_c", "t_e", "c_e", "t_c_e"],
        help="Model configuration to evaluate",
    )
    parser.add_argument("--latent", type=int, default=2, help="Latent dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "cpu" and torch.backends.mps.is_available():
        args.device = "mps"
    
    print(f"Using device: {args.device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print("Please run mnist_ablation.py first to train models.")
        return
    
    run_ood_experiments(
        checkpoint_dir=checkpoint_dir,
        config_name=args.config,
        device=args.device,
        latent_dim=args.latent,
    )


if __name__ == "__main__":
    main()
