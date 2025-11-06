#!/usr/bin/env python3
"""
MNIST Few-Shot Learning - Phase 5.5

Tests how well learned embeddings support few-shot learning:
- k-shot learning: 1, 3, 5, 10 examples per class
- Evaluation on full test set
- Comparison across different model configurations

The hypothesis is that topology-preserving and energy-shaped embeddings
should enable better few-shot generalization through better structured
latent spaces.

Methodology:
1. Train shallow classifier on k examples per class in latent space
2. Evaluate on full test set
3. Compare: logistic regression, kNN, prototypical network
4. Report accuracy and confusion patterns
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def setup_dirs() -> Tuple[Path, Path, Path]:
    """Create output directories."""
    out_dir = Path("outputs/fewshot")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    return out_dir, metrics_dir, plots_dir


class AblationModel(nn.Module):
    """Same model architecture for consistency."""
    
    def __init__(
        self,
        in_dim: int = 784,
        latent_dim: int = 2,
        hidden: int = 128,
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def load_model(ckpt_path: Path, latent_dim: int, device: str) -> AblationModel:
    """Load trained model from checkpoint."""
    model = AblationModel(in_dim=784, latent_dim=latent_dim, hidden=128).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


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
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def embed_dataset(
    model: AblationModel,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed entire dataset."""
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            x = data.view(data.size(0), -1)
            _, z = model(x)
            embeddings_list.append(z.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    return embeddings, labels


def sample_k_shot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample k examples per class.
    
    Returns:
        X_support: (n_classes * k, latent_dim)
        y_support: (n_classes * k,)
    """
    np.random.seed(seed)
    
    n_classes = len(np.unique(labels))
    support_indices = []
    
    for cls in range(n_classes):
        cls_indices = np.where(labels == cls)[0]
        sampled = np.random.choice(cls_indices, size=k, replace=False)
        support_indices.extend(sampled)
    
    support_indices = np.array(support_indices)
    X_support = embeddings[support_indices]
    y_support = labels[support_indices]
    
    return X_support, y_support


def prototypical_classifier(
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
) -> np.ndarray:
    """
    Prototypical network classifier.
    
    Computes class prototypes (centroids) from support set,
    then assigns queries to nearest prototype.
    """
    n_classes = len(np.unique(y_support))
    prototypes = []
    
    for cls in range(n_classes):
        cls_mask = y_support == cls
        cls_prototype = X_support[cls_mask].mean(axis=0)
        prototypes.append(cls_prototype)
    
    prototypes = np.array(prototypes)  # (n_classes, latent_dim)
    
    # Compute distances to prototypes
    # (n_query, latent_dim) @ (latent_dim, n_classes) -> (n_query, n_classes)
    distances = np.linalg.norm(
        X_query[:, None, :] - prototypes[None, :, :], axis=2
    )
    
    # Assign to nearest prototype
    predictions = np.argmin(distances, axis=1)
    
    return predictions


def evaluate_few_shot(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    n_trials: int = 5,
) -> Dict[str, float]:
    """
    Evaluate few-shot learning performance.
    
    Averages over multiple random k-shot samples.
    """
    logreg_accs = []
    knn_accs = []
    proto_accs = []
    
    for trial in range(n_trials):
        # Sample k-shot support set
        X_support, y_support = sample_k_shot(X_train, y_train, k, seed=trial)
        
        # Logistic Regression
        logreg = LogisticRegression(max_iter=1000, random_state=trial)
        logreg.fit(X_support, y_support)
        y_pred_logreg = logreg.predict(X_test)
        logreg_accs.append(accuracy_score(y_test, y_pred_logreg))
        
        # k-Nearest Neighbors (k=5)
        knn = KNeighborsClassifier(n_neighbors=min(5, k))
        knn.fit(X_support, y_support)
        y_pred_knn = knn.predict(X_test)
        knn_accs.append(accuracy_score(y_test, y_pred_knn))
        
        # Prototypical Network
        y_pred_proto = prototypical_classifier(X_support, y_support, X_test)
        proto_accs.append(accuracy_score(y_test, y_pred_proto))
    
    return {
        "logreg_mean": float(np.mean(logreg_accs)),
        "logreg_std": float(np.std(logreg_accs)),
        "knn_mean": float(np.mean(knn_accs)),
        "knn_std": float(np.std(knn_accs)),
        "proto_mean": float(np.mean(proto_accs)),
        "proto_std": float(np.std(proto_accs)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str,
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_fewshot_experiments(
    checkpoint_dir: Path,
    config_name: str = "t_c_e",
    device: str = "cpu",
    latent_dim: int = 2,
    k_shots: List[int] = [1, 3, 5, 10],
    n_trials: int = 5,
):
    """Run few-shot learning experiments."""
    
    out_dir, metrics_dir, plots_dir = setup_dirs()
    
    # Load model
    ckpt_path = checkpoint_dir / f"{config_name}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, latent_dim, device)
    
    # Load data
    print("\n" + "="*60)
    print("Loading and embedding MNIST...")
    print("="*60)
    train_loader, test_loader = get_mnist_loaders()
    
    X_train, y_train = embed_dataset(model, train_loader, device)
    X_test, y_test = embed_dataset(model, test_loader, device)
    
    print(f"Train embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")
    
    # Run few-shot experiments
    results = {}
    
    print("\n" + "="*60)
    print("Running Few-Shot Experiments...")
    print("="*60)
    
    for k in k_shots:
        print(f"\n{k}-shot learning ({n_trials} trials)...")
        metrics = evaluate_few_shot(X_train, y_train, X_test, y_test, k, n_trials)
        results[f"{k}_shot"] = metrics
        
        print(f"  LogReg: {metrics['logreg_mean']:.4f} ± {metrics['logreg_std']:.4f}")
        print(f"  kNN:    {metrics['knn_mean']:.4f} ± {metrics['knn_std']:.4f}")
        print(f"  Proto:  {metrics['proto_mean']:.4f} ± {metrics['proto_std']:.4f}")
    
    # Save results
    results_path = metrics_dir / f"{config_name}_fewshot_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved: {results_path}")
    print(f"{'='*60}")
    
    # Generate confusion matrix for 5-shot prototypical (last trial)
    print("\nGenerating confusion matrix (5-shot, last trial)...")
    X_support, y_support = sample_k_shot(X_train, y_train, k=5, seed=n_trials-1)
    y_pred = prototypical_classifier(X_support, y_support, X_test)
    
    cm_path = plots_dir / f"{config_name}_confusion_5shot.png"
    plot_confusion_matrix(
        y_test,
        y_pred,
        cm_path,
        f"{config_name} - 5-Shot Prototypical (Trial {n_trials})",
    )
    print(f"Confusion matrix saved: {cm_path}")
    
    # Plot learning curves
    print("\nGenerating learning curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ["logreg", "knn", "proto"]
    method_names = ["Logistic Regression", "k-NN", "Prototypical"]
    
    for ax, method, method_name in zip(axes, methods, method_names):
        means = [results[f"{k}_shot"][f"{method}_mean"] for k in k_shots]
        stds = [results[f"{k}_shot"][f"{method}_std"] for k in k_shots]
        
        ax.errorbar(k_shots, means, yerr=stds, marker="o", capsize=5)
        ax.set_xlabel("k (shots per class)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(method_name)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle(f"{config_name} - Few-Shot Learning Curves", fontsize=14)
    plt.tight_layout()
    
    curve_path = plots_dir / f"{config_name}_learning_curves.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Learning curves saved: {curve_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"FEW-SHOT LEARNING SUMMARY - {config_name}")
    print("="*80)
    print(f"{'k-shot':<10} {'LogReg':<20} {'kNN':<20} {'Proto':<20}")
    print("-"*80)
    
    for k in k_shots:
        m = results[f"{k}_shot"]
        print(
            f"{k:<10} "
            f"{m['logreg_mean']:.3f}±{m['logreg_std']:.3f}        "
            f"{m['knn_mean']:.3f}±{m['knn_std']:.3f}        "
            f"{m['proto_mean']:.3f}±{m['proto_std']:.3f}"
        )
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="MNIST Few-Shot Learning - Phase 5.5")
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
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of k-shot values to test",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of random trials per k-shot",
    )
    
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
    
    run_fewshot_experiments(
        checkpoint_dir=checkpoint_dir,
        config_name=args.config,
        device=args.device,
        latent_dim=args.latent,
        k_shots=args.k_shots,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
