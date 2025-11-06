"""Colored Clusters Dataset for T-C Conflict Experiment

Synthetic dataset where topology and causality objectives conflict:
- 2 shape classes (circles vs squares) - causal features
- 2 color classes (red vs blue) - spurious features
- Training: high color-label correlation (90%)
- Test: random color-label correlation (50%)

When spurious features create topology, T (preserve) and C (ignore) conflict.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


def generate_colored_clusters(
    n_samples: int = 1000,
    color_correlation: float = 0.9,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D shapes with spurious color correlation.
    
    Args:
        n_samples: Total number of samples
        color_correlation: Fraction of samples where color matches label
        noise: Gaussian noise std for shape generation
        seed: Random seed for reproducibility
    
    Returns:
        X: (n_samples, 4) - Full features [shape_x, shape_y, color_red, color_blue]
        y: (n_samples,) - True shape labels (0=circle, 1=square)
        X_shape: (n_samples, 2) - Shape features only
        color: (n_samples,) - Color labels (0=red, 1=blue)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_per_class = n_samples // 2
    
    # Generate shape features (causal)
    # Class 0 (circles): points sampled from circle boundary
    theta = np.random.rand(n_per_class) * 2 * np.pi
    circles = np.stack([
        np.cos(theta) + np.random.randn(n_per_class) * noise,
        np.sin(theta) + np.random.randn(n_per_class) * noise
    ], axis=1)
    
    # Class 1 (squares): points sampled uniformly from square
    squares = np.random.rand(n_per_class, 2) * 2 - 1  # [-1, 1]
    squares += np.random.randn(n_per_class, 2) * noise
    
    X_shape = np.vstack([circles, squares])
    y = np.hstack([
        np.zeros(n_per_class, dtype=np.int64),
        np.ones(n_per_class, dtype=np.int64)
    ])
    
    # Generate color features (spurious)
    n_correlated = int(n_samples * color_correlation)
    n_uncorrelated = n_samples - n_correlated
    
    # Correlated: color = label
    color_corr = y[:n_correlated].copy()
    
    # Uncorrelated: random color
    color_uncorr = np.random.randint(0, 2, n_uncorrelated, dtype=np.int64)
    
    # Concatenate and shuffle together
    color = np.hstack([color_corr, color_uncorr])
    indices = np.random.permutation(n_samples)
    
    X_shape = X_shape[indices]
    y = y[indices]
    color = color[indices]
    
    # One-hot encode color
    X_color = np.eye(2)[color]
    
    # Concatenate shape + color for full features
    X = np.hstack([X_shape, X_color]).astype(np.float32)
    
    return X, y, X_shape.astype(np.float32), color


class ColoredClustersDataset(Dataset):
    """PyTorch Dataset wrapper for Colored Clusters."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        color_correlation: float = 0.9,
        noise: float = 0.1,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.X, self.y, self.X_shape, self.color = generate_colored_clusters(
            n_samples=n_samples,
            color_correlation=color_correlation,
            noise=noise,
            seed=seed
        )
        
        # Convert to tensors
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
        self.X_shape = torch.from_numpy(self.X_shape)
        self.color = torch.from_numpy(self.color)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'X': self.X[idx],           # Full features (shape + color)
            'y': self.y[idx],           # Shape label (causal)
            'X_shape': self.X_shape[idx],  # Shape features only
            'color': self.color[idx]    # Color label (spurious)
        }


def create_colored_clusters_loaders(
    train_samples: int = 5000,
    val_samples: int = 1000,
    test_samples: int = 2000,
    train_correlation: float = 0.9,
    test_correlation: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
) -> dict:
    """Create train/val/test dataloaders for Colored Clusters.
    
    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        train_correlation: Color-label correlation in training
        test_correlation: Color-label correlation in test/val
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        seed: Random seed base
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    
    # Create datasets
    train_dataset = ColoredClustersDataset(
        n_samples=train_samples,
        color_correlation=train_correlation,
        noise=0.1,
        seed=seed
    )
    
    val_dataset = ColoredClustersDataset(
        n_samples=val_samples,
        color_correlation=test_correlation,
        noise=0.1,
        seed=seed + 1
    )
    
    test_dataset = ColoredClustersDataset(
        n_samples=test_samples,
        color_correlation=test_correlation,
        noise=0.1,
        seed=seed + 2
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def visualize_colored_clusters(
    X: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    title: str = "Colored Clusters",
    save_path: Optional[str] = None
):
    """Visualize the Colored Clusters dataset.
    
    Args:
        X: (n, 4) features [shape_x, shape_y, color_red, color_blue]
        y: (n,) shape labels
        color: (n,) color labels
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Shape features colored by shape label
    ax = axes[0]
    for shape_label in [0, 1]:
        mask = y == shape_label
        label = 'Circles' if shape_label == 0 else 'Squares'
        ax.scatter(
            X[mask, 0], X[mask, 1],
            label=label,
            alpha=0.6,
            s=50
        )
    ax.set_xlabel('Shape X')
    ax.set_ylabel('Shape Y')
    ax.set_title('Shape Features (Causal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Shape features colored by color label
    ax = axes[1]
    for color_label in [0, 1]:
        mask = color == color_label
        label = 'Red' if color_label == 0 else 'Blue'
        marker = 'o' if color_label == 0 else 's'
        ax.scatter(
            X[mask, 0], X[mask, 1],
            label=label,
            alpha=0.6,
            s=50,
            marker=marker
        )
    ax.set_xlabel('Shape X')
    ax.set_ylabel('Shape Y')
    ax.set_title('Shape Features (Colored by Spurious)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: 4-cluster structure (shape × color)
    ax = axes[2]
    colors_map = {
        (0, 0): 'red',      # Circle + Red
        (0, 1): 'blue',     # Circle + Blue
        (1, 0): 'orange',   # Square + Red
        (1, 1): 'cyan'      # Square + Blue
    }
    markers_map = {
        (0, 0): 'o',
        (0, 1): 'o',
        (1, 0): 's',
        (1, 1): 's'
    }
    
    for shape_label in [0, 1]:
        for color_label in [0, 1]:
            mask = (y == shape_label) & (color == color_label)
            shape_name = 'Circle' if shape_label == 0 else 'Square'
            color_name = 'Red' if color_label == 0 else 'Blue'
            ax.scatter(
                X[mask, 0], X[mask, 1],
                c=colors_map[(shape_label, color_label)],
                marker=markers_map[(shape_label, color_label)],
                label=f'{shape_name} + {color_name}',
                alpha=0.6,
                s=50
            )
    
    ax.set_xlabel('Shape X')
    ax.set_ylabel('Shape Y')
    ax.set_title('4-Cluster Structure (Shape × Color)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Demo: Generate and visualize dataset
    print("Generating Colored Clusters dataset...")
    
    # Training set (high correlation)
    X_train, y_train, X_shape_train, color_train = generate_colored_clusters(
        n_samples=1000,
        color_correlation=0.9,
        noise=0.1,
        seed=42
    )
    
    print(f"Training set:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Shape labels: {np.unique(y_train, return_counts=True)}")
    print(f"  Color labels: {np.unique(color_train, return_counts=True)}")
    print(f"  Color-shape correlation: {(color_train == y_train).mean():.2%}")
    
    # Test set (random correlation)
    X_test, y_test, X_shape_test, color_test = generate_colored_clusters(
        n_samples=1000,
        color_correlation=0.5,
        noise=0.1,
        seed=43
    )
    
    print(f"\nTest set:")
    print(f"  Shape: {X_test.shape}")
    print(f"  Color-shape correlation: {(color_test == y_test).mean():.2%}")
    
    # Visualize
    print("\nVisualizing training set...")
    visualize_colored_clusters(
        X_train, y_train, color_train,
        title="Training Set (90% Color-Shape Correlation)",
        save_path="colored_clusters_train.png"
    )
    
    print("\nVisualizing test set...")
    visualize_colored_clusters(
        X_test, y_test, color_test,
        title="Test Set (50% Color-Shape Correlation)",
        save_path="colored_clusters_test.png"
    )
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    loaders = create_colored_clusters_loaders(
        train_samples=1000,
        val_samples=200,
        test_samples=300,
        train_correlation=0.9,
        test_correlation=0.5,
        batch_size=64
    )
    
    batch = next(iter(loaders['train']))
    print(f"Batch shapes:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape}")
    
    print("\n✅ Colored Clusters dataset ready!")
