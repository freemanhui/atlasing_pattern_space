"""
ColoredMNIST Dataset Generator

Creates a version of MNIST where digit color is spuriously correlated with labels
in training environments but anti-correlated or random in test environments.

This is designed to test causal invariance learning - models must learn to rely on
digit shape (causal feature) rather than color (spurious feature).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
import numpy as np


class ColoredMNIST(Dataset):
    """
    MNIST dataset with colored digits where color correlation with label varies by environment.
    
    Args:
        root: Path to download/load MNIST data
        train: If True, use training split; else test split
        color_correlation: Correlation between digit color and label
            - 1.0 = perfect positive correlation (color perfectly predicts label)
            - 0.5 = random (color independent of label)
            - 0.0 = random (color independent of label)
            - -1.0 = perfect anti-correlation (color predicts WRONG label)
            - Values between control interpolation
        colors: List of RGB color values (one per digit class 0-9)
        transform: Optional transform to apply to images
        download: Whether to download MNIST if not present
        
    Example:
        # Training environment with strong spurious correlation
        train_env = ColoredMNIST(root='./data', train=True, color_correlation=0.8)
        
        # Test environment with random/anti-correlated colors
        test_env = ColoredMNIST(root='./data', train=False, color_correlation=0.1)
    """
    
    # Default color palette: 10 distinct colors for digits 0-9
    DEFAULT_COLORS = [
        [1.0, 0.0, 0.0],  # 0: Red
        [0.0, 1.0, 0.0],  # 1: Green
        [0.0, 0.0, 1.0],  # 2: Blue
        [1.0, 1.0, 0.0],  # 3: Yellow
        [1.0, 0.0, 1.0],  # 4: Magenta
        [0.0, 1.0, 1.0],  # 5: Cyan
        [1.0, 0.5, 0.0],  # 6: Orange
        [0.5, 0.0, 1.0],  # 7: Purple
        [0.0, 0.5, 0.5],  # 8: Teal
        [0.5, 0.5, 0.5],  # 9: Gray
    ]
    
    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        color_correlation: float = 0.8,
        colors: Optional[List[List[float]]] = None,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        seed: Optional[int] = None,
    ):
        self.color_correlation = color_correlation
        self.colors = torch.tensor(colors if colors else self.DEFAULT_COLORS, dtype=torch.float32)
        self.transform = transform
        
        # Load original MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Set random seed for reproducible color assignment
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Pre-compute color assignments for all samples
        self._compute_color_assignments()
        
    def _compute_color_assignments(self):
        """
        Pre-compute which color each sample should have based on correlation rate.
        
        Supports:
        - Positive correlation (0.5-1.0): color matches label with given probability
        - Anti-correlation (-1.0 to 0.0): color predicts WRONG label
        - Random (0.5): color independent of label
        """
        n_samples = len(self.mnist)
        self.color_indices = torch.zeros(n_samples, dtype=torch.long)
        
        for idx in range(n_samples):
            _, label = self.mnist[idx]
            
            if self.color_correlation >= 0:
                # Positive correlation: color matches label with probability
                if torch.rand(1).item() < self.color_correlation:
                    self.color_indices[idx] = label
                else:
                    # Assign random different color
                    wrong_colors = [c for c in range(10) if c != label]
                    self.color_indices[idx] = np.random.choice(wrong_colors)
            else:
                # Anti-correlation: color predicts wrong label
                # -1.0 = always wrong, 0.0 = random
                anti_prob = abs(self.color_correlation)
                if torch.rand(1).item() < anti_prob:
                    # Assign a wrong color systematically (shift by 1)
                    self.color_indices[idx] = (label + 1) % 10
                else:
                    # Random color
                    self.color_indices[idx] = np.random.randint(0, 10)
    
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image: RGB colored digit image (3, 28, 28)
            label: Digit class (0-9)
            color_idx: Assigned color index (for analysis)
        """
        # Get grayscale image and label
        gray_img, label = self.mnist[idx]  # (1, 28, 28), label
        
        # Get assigned color for this sample
        color_idx = self.color_indices[idx].item()
        color = self.colors[color_idx]  # (3,)
        
        # Create RGB image by multiplying grayscale with color
        # gray_img: (1, 28, 28), color: (3,) -> rgb_img: (3, 28, 28)
        rgb_img = gray_img * color.view(3, 1, 1)
        
        # Apply transforms if specified
        if self.transform:
            rgb_img = self.transform(rgb_img)
            
        return rgb_img, label, color_idx


def create_colored_mnist_envs(
    root: str = './data',
    train_correlations: List[float] = [0.8, 0.7],
    test_correlation: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: Optional[int] = 42,
) -> Tuple[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    """
    Create multiple training environments and one test environment with different color correlations.
    
    This setup enables testing Invariant Risk Minimization (IRM) and other causal learning methods.
    
    Args:
        root: Path to download/load MNIST
        train_correlations: List of correlation rates for training environments
            Example: [0.8, 0.7] creates two environments with strong spurious correlation
        test_correlation: Correlation rate for test environment (typically low, e.g., 0.1)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        train_loaders: List of dataloaders for training environments
        test_loader: Single dataloader for test environment
        
    Example:
        >>> train_loaders, test_loader = create_colored_mnist_envs(
        ...     train_correlations=[0.8, 0.7],
        ...     test_correlation=0.1
        ... )
        >>> # Train IRM: minimize worst-case risk across train environments
        >>> # Test: evaluate on low-correlation environment
    """
    train_loaders = []
    
    # Create training environments
    for env_idx, corr in enumerate(train_correlations):
        train_dataset = ColoredMNIST(
            root=root,
            train=True,
            color_correlation=corr,
            download=True,
            seed=seed + env_idx if seed else None,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_loaders.append(train_loader)
    
    # Create test environment
    test_dataset = ColoredMNIST(
        root=root,
        train=False,
        color_correlation=test_correlation,
        download=True,
        seed=seed + len(train_correlations) if seed else None,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loaders, test_loader


def get_color_label_stats(dataset: ColoredMNIST) -> dict:
    """
    Compute statistics about color-label correlation in a ColoredMNIST dataset.
    
    Args:
        dataset: ColoredMNIST dataset instance
        
    Returns:
        Dictionary containing:
            - 'correlation': Overall fraction of samples where color matches label
            - 'per_class': Per-digit correlation rates
            - 'confusion_matrix': Color-label confusion matrix (10x10)
    """
    n_classes = 10
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    
    # Count color-label co-occurrences
    for idx in range(len(dataset)):
        _, label, color_idx = dataset[idx]
        confusion[label, color_idx] += 1
    
    # Compute per-class correlations
    per_class = {}
    for c in range(n_classes):
        total = confusion[c].sum().item()
        correct = confusion[c, c].item()
        per_class[c] = correct / total if total > 0 else 0.0
    
    # Overall correlation
    total_samples = confusion.sum().item()
    correct_samples = torch.diag(confusion).sum().item()
    overall_corr = correct_samples / total_samples if total_samples > 0 else 0.0
    
    return {
        'correlation': overall_corr,
        'per_class': per_class,
        'confusion_matrix': confusion.numpy(),
    }


if __name__ == '__main__':
    """
    Demo: Create ColoredMNIST datasets and visualize color-label correlations.
    """
    import matplotlib.pyplot as plt
    
    # Create datasets with different correlations
    high_corr = ColoredMNIST(root='./data', train=True, color_correlation=0.9, seed=42)
    low_corr = ColoredMNIST(root='./data', train=False, color_correlation=0.1, seed=42)
    
    # Print statistics
    print("High correlation training environment:")
    stats_high = get_color_label_stats(high_corr)
    print(f"  Overall correlation: {stats_high['correlation']:.3f}")
    print(f"  Per-class: {stats_high['per_class']}")
    
    print("\nLow correlation test environment:")
    stats_low = get_color_label_stats(low_corr)
    print(f"  Overall correlation: {stats_low['correlation']:.3f}")
    print(f"  Per-class: {stats_low['per_class']}")
    
    # Visualize samples
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fig.suptitle('ColoredMNIST Samples (Top: High Corr, Bottom: Low Corr)')
    
    for i in range(10):
        # Find a sample with digit i
        for idx in range(len(high_corr)):
            img_high, label_high, _ = high_corr[idx]
            if label_high == i:
                axes[0, i].imshow(img_high.permute(1, 2, 0).numpy())
                axes[0, i].axis('off')
                axes[0, i].set_title(f'{i}')
                break
        
        for idx in range(len(low_corr)):
            img_low, label_low, _ = low_corr[idx]
            if label_low == i:
                axes[1, i].imshow(img_low.permute(1, 2, 0).numpy())
                axes[1, i].axis('off')
                break
    
    plt.tight_layout()
    plt.savefig('outputs/colored_mnist_demo.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to outputs/colored_mnist_demo.png")
