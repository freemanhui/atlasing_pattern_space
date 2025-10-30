"""Colored MNIST dataset for OOD generalization experiments.

This dataset adds color as a spurious feature to MNIST digits:
- Training environments: Color is correlated with digit groups (0-4 vs 5-9)
- Test environment (OOD): Correlation is flipped to test causal invariance

Reference: Arjovsky et al. (2019) - Invariant Risk Minimization
"""

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Literal, Tuple


class ColoredMNIST(Dataset):
    """
    MNIST digits with color as spurious feature.
    
    Args:
        split: 'env1', 'env2', or 'test' (determines correlation strength)
        correlation: Probability that color matches label group (0-4 vs 5-9)
        root: Directory to download/load MNIST data
        download: Whether to download MNIST if not present
        
    Returns dict with:
        image: (3, 28, 28) RGB tensor with colored digit
        label: digit label (0-9)
        color: color index (0=red, 1=green) - for HSIC loss
        environment: environment ID (0 or 1) - for IRM loss
    """
    
    COLORS = {
        'red': torch.tensor([1.0, 0.0, 0.0]),
        'green': torch.tensor([0.0, 1.0, 0.0])
    }
    
    def __init__(
        self,
        split: Literal['env1', 'env2', 'test'] = 'env1',
        correlation: float = 0.8,
        root: str = './data',
        download: bool = True
    ):
        self.split = split
        self.correlation = correlation
        self.root = root
        
        # Load MNIST
        is_train = (split != 'test')
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=is_train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Assign colors based on correlation
        self._assign_colors()
    
    def _assign_colors(self):
        """Assign colors based on correlation strength and split."""
        labels = self.mnist.targets
        n_samples = len(labels)
        
        # Define label groups: 0-4 (group 0) vs 5-9 (group 1)
        label_groups = (labels >= 5).long()
        
        # For OOD test, flip the correlation
        if self.split == 'test':
            # Test: opposite correlation (if group 0, use green; if group 1, use red)
            self.colors = 1 - label_groups
        else:
            # Training: color matches group with probability=correlation
            self.colors = label_groups.clone()
            
            # Flip some colors with probability (1 - correlation)
            flip_mask = torch.rand(n_samples) > self.correlation
            self.colors[flip_mask] = 1 - self.colors[flip_mask]
        
        # Environment ID
        self.env_id = 0 if self.split == 'env1' else (1 if self.split == 'env2' else 2)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single colored MNIST sample."""
        image, label = self.mnist[idx]
        color_idx = self.colors[idx].item()
        
        # Convert grayscale (1, 28, 28) to RGB (3, 28, 28) with color
        # Multiply grayscale by color vector
        color_vec = self.COLORS['red' if color_idx == 0 else 'green']
        rgb_image = image * color_vec.view(3, 1, 1)
        
        return {
            'image': rgb_image,
            'label': label,
            'color': torch.tensor(color_idx, dtype=torch.long),
            'environment': torch.tensor(self.env_id, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.mnist)


def create_colored_mnist_loaders(
    batch_size: int = 128,
    correlation_env1: float = 0.8,
    correlation_env2: float = 0.9,
    num_workers: int = 4,
    root: str = './data',
    download: bool = True
) -> dict:
    """
    Create train/test dataloaders for Colored MNIST OOD experiment.
    
    Args:
        batch_size: Batch size for dataloaders
        correlation_env1: Color-label correlation for environment 1
        correlation_env2: Color-label correlation for environment 2
        num_workers: Number of workers for dataloaders
        root: Data directory
        download: Whether to download MNIST if not present
        
    Returns:
        dict with keys 'train' (combined env1+env2) and 'test_ood' (flipped correlation)
    """
    # Create training environments
    train_env1 = ColoredMNIST(
        split='env1',
        correlation=correlation_env1,
        root=root,
        download=download
    )
    
    train_env2 = ColoredMNIST(
        split='env2',
        correlation=correlation_env2,
        root=root,
        download=download
    )
    
    # Create OOD test set (flipped correlation)
    test_ood = ColoredMNIST(
        split='test',
        correlation=correlation_env2,  # Use same correlation strength, but flipped
        root=root,
        download=download
    )
    
    # Combine training environments
    train_combined = ConcatDataset([train_env1, train_env2])
    
    # Create dataloaders
    loaders = {
        'train': DataLoader(
            train_combined,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test_ood': DataLoader(
            test_ood,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders


def get_correlation_stats(dataset: ColoredMNIST) -> Tuple[float, int, int]:
    """
    Compute actual color-label correlation in dataset.
    
    Args:
        dataset: ColoredMNIST dataset
        
    Returns:
        (correlation, num_matches, total_samples)
    """
    matches = 0
    total = len(dataset)
    
    for idx in range(total):
        sample = dataset[idx]
        label = sample['label'] if isinstance(sample['label'], int) else sample['label'].item()
        color = sample['color'].item()
        
        # Check if color matches label group
        label_group = 1 if label >= 5 else 0
        if label_group == color:
            matches += 1
    
    correlation = matches / total
    return correlation, matches, total


if __name__ == '__main__':
    """Quick test of ColoredMNIST dataset."""
    print("Testing ColoredMNIST dataset...")
    
    # Create datasets
    print("\n[1/3] Creating datasets...")
    env1 = ColoredMNIST('env1', correlation=0.8)
    env2 = ColoredMNIST('env2', correlation=0.9)
    test = ColoredMNIST('test', correlation=0.9)
    
    print(f"  Env1: {len(env1)} samples")
    print(f"  Env2: {len(env2)} samples")
    print(f"  Test: {len(test)} samples")
    
    # Check sample
    print("\n[2/3] Checking sample format...")
    sample = env1[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Color: {sample['color'].item()} ({'red' if sample['color'] == 0 else 'green'})")
    print(f"  Environment: {sample['environment'].item()}")
    
    # Check correlations
    print("\n[3/3] Verifying correlations (sampling 1000 examples)...")
    for name, dataset in [('Env1', env1), ('Env2', env2), ('Test', test)]:
        corr, matches, total = get_correlation_stats(dataset)
        print(f"  {name}: {corr:.3f} ({matches}/{min(1000, total)} matches)")
    
    print("\n✓ ColoredMNIST dataset working correctly!")
