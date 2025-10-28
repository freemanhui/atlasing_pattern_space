"""
Dataset utilities for APS experiments.

Provides standardized data loaders for:
- MNIST (standard and variations)
- Colored MNIST (with spurious correlations)
- Few-shot splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
from pathlib import Path


def get_mnist_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    val_split: float = 0.1,
    flatten: bool = True,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get standard MNIST train/val/test dataloaders.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for dataloaders
        val_split: Fraction of training data for validation
        flatten: If True, flatten images to vectors (784,)
        normalize: If True, normalize to [0, 1]
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Setup transforms
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    
    # Split train into train/val
    n_train = len(train_dataset)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    indices = torch.randperm(len(train_dataset))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader


class ColoredMNIST(Dataset):
    """
    Colored MNIST with spurious correlation for OOD experiments.
    
    Creates MNIST digits with color as a spurious feature:
    - Training: digits 0-4 are red, 5-9 are blue (or customizable)
    - Testing: correlation can be flipped to test OOD robustness
    
    Args:
        mnist_dataset: Base MNIST dataset
        correlation: Probability of correct color assignment (0.9 = 90% correlation)
        flip: If True, flip the color-digit correlation (for test set)
        seed: Random seed
    """
    
    def __init__(
        self,
        mnist_dataset: Dataset,
        correlation: float = 0.9,
        flip: bool = False,
        seed: int = 42,
    ):
        self.mnist_dataset = mnist_dataset
        self.correlation = correlation
        self.flip = flip
        
        # Set random seed
        np.random.seed(seed)
        
        # Pre-generate color assignments
        self.colors = self._generate_colors()
    
    def _generate_colors(self) -> np.ndarray:
        """Generate color assignments for all samples."""
        n_samples = len(self.mnist_dataset)
        colors = np.zeros(n_samples, dtype=np.int64)
        
        for idx in range(n_samples):
            _, label = self.mnist_dataset[idx]
            
            # Determine "correct" color (0 = red, 1 = blue)
            correct_color = 0 if label < 5 else 1
            
            # Flip if needed
            if self.flip:
                correct_color = 1 - correct_color
            
            # Apply spurious correlation
            if np.random.rand() < self.correlation:
                colors[idx] = correct_color
            else:
                colors[idx] = 1 - correct_color
        
        return colors
    
    def __len__(self) -> int:
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image: Grayscale MNIST image (1, 28, 28)
            label: Digit label (0-9)
            color: Color assignment (0=red, 1=blue)
        """
        image, label = self.mnist_dataset[idx]
        color = self.colors[idx]
        
        return image, label, color
    
    @staticmethod
    def apply_color(image: torch.Tensor, color: int) -> torch.Tensor:
        """
        Apply color to grayscale image.
        
        Args:
            image: Grayscale image (1, H, W) or (H, W)
            color: 0 for red, 1 for blue
        
        Returns:
            Colored image (3, H, W)
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        # Create RGB image
        if color == 0:  # Red
            colored = torch.stack([image[0], image[0] * 0.3, image[0] * 0.3])
        else:  # Blue
            colored = torch.stack([image[0] * 0.3, image[0] * 0.3, image[0]])
        
        return colored


def get_colored_mnist_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    train_correlation: float = 0.9,
    test_correlation: float = 0.1,  # Flipped correlation
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get Colored MNIST dataloaders with spurious correlation.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size
        train_correlation: Correlation strength in training (0.9 = strong)
        test_correlation: Correlation in test (0.1 = flipped)
        val_split: Fraction for validation
        seed: Random seed
    
    Returns:
        train_loader, val_loader, test_loader
        Each batch returns (image, label, color)
    """
    # Load base MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_mnist = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    
    test_mnist = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    
    # Create colored versions
    train_colored = ColoredMNIST(
        train_mnist,
        correlation=train_correlation,
        flip=False,
        seed=seed,
    )
    
    test_colored = ColoredMNIST(
        test_mnist,
        correlation=test_correlation,
        flip=True,  # Flip correlation for OOD test
        seed=seed + 1,
    )
    
    # Split train into train/val
    n_train = len(train_colored)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(train_colored))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = Subset(train_colored, train_indices)
    val_subset = Subset(train_colored, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_colored,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader


def create_few_shot_split(
    dataset: Dataset,
    n_way: int = 10,
    k_shot: int = 5,
    n_query: int = 15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Create few-shot learning split from dataset.
    
    Args:
        dataset: PyTorch dataset with (data, label) items
        n_way: Number of classes (e.g., 10 for MNIST)
        k_shot: Number of support examples per class
        n_query: Number of query examples per class
        seed: Random seed
    
    Returns:
        support_indices: Indices for k-shot support set
        query_indices: Indices for query set
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = {i: [] for i in range(n_way)}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[label].append(idx)
    
    support_indices = []
    query_indices = []
    
    for class_id in range(n_way):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        # Take k_shot for support, n_query for query
        support_indices.extend(indices[:k_shot])
        query_indices.extend(indices[k_shot:k_shot + n_query])
    
    return support_indices, query_indices


def get_embeddings_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a trained model.
    
    Args:
        model: Model with encode() method
        dataloader: DataLoader for dataset
        device: Device to use
    
    Returns:
        embeddings: (N, latent_dim)
        labels: (N,)
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                data, labels = batch
            else:
                # Handle colored MNIST (data, labels, color)
                data, labels, _ = batch
            
            data = data.to(device)
            
            # Get embeddings
            embeddings = model.encode(data)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return embeddings, labels
