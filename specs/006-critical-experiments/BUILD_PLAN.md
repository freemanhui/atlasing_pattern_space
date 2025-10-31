# Phase 006: Critical Experiments - Detailed Build Plan
**Addressing Critical Review Gaps**

---

## 🎯 Objectives

This phase addresses the three critical experimental gaps identified in the paper review:
1. **Colored MNIST (OOD)**: Rigorous causality validation with true distribution shift
2. **Text Domain OOD**: High-dimensional validation beyond MNIST
3. **T-C Trade-off Analysis**: Character ize Pareto frontier of competing objectives

---

## 📅 Timeline & Dependencies

### Sub-Phase 6A: Colored MNIST (Week 1)
- **Duration**: 5 days
- **Dependencies**: Phases 001-005 complete
- **Critical Path**: Yes (required for paper)

### Sub-Phase 6B: Text Domain OOD (Weeks 2-3)
- **Duration**: 10 days
- **Dependencies**: 6A complete (design patterns established)
- **Critical Path**: Yes (required for paper)

### Sub-Phase 6C: T-C Trade-off Analysis (Week 4)
- **Duration**: 5 days
- **Dependencies**: 6A, 6B complete (methodology validated)
- **Critical Path**: Should have (strengthens paper significantly)

---

## 📐 Sub-Phase 6A: Colored MNIST Implementation

### Day 1-2: Dataset Generation

#### File: `src/aps/data/colored_mnist.py`
```python
"""Colored MNIST dataset for OOD generalization experiments."""

import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np

class ColoredMNIST(Dataset):
    """
    MNIST digits with color as spurious feature.
    
    Args:
        split: 'env1', 'env2', 'test' (determines correlation strength)
        correlation: Probability that color matches label group
        root: MNIST data directory
        
    Returns:
        image: (3, 28, 28) RGB tensor
        label: digit label (0-9)
        color: color applied (0=red, 1=green) - for HSIC loss
        environment: environment ID (for IRM loss)
    """
    
    COLORS = {
        'red': torch.tensor([1.0, 0.0, 0.0]),
        'green': torch.tensor([0.0, 1.0, 0.0])
    }
    
    def __init__(self, split='env1', correlation=0.8, root='./data'):
        self.mnist = torchvision.datasets.MNIST(root, train=(split != 'test'))
        self.split = split
        self.correlation = correlation
        self._assign_colors()
    
    def _assign_colors(self):
        """Assign colors based on correlation strength."""
        labels = self.mnist.targets
        label_groups = (labels >= 5).long()  # 0-4 vs 5-9
        
        # Correlation determines how often color matches group
        n_samples = len(labels)
        flip_mask = torch.rand(n_samples) > self.correlation
        
        # OOD test flips the correlation
        if self.split == 'test':
            self.colors = 1 - label_groups
        else:
            self.colors = label_groups.clone()
            self.colors[flip_mask] = 1 - self.colors[flip_mask]
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        color_idx = self.colors[idx].item()
        
        # Convert grayscale to RGB with color
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        rgb_image = image_tensor.unsqueeze(0) * self.COLORS[
            'red' if color_idx == 0 else 'green'
        ].view(3, 1, 1)
        
        return {
            'image': rgb_image,
            'label': label,
            'color': torch.tensor(color_idx, dtype=torch.long),
            'environment': 0 if self.split == 'env1' else 1
        }
    
    def __len__(self):
        return len(self.mnist)


def create_colored_mnist_loaders(batch_size=128, correlation_env1=0.8, 
                                  correlation_env2=0.9, num_workers=4):
    """Create train/val/test dataloaders for colored MNIST experiment."""
    train_env1 = ColoredMNIST('env1', correlation_env1)
    train_env2 = ColoredMNIST('env2', correlation_env2)
    test_ood = ColoredMNIST('test', correlation_env2)
    
    # Combine training environments
    from torch.utils.data import ConcatDataset
    train_combined = ConcatDataset([train_env1, train_env2])
    
    loaders = {
        'train': torch.utils.data.DataLoader(
            train_combined, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        'test_ood': torch.utils.data.DataLoader(
            test_ood, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    }
    
    return loaders
```

#### Tests: `tests/test_data/test_colored_mnist.py`
```python
"""Tests for ColoredMNIST dataset."""

def test_colored_mnist_shapes():
    """Check output shapes are correct."""
    dataset = ColoredMNIST('env1', correlation=0.8)
    sample = dataset[0]
    
    assert sample['image'].shape == (3, 28, 28)
    assert sample['label'].shape == ()
    assert sample['color'].shape == ()
    assert sample['environment'] in [0, 1]

def test_correlation_strength():
    """Verify correlation approximately matches specified value."""
    dataset = ColoredMNIST('env1', correlation=0.8)
    
    matches = 0
    total = len(dataset)
    
    for idx in range(min(1000, total)):
        sample = dataset[idx]
        label_group = (sample['label'].item() >= 5)
        color = sample['color'].item()
        
        if (label_group and color == 1) or (not label_group and color == 0):
            matches += 1
    
    observed_corr = matches / min(1000, total)
    assert 0.75 < observed_corr < 0.85, f"Expected ~0.8, got {observed_corr}"

def test_ood_test_flipped():
    """OOD test set should have flipped correlation."""
    train_dataset = ColoredMNIST('env1', correlation=0.8)
    test_dataset = ColoredMNIST('test', correlation=0.8)
    
    # Check that test colors are opposite of what training would expect
    # (This is qualitative - exact verification requires sampling)
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
```

**Deliverable**: Working ColoredMNIST dataset with tests passing.

---

### Day 3-4: Experiment Script

#### File: `experiments/colored_mnist_ood.py`
```python
"""Colored MNIST OOD Generalization Experiment.

Tests whether causality component (HSIC + IRM) enables learning
digit classification invariant to spurious color feature.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
from pathlib import Path

from aps.data.colored_mnist import create_colored_mnist_loaders
from aps.models import APSAutoencoder, APSConfig
from aps.causality import HSICLoss, IRMLoss
from aps.training import Trainer, TrainingConfig
from aps.metrics import compute_accuracy


def train_baseline(loaders, device='cpu', epochs=20):
    """Train baseline model without causality regularization."""
    # Simple classifier on raw pixels
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in loaders['train']:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def train_aps_causal(loaders, device='cpu', epochs=20, lambda_C=1.0):
    """Train APS model with causality regularization (HSIC + IRM)."""
    config = APSConfig(
        in_dim=3 * 28 * 28,
        latent_dim=32,
        hidden_dims=[128, 64],
        lambda_T=0.0,  # Focus on causality for this experiment
        lambda_C=lambda_C,
        lambda_E=0.0
    )
    
    model = APSAutoencoder(config).to(device)
    
    # Causality losses
    hsic_loss = HSICLoss(sigma=1.0)
    irm_loss = IRMLoss()
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for batch in loaders['train']:
            images = batch['image'].to(device).flatten(1)
            labels = batch['label'].to(device)
            colors = batch['color'].to(device).float().unsqueeze(1)
            envs = batch['environment'].to(device)
            
            # Forward pass
            z = model.encode(images)
            x_recon = model.decode(z)
            
            # Task loss (reconstruction)
            recon_loss = nn.functional.mse_loss(x_recon, images)
            
            # Causality losses
            hsic = hsic_loss(z, colors)  # Independence from color
            irm_penalty = irm_loss(z, labels, envs)  # Environment invariance
            
            # Total loss
            loss = recon_loss + lambda_C * (hsic + irm_penalty)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_ood(model, loader, device='cpu', is_aps=False):
    """Evaluate accuracy on OOD test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if is_aps:
                images_flat = images.flatten(1)
                z = model.encode(images_flat)
                # Use simple linear classifier on latent
                logits = nn.Linear(z.size(1), 10).to(device)(z)
            else:
                logits = model(images)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def main(args):
    """Run colored MNIST OOD experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    loaders = create_colored_mnist_loaders(
        batch_size=128,
        correlation_env1=0.8,
        correlation_env2=0.9
    )
    
    print("="*60)
    print("COLORED MNIST OOD GENERALIZATION EXPERIMENT")
    print("="*60)
    
    # Train baseline
    print("\n[1/2] Training Baseline Model (no causality)...")
    model_baseline = train_baseline(loaders, device, epochs=args.epochs)
    acc_baseline = evaluate_ood(model_baseline, loaders['test_ood'], device, is_aps=False)
    print(f"✓ Baseline OOD Accuracy: {acc_baseline:.4f}")
    
    # Train APS with causality
    print("\n[2/2] Training APS Model (with causality: HSIC + IRM)...")
    model_aps = train_aps_causal(loaders, device, epochs=args.epochs, lambda_C=args.lambda_C)
    acc_aps = evaluate_ood(model_aps, loaders['test_ood'], device, is_aps=True)
    print(f"✓ APS OOD Accuracy: {acc_aps:.4f}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline OOD Accuracy:  {acc_baseline:.4f}")
    print(f"APS OOD Accuracy:       {acc_aps:.4f}")
    print(f"Improvement (Δ):        {acc_aps - acc_baseline:.4f} ({(acc_aps/acc_baseline - 1)*100:.1f}%)")
    
    # Success criteria
    success = (acc_aps > 0.85) and (acc_baseline < 0.60) and ((acc_aps - acc_baseline) > 0.25)
    if success:
        print("\n✅ SUCCESS: Causality component demonstrates OOD generalization!")
    else:
        print("\n❌ FAIL: Did not meet success criteria (APS > 85%, Baseline < 60%, Δ > 25%)")
    
    # Save results
    results_dir = Path('outputs/colored_mnist_ood')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'baseline_ood_accuracy': acc_baseline,
        'aps_ood_accuracy': acc_aps,
        'improvement': acc_aps - acc_baseline,
        'success': success
    }
    
    import json
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lambda-C', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    main(args)
```

**Deliverable**: Colored MNIST experiment script that validates causality claims.

---

### Day 5: Analysis & Documentation

#### Notebook: `notebooks/colored_mnist_analysis.ipynb`
- Visualizations:
  - Latent space embeddings colored by digit label
  - Latent space embeddings colored by spurious color feature
  - Demonstration that APS separates by digit, not color
- Quantitative analysis:
  - Accuracy vs epoch (train/test)
  - HSIC(Z, color) over training (should decrease)
  - Ablation: HSIC-only vs IRM-only vs both

#### Documentation: `specs/006-critical-experiments/6a-colored-mnist-results.md`
- Summarize findings
- Include key figures
- Compare to paper claims
- Prepare text for Section 5.2 of paper

**Deliverable**: Complete analysis ready for paper inclusion.

---

## 📊 Sub-Phase 6B: Text Domain OOD Implementation

### Day 1-3: Dataset Preparation

#### File: `src/aps/data/wilds_amazon.py`
```python
"""WILDS Amazon Reviews dataset loader for domain shift experiments."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from pathlib import Path


class WILDSAmazon(Dataset):
    """
    Amazon product reviews with domain shift.
    
    Domains (product categories):
        - Training: Books, Electronics, Movies
        - Test (OOD): Clothing
    
    Task: Binary sentiment classification (positive/negative)
    """
    
    def __init__(self, domains, split='train', root='./data/wilds_amazon', 
                 use_embeddings=True, bert_model='bert-base-uncased'):
        self.domains = domains
        self.split = split
        self.root = Path(root)
        self.use_embeddings = use_embeddings
        
        # Load data
        self.data = self._load_data()
        
        # Precompute BERT embeddings if requested
        if use_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.bert = AutoModel.from_pretrained(bert_model)
            self.embeddings = self._precompute_embeddings()
    
    def _load_data(self):
        """Load Amazon reviews for specified domains."""
        data_frames = []
        
        for domain in self.domains:
            file_path = self.root / f"{domain}_{self.split}.csv"
            df = pd.read_csv(file_path)
            df['domain'] = domain
            data_frames.append(df)
        
        return pd.concat(data_frames, ignore_index=True)
    
    def _precompute_embeddings(self):
        """Precompute BERT [CLS] embeddings for all reviews."""
        embeddings = []
        
        self.bert.eval()
        with torch.no_grad():
            for idx in range(len(self.data)):
                text = self.data.iloc[idx]['review_text']
                inputs = self.tokenizer(text, return_tensors='pt', 
                                       truncation=True, max_length=128)
                outputs = self.bert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings.append(embedding.squeeze(0))
        
        return torch.stack(embeddings)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        return {
            'embedding': self.embeddings[idx] if self.use_embeddings else row['review_text'],
            'label': torch.tensor(row['sentiment'], dtype=torch.long),  # 0=neg, 1=pos
            'domain': row['domain'],
            'domain_id': self.domains.index(row['domain'])
        }
    
    def __len__(self):
        return len(self.data)


def create_wilds_amazon_loaders(batch_size=64, num_workers=4):
    """Create train/val/test loaders for domain shift experiment."""
    train_domains = ['books', 'electronics', 'movies']
    test_domain = ['clothing']
    
    train_dataset = WILDSAmazon(train_domains, split='train')
    test_ood_dataset = WILDSAmazon(test_domain, split='test')
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, 
                           shuffle=True, num_workers=num_workers),
        'test_ood': DataLoader(test_ood_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    }
    
    return loaders
```

**Note**: If WILDS Amazon is not available, use AG News with multi-domain splits or create synthetic domain shift from existing datasets.

---

### Day 4-7: Experiment Script

#### File: `experiments/text_domain_ood.py`
```python
"""Text Domain OOD Generalization Experiment.

Tests APS on high-dimensional text data with domain shift.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import argparse

from aps.data.wilds_amazon import create_wilds_amazon_loaders
from aps.models import APSAutoencoder, APSConfig
from aps.causality import HSICLoss
from aps.topology import KNNTopoLoss
from aps.training import Trainer, TrainingConfig


def train_baseline(loaders, device='cpu', epochs=20):
    """Train baseline sentiment classifier without domain invariance."""
    model = nn.Sequential(
        nn.Linear(768, 256),  # BERT embeddings are 768-dim
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)  # Binary sentiment
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in loaders['train']:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def train_aps_causal(loaders, device='cpu', epochs=20, lambda_T=1.0, lambda_C=1.0):
    """Train APS with domain-invariant causality."""
    config = APSConfig(
        in_dim=768,
        latent_dim=32,
        hidden_dims=[256, 128],
        lambda_T=lambda_T,
        lambda_C=lambda_C,
        lambda_E=0.0
    )
    
    model = APSAutoencoder(config).to(device)
    
    # Losses
    topo_loss = KNNTopoLoss(k=15)
    hsic_loss = HSICLoss(sigma=1.0)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for batch in loaders['train']:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            domain_ids = batch['domain_id'].to(device).float().unsqueeze(1)
            
            # Forward
            z = model.encode(embeddings)
            x_recon = model.decode(z)
            
            # Losses
            recon_loss = nn.functional.mse_loss(x_recon, embeddings)
            topo = topo_loss(embeddings, z)
            hsic = hsic_loss(z, domain_ids)  # Independence from domain
            
            loss = recon_loss + lambda_T * topo + lambda_C * hsic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_sentiment_ood(model, loader, device='cpu', is_aps=False):
    """Evaluate sentiment classification on OOD domain."""
    model.eval()
    correct = 0
    total = 0
    
    # Need sentiment classifier head
    classifier = nn.Linear(32 if is_aps else 768, 2).to(device)
    
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            if is_aps:
                z = model.encode(embeddings)
                logits = classifier(z)
            else:
                logits = model(embeddings)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def main(args):
    """Run text domain OOD experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loaders = create_wilds_amazon_loaders(batch_size=64)
    
    print("="*60)
    print("TEXT DOMAIN OOD GENERALIZATION EXPERIMENT")
    print("Training domains: Books, Electronics, Movies")
    print("Test domain (OOD): Clothing")
    print("="*60)
    
    # Baseline
    print("\n[1/2] Training Baseline...")
    model_baseline = train_baseline(loaders, device, args.epochs)
    acc_baseline = evaluate_sentiment_ood(model_baseline, loaders['test_ood'], device, is_aps=False)
    print(f"✓ Baseline OOD Accuracy: {acc_baseline:.4f}")
    
    # APS
    print("\n[2/2] Training APS (T+C: Topology + Domain-Invariant Causality)...")
    model_aps = train_aps_causal(loaders, device, args.epochs, lambda_T=1.0, lambda_C=1.0)
    acc_aps = evaluate_sentiment_ood(model_aps, loaders['test_ood'], device, is_aps=True)
    print(f"✓ APS OOD Accuracy: {acc_aps:.4f}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline OOD Accuracy:  {acc_baseline:.4f}")
    print(f"APS OOD Accuracy:       {acc_aps:.4f}")
    print(f"Improvement (Δ):        {acc_aps - acc_baseline:.4f}")
    
    success = (acc_aps > 0.70) and (acc_baseline < 0.60) and ((acc_aps - acc_baseline) > 0.10)
    if success:
        print("\n✅ SUCCESS: APS generalizes to OOD text domain!")
    else:
        print("\n❌ Targets not met (APS > 70%, Baseline < 60%, Δ > 10%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)
```

---

### Day 8-10: Analysis & Documentation

Similar structure to 6A, but focused on:
- High-dimensional latent space analysis
- HSIC independence verification
- Ablation: T-only vs C-only vs T+C

**Deliverable**: Text OOD experiment ready for paper Section 5.3.

---

## 📈 Sub-Phase 6C: T-C Trade-off Analysis

### Day 1-2: Synthetic Dataset

#### File: `src/aps/data/spurious_shapes.py`
```python
"""Synthetic shapes dataset with spurious color features."""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw


def generate_circle(size=28, color='red'):
    """Generate circle image with colored background."""
    img = Image.new('RGB', (size, size), color=color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size-4, size-4], fill='white', outline='black')
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0


def generate_square(size=28, color='blue'):
    """Generate square image with colored background."""
    img = Image.new('RGB', (size, size), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, size-4, size-4], fill='white', outline='black')
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0


class SpuriousShapesDataset(Dataset):
    """
    Shapes with spurious color correlation.
    
    Training: Circles always red, Squares always blue
    Test: Circles blue, Squares red (flipped)
    """
    
    def __init__(self, n_samples=1000, split='train', size=28):
        self.n_samples = n_samples
        self.split = split
        self.size = size
        self._generate_data()
    
    def _generate_data(self):
        self.images = []
        self.labels = []  # 0=circle, 1=square
        self.colors = []  # 0=red, 1=blue
        
        for _ in range(self.n_samples // 2):
            if self.split == 'train':
                # Circles are red
                self.images.append(generate_circle(self.size, color='red'))
                self.labels.append(0)
                self.colors.append(0)
            else:
                # Test: circles are blue (flipped)
                self.images.append(generate_circle(self.size, color='blue'))
                self.labels.append(0)
                self.colors.append(1)
        
        for _ in range(self.n_samples // 2):
            if self.split == 'train':
                # Squares are blue
                self.images.append(generate_square(self.size, color='blue'))
                self.labels.append(1)
                self.colors.append(1)
            else:
                # Test: squares are red (flipped)
                self.images.append(generate_square(self.size, color='red'))
                self.labels.append(1)
                self.colors.append(0)
        
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.colors = torch.tensor(self.colors, dtype=torch.long)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'color': self.colors[idx]
        }
    
    def __len__(self):
        return self.n_samples
```

---

### Day 3-4: Hyperparameter Sweep

#### File: `experiments/tc_tradeoff_sweep.py`
```python
"""T-C Trade-off Analysis: Pareto Frontier Experiment."""

import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
import numpy as np
from pathlib import Path

from aps.data.spurious_shapes import SpuriousShapesDataset
from aps.models import APSAutoencoder, APSConfig
from aps.topology import KNNTopoLoss
from aps.causality import HSICLoss
from aps.metrics import knn_preservation


def train_aps_config(train_loader, lambda_T, lambda_C, device='cpu', epochs=50):
    """Train APS with specific λ_T and λ_C values."""
    config = APSConfig(
        in_dim=3 * 28 * 28,
        latent_dim=8,
        hidden_dims=[128, 64],
        lambda_T=lambda_T,
        lambda_C=lambda_C,
        lambda_E=0.0
    )
    
    model = APSAutoencoder(config).to(device)
    
    topo_loss = KNNTopoLoss(k=10)
    hsic_loss = HSICLoss(sigma=1.0)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device).flatten(1)
            colors = batch['color'].to(device).float().unsqueeze(1)
            
            z = model.encode(images)
            x_recon = model.decode(z)
            
            recon_loss = nn.functional.mse_loss(x_recon, images)
            topo = topo_loss(images, z)
            hsic = hsic_loss(z, colors)
            
            loss = recon_loss + lambda_T * topo + lambda_C * hsic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_metrics(model, test_loader, device='cpu'):
    """Compute topological preservation and causal accuracy."""
    model.eval()
    
    all_images = []
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device).flatten(1)
            labels = batch['label'].to(device)
            
            z = model.encode(images)
            
            all_images.append(images.cpu())
            all_latents.append(z.cpu())
            all_labels.append(labels.cpu())
    
    images_all = torch.cat(all_images)
    latents_all = torch.cat(all_latents)
    labels_all = torch.cat(all_labels)
    
    # Topology preservation (k-NN preservation score)
    topo_score = knn_preservation(images_all, latents_all, k=10)
    
    # Causal accuracy (linear probe on latent for shape classification)
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(latents_all.numpy(), labels_all.numpy())
    causal_acc = classifier.score(latents_all.numpy(), labels_all.numpy())
    
    return topo_score, causal_acc


def main(args):
    """Run hyperparameter sweep for T-C trade-off analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    train_dataset = SpuriousShapesDataset(n_samples=1000, split='train')
    test_dataset = SpuriousShapesDataset(n_samples=200, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Hyperparameter grid
    lambda_T_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    lambda_C_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    print("="*60)
    print("T-C TRADE-OFF ANALYSIS: Pareto Frontier")
    print(f"Configurations: {len(lambda_T_values) * len(lambda_C_values)}")
    print("="*60)
    
    for lambda_T in lambda_T_values:
        for lambda_C in lambda_C_values:
            print(f"\nTraining λ_T={lambda_T:.1f}, λ_C={lambda_C:.1f}...")
            
            model = train_aps_config(train_loader, lambda_T, lambda_C, device, epochs=args.epochs)
            topo_score, causal_acc = evaluate_metrics(model, test_loader, device)
            
            results.append({
                'lambda_T': lambda_T,
                'lambda_C': lambda_C,
                'topo_score': topo_score,
                'causal_acc': causal_acc
            })
            
            print(f"  ✓ Topo Preservation: {topo_score:.4f}, Causal Accuracy: {causal_acc:.4f}")
    
    # Save results
    results_dir = Path('outputs/tc_tradeoff')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(results_dir / 'pareto_frontier_data.csv', index=False)
    
    print(f"\nResults saved to {results_dir / 'pareto_frontier_data.csv'}")
    print("Run `notebooks/pareto_frontier_analysis.ipynb` to visualize.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
```

---

### Day 5: Visualization & Analysis

#### Notebook: `notebooks/pareto_frontier_analysis.ipynb`
```python
"""Pareto Frontier Visualization."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('outputs/tc_tradeoff/pareto_frontier_data.csv')

# 2D scatter: topo vs causal
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    df['topo_score'],
    df['causal_acc'],
    c=df['lambda_T'] + df['lambda_C'],  # Color by total regularization
    s=100,
    alpha=0.7,
    cmap='viridis'
)

ax.set_xlabel('Topological Preservation (k-NN score)', fontsize=12)
ax.set_ylabel('Causal Accuracy (shape classification)', fontsize=12)
ax.set_title('T-C Trade-off: Pareto Frontier', fontsize=14)
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, label='λ_T + λ_C (total regularization)')

# Annotate extreme points
best_topo = df.loc[df['topo_score'].idxmax()]
best_causal = df.loc[df['causal_acc'].idxmax()]

ax.annotate(f"Best Topo\n(λ_T={best_topo['lambda_T']:.1f}, λ_C={best_topo['lambda_C']:.1f})",
            xy=(best_topo['topo_score'], best_topo['causal_acc']),
            xytext=(10, -20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'))

ax.annotate(f"Best Causal\n(λ_T={best_causal['lambda_T']:.1f}, λ_C={best_causal['lambda_C']:.1f})",
            xy=(best_causal['topo_score'], best_causal['causal_acc']),
            xytext=(-80, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('outputs/tc_tradeoff/pareto_frontier.png', dpi=300)
plt.show()

print("Pareto frontier plot saved!")
```

**Deliverable**: Pareto frontier visualization ready for paper Figure 5.4.

---

## ✅ Phase 006 Success Criteria

### 6A: Colored MNIST
- [ ] APS OOD accuracy > 85%
- [ ] Baseline OOD accuracy < 60%
- [ ] Δ Accuracy > 25%
- [ ] HSIC(Z, color) < 0.1 after training
- [ ] Visualizations show digit-based clustering, not color-based

### 6B: Text Domain OOD
- [ ] APS OOD accuracy > 70%
- [ ] Baseline OOD accuracy < 60%
- [ ] Δ Accuracy > 10%
- [ ] HSIC(Z, domain_id) < 0.1 after training
- [ ] Latent space preserves sentiment structure across domains

### 6C: T-C Trade-off
- [ ] 36 configurations trained successfully
- [ ] Clear Pareto frontier identified
- [ ] Guidance provided for λ_T, λ_C selection based on task
- [ ] Figure suitable for paper inclusion

---

## 📦 Deliverables Summary

### Code
- `src/aps/data/colored_mnist.py` ✅
- `src/aps/data/wilds_amazon.py` ✅
- `src/aps/data/spurious_shapes.py` ✅
- `experiments/colored_mnist_ood.py` ✅
- `experiments/text_domain_ood.py` ✅
- `experiments/tc_tradeoff_sweep.py` ✅

### Documentation
- `specs/006-critical-experiments/6a-colored-mnist-results.md`
- `specs/006-critical-experiments/6b-text-ood-results.md`
- `specs/006-critical-experiments/6c-tc-tradeoff-results.md`

### Notebooks
- `notebooks/colored_mnist_analysis.ipynb`
- `notebooks/text_ood_analysis.ipynb`
- `notebooks/pareto_frontier_analysis.ipynb`

### Tests
- `tests/test_data/test_colored_mnist.py` (5 tests)
- `tests/test_data/test_wilds_amazon.py` (5 tests)
- `tests/test_data/test_spurious_shapes.py` (3 tests)

### Paper Materials
- Section 5.2: Colored MNIST OOD (1-2 pages, 2 figures)
- Section 5.3: Text Domain OOD (1-2 pages, 2 figures)
- Section 5.4: T-C Trade-off Analysis (1 page, 1 figure)
- Updated Section 6: Discussion incorporating new findings

---

## 🚀 Getting Started

```bash
# Create branch
git checkout -b 006-critical-experiments

# Install dependencies
pip install transformers datasets wilds scikit-learn

# Start with 6A
cd specs/006-critical-experiments
cat 6a-colored-mnist-plan.md

# Implement dataset
code src/aps/data/colored_mnist.py

# Write tests first (TDD)
code tests/test_data/test_colored_mnist.py
pytest tests/test_data/test_colored_mnist.py

# Implement experiment
code experiments/colored_mnist_ood.py
python experiments/colored_mnist_ood.py --epochs 20

# Analyze results
jupyter notebook notebooks/colored_mnist_analysis.ipynb
```

---

**Last Updated**: 2025-01-30  
**Phase**: 006-critical-experiments  
**Timeline**: 4 weeks (parallel work possible)  
**Priority**: CRITICAL for paper submission
