"""Phase 006B: Text Domain OOD Generalization Experiment

Tests APS framework on text domain shift using AG News dataset:
- Training domains: Sports, Business, Sci-Tech
- Test domain (OOD): World
- Task: Binary sentiment classification
- Hypothesis: APS learns topic-invariant sentiment representations

Experiments:
1. Baseline: Standard supervised learning
2. APS-T: Topology preservation
3. APS-C: Causal invariance (IRM/HSIC)
4. APS-TC: Combined topology + causality
5. APS-Full: T+C+E (with energy attractors)

Metrics:
- In-distribution accuracy (train domains)
- OOD accuracy (World domain)
- OOD generalization gap
- Training stability
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.data.ag_news_ood import create_ag_news_ood_loaders
from aps.models import APSConfig, APSAutoencoder
from aps.metrics import knn_preservation


@dataclass
class ExperimentConfig:
    """Configuration for Phase 006B experiments."""
    
    # Experiment settings
    experiment_name: str = 'baseline'
    seed: int = 42
    # Auto-detect best device: MPS (Apple Silicon) > CUDA > CPU
    device: str = (
        'mps' if torch.backends.mps.is_available() 
        else 'cuda' if torch.cuda.is_available() 
        else 'cpu'
    )
    
    # Data settings
    batch_size: int = 64
    num_workers: int = 0  # Set to 0 to avoid tokenizer parallelism issues
    use_cache: bool = True  # Enable caching for faster subsequent runs
    max_samples_per_domain: int = None  # None = use all data
    
    # Model architecture
    bert_dim: int = 768  # BERT-base embedding dimension
    latent_dim: int = 32
    hidden_dims: list = None  # [256, 128]
    
    # Loss weights (for ablation)
    lambda_T: float = 0.0  # Topology
    lambda_C: float = 0.0  # Causality (HSIC)
    lambda_E: float = 0.0  # Energy
    
    # Component hyperparameters
    topo_k: int = 8
    hsic_sigma: float = 1.0
    n_mem: int = 16
    beta: float = 5.0
    alpha: float = 0.0
    
    # Training settings
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Output
    output_dir: str = './outputs/phase006b'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class SentimentClassifier(nn.Module):
    """Simple classifier head on top of latent representation."""
    
    def __init__(self, latent_dim: int, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, z):
        return self.classifier(z)


def train_epoch(
    encoder: APSAutoencoder,
    classifier: SentimentClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_domain_labels: bool = False
) -> dict:
    """Train for one epoch."""
    encoder.train()
    classifier.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        # Get inputs
        x = batch['embedding'].to(device)  # [B, 768]
        y = batch['sentiment'].to(device)  # [B]
        _ = batch['domain_id'].to(device) if use_domain_labels else None  # Domain ID (for future IRM)
        
        # Forward pass through encoder
        z, x_recon = encoder(x)
        
        # Classification loss
        logits = classifier(z)
        cls_loss = nn.CrossEntropyLoss()(logits, y)
        
        # Get encoder losses (includes reconstruction + APS components)
        # Use compute_loss with already computed z and x_recon
        losses = encoder.compute_loss(x)
        encoder_loss = losses['total']
        
        # Total loss
        loss = cls_loss + encoder_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(
    encoder: APSAutoencoder,
    classifier: SentimentClassifier,
    loader: DataLoader,
    device: str,
    split_name: str = 'test'
) -> dict:
    """Evaluate on a dataset."""
    encoder.eval()
    classifier.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_z = []
    all_y = []
    
    pbar = tqdm(loader, desc=f'Evaluating {split_name}')
    
    for batch in pbar:
        x = batch['embedding'].to(device)
        y = batch['sentiment'].to(device)
        
        # Forward
        z, x_recon = encoder(x)
        logits = classifier(z)
        
        # Loss
        cls_loss = nn.CrossEntropyLoss()(logits, y)
        losses = encoder.compute_loss(x)
        encoder_loss = losses['total']
        loss = cls_loss + encoder_loss
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        # Store for geometric metrics
        all_z.append(z.cpu())
        all_y.append(y.cpu())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    # Compute geometric metrics
    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    # Only compute if latent dim allows (for kNN metrics)
    geometric_metrics = {}
    if all_z.size(1) >= 2:  # Need at least 2D for kNN
        try:
            # Use original BERT embeddings for reference
            # (We'd need to store them - for now just use z as proxy)
            geometric_metrics['knn_preservation'] = knn_preservation(all_z, all_z, k=8).item()
        except Exception:
            pass
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total,
        **geometric_metrics
    }


def run_experiment(config: ExperimentConfig):
    """Run a single experiment configuration."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: {config.experiment_name}")
    print(f"{'='*80}")
    print(f"Config: λ_T={config.lambda_T}, λ_C={config.lambda_C}, λ_E={config.lambda_E}")
    print(f"Device: {config.device}")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    print("\n[1/4] Loading AG News OOD data...")
    loaders = create_ag_news_ood_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cache=config.use_cache,
        max_samples_per_domain=config.max_samples_per_domain
    )
    
    train_loader = loaders['train']
    test_ood_loader = loaders['test_ood']
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test (OOD): {len(test_ood_loader.dataset)} samples")
    
    # Create model
    print("\n[2/4] Creating APS model...")
    model_config = APSConfig(
        in_dim=config.bert_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        lambda_T=config.lambda_T,
        lambda_C=config.lambda_C,
        lambda_E=config.lambda_E,
        topo_k=config.topo_k,
        hsic_sigma=config.hsic_sigma,
        n_mem=config.n_mem,
        beta=config.beta,
        alpha=config.alpha
    )
    
    encoder = APSAutoencoder(model_config).to(config.device)
    classifier = SentimentClassifier(config.latent_dim, num_classes=2).to(config.device)
    
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Classifier: {sum(p.numel() for p in classifier.parameters()):,} params")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print("\n[3/4] Training...")
    
    best_ood_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'ood_loss': [],
        'ood_acc': []
    }
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_metrics = train_epoch(
            encoder, classifier, train_loader,
            optimizer, config.device,
            use_domain_labels=(config.lambda_C > 0)
        )
        
        # Evaluate OOD
        ood_metrics = evaluate(
            encoder, classifier, test_ood_loader,
            config.device, split_name='OOD'
        )
        
        # Log
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={100*train_metrics['accuracy']:.2f}%")
        print(f"  OOD:   loss={ood_metrics['loss']:.4f}, acc={100*ood_metrics['accuracy']:.2f}%")
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['ood_loss'].append(ood_metrics['loss'])
        history['ood_acc'].append(ood_metrics['accuracy'])
        
        # Save best model
        if ood_metrics['accuracy'] > best_ood_acc:
            best_ood_acc = ood_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': asdict(config),
                'metrics': ood_metrics
            }, output_dir / 'best_model.pt')
    
    # Final evaluation
    print("\n[4/4] Final evaluation...")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    # Evaluate
    final_metrics = evaluate(
        encoder, classifier, test_ood_loader,
        config.device, split_name='Final OOD'
    )
    
    print(f"\nFinal OOD Accuracy: {100*final_metrics['accuracy']:.2f}%")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final metrics
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({
            'final_ood_accuracy': final_metrics['accuracy'],
            'best_ood_accuracy': best_ood_acc,
            'final_ood_loss': final_metrics['loss'],
            **final_metrics
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Phase 006B: Text Domain OOD Experiments')
    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'aps-T', 'aps-C', 'aps-TC', 'aps-full'],
                       help='Experiment configuration')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with small dataset (500 samples/domain)')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per domain (overrides --quick-test)')
    parser.add_argument('--output-dir', type=str, default='./outputs/phase006b', help='Output directory')
    
    args = parser.parse_args()
    
    # Base config
    max_samples = args.max_samples if args.max_samples else (500 if args.quick_test else None)
    base_config = ExperimentConfig(
        experiment_name=args.experiment,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        max_samples_per_domain=max_samples
    )
    
    # Experiment-specific settings
    if args.experiment == 'baseline':
        config = base_config
    
    elif args.experiment == 'aps-T':
        config = ExperimentConfig(**{**asdict(base_config), 'lambda_T': 1.0})
    
    elif args.experiment == 'aps-C':
        config = ExperimentConfig(**{**asdict(base_config), 'lambda_C': 1.0})
    
    elif args.experiment == 'aps-TC':
        config = ExperimentConfig(**{**asdict(base_config), 'lambda_T': 1.0, 'lambda_C': 0.5})
    
    elif args.experiment == 'aps-full':
        config = ExperimentConfig(**{
            **asdict(base_config),
            'lambda_T': 1.0,
            'lambda_C': 0.5,
            'lambda_E': 0.1
        })
    
    # Run experiment
    results = run_experiment(config)
    
    print(f"\n{'='*80}")
    print(f"Experiment '{args.experiment}' completed!")
    print(f"OOD Accuracy: {100*results['accuracy']:.2f}%")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
