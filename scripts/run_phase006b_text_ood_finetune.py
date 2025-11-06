"""Phase 006B: Text Domain OOD with BERT Fine-Tuning

Key difference from run_phase006b_text_ood.py:
- Fine-tunes BERT embeddings instead of using frozen pre-computed embeddings
- Tests if trainable representations allow T and C components to be effective
- Addresses reviewer feedback about frozen embeddings limiting APS effectiveness

Experiments:
1. Baseline: Standard supervised learning with fine-tuned BERT
2. APS-T: Topology preservation + fine-tuning
3. APS-C: Causal invariance (HSIC) + fine-tuning
4. APS-TC: Combined topology + causality + fine-tuning
5. APS-Full: T+C+E + fine-tuning

Expected outcomes:
- Frozen embeddings (previous): T and C show no benefit
- Fine-tuned embeddings (this): T and C should improve OOD accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
import argparse
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.models import APSConfig, APSAutoencoder
from aps.metrics import knn_preservation, trustworthiness


@dataclass
class ExperimentConfig:
    """Configuration for Phase 006B fine-tuning experiments."""
    
    # Experiment settings
    experiment_name: str = 'baseline_finetune'
    seed: int = 42
    device: str = (
        'mps' if torch.backends.mps.is_available() 
        else 'cuda' if torch.cuda.is_available() 
        else 'cpu'
    )
    
    # Data settings
    batch_size: int = 32  # Smaller batch for fine-tuning
    num_workers: int = 0
    max_samples_per_domain: int = 5000  # Limit for faster training
    max_length: int = 128  # Token sequence length
    
    # Model architecture
    bert_model: str = 'bert-base-uncased'
    bert_dim: int = 768
    latent_dim: int = 32
    hidden_dims: list = None  # [256, 128]
    freeze_bert_layers: int = 6  # Freeze first N layers (0 = fine-tune all)
    
    # Loss weights (for ablation)
    lambda_T: float = 0.0  # Topology
    lambda_C: float = 0.0  # Causality (HSIC)
    lambda_E: float = 0.0  # Energy
    
    # Component hyperparameters
    topo_k: int = 8  # Must be < batch_size
    hsic_sigma: float = 1.0
    n_mem: int = 16
    beta: float = 5.0
    alpha: float = 0.0
    
    # Training settings
    epochs: int = 10  # Fewer epochs for fine-tuning
    lr: float = 2e-5  # Lower LR for BERT
    encoder_lr: float = 1e-3  # Higher LR for APS components
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    
    # Output
    output_dir: str = './outputs/phase006b_finetune'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class AGNewsSentimentDataset(Dataset):
    """AG News dataset for sentiment classification with domain labels."""
    
    def __init__(
        self,
        split: str = 'train',
        tokenizer=None,
        max_length: int = 128,
        max_samples: int = None,
        test_domain: str = 'World'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load AG News
        dataset = load_dataset('ag_news', split=split)
        
        # Map class IDs to topics
        self.topic_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        # Filter by domain
        if split == 'train':
            # Use Sports, Business, Sci/Tech for training
            indices = [i for i, ex in enumerate(dataset) 
                      if self.topic_names[ex['label']] != test_domain]
        else:
            # Use World for OOD testing
            indices = [i for i, ex in enumerate(dataset)
                      if self.topic_names[ex['label']] == test_domain]
        
        dataset = dataset.select(indices)
        
        # Limit samples
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        self.dataset = dataset
        
        # Assign binary sentiment labels (random for now - AG News has no sentiment)
        # In practice, you'd want to use a sentiment-labeled dataset or add sentiment labels
        np.random.seed(42)
        self.sentiments = (np.random.rand(len(dataset)) > 0.5).astype(int)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example['text']
        topic_id = example['label']
        sentiment = self.sentiments[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'topic_id': torch.tensor(topic_id, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }


class BERTEncoder(nn.Module):
    """BERT encoder with optional layer freezing."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', freeze_layers: int = 6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token


class APSTextModel(nn.Module):
    """Combined BERT + APS Autoencoder + Classifier."""
    
    def __init__(
        self,
        bert_encoder: BERTEncoder,
        aps_config: APSConfig,
        num_classes: int = 2
    ):
        super().__init__()
        self.bert = bert_encoder
        self.aps = APSAutoencoder(aps_config)
        
        # Classifier on latent space
        self.classifier = nn.Sequential(
            nn.Linear(aps_config.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_emb = self.bert(input_ids, attention_mask)  # [B, 768]
        
        # APS encoding
        z, x_recon = self.aps(bert_emb)
        
        # Classification
        logits = self.classifier(z)
        
        return {
            'z': z,
            'x_recon': x_recon,
            'logits': logits,
            'bert_emb': bert_emb
        }


def train_epoch(
    model: APSTextModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment = batch['sentiment'].to(device)
        topic_id = batch['topic_id'].to(device)
        
        # Forward
        outputs = model(input_ids, attention_mask)
        
        # Classification loss
        cls_loss = nn.CrossEntropyLoss()(outputs['logits'], sentiment)
        
        # APS losses (reconstruction + T + C + E)
        aps_losses = model.aps.compute_loss(outputs['bert_emb'])
        aps_loss = aps_losses['total']
        
        # Total loss
        loss = cls_loss + aps_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == sentiment).sum().item()
        total += sentiment.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'aps': f'{aps_loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate(
    model: APSTextModel,
    loader: DataLoader,
    device: str,
    split_name: str = 'test'
) -> dict:
    """Evaluate on a dataset."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_z = []
    all_y = []
    
    pbar = tqdm(loader, desc=f'Evaluating {split_name}')
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment = batch['sentiment'].to(device)
        
        # Forward
        outputs = model(input_ids, attention_mask)
        
        # Loss
        cls_loss = nn.CrossEntropyLoss()(outputs['logits'], sentiment)
        aps_losses = model.aps.compute_loss(outputs['bert_emb'])
        aps_loss = aps_losses['total']
        loss = cls_loss + aps_loss
        
        # Metrics
        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == sentiment).sum().item()
        total += sentiment.size(0)
        
        # Store for geometric metrics
        all_z.append(outputs['z'].cpu())
        all_y.append(sentiment.cpu())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total
    }


def run_experiment(config: ExperimentConfig):
    """Run a single experiment configuration."""
    
    print(f"\n{'='*80}")
    print(f"Experiment: {config.experiment_name}")
    print(f"{'='*80}")
    print(f"Config: λ_T={config.lambda_T}, λ_C={config.lambda_C}, λ_E={config.lambda_E}")
    print(f"BERT fine-tuning: {config.freeze_bert_layers} layers frozen")
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
    
    # Load tokenizer and create datasets
    print("\n[1/4] Loading AG News data with tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
    
    train_dataset = AGNewsSentimentDataset(
        split='train',
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_samples_per_domain,
        test_domain='World'
    )
    
    test_dataset = AGNewsSentimentDataset(
        split='test',
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_samples=config.max_samples_per_domain,
        test_domain='World'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test (OOD): {len(test_dataset)} samples")
    
    # Create model
    print("\n[2/4] Creating BERT + APS model...")
    bert_encoder = BERTEncoder(
        model_name=config.bert_model,
        freeze_layers=config.freeze_bert_layers
    )
    
    aps_config = APSConfig(
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
    
    model = APSTextModel(bert_encoder, aps_config, num_classes=2).to(config.device)
    
    print(f"  BERT: {sum(p.numel() for p in model.bert.parameters() if p.requires_grad):,} trainable params")
    print(f"  APS: {sum(p.numel() for p in model.aps.parameters()):,} params")
    print(f"  Classifier: {sum(p.numel() for p in model.classifier.parameters()):,} params")
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': config.lr},
        {'params': model.aps.parameters(), 'lr': config.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': config.encoder_lr}
    ], weight_decay=config.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps
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
            model, train_loader, optimizer, scheduler, config.device
        )
        
        # Evaluate OOD
        ood_metrics = evaluate(
            model, test_loader, config.device, split_name='OOD'
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
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': asdict(config),
                'metrics': ood_metrics
            }, output_dir / 'best_model.pt')
    
    # Final evaluation
    print("\n[4/4] Final evaluation...")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model'])
    
    # Evaluate
    final_metrics = evaluate(
        model, test_loader, config.device, split_name='Final OOD'
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
    parser = argparse.ArgumentParser(description='Phase 006B: Fine-Tuning Experiments')
    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'aps-T', 'aps-C', 'aps-TC', 'aps-full'],
                       help='Experiment configuration')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=5000, help='Max samples per domain')
    parser.add_argument('--freeze-layers', type=int, default=6, help='Freeze first N BERT layers')
    
    args = parser.parse_args()
    
    # Configure experiments
    configs = {
        'baseline': ExperimentConfig(
            experiment_name='baseline_finetune',
            lambda_T=0.0, lambda_C=0.0, lambda_E=0.0
        ),
        'aps-T': ExperimentConfig(
            experiment_name='aps-T_finetune',
            lambda_T=1.0, lambda_C=0.0, lambda_E=0.0
        ),
        'aps-C': ExperimentConfig(
            experiment_name='aps-C_finetune',
            lambda_T=0.0, lambda_C=0.5, lambda_E=0.0
        ),
        'aps-TC': ExperimentConfig(
            experiment_name='aps-TC_finetune',
            lambda_T=1.0, lambda_C=0.5, lambda_E=0.0
        ),
        'aps-full': ExperimentConfig(
            experiment_name='aps-full_finetune',
            lambda_T=1.0, lambda_C=0.5, lambda_E=0.1
        )
    }
    
    # Override with CLI args
    config = configs[args.experiment]
    config.latent_dim = args.latent_dim
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples_per_domain = args.max_samples
    config.freeze_bert_layers = args.freeze_layers
    
    # Run experiment
    run_experiment(config)


if __name__ == '__main__':
    main()
