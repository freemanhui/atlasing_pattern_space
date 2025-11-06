"""Colored MNIST OOD Generalization Experiment.

Tests whether causality component (HSIC + IRM) enables learning
digit classification invariant to spurious color feature.

Success Criteria:
- APS OOD accuracy > 85%
- Baseline OOD accuracy < 60%
- Improvement (Δ) > 25%
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
from pathlib import Path
import json
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aps.data.colored_mnist import create_colored_mnist_loaders
from aps.models import APSAutoencoder, APSConfig
from aps.causality import HSICLoss


class BaselineClassifier(nn.Module):
    """Simple classifier without causality regularization."""
    
    def __init__(self, input_dim=3*28*28, hidden_dim=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


def train_baseline(loaders, device='cpu', epochs=20, lr=1e-3):
    """Train baseline model without causality regularization."""
    print("Training Baseline (no causality)...")
    
    model = BaselineClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.1f}%'
            })
    
    return model


def train_aps_causal(loaders, device='cpu', epochs=20, lambda_C=1.0, lr=1e-3):
    """Train APS model with causality regularization (HSIC independence)."""
    print("Training APS (with causality: HSIC independence from color)...")
    
    # APS configuration
    config = APSConfig(
        in_dim=3 * 28 * 28,
        latent_dim=32,
        hidden_dims=[128, 64],
        lambda_T=0.0,  # Focus on causality for this experiment
        lambda_C=lambda_C,
        lambda_E=0.0
    )
    
    model = APSAutoencoder(config).to(device)
    
    # Causality loss (HSIC independence)
    hsic_loss = HSICLoss(sigma=1.0)
    
    # Classification head (linear probe on latent)
    classifier = nn.Linear(config.latent_dim, 10).to(device)
    
    # Optimizers
    optimizer_ae = Adam(model.parameters(), lr=lr)
    optimizer_clf = Adam(classifier.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device).flatten(1)
            labels = batch['label'].to(device)
            colors = batch['color'].to(device).float().unsqueeze(1)
            _ = batch['environment'].to(device)  # Environment ID (unused in this simple version)
            
            # Forward pass through APS
            z = model.encode(images)
            x_recon = model.decode(z)
            
            # Classification on latent
            logits = classifier(z)
            
            # Task losses
            recon_loss = nn.functional.mse_loss(x_recon, images)
            clf_loss = nn.functional.cross_entropy(logits, labels)
            
            # Causality losses
            hsic = hsic_loss(z, colors)  # Independence from color
            
            # For IRM, we need to separate by environment
            # Since this is per-batch, we'll simplify: just use HSIC for now
            # Full IRM would require environment-specific batches
            
            # Total loss
            loss = recon_loss + clf_loss + lambda_C * hsic
            
            # Backward
            optimizer_ae.zero_grad()
            optimizer_clf.zero_grad()
            loss.backward()
            optimizer_ae.step()
            optimizer_clf.step()
            
            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.1f}%',
                'hsic': f'{hsic.item():.4f}'
            })
    
    return model, classifier


def evaluate_ood(model, loader, device='cpu', is_aps=False, classifier=None):
    """Evaluate accuracy on OOD test set."""
    model.eval()
    if classifier is not None:
        classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if is_aps:
                # APS: encode + classify on latent
                images_flat = images.flatten(1)
                z = model.encode(images_flat)
                logits = classifier(z)
            else:
                # Baseline: direct classification
                logits = model(images)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def compute_hsic_independence(model, loader, device='cpu'):
    """Compute HSIC independence between latent and color."""
    model.eval()
    hsic_loss = HSICLoss(sigma=1.0)
    
    all_z = []
    all_colors = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device).flatten(1)
            colors = batch['color'].to(device).float().unsqueeze(1)
            
            z = model.encode(images)
            all_z.append(z)
            all_colors.append(colors)
    
    z_all = torch.cat(all_z)
    colors_all = torch.cat(all_colors)
    
    hsic_val = hsic_loss(z_all, colors_all)
    return hsic_val.item()


def main(args):
    """Run colored MNIST OOD experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating Colored MNIST dataloaders...")
    loaders = create_colored_mnist_loaders(
        batch_size=args.batch_size,
        correlation_env1=0.8,
        correlation_env2=0.9,
        num_workers=4
    )
    
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Test (OOD): {len(loaders['test_ood'].dataset)} samples")
    
    print("\n" + "="*60)
    print("COLORED MNIST OOD GENERALIZATION EXPERIMENT")
    print("="*60)
    
    # Train baseline
    print("\n[1/2] Training Baseline Model (no causality)...")
    model_baseline = train_baseline(loaders, device, epochs=args.epochs, lr=args.lr)
    acc_baseline = evaluate_ood(model_baseline, loaders['test_ood'], device, is_aps=False)
    print(f"\n✓ Baseline OOD Accuracy: {acc_baseline:.4f} ({acc_baseline*100:.1f}%)")
    
    # Train APS with causality
    print("\n[2/2] Training APS Model (with causality: HSIC independence)...")
    model_aps, classifier_aps = train_aps_causal(
        loaders, device, epochs=args.epochs, lambda_C=args.lambda_C, lr=args.lr
    )
    acc_aps = evaluate_ood(
        model_aps, loaders['test_ood'], device, is_aps=True, classifier=classifier_aps
    )
    print(f"\n✓ APS OOD Accuracy: {acc_aps:.4f} ({acc_aps*100:.1f}%)")
    
    # Check HSIC independence
    print("\nComputing HSIC(Z, color) for independence verification...")
    hsic_val = compute_hsic_independence(model_aps, loaders['test_ood'], device)
    print(f"  HSIC(Z, color) = {hsic_val:.6f}")
    
    # Results
    improvement = acc_aps - acc_baseline
    improvement_pct = (acc_aps / acc_baseline - 1) * 100 if acc_baseline > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline OOD Accuracy:  {acc_baseline:.4f} ({acc_baseline*100:.1f}%)")
    print(f"APS OOD Accuracy:       {acc_aps:.4f} ({acc_aps*100:.1f}%)")
    print(f"Improvement (Δ):        {improvement:.4f} ({improvement*100:.1f}%)")
    print(f"Relative Improvement:   {improvement_pct:.1f}%")
    print(f"HSIC(Z, color):         {hsic_val:.6f}")
    
    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    criteria = {
        'aps_acc_gt_85': acc_aps > 0.85,
        'baseline_acc_lt_60': acc_baseline < 0.60,
        'improvement_gt_25': improvement > 0.25,
        'hsic_lt_01': hsic_val < 0.1
    }
    
    print(f"✓ APS OOD Accuracy > 85%:     {'PASS' if criteria['aps_acc_gt_85'] else 'FAIL'} ({acc_aps*100:.1f}%)")
    print(f"✓ Baseline OOD Accuracy < 60%: {'PASS' if criteria['baseline_acc_lt_60'] else 'FAIL'} ({acc_baseline*100:.1f}%)")
    print(f"✓ Improvement (Δ) > 25%:       {'PASS' if criteria['improvement_gt_25'] else 'FAIL'} ({improvement*100:.1f}%)")
    print(f"✓ HSIC(Z, color) < 0.1:        {'PASS' if criteria['hsic_lt_01'] else 'FAIL'} ({hsic_val:.6f})")
    
    all_pass = all(criteria.values())
    
    if all_pass:
        print("\n✅ SUCCESS: All criteria met! Causality component demonstrates OOD generalization!")
    else:
        print("\n⚠️  PARTIAL: Some criteria not met. May need hyperparameter tuning.")
    
    # Save results
    results_dir = Path('outputs/colored_mnist_ood')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'baseline_ood_accuracy': float(acc_baseline),
        'aps_ood_accuracy': float(acc_aps),
        'improvement': float(improvement),
        'improvement_pct': float(improvement_pct),
        'hsic_z_color': float(hsic_val),
        'criteria': {k: bool(v) for k, v in criteria.items()},
        'all_criteria_met': bool(all_pass),
        'hyperparameters': {
            'epochs': args.epochs,
            'lambda_C': args.lambda_C,
            'lr': args.lr,
            'batch_size': args.batch_size
        }
    }
    
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    # Save models
    if args.save_models:
        torch.save(model_baseline.state_dict(), results_dir / 'baseline_model.pt')
        torch.save({
            'autoencoder': model_aps.state_dict(),
            'classifier': classifier_aps.state_dict()
        }, results_dir / 'aps_model.pt')
        print(f"💾 Models saved to {results_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Colored MNIST OOD Generalization Experiment'
    )
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lambda-C', type=float, default=1.0,
                       help='Weight for causality loss (default: 1.0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--device', default='cpu',
                       help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    main(args)
