"""Visualize Phase 006B experiment results.

Creates plots for:
- Training/OOD accuracy curves
- Loss curves
- OOD gap comparison
- Component contribution analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict

# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 300


def load_experiment_results(output_dir: str = './outputs/phase006b') -> Dict:
    """Load results from all experiments."""
    output_dir = Path(output_dir)
    
    experiments = ['baseline', 'aps-T', 'aps-C', 'aps-TC', 'aps-full']
    results = {}
    
    for exp in experiments:
        exp_dir = output_dir / exp
        
        metrics_file = exp_dir / 'final_metrics.json'
        history_file = exp_dir / 'history.json'
        config_file = exp_dir / 'config.json'
        
        if not metrics_file.exists():
            print(f"Warning: {exp} results not found, skipping...")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        results[exp] = {
            'metrics': metrics,
            'history': history,
            'config': config
        }
    
    return results


def plot_accuracy_curves(results: Dict, output_dir: str):
    """Plot training and OOD accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'baseline': '#1f77b4',
        'aps-T': '#ff7f0e', 
        'aps-C': '#2ca02c',
        'aps-TC': '#d62728',
        'aps-full': '#9467bd'
    }
    
    for exp_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history['train_acc']) + 1)
        color = colors.get(exp_name, 'gray')
        
        # Train accuracy
        ax1.plot(epochs, [100*x for x in history['train_acc']], 
                label=exp_name, color=color, linewidth=2)
        
        # OOD accuracy
        ax2.plot(epochs, [100*x for x in history['ood_acc']], 
                label=exp_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('OOD Accuracy (World Domain)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_curves.png")
    plt.close()


def plot_loss_curves(results: Dict, output_dir: str):
    """Plot training and OOD loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'baseline': '#1f77b4',
        'aps-T': '#ff7f0e',
        'aps-C': '#2ca02c',
        'aps-TC': '#d62728',
        'aps-full': '#9467bd'
    }
    
    for exp_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors.get(exp_name, 'gray')
        
        # Train loss
        ax1.plot(epochs, history['train_loss'], 
                label=exp_name, color=color, linewidth=2)
        
        # OOD loss
        ax2.plot(epochs, history['ood_loss'], 
                label=exp_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('OOD Loss (World Domain)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/loss_curves.png")
    plt.close()


def plot_ood_comparison(results: Dict, output_dir: str):
    """Bar plot comparing final OOD accuracies."""
    experiments = []
    ood_accs = []
    train_accs = []
    
    for exp_name, data in results.items():
        experiments.append(exp_name)
        ood_accs.append(100 * data['metrics']['final_ood_accuracy'])
        train_acc = data['history']['train_acc'][-1] if data['history']['train_acc'] else 0
        train_accs.append(100 * train_acc)
    
    x = np.arange(len(experiments))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, ood_accs, width, label='OOD', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'ood_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/ood_comparison.png")
    plt.close()


def plot_ood_gap(results: Dict, output_dir: str):
    """Plot OOD generalization gap."""
    experiments = []
    gaps = []
    
    for exp_name, data in results.items():
        experiments.append(exp_name)
        train_acc = data['history']['train_acc'][-1] if data['history']['train_acc'] else 0
        ood_acc = data['metrics']['final_ood_accuracy']
        gap = 100 * (train_acc - ood_acc)
        gaps.append(gap)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728' if g > 10 else '#2ca02c' for g in gaps]
    bars = ax.bar(experiments, gaps, color=colors, alpha=0.7)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('OOD Gap (%)')
    ax.set_title('Generalization Gap (Train Acc - OOD Acc)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'ood_gap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/ood_gap.png")
    plt.close()


def plot_improvement_over_baseline(results: Dict, output_dir: str):
    """Plot improvement over baseline."""
    if 'baseline' not in results:
        print("Baseline not found, skipping improvement plot")
        return
    
    baseline_ood = results['baseline']['metrics']['final_ood_accuracy']
    
    experiments = []
    improvements = []
    
    for exp_name, data in results.items():
        if exp_name == 'baseline':
            continue
        experiments.append(exp_name)
        ood_acc = data['metrics']['final_ood_accuracy']
        imp = 100 * (ood_acc - baseline_ood)
        improvements.append(imp)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ca02c' if i > 0 else '#d62728' for i in improvements]
    bars = ax.bar(experiments, improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Improvement (%)')
    ax.set_title(f'OOD Accuracy Improvement over Baseline ({100*baseline_ood:.2f}%)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.2f}',
               ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'improvement_over_baseline.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/improvement_over_baseline.png")
    plt.close()


def create_all_plots(output_dir: str = './outputs/phase006b'):
    """Create all visualization plots."""
    print(f"Loading results from {output_dir}...")
    results = load_experiment_results(output_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} experiments: {list(results.keys())}")
    print("\nCreating plots...")
    
    plot_accuracy_curves(results, output_dir)
    plot_loss_curves(results, output_dir)
    plot_ood_comparison(results, output_dir)
    plot_ood_gap(results, output_dir)
    plot_improvement_over_baseline(results, output_dir)
    
    print("\n✓ All plots created!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Phase 006B results')
    parser.add_argument('--output-dir', type=str, default='./outputs/phase006b',
                       help='Output directory')
    args = parser.parse_args()
    
    create_all_plots(args.output_dir)


if __name__ == '__main__':
    main()
