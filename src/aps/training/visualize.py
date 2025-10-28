"""
Visualization utilities for training analysis.

Provides functions to plot training curves, loss components, and metrics.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    log_dir: Path,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
):
    """
    Plot training and validation loss curves from logged metrics.
    
    Args:
        log_dir: Directory containing metric JSON files
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    """
    # Load metrics
    epoch_file = Path(log_dir) / 'epoch_metrics.json'
    if not epoch_file.exists():
        raise FileNotFoundError(f"Epoch metrics not found: {epoch_file}")
    
    with open(epoch_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_metrics = data['train_metrics']
    val_metrics = data['val_metrics']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot total loss
    if 'total' in train_metrics:
        ax1.plot(epochs, train_metrics['total'], label='Train', marker='o')
        if 'total' in val_metrics:
            ax1.plot(epochs, val_metrics['total'], label='Val', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total Loss over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot loss components
    components = [k for k in train_metrics.keys() if k != 'total']
    for component in components:
        ax2.plot(epochs, train_metrics[component], label=component, marker='o')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to: {save_path}")
    else:
        plt.show()
    
    return fig


def plot_step_metrics(
    log_dir: Path,
    metric: str = 'total',
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 4),
    smoothing: float = 0.0,
):
    """
    Plot step-level metrics (useful for seeing within-epoch dynamics).
    
    Args:
        log_dir: Directory containing metric JSON files
        metric: Metric name to plot
        save_path: Path to save figure (optional)
        figsize: Figure size
        smoothing: Exponential smoothing factor (0 = no smoothing, 0.9 = heavy)
    """
    # Load metrics
    step_file = Path(log_dir) / 'step_metrics.json'
    if not step_file.exists():
        raise FileNotFoundError(f"Step metrics not found: {step_file}")
    
    with open(step_file, 'r') as f:
        data = json.load(f)
    
    steps = data['steps']
    metrics = data['metrics']
    
    if metric not in metrics:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(metrics.keys())}")
    
    values = metrics[metric]
    
    # Apply smoothing if requested
    if smoothing > 0:
        smoothed = []
        last = values[0]
        for v in values:
            last = smoothing * last + (1 - smoothing) * v
            smoothed.append(last)
        values_smooth = smoothed
    else:
        values_smooth = values
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if smoothing > 0:
        ax.plot(steps, values, alpha=0.3, label='Raw')
        ax.plot(steps, values_smooth, label=f'Smoothed ({smoothing})')
    else:
        ax.plot(steps, values, label=metric)
    
    ax.set_xlabel('Step')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} over Training Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved step metrics to: {save_path}")
    else:
        plt.show()
    
    return fig


def plot_ablation_comparison(
    experiment_dirs: Dict[str, Path],
    metric: str = 'total',
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
):
    """
    Compare multiple ablation experiments.
    
    Args:
        experiment_dirs: Dict mapping experiment names to log directories
        metric: Metric to compare (default: 'total')
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, log_dir in experiment_dirs.items():
        epoch_file = Path(log_dir) / 'epoch_metrics.json'
        if not epoch_file.exists():
            print(f"Warning: {name} metrics not found, skipping")
            continue
        
        with open(epoch_file, 'r') as f:
            data = json.load(f)
        
        epochs = data['epochs']
        val_metrics = data.get('val_metrics', {})
        
        if metric in val_metrics:
            ax.plot(epochs, val_metrics[metric], label=name, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Validation {metric.capitalize()}')
    ax.set_title('Ablation Study Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation comparison to: {save_path}")
    else:
        plt.show()
    
    return fig


def plot_all_components(
    log_dir: Path,
    save_dir: Optional[Path] = None,
):
    """
    Generate all standard training plots.
    
    Args:
        log_dir: Directory containing metric JSON files
        save_dir: Directory to save plots (optional)
    """
    log_dir = Path(log_dir)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    try:
        save_path = save_dir / 'loss_curves.png' if save_dir else None
        plot_loss_curves(log_dir, save_path=save_path)
        plt.close()
    except Exception as e:
        print(f"Could not plot loss curves: {e}")
    
    # Plot step metrics for total loss
    try:
        save_path = save_dir / 'step_metrics.png' if save_dir else None
        plot_step_metrics(log_dir, metric='total', save_path=save_path, smoothing=0.7)
        plt.close()
    except Exception as e:
        print(f"Could not plot step metrics: {e}")
    
    print(f"Visualization complete! Plots saved to: {save_dir or 'display'}")


def create_training_summary(
    log_dir: Path,
    save_path: Optional[Path] = None,
) -> str:
    """
    Create a text summary of training results.
    
    Args:
        log_dir: Directory containing metric JSON files
        save_path: Path to save summary (optional)
    
    Returns:
        Summary string
    """
    log_dir = Path(log_dir)
    epoch_file = log_dir / 'epoch_metrics.json'
    
    if not epoch_file.exists():
        return "No training metrics found"
    
    with open(epoch_file, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_metrics = data['train_metrics']
    val_metrics = data['val_metrics']
    
    # Build summary
    lines = []
    lines.append("="*50)
    lines.append("TRAINING SUMMARY")
    lines.append("="*50)
    lines.append(f"Total epochs: {len(epochs)}")
    lines.append("")
    
    # Final metrics
    lines.append("Final Metrics:")
    lines.append("-" * 30)
    for key in train_metrics:
        train_val = train_metrics[key][-1]
        lines.append(f"  Train {key}: {train_val:.4f}")
        if key in val_metrics and val_metrics[key]:
            val_val = val_metrics[key][-1]
            lines.append(f"  Val {key}: {val_val:.4f}")
    lines.append("")
    
    # Best metrics
    if 'total' in val_metrics and val_metrics['total']:
        best_epoch = val_metrics['total'].index(min(val_metrics['total']))
        best_loss = val_metrics['total'][best_epoch]
        lines.append(f"Best validation loss: {best_loss:.4f} (epoch {epochs[best_epoch]})")
    
    lines.append("="*50)
    
    summary = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary to: {save_path}")
    
    return summary
