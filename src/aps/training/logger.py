"""
Metrics logging for APS training.

Tracks loss components and training metrics over time.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import torch


class MetricsLogger:
    """
    Logger for tracking training metrics.
    
    Tracks all loss components (recon, topo, hsic, energy) and additional metrics
    over training steps and epochs.
    
    Example:
        >>> logger = MetricsLogger(log_dir='./logs')
        >>> logger.log_step(step=0, metrics={'loss': 0.5, 'recon': 0.3})
        >>> logger.log_epoch(epoch=1, train_metrics={'loss': 0.4}, val_metrics={'loss': 0.45})
        >>> logger.save()
    """
    
    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Enable tensorboard logging
            use_wandb: Enable wandb logging
            wandb_project: Wandb project name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Step-level metrics
        self.step_metrics: Dict[str, List[float]] = {}
        self.steps: List[int] = []
        
        # Epoch-level metrics
        self.epoch_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}
        self.epochs: List[int] = []
        
        # Optional integrations
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
            except ImportError:
                print("Warning: tensorboard not installed, skipping tensorboard logging")
                self.use_tensorboard = False
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = None  # Will be initialized when config is available
                self.wandb_project = wandb_project
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")
                self.use_wandb = False
    
    def init_wandb(self, config: Dict, name: Optional[str] = None):
        """
        Initialize wandb run.
        
        Args:
            config: Training configuration dict
            name: Run name (optional)
        """
        if self.use_wandb:
            self.wandb_run = self.wandb.init(
                project=self.wandb_project,
                name=name,
                config=config,
            )
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a training step.
        
        Args:
            step: Current step number
            metrics: Dictionary of metric names and values
        """
        self.steps.append(step)
        
        for name, value in metrics.items():
            if name not in self.step_metrics:
                self.step_metrics[name] = []
            self.step_metrics[name].append(float(value))
        
        # Log to tensorboard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f'step/{name}', value, step)
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            self.wandb.log({f'step/{k}': v for k, v in metrics.items()}, step=step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        self.epochs.append(epoch)
        
        # Log training metrics
        for name, value in train_metrics.items():
            if name not in self.epoch_metrics:
                self.epoch_metrics[name] = []
            self.epoch_metrics[name].append(float(value))
        
        # Log validation metrics
        if val_metrics is not None:
            for name, value in val_metrics.items():
                if name not in self.val_metrics:
                    self.val_metrics[name] = []
                self.val_metrics[name].append(float(value))
        
        # Log to tensorboard
        if self.use_tensorboard:
            for name, value in train_metrics.items():
                self.tb_writer.add_scalar(f'epoch/train_{name}', value, epoch)
            if val_metrics is not None:
                for name, value in val_metrics.items():
                    self.tb_writer.add_scalar(f'epoch/val_{name}', value, epoch)
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            log_dict = {f'epoch/train_{k}': v for k, v in train_metrics.items()}
            if val_metrics is not None:
                log_dict.update({f'epoch/val_{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            self.wandb.log(log_dict)
    
    def log_learning_rate(self, step: int, lr: float):
        """Log learning rate."""
        if self.use_tensorboard:
            self.tb_writer.add_scalar('learning_rate', lr, step)
        if self.use_wandb and self.wandb_run:
            self.wandb.log({'learning_rate': lr}, step=step)
    
    def save(self):
        """Save metrics to JSON files."""
        # Save step metrics
        if self.step_metrics:
            step_data = {
                'steps': self.steps,
                'metrics': self.step_metrics,
            }
            with open(self.log_dir / 'step_metrics.json', 'w') as f:
                json.dump(step_data, f, indent=2)
        
        # Save epoch metrics
        if self.epoch_metrics:
            epoch_data = {
                'epochs': self.epochs,
                'train_metrics': self.epoch_metrics,
                'val_metrics': self.val_metrics,
            }
            with open(self.log_dir / 'epoch_metrics.json', 'w') as f:
                json.dump(epoch_data, f, indent=2)
    
    def load(self):
        """Load metrics from JSON files."""
        # Load step metrics
        step_file = self.log_dir / 'step_metrics.json'
        if step_file.exists():
            with open(step_file, 'r') as f:
                step_data = json.load(f)
                self.steps = step_data['steps']
                self.step_metrics = step_data['metrics']
        
        # Load epoch metrics
        epoch_file = self.log_dir / 'epoch_metrics.json'
        if epoch_file.exists():
            with open(epoch_file, 'r') as f:
                epoch_data = json.load(f)
                self.epochs = epoch_data['epochs']
                self.epoch_metrics = epoch_data['train_metrics']
                self.val_metrics = epoch_data['val_metrics']
    
    def get_best_epoch(self, metric: str = 'total', mode: str = 'min') -> Optional[int]:
        """
        Get epoch with best validation metric.
        
        Args:
            metric: Metric name to use
            mode: 'min' or 'max'
        
        Returns:
            Best epoch number, or None if no validation metrics
        """
        if metric not in self.val_metrics or not self.val_metrics[metric]:
            return None
        
        values = self.val_metrics[metric]
        if mode == 'min':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        return self.epochs[best_idx]
    
    def close(self):
        """Close logger and cleanup resources."""
        self.save()
        
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb and self.wandb_run:
            self.wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
