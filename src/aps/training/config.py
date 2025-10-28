"""
Training configuration for APS models.

Provides comprehensive configuration for training, optimization, and logging.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    name: str = 'adam'  # 'adam', 'sgd', 'adamw'
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    momentum: float = 0.9  # For SGD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'momentum': self.momentum,
        }


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: Optional[str] = None  # 'step', 'cosine', 'exponential', None
    step_size: int = 30  # For StepLR
    gamma: float = 0.1  # For StepLR and ExponentialLR
    T_max: Optional[int] = None  # For CosineAnnealingLR (defaults to epochs)
    eta_min: float = 0.0  # For CosineAnnealingLR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
        }


@dataclass
class TrainingConfig:
    """
    Configuration for training APS models.
    
    Training:
        epochs: Number of training epochs
        batch_size: Batch size for training
        val_batch_size: Batch size for validation (defaults to batch_size)
        val_freq: Validation frequency (every N epochs)
        grad_clip: Gradient clipping value (None = no clipping)
        
    Device:
        device: Device to use ('cpu', 'cuda', 'mps', or specific like 'cuda:0')
        mixed_precision: Use mixed precision training (fp16)
        
    Logging:
        log_freq: Logging frequency (every N steps)
        save_freq: Checkpoint saving frequency (every N epochs)
        output_dir: Directory for outputs (checkpoints, logs, plots)
        experiment_name: Name for this experiment
        use_tensorboard: Enable tensorboard logging
        use_wandb: Enable wandb logging
        wandb_project: Wandb project name
        
    Checkpointing:
        resume_from: Path to checkpoint to resume from
        save_best_only: Only save checkpoint if validation loss improves
        patience: Early stopping patience (None = no early stopping)
    """
    # Training
    epochs: int = 100
    batch_size: int = 32
    val_batch_size: Optional[int] = None
    val_freq: int = 1
    grad_clip: Optional[float] = None
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Device
    device: str = 'cpu'
    mixed_precision: bool = False
    
    # Logging
    log_freq: int = 10
    save_freq: int = 10
    output_dir: str = './outputs'
    experiment_name: str = 'aps_experiment'
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    # Checkpointing
    resume_from: Optional[str] = None
    save_best_only: bool = False
    patience: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set val_batch_size to batch_size if not specified
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        
        # Create output directory
        self.output_dir = Path(self.output_dir)
        
        # Set T_max for cosine scheduler if not specified
        if self.scheduler.name == 'cosine' and self.scheduler.T_max is None:
            self.scheduler.T_max = self.epochs
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self.output_dir / self.experiment_name / 'checkpoints'
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return self.output_dir / self.experiment_name / 'logs'
    
    @property
    def plot_dir(self) -> Path:
        """Get plot directory path."""
        return self.output_dir / self.experiment_name / 'plots'
    
    def create_directories(self):
        """Create all necessary directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'val_batch_size': self.val_batch_size,
            'val_freq': self.val_freq,
            'grad_clip': self.grad_clip,
            'optimizer': self.optimizer.to_dict(),
            'scheduler': self.scheduler.to_dict(),
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'log_freq': self.log_freq,
            'save_freq': self.save_freq,
            'experiment_name': self.experiment_name,
            'save_best_only': self.save_best_only,
            'patience': self.patience,
        }
