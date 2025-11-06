"""
Unified trainer for APS models.

Provides training loop, validation, checkpointing, and metrics tracking.
"""

from typing import Optional, Dict, Callable
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .logger import MetricsLogger
from ..models.aps_autoencoder import APSAutoencoder


class Trainer:
    """
    Unified trainer for APS models.
    
    Handles:
    - Training loop with progress tracking
    - Validation and early stopping
    - Checkpointing and resuming
    - Metrics logging (tensorboard, wandb, JSON)
    - Learning rate scheduling
    - Gradient clipping
    
    Example:
        >>> from aps.models import APSAutoencoder, APSConfig
        >>> from aps.training import Trainer, TrainingConfig
        >>> 
        >>> model_cfg = APSConfig(in_dim=784, latent_dim=2)
        >>> model = APSAutoencoder(model_cfg)
        >>> 
        >>> train_cfg = TrainingConfig(epochs=100, batch_size=32)
        >>> trainer = Trainer(model, train_cfg)
        >>> 
        >>> trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: APSAutoencoder,
        config: TrainingConfig,
    ):
        """
        Initialize trainer.
        
        Args:
            model: APSAutoencoder model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logger
        config.create_directories()
        self.logger = MetricsLogger(
            log_dir=config.log_dir,
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_cfg = self.config.optimizer
        
        if opt_cfg.name.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas,
            )
        elif opt_cfg.name.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas,
            )
        elif opt_cfg.name.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                momentum=opt_cfg.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_cfg = self.config.scheduler
        
        if sched_cfg.name is None:
            return None
        elif sched_cfg.name.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
            )
        elif sched_cfg.name.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.T_max,
                eta_min=sched_cfg.eta_min,
            )
        elif sched_cfg.name.lower() == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_cfg.gamma,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        nuisance_fn: Optional[Callable] = None,
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            nuisance_fn: Function to extract nuisance variables from batch (optional)
                         Should take batch and return nuisance tensor
        """
        # Initialize wandb if needed
        if self.config.use_wandb:
            wandb_config = {
                'model': self.model.config.__dict__,
                'training': self.config.to_dict(),
            }
            self.logger.init_wandb(wandb_config, name=self.config.experiment_name)
        
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.config.output_dir / self.config.experiment_name}")
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, nuisance_fn)
                
                # Validation phase
                val_metrics = None
                if val_loader and (epoch + 1) % self.config.val_freq == 0:
                    val_metrics = self._validate(val_loader, nuisance_fn)
                
                # Log epoch metrics
                self.logger.log_epoch(epoch, train_metrics, val_metrics)
                
                # Print progress
                self._print_progress(epoch, train_metrics, val_metrics)
                
                # Step scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_freq == 0:
                    self._save_checkpoint(epoch, val_metrics)
                
                # Early stopping check
                if val_metrics and self.config.patience:
                    if val_metrics['total'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['total']
                        self.patience_counter = 0
                        # Save best model
                        self._save_checkpoint(epoch, val_metrics, is_best=True)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.patience:
                            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                            break
        
        finally:
            # Always save final checkpoint and close logger
            self._save_checkpoint(self.current_epoch, None, is_final=True)
            self.logger.close()
            print(f"\nTraining complete! Results saved to: {self.config.output_dir / self.config.experiment_name}")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        nuisance_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Extract data and move to device
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
                nuisance = nuisance_fn(batch) if nuisance_fn else None
            else:
                x = batch.to(self.device)
                nuisance = None
            
            # Forward pass and compute losses
            losses = self.model.compute_loss(x, nuisance)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            for name, value in losses.items():
                if name not in epoch_metrics:
                    epoch_metrics[name] = 0.0
                epoch_metrics[name] += value.item()
            
            # Log step metrics
            if self.global_step % self.config.log_freq == 0:
                step_metrics = {k: v.item() for k, v in losses.items()}
                self.logger.log_step(self.global_step, step_metrics)
                self.logger.log_learning_rate(
                    self.global_step,
                    self.optimizer.param_groups[0]['lr']
                )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.global_step += 1
        
        # Average metrics over epoch
        num_batches = len(train_loader)
        return {k: v / num_batches for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        nuisance_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {}
        
        for batch in val_loader:
            # Extract data and move to device
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
                nuisance = nuisance_fn(batch) if nuisance_fn else None
            else:
                x = batch.to(self.device)
                nuisance = None
            
            # Compute losses
            losses = self.model.compute_loss(x, nuisance)
            
            # Accumulate metrics
            for name, value in losses.items():
                if name not in val_metrics:
                    val_metrics[name] = 0.0
                val_metrics[name] += value.item()
        
        # Average metrics
        num_batches = len(val_loader)
        return {k: v / num_batches for k, v in val_metrics.items()}
    
    def _print_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Print training progress."""
        msg = f"Epoch {epoch}: "
        msg += f"train_loss={train_metrics['total']:.4f}"
        
        if val_metrics:
            msg += f", val_loss={val_metrics['total']:.4f}"
        
        print(msg)
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save model checkpoint."""
        if self.config.save_best_only and not (is_best or is_final):
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.config.__dict__,
            'training_config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if val_metrics:
            checkpoint['val_metrics'] = val_metrics
        
        # Save checkpoint
        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
        elif is_final:
            path = self.config.checkpoint_dir / 'final_model.pt'
        else:
            path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resumed from checkpoint: {path}")
        print(f"Starting from epoch {self.current_epoch}")
