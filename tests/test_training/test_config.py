"""
Tests for training configuration.
"""

import pytest
from pathlib import Path
from aps.training.config import TrainingConfig, OptimizerConfig, SchedulerConfig


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""
    
    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        
        assert config.name == 'adam'
        assert config.lr == 1e-3
        assert config.weight_decay == 0.0
        assert config.betas == (0.9, 0.999)
        assert config.momentum == 0.9
    
    def test_custom_config(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            name='sgd',
            lr=0.01,
            weight_decay=1e-4,
            momentum=0.95,
        )
        
        assert config.name == 'sgd'
        assert config.lr == 0.01
        assert config.weight_decay == 1e-4
        assert config.momentum == 0.95
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig(name='adamw', lr=1e-4)
        d = config.to_dict()
        
        assert d['name'] == 'adamw'
        assert d['lr'] == 1e-4
        assert 'weight_decay' in d
        assert 'betas' in d


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""
    
    def test_default_config(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        
        assert config.name is None
        assert config.step_size == 30
        assert config.gamma == 0.1
        assert config.T_max is None
        assert config.eta_min == 0.0
    
    def test_step_scheduler_config(self):
        """Test step scheduler configuration."""
        config = SchedulerConfig(
            name='step',
            step_size=10,
            gamma=0.5,
        )
        
        assert config.name == 'step'
        assert config.step_size == 10
        assert config.gamma == 0.5
    
    def test_cosine_scheduler_config(self):
        """Test cosine scheduler configuration."""
        config = SchedulerConfig(
            name='cosine',
            T_max=100,
            eta_min=1e-6,
        )
        
        assert config.name == 'cosine'
        assert config.T_max == 100
        assert config.eta_min == 1e-6
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SchedulerConfig(name='exponential', gamma=0.95)
        d = config.to_dict()
        
        assert d['name'] == 'exponential'
        assert d['gamma'] == 0.95


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.val_batch_size == 32  # Same as batch_size
        assert config.val_freq == 1
        assert config.grad_clip is None
        assert config.device == 'cpu'
        assert config.log_freq == 10
        assert config.save_freq == 10
        assert config.output_dir == Path('./outputs')
        assert config.experiment_name == 'aps_experiment'
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            epochs=50,
            batch_size=64,
            val_batch_size=128,
            device='cuda',
            experiment_name='test_exp',
        )
        
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.val_batch_size == 128
        assert config.device == 'cuda'
        assert config.experiment_name == 'test_exp'
    
    def test_val_batch_size_default(self):
        """Test that val_batch_size defaults to batch_size."""
        config = TrainingConfig(batch_size=128)
        assert config.val_batch_size == 128
    
    def test_optimizer_config(self):
        """Test optimizer configuration within training config."""
        opt_cfg = OptimizerConfig(name='adamw', lr=1e-4)
        config = TrainingConfig(optimizer=opt_cfg)
        
        assert config.optimizer.name == 'adamw'
        assert config.optimizer.lr == 1e-4
    
    def test_scheduler_config(self):
        """Test scheduler configuration within training config."""
        sched_cfg = SchedulerConfig(name='cosine', T_max=100)
        config = TrainingConfig(scheduler=sched_cfg)
        
        assert config.scheduler.name == 'cosine'
        assert config.scheduler.T_max == 100
    
    def test_cosine_scheduler_t_max_default(self):
        """Test that T_max defaults to epochs for cosine scheduler."""
        sched_cfg = SchedulerConfig(name='cosine')
        config = TrainingConfig(epochs=200, scheduler=sched_cfg)
        
        assert config.scheduler.T_max == 200
    
    def test_directory_properties(self):
        """Test directory path properties."""
        config = TrainingConfig(
            output_dir='./test_outputs',
            experiment_name='my_exp',
        )
        
        assert config.checkpoint_dir == Path('./test_outputs/my_exp/checkpoints')
        assert config.log_dir == Path('./test_outputs/my_exp/logs')
        assert config.plot_dir == Path('./test_outputs/my_exp/plots')
    
    def test_create_directories(self, tmp_path):
        """Test directory creation."""
        config = TrainingConfig(
            output_dir=str(tmp_path),
            experiment_name='test',
        )
        
        config.create_directories()
        
        assert config.checkpoint_dir.exists()
        assert config.log_dir.exists()
        assert config.plot_dir.exists()
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(
            epochs=50,
            batch_size=64,
            experiment_name='test',
        )
        
        d = config.to_dict()
        
        assert d['epochs'] == 50
        assert d['batch_size'] == 64
        assert d['experiment_name'] == 'test'
        assert 'optimizer' in d
        assert 'scheduler' in d
    
    def test_logging_config(self):
        """Test logging configuration."""
        config = TrainingConfig(
            use_tensorboard=True,
            use_wandb=True,
            wandb_project='test_project',
        )
        
        assert config.use_tensorboard is True
        assert config.use_wandb is True
        assert config.wandb_project == 'test_project'
    
    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            patience=10,
            save_best_only=True,
        )
        
        assert config.patience == 10
        assert config.save_best_only is True
    
    def test_resume_config(self):
        """Test resume configuration."""
        config = TrainingConfig(
            resume_from='/path/to/checkpoint.pt',
        )
        
        assert config.resume_from == '/path/to/checkpoint.pt'
