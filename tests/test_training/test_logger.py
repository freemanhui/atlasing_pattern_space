"""
Tests for metrics logger.
"""

import pytest
import json
from pathlib import Path
from aps.training.logger import MetricsLogger


class TestMetricsLogger:
    """Tests for MetricsLogger."""
    
    def test_initialization(self, tmp_path):
        """Test logger initialization."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        assert logger.log_dir == tmp_path
        assert logger.log_dir.exists()
        assert logger.use_tensorboard is False
        assert logger.use_wandb is False
        assert len(logger.steps) == 0
        assert len(logger.epochs) == 0
    
    def test_log_step(self, tmp_path):
        """Test logging step metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        logger.log_step(0, {'loss': 1.0, 'recon': 0.8})
        logger.log_step(1, {'loss': 0.9, 'recon': 0.7})
        
        assert len(logger.steps) == 2
        assert logger.steps == [0, 1]
        assert logger.step_metrics['loss'] == [1.0, 0.9]
        assert logger.step_metrics['recon'] == [0.8, 0.7]
    
    def test_log_epoch(self, tmp_path):
        """Test logging epoch metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        logger.log_epoch(
            epoch=0,
            train_metrics={'loss': 1.0, 'recon': 0.8},
            val_metrics={'loss': 1.1, 'recon': 0.9},
        )
        
        assert len(logger.epochs) == 1
        assert logger.epochs == [0]
        assert logger.epoch_metrics['loss'] == [1.0]
        assert logger.val_metrics['loss'] == [1.1]
    
    def test_log_epoch_no_validation(self, tmp_path):
        """Test logging epoch without validation metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        logger.log_epoch(
            epoch=0,
            train_metrics={'loss': 1.0},
        )
        
        assert len(logger.epochs) == 1
        assert logger.epoch_metrics['loss'] == [1.0]
        assert len(logger.val_metrics) == 0
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        # Log some metrics
        logger.log_step(0, {'loss': 1.0})
        logger.log_step(1, {'loss': 0.9})
        logger.log_epoch(0, {'loss': 0.95}, {'loss': 1.0})
        
        # Save
        logger.save()
        
        # Check files exist
        assert (tmp_path / 'step_metrics.json').exists()
        assert (tmp_path / 'epoch_metrics.json').exists()
        
        # Load into new logger
        new_logger = MetricsLogger(log_dir=tmp_path)
        new_logger.load()
        
        assert new_logger.steps == [0, 1]
        assert new_logger.step_metrics['loss'] == [1.0, 0.9]
        assert new_logger.epochs == [0]
        assert new_logger.epoch_metrics['loss'] == [0.95]
        assert new_logger.val_metrics['loss'] == [1.0]
    
    def test_get_best_epoch(self, tmp_path):
        """Test getting best epoch."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        # Log multiple epochs
        logger.log_epoch(0, {'loss': 1.0}, {'loss': 1.1})
        logger.log_epoch(1, {'loss': 0.9}, {'loss': 0.8})  # Best
        logger.log_epoch(2, {'loss': 0.85}, {'loss': 0.9})
        
        best_epoch = logger.get_best_epoch(metric='loss', mode='min')
        assert best_epoch == 1
    
    def test_get_best_epoch_max(self, tmp_path):
        """Test getting best epoch with max mode."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        # Log multiple epochs with accuracy
        logger.log_epoch(0, {'acc': 0.7}, {'acc': 0.65})
        logger.log_epoch(1, {'acc': 0.8}, {'acc': 0.82})  # Best
        logger.log_epoch(2, {'acc': 0.85}, {'acc': 0.78})
        
        best_epoch = logger.get_best_epoch(metric='acc', mode='max')
        assert best_epoch == 1
    
    def test_get_best_epoch_no_val_metrics(self, tmp_path):
        """Test getting best epoch with no validation metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        best_epoch = logger.get_best_epoch(metric='loss', mode='min')
        assert best_epoch is None
    
    def test_context_manager(self, tmp_path):
        """Test logger as context manager."""
        with MetricsLogger(log_dir=tmp_path) as logger:
            logger.log_step(0, {'loss': 1.0})
            logger.log_epoch(0, {'loss': 0.95})
        
        # Files should be saved automatically
        assert (tmp_path / 'step_metrics.json').exists()
        assert (tmp_path / 'epoch_metrics.json').exists()
    
    def test_multiple_metrics(self, tmp_path):
        """Test logging multiple metrics."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        metrics = {
            'total': 1.0,
            'recon': 0.5,
            'topo': 0.3,
            'energy': 0.2,
        }
        
        logger.log_step(0, metrics)
        
        assert len(logger.step_metrics) == 4
        for key in metrics:
            assert key in logger.step_metrics
            assert logger.step_metrics[key] == [metrics[key]]
    
    def test_incremental_metrics(self, tmp_path):
        """Test that new metrics can be added incrementally."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        # First step has only loss
        logger.log_step(0, {'loss': 1.0})
        
        # Second step adds new metric
        logger.log_step(1, {'loss': 0.9, 'recon': 0.7})
        
        assert logger.step_metrics['loss'] == [1.0, 0.9]
        assert logger.step_metrics['recon'] == [0.7]
    
    def test_json_serialization(self, tmp_path):
        """Test that saved JSON is valid."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        logger.log_step(0, {'loss': 1.0})
        logger.log_epoch(0, {'loss': 0.95}, {'loss': 1.0})
        logger.save()
        
        # Load and verify JSON
        with open(tmp_path / 'step_metrics.json', 'r') as f:
            step_data = json.load(f)
            assert 'steps' in step_data
            assert 'metrics' in step_data
        
        with open(tmp_path / 'epoch_metrics.json', 'r') as f:
            epoch_data = json.load(f)
            assert 'epochs' in epoch_data
            assert 'train_metrics' in epoch_data
            assert 'val_metrics' in epoch_data


class TestMetricsLoggerIntegration:
    """Integration tests for MetricsLogger."""
    
    def test_full_training_simulation(self, tmp_path):
        """Simulate a full training run."""
        logger = MetricsLogger(log_dir=tmp_path)
        
        # Simulate 3 epochs with 5 steps each
        global_step = 0
        for epoch in range(3):
            epoch_train_losses = []
            
            for step in range(5):
                loss = 1.0 - (global_step * 0.01)
                logger.log_step(global_step, {'loss': loss})
                epoch_train_losses.append(loss)
                global_step += 1
            
            # Log epoch metrics
            train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            val_loss = train_loss + 0.1
            
            logger.log_epoch(
                epoch,
                {'loss': train_loss},
                {'loss': val_loss},
            )
        
        # Verify
        assert len(logger.steps) == 15
        assert len(logger.epochs) == 3
        assert logger.step_metrics['loss'][0] == 1.0
        assert logger.step_metrics['loss'][-1] < 1.0
        
        # Save and verify
        logger.save()
        assert (tmp_path / 'step_metrics.json').exists()
        assert (tmp_path / 'epoch_metrics.json').exists()
