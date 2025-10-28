"""
APS training module.

Provides unified training pipeline with configuration, logging, and checkpointing.
"""

from .config import TrainingConfig, OptimizerConfig, SchedulerConfig
from .logger import MetricsLogger
from .trainer import Trainer
from .visualize import (
    plot_loss_curves,
    plot_step_metrics,
    plot_ablation_comparison,
    plot_all_components,
    create_training_summary,
)

__all__ = [
    'TrainingConfig',
    'OptimizerConfig',
    'SchedulerConfig',
    'MetricsLogger',
    'Trainer',
    'plot_loss_curves',
    'plot_step_metrics',
    'plot_ablation_comparison',
    'plot_all_components',
    'create_training_summary',
]
