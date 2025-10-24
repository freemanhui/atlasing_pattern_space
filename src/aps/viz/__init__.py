"""
Energy basin visualization module for APS.

This module provides interactive visualization tools for energy landscapes
in latent space, including basin identification, trajectory simulation,
and cross-sectional analysis.
"""

from .visualizer import EnergyLandscapeVisualizer
from .config import VisualizationConfig
from .interactions import InteractionHandler, PointInfo, ComparisonResult
from .data_structures import (
    EnergyLandscape,
    MemoryPattern,
    Basin,
    Point,
    Trajectory,
    CrossSection
)

__all__ = [
    'EnergyLandscapeVisualizer',
    'VisualizationConfig',
    'InteractionHandler',
    'PointInfo',
    'ComparisonResult',
    'EnergyLandscape',
    'MemoryPattern',
    'Basin',
    'Point',
    'Trajectory',
    'CrossSection'
]
