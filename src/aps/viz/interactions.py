"""
Interactive event handling for energy landscape visualizations.

This module provides the InteractionHandler class for managing hover and click
events on energy landscape visualizations.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from .data_structures import EnergyLandscape, Basin


@dataclass
class PointInfo:
    """Information about a point in the energy landscape."""
    position: Tuple[float, float]
    energy: float
    basin_id: Optional[int] = None
    nearest_pattern_id: Optional[int] = None
    distance_to_pattern: Optional[float] = None


@dataclass
class ComparisonResult:
    """Result of comparing multiple points."""
    points: List[PointInfo]
    energy_range: Tuple[float, float]
    basin_diversity: int  # Number of unique basins


class InteractionHandler:
    """
    Handles interactive events (hover, click) on energy landscape visualizations.
    
    Provides methods for responding to user interactions with energy landscape
    plots, including hover tooltips and click-to-inspect functionality.
    
    Example:
        >>> handler = InteractionHandler()
        >>> point_info = handler.on_click(event, landscape, basins)
        >>> print(f"Energy: {point_info.energy:.3f}")
    """
    
    def __init__(self):
        """Initialize the interaction handler."""
        self._clicked_points: List[PointInfo] = []
        self._compare_mode = False
    
    def on_hover(
        self,
        x: float,
        y: float,
        landscape: EnergyLandscape,
        basins: Optional[List[Basin]] = None
    ) -> PointInfo:
        """
        Handle hover event at specified coordinates.
        
        Args:
            x: X coordinate in latent space
            y: Y coordinate in latent space
            landscape: EnergyLandscape object
            basins: Optional list of identified basins
            
        Returns:
            PointInfo object with details about the hovered point
            
        Performance:
            Must complete in < 100ms (SC-002)
        """
        # TODO: Implement hover logic
        energy = self._get_energy_at(x, y, landscape)
        basin_id = self._find_basin_id(x, y, basins) if basins else None
        nearest_pattern_id, distance = self._find_nearest_pattern(x, y, landscape)
        
        return PointInfo(
            position=(x, y),
            energy=energy,
            basin_id=basin_id,
            nearest_pattern_id=nearest_pattern_id,
            distance_to_pattern=distance
        )
    
    def on_click(
        self,
        x: float,
        y: float,
        landscape: EnergyLandscape,
        basins: Optional[List[Basin]] = None
    ) -> PointInfo:
        """
        Handle click event at specified coordinates.
        
        Args:
            x: X coordinate in latent space
            y: Y coordinate in latent space
            landscape: EnergyLandscape object
            basins: Optional list of identified basins
            
        Returns:
            PointInfo object with details about the clicked point
            
        Performance:
            Must complete in < 100ms (SC-002)
        """
        # TODO: Implement click logic
        point_info = self.on_hover(x, y, landscape, basins)
        
        if self._compare_mode:
            self._clicked_points.append(point_info)
        else:
            self._clicked_points = [point_info]
        
        return point_info
    
    def enable_compare_mode(self):
        """Enable compare mode to track multiple clicked points."""
        self._compare_mode = True
    
    def disable_compare_mode(self):
        """Disable compare mode and clear comparison points."""
        self._compare_mode = False
        self._clicked_points = []
    
    def get_comparison(self) -> Optional[ComparisonResult]:
        """
        Get comparison result for all clicked points.
        
        Returns:
            ComparisonResult if multiple points clicked, None otherwise
        """
        if len(self._clicked_points) < 2:
            return None
        
        energies = [p.energy for p in self._clicked_points]
        basin_ids = set(p.basin_id for p in self._clicked_points if p.basin_id is not None)
        
        return ComparisonResult(
            points=self._clicked_points.copy(),
            energy_range=(min(energies), max(energies)),
            basin_diversity=len(basin_ids)
        )
    
    def clear_clicks(self):
        """Clear all clicked points."""
        self._clicked_points = []
    
    # Helper methods
    
    def _get_energy_at(self, x: float, y: float, landscape: EnergyLandscape) -> float:
        """Get energy value at specified coordinates using interpolation."""
        # TODO: Implement energy interpolation
        # For now, use get_energy_at from landscape
        return landscape.get_energy_at(x, y)
    
    def _find_basin_id(
        self,
        x: float,
        y: float,
        basins: Optional[List[Basin]]
    ) -> Optional[int]:
        """Find which basin a point belongs to."""
        if not basins:
            return None
        
        for basin in basins:
            if basin.contains_point(x, y):
                return basin.pattern_id
        
        return None
    
    def _find_nearest_pattern(
        self,
        x: float,
        y: float,
        landscape: EnergyLandscape
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find the nearest memory pattern to a point."""
        if not landscape.memory_patterns:
            return None, None
        
        min_dist = float('inf')
        nearest_id = None
        
        for pattern in landscape.memory_patterns:
            px, py = pattern.position_2d
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_id = pattern.id
        
        return nearest_id, min_dist
