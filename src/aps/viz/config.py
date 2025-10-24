"""
Configuration for energy landscape visualizations.
"""

from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """
    User preferences for visualization appearance.
    
    Attributes:
        color_scheme: Matplotlib/Plotly colormap name (default: "viridis")
        color_scale: "linear" or "log" for color scaling
        show_memory_markers: Whether to display memory pattern markers
        marker_size: Size of memory pattern markers
        show_grid: Whether to show grid lines
        grid_alpha: Transparency of grid lines (0-1)
        tooltip_enabled: Enable hover tooltips
        trajectory_color_by_basin: Color trajectories by their destination basin
    """
    color_scheme: str = "viridis"
    color_scale: str = "linear"
    show_memory_markers: bool = True
    marker_size: int = 10
    show_grid: bool = False
    grid_alpha: float = 0.3
    tooltip_enabled: bool = True
    trajectory_color_by_basin: bool = True
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration values are invalid
        """
        if self.color_scale not in ["linear", "log"]:
            raise ValueError("color_scale must be 'linear' or 'log'")
        
        if not (0 <= self.grid_alpha <= 1):
            raise ValueError("grid_alpha must be in [0, 1]")
        
        if self.marker_size < 1:
            raise ValueError("marker_size must be positive")
