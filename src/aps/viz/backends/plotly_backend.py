"""
Plotly backend for interactive energy landscape visualizations.
"""

import plotly.graph_objects as go
from ..data_structures import EnergyLandscape
from ..config import VisualizationConfig


class PlotlyBackend:
    """
    Plotly backend for rendering energy landscapes.
    
    Provides interactive heatmaps with hover tooltips and memory pattern markers.
    """
    
    def plot_heatmap(
        self,
        landscape: EnergyLandscape,
        config: VisualizationConfig,
        **kwargs
    ) -> go.Figure:
        """
        Create 2D heatmap visualization with Plotly.
        
        Args:
            landscape: EnergyLandscape to visualize
            config: Visualization configuration
            **kwargs: Additional Plotly arguments
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add heatmap trace
        heatmap = go.Heatmap(
            x=landscape.grid_x[0, :],
            y=landscape.grid_y[:, 0],
            z=landscape.energy_values,
            colorscale=config.color_scheme,
            colorbar=dict(title="Energy"),
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>Energy: %{z:.3f}<extra></extra>",
            name="Energy"
        )
        fig.add_trace(heatmap)
        
        # Add memory pattern markers if enabled
        if config.show_memory_markers and landscape.memory_patterns:
            x_markers = [mp.position_2d[0] for mp in landscape.memory_patterns]
            y_markers = [mp.position_2d[1] for mp in landscape.memory_patterns]
            [mp.energy for mp in landscape.memory_patterns]
            
            markers = go.Scatter(
                x=x_markers,
                y=y_markers,
                mode='markers',
                marker=dict(
                    size=config.marker_size,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                text=[f"Pattern {mp.id}<br>E={mp.energy:.3f}" for mp in landscape.memory_patterns],
                hovertemplate="%{text}<extra></extra>",
                name="Memory Patterns"
            )
            fig.add_trace(markers)
        
        # Update layout
        fig.update_layout(
            title="Energy Landscape",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            width=kwargs.get('width', 800),
            height=kwargs.get('height', 600),
            hovermode='closest'
        )
        
        # Add grid if requested
        if config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,{})'.format(config.grid_alpha))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,{})'.format(config.grid_alpha))
        
        return fig
