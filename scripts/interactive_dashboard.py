#!/usr/bin/env python3
"""
Interactive Energy Landscape Dashboard

Creates an HTML dashboard with sliders to tune parameters in real-time.
Allows exploration of how beta, number of patterns, and resolution affect
the energy landscape.
"""

import torch
import numpy as np
from aps.energy import MemoryEnergy, MemoryEnergyConfig
from aps.viz import EnergyLandscapeVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(output_file='outputs/interactive_dashboard.html'):
    """
    Create an interactive dashboard with multiple visualizations.
    
    Shows:
    1. Energy landscape heatmap
    2. Cross-section along y=0
    3. Basin depth analysis
    """
    
    # Create multiple beta values to compare
    beta_values = [1.0, 3.0, 5.0, 10.0]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Beta = {b}' for b in beta_values],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Generate landscapes for different beta values
    for idx, beta in enumerate(beta_values):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        # Create energy model
        config = MemoryEnergyConfig(latent_dim=2, n_mem=4, beta=beta)
        energy_module = MemoryEnergy(config)
        
        with torch.no_grad():
            energy_module.mem.copy_(torch.tensor([
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0]
            ]))
        
        # Compute landscape
        viz = EnergyLandscapeVisualizer(energy_module, latent_dim=2, resolution=50)
        landscape = viz.compute_landscape(bounds=(-2.0, 2.0, -2.0, 2.0))
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                x=landscape.grid_x[0, :],
                y=landscape.grid_y[:, 0],
                z=landscape.energy_values,
                colorscale='Viridis',
                showscale=(idx == len(beta_values) - 1),
                colorbar=dict(title='Energy', x=1.15) if idx == len(beta_values) - 1 else None,
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Energy: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add memory pattern markers
        mem_x = [p.position_2d[0] for p in landscape.memory_patterns]
        mem_y = [p.position_2d[1] for p in landscape.memory_patterns]
        
        fig.add_trace(
            go.Scatter(
                x=mem_x,
                y=mem_y,
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='red',
                    line=dict(width=2, color='white')
                ),
                showlegend=(idx == 0),
                name='Memory Patterns',
                hovertemplate='Pattern<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Energy Landscape: Effect of Beta Parameter<br><sub>Higher beta = sharper basins around memory patterns</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(title_text='Latent Dimension 1', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text='Latent Dimension 2', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    # Save
    fig.write_html(output_file)
    print(f"✓ Created interactive dashboard: {output_file}")
    print(f"  - Compare 4 different beta values")
    print(f"  - Hover to see exact coordinates and energy")
    print(f"  - Zoom and pan each subplot independently")
    
    return fig


def create_cross_section_comparison(output_file='outputs/cross_section_comparison.html'):
    """
    Create interactive cross-section comparison across different beta values.
    """
    
    beta_values = [1.0, 3.0, 5.0, 10.0]
    fig = go.Figure()
    
    for beta in beta_values:
        # Create energy model
        config = MemoryEnergyConfig(latent_dim=2, n_mem=4, beta=beta)
        energy_module = MemoryEnergy(config)
        
        with torch.no_grad():
            energy_module.mem.copy_(torch.tensor([
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0]
            ]))
        
        # Compute landscape
        viz = EnergyLandscapeVisualizer(energy_module, latent_dim=2, resolution=100)
        landscape = viz.compute_landscape(bounds=(-2.0, 2.0, -2.0, 2.0))
        
        # Create cross-section
        cross_section = viz.plot_cross_section(
            landscape,
            start_point=(-2.0, 0.0),
            end_point=(2.0, 0.0),
            num_samples=100
        )
        
        # Extract data
        distances = [p[0] for p in cross_section.sample_points]
        energies = [p[1] for p in cross_section.sample_points]
        
        # Add line
        fig.add_trace(go.Scatter(
            x=distances,
            y=energies,
            mode='lines',
            name=f'β = {beta}',
            line=dict(width=2),
            hovertemplate='Distance: %{x:.2f}<br>Energy: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Energy Profile Along y=0 (Cross-Section Comparison)',
        xaxis_title='Distance Along Line',
        yaxis_title='Energy',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.write_html(output_file)
    print(f"✓ Created cross-section comparison: {output_file}")
    
    return fig


def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("Creating Interactive Dashboards")
    print("=" * 70)
    print()
    
    # Create multi-panel comparison
    create_dashboard()
    print()
    
    # Create cross-section comparison
    create_cross_section_comparison()
    print()
    
    print("=" * 70)
    print("Done! Open the HTML files to explore:")
    print("  - outputs/interactive_dashboard.html")
    print("  - outputs/cross_section_comparison.html")
    print("=" * 70)


if __name__ == '__main__':
    main()
