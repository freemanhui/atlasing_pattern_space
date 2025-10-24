#!/usr/bin/env python3
"""
3D Surface Visualization for Energy Landscapes

Shows energy as height (z-axis) over 2D latent space (x, y).
Creates intuitive "terrain" view of basins and barriers.
"""

import torch
import numpy as np
from aps.energy import MemoryEnergy, MemoryEnergyConfig
from aps.viz import EnergyLandscapeVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_3d_surface_comparison(output_file='outputs/energy_3d_surfaces.html'):
    """
    Create side-by-side 3D surface plots for different beta values.
    """
    
    beta_values = [1.0, 3.0, 5.0, 10.0]
    
    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[f'Beta = {b}' for b in beta_values],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Generate surfaces for each beta
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
        
        # Add 3D surface
        fig.add_trace(
            go.Surface(
                x=landscape.grid_x[0, :],
                y=landscape.grid_y[:, 0],
                z=landscape.energy_values,
                colorscale='Viridis',
                showscale=(idx == len(beta_values) - 1),
                colorbar=dict(
                    title='Energy',
                    x=1.15,
                    len=0.45,
                    y=0.25
                ) if idx == len(beta_values) - 1 else None,
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Energy: %{z:.2f}<extra></extra>',
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
                )
            ),
            row=row, col=col
        )
        
        # Add memory pattern markers as 3D scatter
        mem_x = [p.position_2d[0] for p in landscape.memory_patterns]
        mem_y = [p.position_2d[1] for p in landscape.memory_patterns]
        # Compute energy at memory locations
        mem_points = torch.tensor([[x, y] for x, y in zip(mem_x, mem_y)], dtype=torch.float32)
        with torch.no_grad():
            mem_energy = energy_module(mem_points).numpy()
        
        fig.add_trace(
            go.Scatter3d(
                x=mem_x,
                y=mem_y,
                z=mem_energy,
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=8,
                    color='red',
                    line=dict(width=4, color='white')
                ),
                showlegend=(idx == 0),
                name='Memory Patterns',
                hovertemplate='Pattern<br>x: %{x:.2f}<br>y: %{y:.2f}<br>E: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '3D Energy Surfaces: Effect of Beta Parameter<br><sub>Height = Energy | Valleys = Basins | Peaks = Barriers</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=900,
        showlegend=True,
        scene=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Energy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        scene2=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Energy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        scene3=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Energy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        scene4=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Energy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        )
    )
    
    fig.write_html(output_file)
    print(f"✓ Created 3D surface comparison: {output_file}")
    print(f"  - Rotate by clicking and dragging")
    print(f"  - Zoom with scroll wheel")
    print(f"  - See basin depth as actual valleys")
    
    return fig


def create_single_detailed_3d(output_file='outputs/energy_3d_detailed.html'):
    """
    Create a single detailed 3D surface with multiple viewing angles.
    """
    
    beta = 5.0
    
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
    
    # Compute high-resolution landscape
    viz = EnergyLandscapeVisualizer(energy_module, latent_dim=2, resolution=100)
    landscape = viz.compute_landscape(bounds=(-2.0, 2.0, -2.0, 2.0))
    
    # Create figure with surface and contour projection
    fig = go.Figure()
    
    # Add main surface
    fig.add_trace(go.Surface(
        x=landscape.grid_x[0, :],
        y=landscape.grid_y[:, 0],
        z=landscape.energy_values,
        colorscale='Viridis',
        name='Energy Surface',
        showscale=True,
        colorbar=dict(title='Energy', x=1.1),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Energy: %{z:.2f}<extra></extra>',
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    ))
    
    # Add memory pattern markers
    mem_x = [p.position_2d[0] for p in landscape.memory_patterns]
    mem_y = [p.position_2d[1] for p in landscape.memory_patterns]
    mem_points = torch.tensor([[x, y] for x, y in zip(mem_x, mem_y)], dtype=torch.float32)
    with torch.no_grad():
        mem_energy = energy_module(mem_points).numpy()
    
    fig.add_trace(go.Scatter3d(
        x=mem_x,
        y=mem_y,
        z=mem_energy,
        mode='markers+text',
        marker=dict(
            symbol='x',
            size=10,
            color='red',
            line=dict(width=4, color='white')
        ),
        text=[f'M{i+1}' for i in range(len(mem_x))],
        textposition='top center',
        name='Memory Patterns',
        hovertemplate='Pattern %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>E: %{z:.2f}<extra></extra>'
    ))
    
    # Update layout with detailed controls
    fig.update_layout(
        title={
            'text': f'Detailed 3D Energy Landscape (β = {beta})<br><sub>Rotate to explore • Valleys = attraction basins • Red X = memory patterns</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        scene=dict(
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Energy',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=-0.1)
            ),
            aspectmode='cube'
        ),
        showlegend=True
    )
    
    fig.write_html(output_file)
    print(f"✓ Created detailed 3D surface: {output_file}")
    print(f"  - High resolution (100x100 grid)")
    print(f"  - Contour projections on base")
    print(f"  - Labeled memory patterns")
    
    return fig


def create_wireframe_comparison(output_file='outputs/energy_3d_wireframe.html'):
    """
    Create wireframe view to emphasize basin structure.
    """
    
    beta_values = [1.0, 5.0, 10.0]
    
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
        
        # Compute landscape (lower resolution for clearer wireframe)
        viz = EnergyLandscapeVisualizer(energy_module, latent_dim=2, resolution=30)
        landscape = viz.compute_landscape(bounds=(-2.0, 2.0, -2.0, 2.0))
        
        # Add wireframe surface
        fig.add_trace(go.Surface(
            x=landscape.grid_x[0, :],
            y=landscape.grid_y[:, 0],
            z=landscape.energy_values,
            name=f'β = {beta}',
            showscale=False,
            visible=(beta == 5.0),  # Only show beta=5.0 initially
            hovertemplate='β=%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>E: %{z:.2f}<extra></extra>',
            text=[[beta]*landscape.grid_x.shape[1]]*landscape.grid_x.shape[0],
            hidesurface=True,
            contours=dict(
                x=dict(show=True, color='gray', width=2),
                y=dict(show=True, color='gray', width=2),
                z=dict(show=True, usecolormap=False, width=2)
            )
        ))
    
    # Add buttons to switch between beta values
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[{"visible": [i == idx for i in range(len(beta_values))]}],
                        label=f"β = {beta}",
                        method="update"
                    )
                    for idx, beta in enumerate(beta_values)
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title={
            'text': 'Wireframe View: Basin Structure<br><sub>Use dropdown to switch beta values</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        scene=dict(
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Energy',
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        )
    )
    
    fig.write_html(output_file)
    print(f"✓ Created wireframe comparison: {output_file}")
    print(f"  - Wireframe emphasizes structure")
    print(f"  - Use dropdown to switch beta")
    
    return fig


def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("Creating 3D Surface Visualizations")
    print("=" * 70)
    print()
    
    # Create multi-panel 3D comparison
    print("[1/3] Multi-panel 3D surfaces...")
    create_3d_surface_comparison()
    print()
    
    # Create detailed single view
    print("[2/3] Detailed 3D surface...")
    create_single_detailed_3d()
    print()
    
    # Create wireframe view
    print("[3/3] Wireframe structure view...")
    create_wireframe_comparison()
    print()
    
    print("=" * 70)
    print("Done! Open the HTML files:")
    print("  - outputs/energy_3d_surfaces.html    (4-panel comparison)")
    print("  - outputs/energy_3d_detailed.html    (high-res single view)")
    print("  - outputs/energy_3d_wireframe.html   (structure emphasis)")
    print("=" * 70)


if __name__ == '__main__':
    main()
