#!/usr/bin/env python3
"""
True 3D Latent Space Visualization

Demonstrates energy landscapes in actual 3D latent space (latent_dim=3).
Includes cross-sectional slices and 3D scatter plots of embeddings.
"""

import torch
import numpy as np
from aps.energy import MemoryEnergy, MemoryEnergyConfig
from aps.viz import EnergyLandscapeVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_3d_memory_cube(output_file='outputs/energy_3d_cube.html'):
    """
    Create 3D latent space with memory patterns at cube vertices.
    Visualize with 3D scatter and slice views.
    """
    
    beta = 5.0
    
    # Create 3D energy model
    config = MemoryEnergyConfig(latent_dim=3, n_mem=8, beta=beta)
    energy_module = MemoryEnergy(config)
    
    # Place memory patterns at cube corners
    cube_corners = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    with torch.no_grad():
        energy_module.mem.copy_(cube_corners)
    
    # Generate sample points in 3D space
    n_samples = 2000
    sample_points = torch.randn(n_samples, 3) * 1.5
    
    with torch.no_grad():
        sample_energy = energy_module(sample_points).numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Add sample points colored by energy
    fig.add_trace(go.Scatter3d(
        x=sample_points[:, 0].numpy(),
        y=sample_points[:, 1].numpy(),
        z=sample_points[:, 2].numpy(),
        mode='markers',
        marker=dict(
            size=2,
            color=sample_energy,
            colorscale='Viridis',
            colorbar=dict(title='Energy'),
            showscale=True
        ),
        name='Sample Points',
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>E: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Add memory patterns as large markers
    with torch.no_grad():
        mem_energy = energy_module(cube_corners).numpy()
    
    fig.add_trace(go.Scatter3d(
        x=cube_corners[:, 0].numpy(),
        y=cube_corners[:, 1].numpy(),
        z=cube_corners[:, 2].numpy(),
        mode='markers+text',
        marker=dict(
            size=10,
            color='red',
            symbol='x',
            line=dict(width=4, color='white')
        ),
        text=[f'M{i+1}' for i in range(8)],
        textposition='top center',
        name='Memory Patterns',
        hovertemplate='Pattern %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>E: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Add cube edges for reference
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        i, j = edge
        fig.add_trace(go.Scatter3d(
            x=[cube_corners[i, 0], cube_corners[j, 0]],
            y=[cube_corners[i, 1], cube_corners[j, 1]],
            z=[cube_corners[i, 2], cube_corners[j, 2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={
            'text': f'3D Latent Space with Energy (β = {beta})<br><sub>Memory patterns at cube vertices • Color = Energy</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        scene=dict(
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Latent Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        )
    )
    
    fig.write_html(output_file)
    print(f"✓ Created 3D cube visualization: {output_file}")
    print(f"  - 3D latent space (latent_dim=3)")
    print(f"  - {n_samples} sample points colored by energy")
    print(f"  - Memory patterns at cube vertices")
    
    return fig


def create_3d_slices(output_file='outputs/energy_3d_slices.html'):
    """
    Create cross-sectional slices through 3D energy landscape.
    Shows how energy varies at different z-levels.
    """
    
    beta = 5.0
    z_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Create 3D energy model
    config = MemoryEnergyConfig(latent_dim=3, n_mem=8, beta=beta)
    energy_module = MemoryEnergy(config)
    
    # Place memory patterns at cube corners
    cube_corners = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    with torch.no_grad():
        energy_module.mem.copy_(cube_corners)
    
    # Create subplots for each z-level
    n_slices = len(z_levels)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'z = {z:.1f}' for z in z_levels] + [''],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    for idx, z_val in enumerate(z_levels):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # Create grid at this z-level
        resolution = 50
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Compute energy on this slice
        points = torch.tensor(
            np.stack([xx.flatten(), yy.flatten(), np.full(resolution**2, z_val)], axis=1),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            energy = energy_module(points).numpy().reshape(resolution, resolution)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=energy,
                colorscale='Viridis',
                showscale=(idx == len(z_levels) - 1),
                colorbar=dict(title='Energy', x=1.15, len=0.4, y=0.75) if idx == len(z_levels) - 1 else None,
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>E: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add memory patterns that are at or near this z-level
        nearby_patterns = cube_corners[torch.abs(cube_corners[:, 2] - z_val) < 0.3]
        if len(nearby_patterns) > 0:
            fig.add_trace(
                go.Scatter(
                    x=nearby_patterns[:, 0].numpy(),
                    y=nearby_patterns[:, 1].numpy(),
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color='red',
                        line=dict(width=2, color='white')
                    ),
                    showlegend=(idx == 0),
                    name='Memory Patterns',
                    hovertemplate='Pattern<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title={
            'text': f'Energy Landscape Cross-Sections (β = {beta})<br><sub>Slices through 3D space at different z-levels</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        showlegend=True
    )
    
    # Update axes
    for i in range(n_slices):
        fig.update_xaxes(title_text='Latent Dim 1', row=(i//3)+1, col=(i%3)+1)
        fig.update_yaxes(title_text='Latent Dim 2', row=(i//3)+1, col=(i%3)+1)
    
    fig.write_html(output_file)
    print(f"✓ Created 3D slice visualization: {output_file}")
    print(f"  - {len(z_levels)} cross-sections at different z-levels")
    print(f"  - Shows how energy structure varies through 3D space")
    
    return fig


def create_3d_trajectory(output_file='outputs/energy_3d_trajectory.html'):
    """
    Visualize gradient descent trajectories in 3D space.
    Shows how points flow toward nearest memory basin.
    """
    
    beta = 5.0
    
    # Create 3D energy model
    config = MemoryEnergyConfig(latent_dim=3, n_mem=8, beta=beta)
    energy_module = MemoryEnergy(config)
    
    # Place memory patterns at cube corners
    cube_corners = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    with torch.no_grad():
        energy_module.mem.copy_(cube_corners)
    
    # Create figure
    fig = go.Figure()
    
    # Generate initial points and simulate gradient descent
    n_trajectories = 20
    n_steps = 50
    lr = 0.1
    
    initial_points = torch.randn(n_trajectories, 3) * 0.8
    
    for i in range(n_trajectories):
        trajectory = [initial_points[i].clone()]
        current = initial_points[i].clone()
        current.requires_grad_(True)
        
        for step in range(n_steps):
            energy = energy_module(current.unsqueeze(0))
            energy.backward()
            
            with torch.no_grad():
                current -= lr * current.grad
                current.grad.zero_()
                trajectory.append(current.clone())
        
        trajectory_array = torch.stack(trajectory).detach().numpy()
        
        # Compute energy along trajectory
        with torch.no_grad():
            traj_energy = energy_module(torch.stack(trajectory)).numpy()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory_array[:, 0],
            y=trajectory_array[:, 1],
            z=trajectory_array[:, 2],
            mode='lines+markers',
            line=dict(width=2, color=traj_energy, colorscale='Plasma', showscale=(i == 0)),
            marker=dict(size=2, color=traj_energy, colorscale='Plasma', showscale=False),
            showlegend=False,
            hovertemplate='Step %{pointNumber}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
        ))
    
    # Add memory patterns
    fig.add_trace(go.Scatter3d(
        x=cube_corners[:, 0].numpy(),
        y=cube_corners[:, 1].numpy(),
        z=cube_corners[:, 2].numpy(),
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond',
            line=dict(width=4, color='white')
        ),
        name='Memory Patterns',
        hovertemplate='Pattern<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'Gradient Descent Trajectories in 3D (β = {beta})<br><sub>Lines show flow toward nearest memory basin • Color = Energy</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        scene=dict(
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Latent Dimension 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        )
    )
    
    fig.write_html(output_file)
    print(f"✓ Created 3D trajectory visualization: {output_file}")
    print(f"  - {n_trajectories} gradient descent trajectories")
    print(f"  - Shows basin attraction in 3D space")
    
    return fig


def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("Creating True 3D Latent Space Visualizations")
    print("=" * 70)
    print()
    
    # Create 3D cube visualization
    print("[1/3] 3D cube with energy...")
    create_3d_memory_cube()
    print()
    
    # Create cross-sectional slices
    print("[2/3] Cross-sectional slices...")
    create_3d_slices()
    print()
    
    # Create trajectory visualization
    print("[3/3] Gradient descent trajectories...")
    create_3d_trajectory()
    print()
    
    print("=" * 70)
    print("Done! Open the HTML files:")
    print("  - outputs/energy_3d_cube.html        (3D scatter with energy)")
    print("  - outputs/energy_3d_slices.html      (cross-sections)")
    print("  - outputs/energy_3d_trajectory.html  (basin dynamics)")
    print("=" * 70)
    print()
    print("Key insights:")
    print("  • Memory patterns form cube vertices in 3D space")
    print("  • Cross-sections show energy structure at each z-level")
    print("  • Trajectories demonstrate basin attraction")


if __name__ == '__main__':
    main()
