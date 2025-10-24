#!/usr/bin/env python3
"""
Compare different energy variants in 3D (energy as height).
"""

from aps.energy import *
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Creating 3D energy surface comparison...")

# Create 3 energy models with same initialization
memory_cfg = MemoryEnergyConfig(latent_dim=2, n_mem=4, init_method='grid', beta=5.0)
rbf_cfg = RBFEnergyConfig(latent_dim=2, n_mem=4, sigma=0.5, init_method='grid')
mix_cfg = MixtureEnergyConfig(latent_dim=2, n_mem=4, init_method='grid')

energies = {
    'MemoryEnergy<br>(dot product)': MemoryEnergy(memory_cfg),
    'RBFEnergy<br>(Gaussian)': RBFEnergy(rbf_cfg),
    'MixtureEnergy<br>(learnable params)': MixtureEnergy(mix_cfg)
}

# Compute energy landscapes
resolution = 80
x = np.linspace(-2, 2, resolution)
y = np.linspace(-2, 2, resolution)
xx, yy = np.meshgrid(x, y)
grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=list(energies.keys()),
    horizontal_spacing=0.05
)

for idx, (name, energy_model) in enumerate(energies.items()):
    col = idx + 1
    
    with torch.no_grad():
        e = energy_model.energy(grid).numpy().reshape(resolution, resolution)
    
    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=e,
            colorscale='Viridis',
            showscale=(idx == 2),  # Only show colorbar for last plot
            colorbar=dict(title='Energy', x=1.15, len=0.8) if idx == 2 else None,
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Energy: %{z:.2f}<extra></extra>',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            )
        ),
        row=1, col=col
    )
    
    # Add memory pattern markers in 3D
    mem_np = energy_model.mem.detach().cpu().numpy()
    with torch.no_grad():
        mem_points = torch.tensor(mem_np, dtype=torch.float32)
        mem_energy = energy_model.energy(mem_points).numpy()
    
    fig.add_trace(
        go.Scatter3d(
            x=mem_np[:, 0],
            y=mem_np[:, 1],
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
        row=1, col=col
    )

# Update layout
fig.update_layout(
    title={
        'text': '3D Energy Surface Comparison: Different Variants<br><sub>Height = Energy | Valleys = Basins | Red X = Memory Patterns</sub>',
        'x': 0.5,
        'xanchor': 'center'
    },
    height=600,
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
    )
)

# Save
import os
os.makedirs('outputs', exist_ok=True)
output_file = 'outputs/energy_comparison_3d.html'
fig.write_html(output_file)

print(f"✓ Saved 3D comparison to {output_file}")
print("  Features:")
print("    - Rotate by clicking and dragging")
print("    - Zoom with scroll wheel")
print("    - Compare basin shapes directly")
print("    - Valleys show where patterns attract points")
