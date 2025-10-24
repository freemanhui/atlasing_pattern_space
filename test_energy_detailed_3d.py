#!/usr/bin/env python3
"""
Create detailed 3D view of each energy variant separately.
"""

from aps.energy import *
import torch
import numpy as np
import plotly.graph_objects as go

print("Creating detailed 3D views for each variant...")

# Define energy models
variants = {
    'memory': {
        'name': 'MemoryEnergy (Dot Product)',
        'config': MemoryEnergyConfig(latent_dim=2, n_mem=4, init_method='grid', beta=5.0),
        'model_class': MemoryEnergy,
        'description': 'Log-sum-exp with dot product similarity'
    },
    'rbf': {
        'name': 'RBFEnergy (Gaussian Basins)',
        'config': RBFEnergyConfig(latent_dim=2, n_mem=4, sigma=0.5, init_method='grid'),
        'model_class': RBFEnergy,
        'description': 'Gaussian kernels with learnable radius'
    },
    'mixture': {
        'name': 'MixtureEnergy (Learnable Parameters)',
        'config': MixtureEnergyConfig(latent_dim=2, n_mem=4, init_method='grid'),
        'model_class': MixtureEnergy,
        'description': 'Per-pattern weights and sharpness'
    }
}

# Compute high-resolution landscape
resolution = 100
x = np.linspace(-2, 2, resolution)
y = np.linspace(-2, 2, resolution)
xx, yy = np.meshgrid(x, y)
grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)

for variant_id, info in variants.items():
    print(f"  Creating {info['name']}...")
    
    # Create model
    energy_model = info['model_class'](info['config'])
    
    # Compute energy
    with torch.no_grad():
        e = energy_model.energy(grid).numpy().reshape(resolution, resolution)
    
    # Create figure
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=e,
        colorscale='Viridis',
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
    
    # Add memory patterns
    mem_np = energy_model.mem.detach().cpu().numpy()
    with torch.no_grad():
        mem_points = torch.tensor(mem_np, dtype=torch.float32)
        mem_energy = energy_model.energy(mem_points).numpy()
    
    fig.add_trace(go.Scatter3d(
        x=mem_np[:, 0],
        y=mem_np[:, 1],
        z=mem_energy,
        mode='markers+text',
        marker=dict(
            symbol='x',
            size=10,
            color='red',
            line=dict(width=4, color='white')
        ),
        text=[f'M{i+1}' for i in range(len(mem_np))],
        textposition='top center',
        name='Memory Patterns',
        hovertemplate='Pattern %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>E: %{z:.2f}<extra></extra>'
    ))
    
    # Get variant-specific info
    extra_info = []
    if variant_id == 'rbf':
        extra_info.append(f"σ = {energy_model.sigma.mean().item():.2f}")
    elif variant_id == 'mixture':
        extra_info.append(f"Weight entropy: {-torch.sum(energy_model.weights * torch.log(energy_model.weights + 1e-8)).item():.3f}")
    
    extra_text = " | ".join(extra_info) if extra_info else ""
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'{info["name"]}<br><sub>{info["description"]}{" | " + extra_text if extra_text else ""}</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
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
    
    # Save
    output_file = f'outputs/energy_detailed_{variant_id}_3d.html'
    fig.write_html(output_file)
    print(f"    ✓ Saved {output_file}")

print()
print("=" * 60)
print("Created 3 detailed views:")
print("  - outputs/energy_detailed_memory_3d.html")
print("  - outputs/energy_detailed_rbf_3d.html")
print("  - outputs/energy_detailed_mixture_3d.html")
print("=" * 60)
