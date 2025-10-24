#!/usr/bin/env python3
"""
Compare different energy variants visually.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from aps.energy import (
    MemoryEnergy, MemoryEnergyConfig,
    RBFEnergy, RBFEnergyConfig,
    MixtureEnergy, MixtureEnergyConfig
)

print("Creating energy landscape comparison...")

# Create 3 energy models with same initialization
memory_cfg = MemoryEnergyConfig(latent_dim=2, n_mem=4, init_method='grid', beta=5.0)
rbf_cfg = RBFEnergyConfig(latent_dim=2, n_mem=4, sigma=0.5, init_method='grid')
mix_cfg = MixtureEnergyConfig(latent_dim=2, n_mem=4, init_method='grid')

energies = {
    'MemoryEnergy\n(dot product)': MemoryEnergy(memory_cfg),
    'RBFEnergy\n(Gaussian)': RBFEnergy(rbf_cfg),
    'MixtureEnergy\n(learnable params)': MixtureEnergy(mix_cfg)
}

# Compute energy landscapes
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
xx, yy = np.meshgrid(x, y)
grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, energy_model) in zip(axes, energies.items()):
    with torch.no_grad():
        e = energy_model.energy(grid).numpy().reshape(100, 100)
    
    im = ax.imshow(e, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
    mem_np = energy_model.mem.detach().cpu().numpy()
    ax.scatter(mem_np[:, 0], mem_np[:, 1], 
               c='red', marker='x', s=150, linewidths=3, label='Memory patterns')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Energy')

plt.suptitle('Energy Landscape Comparison: Different Variants', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/energy_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved comparison to outputs/energy_comparison.png")
print("  Open the file to see the differences between energy variants!")
