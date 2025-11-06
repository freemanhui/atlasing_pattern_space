#!/usr/bin/env python3
"""
Demo: Topology-Aware Energy vs Memory Energy

This script demonstrates the difference between:
1. MemoryEnergy: Creates arbitrary attractor basins
2. TopologyEnergy: Reinforces topology preservation

Shows how TopologyEnergy aligns with topology objectives rather than competing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from aps.energy import (
    MemoryEnergy, MemoryEnergyConfig,
    TopologyEnergy, TopologyEnergyConfig
)
from aps.topology import knn_indices, adjacency_from_knn


def generate_data(n_samples=200, noise=0.1):
    """Generate 2D moons dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X, dtype=torch.float32), y


def compute_knn_preservation(X_original, Z_latent, k=8):
    """Compute k-NN preservation ratio."""
    # Get kNN in both spaces
    knn_orig = knn_indices(X_original, k=k)
    knn_latent = knn_indices(Z_latent, k=k)
    
    # Convert to adjacency
    adj_orig = adjacency_from_knn(knn_orig, n_samples=X_original.shape[0])
    adj_latent = adjacency_from_knn(knn_latent, n_samples=Z_latent.shape[0])
    
    # Compute agreement
    agreement = (adj_orig * adj_latent).sum()
    total = X_original.shape[0] * k
    
    return (agreement / total).item()


def visualize_energies(X, y, energy_fn, title, ax, is_topology_energy=False):
    """Visualize energy landscape and data points."""
    # Create grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], 
        dtype=torch.float32
    )
    
    # Compute energy on grid
    with torch.no_grad():
        if is_topology_energy:
            # For TopologyEnergy, compute local neighborhood preservation
            # by measuring distance to nearest data points and their neighborhoods
            energies = []
            batch_size = 100  # Process in batches
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                # For each grid point, find distance to nearest actual data point
                dists = torch.cdist(batch, X)  # (batch, n_data)
                # Use negative min distance as proxy for topology alignment
                energy_batch = -torch.min(dists, dim=1)[0]
                energies.append(energy_batch)
            energies = torch.cat(energies).numpy()
        else:
            energies = energy_fn.energy(grid_points).numpy()
    energies = energies.reshape(xx.shape)
    
    # Plot energy landscape
    contour = ax.contourf(xx, yy, energies, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Energy')
    
    # Plot data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], 
        c=y, cmap='coolwarm', 
        edgecolors='black', s=50, alpha=0.8
    )
    
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')


def main():
    print("=" * 60)
    print("Topology-Aware Energy Demo")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating 2D moons dataset...")
    X, y = generate_data(n_samples=200)
    print(f"   Shape: {X.shape}")
    
    # Configuration
    k = 8
    latent_dim = 2
    
    # Setup MemoryEnergy
    print("\n2. Setting up MemoryEnergy (arbitrary attractors)...")
    mem_cfg = MemoryEnergyConfig(
        latent_dim=latent_dim,
        n_mem=8,
        beta=5.0,
        init_method='random'
    )
    mem_energy = MemoryEnergy(mem_cfg)
    print(f"   Memory patterns shape: {mem_energy.mem.shape}")
    
    # Setup TopologyEnergy
    print("\n3. Setting up TopologyEnergy (data-driven)...")
    topo_cfg = TopologyEnergyConfig(
        latent_dim=latent_dim,
        k=k,
        mode='agreement',
        continuous=True
    )
    topo_energy = TopologyEnergy(topo_cfg)
    topo_energy.set_target_adjacency(X)
    print(f"   Target adjacency shape: {topo_energy.target_adjacency.shape}")
    
    # Compute energies
    print("\n4. Computing energy values...")
    with torch.no_grad():
        mem_e = mem_energy.energy(X).mean().item()
        topo_e = topo_energy.energy(X).mean().item()
    
    print(f"   MemoryEnergy (mean): {mem_e:.4f}")
    print(f"   TopologyEnergy (mean): {topo_e:.4f}")
    
    # Test on perturbed data
    print("\n5. Testing on perturbed data (simulating learned latent)...")
    X_perturbed = X + 0.05 * torch.randn_like(X)
    
    with torch.no_grad():
        mem_e_pert = mem_energy.energy(X_perturbed).mean().item()
        topo_e_pert = topo_energy.energy(X_perturbed).mean().item()
    
    print(f"   MemoryEnergy (perturbed): {mem_e_pert:.4f}")
    print(f"   TopologyEnergy (perturbed): {topo_e_pert:.4f}")
    
    # Compute k-NN preservation
    print("\n6. Computing k-NN preservation...")
    preservation_orig = compute_knn_preservation(X, X, k=k)
    preservation_pert = compute_knn_preservation(X, X_perturbed, k=k)
    
    print(f"   Original (perfect): {preservation_orig:.4f}")
    print(f"   Perturbed: {preservation_pert:.4f}")
    
    # Visualize
    print("\n7. Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original data
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                   edgecolors='black', s=50, alpha=0.8)
    axes[0].set_title('Original Data (2D Moons)')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    
    # MemoryEnergy landscape
    visualize_energies(X, y, mem_energy, 
                      'MemoryEnergy\n(Arbitrary Attractors)', axes[1],
                      is_topology_energy=False)
    
    # TopologyEnergy landscape (show proximity to data)
    visualize_energies(X, y, topo_energy, 
                      'TopologyEnergy\n(Data-Driven)', axes[2],
                      is_topology_energy=True)
    
    plt.tight_layout()
    output_path = 'outputs/energy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("MemoryEnergy:")
    print("  - Creates fixed attractor basins at learned memory locations")
    print("  - Independent of data topology")
    print("  - Can compete with topology preservation objective")
    print()
    print("TopologyEnergy:")
    print("  - Rewards preservation of k-NN adjacency structure")
    print("  - Aligns with topology preservation objective")
    print("  - Data-driven (no arbitrary attractors)")
    print("  - Should maintain better ARI and trustworthiness")
    print("=" * 60)


if __name__ == '__main__':
    main()
