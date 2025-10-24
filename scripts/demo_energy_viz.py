#!/usr/bin/env python3
"""
Comprehensive demo of energy basin visualization features.

Demonstrates all four user stories:
1. Static energy landscape visualization
2. Interactive point exploration
3. Trajectory visualization  
4. Cross-sectional analysis
"""

import torch
from aps.energy import MemoryEnergy, MemoryEnergyConfig
from aps.viz import EnergyLandscapeVisualizer

def main():
    print("=" * 70)
    print("Energy Basin Visualization Demo")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Setup: Create a simple memory energy model
    # ========================================================================
    print("1. Setting up energy model...")
    config = MemoryEnergyConfig(latent_dim=2, n_mem=4, beta=5.0)
    energy_module = MemoryEnergy(config)
    
    # Initialize memory patterns at corners of a square
    with torch.no_grad():
        energy_module.mem.copy_(torch.tensor([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0]
        ]))
    
    print(f"   ✓ Created MemoryEnergy with {config.n_mem} patterns")
    print()
    
    # ========================================================================
    # User Story 1: Static Energy Landscape Visualization
    # ========================================================================
    print("2. Computing energy landscape (User Story 1)...")
    viz = EnergyLandscapeVisualizer(
        energy_module=energy_module,
        latent_dim=2,
        resolution=100
    )
    
    landscape = viz.compute_landscape(bounds=(-2.0, 2.0, -2.0, 2.0))
    print(f"   ✓ Computed {landscape.grid_x.shape[0]}x{landscape.grid_x.shape[1]} grid")
    print(f"   ✓ Found {len(landscape.memory_patterns)} memory patterns")
    print()
    
    # Export landscape
    print("3. Exporting visualizations...")
    try:
        viz.export(landscape, 'outputs/energy_landscape', format='html')
        print("   ✓ Exported HTML: outputs/energy_landscape.html")
    except Exception as e:
        print(f"   ⚠ HTML export: {e}")
    
    print()
    
    # ========================================================================
    # User Story 2: Interactive Point Exploration
    # ========================================================================
    print("4. Identifying energy basins (User Story 2)...")
    basins = viz.identify_basins(landscape, num_samples=50)
    
    print(f"   ✓ Identified {len(basins)} basins")
    for basin in basins:
        print(f"     Basin {basin.pattern_id}: depth={basin.depth:.3f}, "
              f"center={basin.center}")
    print()
    
    # Demonstrate point inspection
    print("5. Inspecting points...")
    from aps.viz import InteractionHandler
    
    handler = InteractionHandler()
    test_points = [(0.0, 0.0), (0.8, 0.8), (-0.8, -0.8)]
    
    for x, y in test_points:
        point_info = handler.on_hover(x, y, landscape, basins)
        print(f"   Point ({x:.1f}, {y:.1f}): "
              f"energy={point_info.energy:.3f}, "
              f"nearest_pattern={point_info.nearest_pattern_id}")
    print()
    
    # ========================================================================
    # User Story 3: Trajectory Visualization
    # ========================================================================
    print("6. Simulating gradient descent trajectories (User Story 3)...")
    
    # Create trajectories from different starting points
    start_points = [
        (1.5, 1.5),
        (-1.5, -1.5),
        (0.3, -0.3)
    ]
    
    trajectories = []
    for i, start in enumerate(start_points):
        traj = viz.add_trajectory(landscape, start)
        trajectories.append(traj)
        print(f"   Trajectory {i+1}: {traj.num_steps} steps, "
              f"converged={traj.converged}, "
              f"basin={traj.destination_basin_id}")
    print()
    
    # ========================================================================
    # User Story 4: Cross-Sectional Analysis
    # ========================================================================
    print("7. Creating cross-sectional views (User Story 4)...")
    
    # Horizontal cross-section
    cross_section = viz.plot_cross_section(
        landscape,
        start_point=(-1.5, 0.0),
        end_point=(1.5, 0.0),
        num_samples=50,
        basins=basins
    )
    
    print(f"   ✓ Cross-section with {len(cross_section.sample_points)} points")
    print(f"   ✓ Basins crossed: {cross_section.basins_crossed}")
    
    # Export to DataFrame
    df = cross_section.to_dataframe()
    print(f"   ✓ Converted to DataFrame: {df.shape}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Summary of features demonstrated:")
    print("  ✓ Static energy landscape computation")
    print("  ✓ Basin identification via gradient descent")
    print("  ✓ Interactive point inspection")
    print("  ✓ Trajectory simulation and convergence")
    print("  ✓ Cross-sectional energy profiles")
    print("  ✓ Data export (HTML, DataFrame)")
    print()
    print("Check outputs/ directory for exported visualizations.")
    print()

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()
