#!/usr/bin/env python
"""
Validation script for Energy Basin Visualization MVP (User Story 1).

Tests the basic usage from quickstart.md to ensure MVP delivers value.
"""

import sys
import time
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aps.viz import EnergyLandscapeVisualizer, VisualizationConfig

# Import shared test fixture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_viz.fixtures import MockMemoryEnergy


def main():
    """Run MVP validation based on quickstart.md."""
    print("=" * 60)
    print("Energy Basin Visualization MVP Validation")
    print("User Story 1: Static Energy Landscape Visualization")
    print("=" * 60)
    print()
    
    # Step 1: Load trained model (use mock for validation)
    print("Step 1: Loading trained model with MemoryEnergy...")
    energy_module = MockMemoryEnergy(n_mem=4, latent_dim=2, beta=5.0)
    latent_dim = 2
    print(f"  ✓ Loaded model with {energy_module.n_mem} memory patterns")
    print(f"  ✓ Latent dimensionality: {latent_dim}")
    print()
    
    # Step 2: Create Visualizer
    print("Step 2: Creating EnergyLandscapeVisualizer...")
    viz = EnergyLandscapeVisualizer(
        energy_module=energy_module,
        latent_dim=latent_dim,
        resolution=100  # 100x100 grid
    )
    print(f"  ✓ Visualizer created with {viz.resolution}x{viz.resolution} resolution")
    print()
    
    # Step 3: Compute and Plot Energy Landscape
    print("Step 3: Computing energy landscape...")
    start_time = time.time()
    landscape = viz.compute_landscape()
    elapsed = time.time() - start_time
    
    print(f"  ✓ Landscape computed in {elapsed:.2f}s")
    print(f"  ✓ Performance requirement: < 5s (SC-001) {'✓ PASS' if elapsed < 5.0 else '✗ FAIL'}")
    print(f"  ✓ Grid shape: {landscape.grid_x.shape}")
    print(f"  ✓ Energy values shape: {landscape.energy_values.shape}")
    print(f"  ✓ Memory patterns: {len(landscape.memory_patterns)}")
    print(f"  ✓ Bounds: {landscape.bounds}")
    print()
    
    # Verify basins are visible
    print("Step 4: Verifying basins are visible...")
    avg_energy = landscape.energy_values.mean()
    min_energy = landscape.energy_values.min()
    max_energy = landscape.energy_values.max()
    
    print(f"  ✓ Energy range: [{min_energy:.3f}, {max_energy:.3f}]")
    print(f"  ✓ Average energy: {avg_energy:.3f}")
    
    # Check that memory patterns are at low-energy regions
    basins_visible = True
    for mp in landscape.memory_patterns:
        if mp.energy >= avg_energy:
            basins_visible = False
            print(f"  ✗ Pattern {mp.id} not a basin (energy {mp.energy:.3f} >= avg)")
        else:
            print(f"  ✓ Pattern {mp.id} at {mp.position_2d}: energy {mp.energy:.3f} (basin)")
    
    print(f"\n  Basins visible as low-energy regions: {'✓ PASS' if basins_visible else '✗ FAIL'}")
    print()
    
    # Step 5: Create heatmap visualization
    print("Step 5: Creating heatmap visualization...")
    config = VisualizationConfig(show_memory_markers=True)
    fig = viz.plot_heatmap(landscape, config=config)
    
    print(f"  ✓ Heatmap created with {len(fig.data)} traces")
    print(f"  ✓ Memory markers overlay: {'✓ enabled' if config.show_memory_markers else '✗ disabled'}")
    print(f"  ✓ Figure type: {type(fig).__name__}")
    print()
    
    # Summary
    print("=" * 60)
    print("MVP VALIDATION SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Model loading", True),
        ("Visualizer initialization", True),
        ("Landscape computation", True),
        ("Performance < 5s (SC-001)", elapsed < 5.0),
        ("Basins visible", basins_visible),
        ("Heatmap generation", fig is not None),
        ("Memory markers overlay", len(fig.data) >= 2)
    ]
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for check_name, status in checks:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {check_name}")
    
    print()
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 MVP VALIDATION SUCCESSFUL!")
        print("User Story 1 delivers expected value:")
        print("  - Researchers can visualize energy landscapes in < 5s")
        print("  - Memory patterns marked as basin centers")
        print("  - Interactive heatmap ready for exploration")
        return 0
    else:
        print(f"\n✗ MVP VALIDATION FAILED ({passed}/{total} passed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
