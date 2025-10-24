# Implementation Plan: Interactive Energy Basin Visualization System

**Branch**: `001-energy-basin-viz` | **Date**: 2025-10-23 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `specs/001-energy-basin-viz/spec.md`

## Summary

Create an interactive visualization system for energy basin landscapes in latent space. The system will compute energy values across a 2D grid, identify attractor basins around memory patterns, and provide interactive exploration tools including point inspection, trajectory visualization, and cross-sectional analysis. The MVP (P1) focuses on static 2D heatmap visualization with memory pattern markers, delivering immediate value for understanding model behavior.

## Technical Context

**Language/Version**: Python 3.9+  
**Primary Dependencies**: 
- PyTorch (existing, for energy computation and gradients)
- NumPy (grid generation, numerical operations)
- Plotly (primary visualization backend for interactivity)
- Matplotlib (alternative/static backend)
- scikit-learn (optional, for PCA/t-SNE dimensionality reduction)

**Storage**: File-based (.npz format for caching computed landscapes)  
**Testing**: pytest with fixtures for trained models  
**Target Platform**: Local development (macOS/Linux), Jupyter notebooks, Python scripts  
**Project Type**: Single library package (adds `aps.viz` module to existing APS package)  
**Performance Goals**:
- Landscape computation: < 5 seconds for 100x100 grid (SC-001)
- Interactive hover/click: < 100ms response (SC-002)
- Trajectory simulation: < 2 seconds (SC-003)
- Energy accuracy: < 1% error vs MemoryEnergy ground truth (SC-005)

**Constraints**:
- Must integrate with existing `aps.energy.MemoryEnergy` module
- No modification to core APS package structure
- Jupyter notebook compatibility required
- Memory efficient for grids up to 200x200

**Scale/Scope**:
- Support latent dimensions from 2 to ~10 (with projection)
- Handle up to 50 memory patterns
- Reasonable performance on laptop CPUs

## Constitution Check

*Note: Constitution file contains template placeholders only. Proceeding with standard best practices.*

**Assumed Principles** (pending constitution definition):
- ✅ **Library-First**: Adding `aps.viz` as self-contained visualization module
- ✅ **Test-First**: Will write tests for core computation and API contracts before implementation
- ✅ **Clear Purpose**: Focused on energy landscape visualization, no scope creep
- ✅ **Observability**: Visualizations themselves provide debugging insight

**Re-check after implementation**: Verify test coverage and library independence.

## Project Structure

### Documentation (this feature)

```text
specs/001-energy-basin-viz/
├── spec.md              # Feature specification (COMPLETE)
├── plan.md              # This file (COMPLETE)
├── data-model.md        # Entity and relationship design (COMPLETE)
├── quickstart.md        # Usage guide and examples (COMPLETE)
├── contracts/
│   └── visualizer_api.md  # API contracts (COMPLETE)
└── checklists/
    └── requirements.md   # Quality validation checklist (COMPLETE)
```

### Source Code (repository root)

Adding new `aps.viz` module to existing `src/aps/` structure:

```text
src/aps/
├── energy/              # Existing: MemoryEnergy and related
├── topology/            # Existing: kNN topology loss
├── causality/           # Existing: causal learning
├── metrics/             # Existing: evaluation metrics
├── utils/               # Existing: data generation, visualization helpers
└── viz/                 # NEW: Energy basin visualization
    ├── __init__.py      # Export main classes
    ├── visualizer.py    # EnergyLandscapeVisualizer (main API)
    ├── data_structures.py  # EnergyLandscape, Basin, Trajectory, etc.
    ├── interactions.py  # InteractionHandler for hover/click
    ├── config.py        # VisualizationConfig
    ├── backends/
    │   ├── __init__.py
    │   ├── plotly_backend.py   # Plotly implementation
    │   └── mpl_backend.py      # Matplotlib implementation
    └── utils.py         # Helper functions (interpolation, basin identification)

tests/
├── test_energy.py       # Existing
├── test_topology.py     # Existing
└── test_viz/            # NEW: Tests for visualization module
    ├── __init__.py
    ├── test_visualizer.py     # Test EnergyLandscapeVisualizer API
    ├── test_data_structures.py  # Test data classes
    ├── test_interactions.py   # Test hover/click handlers
    ├── test_backends.py       # Test rendering backends
    └── fixtures.py            # Shared test fixtures (mock models)
```

**Structure Decision**: Single project structure (Option 1) selected. Adding `aps.viz` as new subpackage within existing `src/aps/` follows the established pattern of `aps.energy`, `aps.topology`, etc. No web/mobile components needed - this is a library for local visualization in notebooks/scripts.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     aps.viz.visualizer                          │
│                 EnergyLandscapeVisualizer                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ - compute_landscape()                                     │ │
│  │ - plot_heatmap()                                          │ │
│  │ - identify_basins()                                       │ │
│  │ - add_trajectory()                                        │ │
│  │ - plot_cross_section()                                    │ │
│  │ - export()                                                │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌───▼───┐  ┌────▼────────┐
    │ aps.    │  │ Data  │  │ Backends    │
    │ energy  │  │ Struct│  │ (Plotly/MPL)│
    └─────────┘  └───────┘  └─────────────┘
         │
    MemoryEnergy
```

### Key Design Decisions

1. **Backend Abstraction**: Support both Plotly (interactive) and Matplotlib (static) via pluggable backends
2. **Lazy Computation**: Only compute basins when explicitly requested, not during landscape computation
3. **Caching**: EnergyLandscape objects serializable to .npz for reuse
4. **Projection Strategy**: For high-dim spaces, apply PCA/t-SNE to memory patterns first, then compute energy on projection

### Data Flow

```
1. User loads trained model → Extract MemoryEnergy module
2. Create EnergyLandscapeVisualizer(energy_module, latent_dim)
3. compute_landscape() → Generate 2D grid → Compute energy at each point
4. plot_heatmap(landscape) → Pass to backend → Render interactive figure
5. (Optional) identify_basins(landscape) → Run gradient descent from grid points
6. (Optional) add_trajectory(start) → Simulate path → Overlay on plot
```

## Implementation Phases

### Phase 0: Research & Setup (Already Complete)

✅ Reviewed existing `aps.energy.MemoryEnergy` interface  
✅ Confirmed PyTorch gradient computation available  
✅ Identified Plotly as best interactive visualization library  
✅ Verified dimensionality reduction approach (sklearn PCA/t-SNE)

### Phase 1: Core Infrastructure (MVP - User Story P1)

**Goal**: Static 2D heatmap visualization with memory pattern markers

**Tasks**:
1. Create `aps.viz` module structure
2. Implement `EnergyLandscape` data structure
3. Implement `compute_landscape()` method
4. Implement Plotly backend for heatmap rendering
5. Implement `plot_heatmap()` with memory pattern markers
6. Write unit tests for landscape computation
7. Write contract tests for API

**Deliverable**: Researchers can visualize energy landscapes in < 5 seconds

### Phase 2: Interactivity (User Story P2)

**Goal**: Click/hover exploration of energy landscape

**Tasks**:
1. Implement `InteractionHandler` class
2. Add hover event handling (tooltip with energy value)
3. Add click event handling (detailed point info)
4. Implement basin depth calculation
5. Add "compare mode" for multiple clicked points
6. Write interaction tests

**Deliverable**: Interactive exploration with < 100ms response

### Phase 3: Trajectory Visualization (User Story P3)

**Goal**: Gradient descent trajectory overlay

**Tasks**:
1. Implement `Trajectory` data structure
2. Implement `add_trajectory()` method with gradient descent simulation
3. Add trajectory rendering to backends
4. Implement color-coding by destination basin
5. Add trajectory hover interactions
6. Write trajectory simulation tests

**Deliverable**: Trajectories complete in < 2 seconds

### Phase 4: Cross-Sectional Analysis (User Story P4)

**Goal**: 1D energy profile slices

**Tasks**:
1. Implement `CrossSection` data structure
2. Implement `plot_cross_section()` method
3. Add line sampling logic
4. Implement linked highlighting (click on 1D plot → highlight on 2D)
5. Add basin crossing detection
6. Write cross-section tests

**Deliverable**: 1D profiles for detailed basin analysis

### Phase 5: Polish & Export

**Goal**: Production-ready features

**Tasks**:
1. Implement `export()` method (HTML, PNG, SVG)
2. Add high-dimensional support (PCA/t-SNE projection)
3. Implement adaptive grid resolution
4. Add logarithmic color scale option
5. Add basin clustering for >20 memory patterns
6. Performance optimization
7. Documentation and examples

**Deliverable**: Complete feature with all P1-P4 user stories

## Dependencies

### Internal (Existing APS Modules)
- `aps.energy.MemoryEnergy`: Core energy computation
- `aps.utils`: Potentially reuse scatter_labels() or similar utilities

### External (New Dependencies)
- `plotly>=5.0`: Interactive visualizations
- `matplotlib>=3.5`: Alternative static backend
- `scipy>=1.7`: For interpolation and optimization
- `scikit-learn>=1.0`: Optional, for PCA/t-SNE
- `ipywidgets>=8.0`: Optional, for Jupyter interactivity enhancements

### Development
- `pytest>=7.0`: Testing framework
- `pytest-mock`: Mocking trained models in tests

## Testing Strategy

### Unit Tests
- Energy landscape computation correctness
- Data structure serialization/deserialization
- Grid generation and bounds inference
- Basin identification algorithm
- Trajectory gradient descent convergence

### Contract Tests
- All public API methods match contracts
- Performance benchmarks (< 5s landscape, < 100ms interaction, < 2s trajectory)
- Energy accuracy < 1% error

### Integration Tests
- End-to-end visualization workflow
- Backend rendering (Plotly and Matplotlib)
- Export functionality (HTML preserves interactivity, PNG/SVG quality)

### Test Fixtures
- Mock trained models with known MemoryEnergy configurations
- Synthetic energy landscapes with known basin locations
- Pre-computed trajectories for validation

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Energy computation too slow for dense grids | High | Implement adaptive resolution, start coarse |
| High-dimensional projection loses basin structure | Medium | Provide projection quality metrics, allow manual bounds |
| Plotly interactivity not responsive enough | Medium | Optimize data transfer, use decimation for large grids |
| Basin identification ambiguous at boundaries | Low | Document limitations, provide convergence tolerance tuning |
| Memory patterns too cluttered (>20) | Low | Implement clustering/grouping in Phase 5 |

## Success Metrics

Aligned with Success Criteria from spec.md:

- **SC-001**: ✅ Landscape computation < 5 seconds (100x100 grid)
- **SC-002**: ✅ Interaction response < 100ms
- **SC-003**: ✅ Trajectory computation < 2 seconds
- **SC-004**: ✅ Support latent dim up to 10
- **SC-005**: ✅ Energy accuracy < 1% error
- **SC-006**: ✅ 90% usability (basin identification in 30 seconds)
- **SC-007**: ✅ Export quality (HTML interactive, PNG/SVG high-res)

## Next Steps

After plan approval:

1. Run `/speckit.tasks` to generate actionable task breakdown
2. Begin Phase 1 implementation (MVP - P1 user story)
3. Write tests first for core computation logic
4. Implement and validate against performance contracts
