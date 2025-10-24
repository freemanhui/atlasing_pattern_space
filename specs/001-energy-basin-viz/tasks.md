# Tasks: Interactive Energy Basin Visualization System

**Input**: Design documents from `/specs/001-energy-basin-viz/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/visualizer_api.md, quickstart.md

**Tests**: Tests are included based on plan.md Phase 1 requirements ("Write unit tests for landscape computation" and "Write contract tests for API")

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [ ] T001 Create `src/aps/viz/` directory structure with `__init__.py`
- [ ] T002 Create `src/aps/viz/backends/` directory with `__init__.py`
- [ ] T003 Create `tests/test_viz/` directory with `__init__.py`
- [ ] T004 [P] Add visualization dependencies to pyproject.toml (plotly>=5.0, matplotlib>=3.5, scipy>=1.7)
- [ ] T005 [P] Add optional dependencies group `[visualization]` in pyproject.toml with scikit-learn, ipywidgets
- [ ] T006 Install development dependencies with `pip install -e ".[visualization,dev]"`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures and base infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Create data structures file `src/aps/viz/data_structures.py` with EnergyLandscape dataclass
- [ ] T008 [P] Add MemoryPattern dataclass to `src/aps/viz/data_structures.py`
- [ ] T009 [P] Add Basin dataclass to `src/aps/viz/data_structures.py`
- [ ] T010 [P] Add Point dataclass to `src/aps/viz/data_structures.py`
- [ ] T011 [P] Add Trajectory dataclass to `src/aps/viz/data_structures.py`
- [ ] T012 [P] Add CrossSection dataclass to `src/aps/viz/data_structures.py`
- [ ] T013 [P] Create VisualizationConfig dataclass in `src/aps/viz/config.py` with validation method
- [ ] T014 [P] Create utility functions file `src/aps/viz/utils.py` with grid generation helper
- [ ] T015 Create test fixtures in `tests/test_viz/fixtures.py` for mock MemoryEnergy modules
- [ ] T016 [P] Write unit tests for data structure serialization in `tests/test_viz/test_data_structures.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Static Energy Landscape Visualization (Priority: P1) 🎯 MVP

**Goal**: Enable researchers to visualize energy landscapes as 2D heatmaps with memory pattern markers in under 5 seconds

**Independent Test**: Load a trained model with MemoryEnergy, call `compute_landscape()` and `plot_heatmap()`, verify that memory patterns appear as low-energy basins and visualization completes in < 5 seconds

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T017 [P] [US1] Contract test for EnergyLandscapeVisualizer initialization in `tests/test_viz/test_visualizer.py`
- [ ] T018 [P] [US1] Contract test for compute_landscape() method in `tests/test_viz/test_visualizer.py`
- [ ] T019 [P] [US1] Unit test for grid bounds inference in `tests/test_viz/test_utils.py`
- [ ] T020 [P] [US1] Unit test for energy computation accuracy (<1% error) in `tests/test_viz/test_visualizer.py`
- [ ] T021 [P] [US1] Performance test for 100x100 grid (<5 seconds) in `tests/test_viz/test_visualizer.py`

### Implementation for User Story 1

- [ ] T022 [US1] Implement EnergyLandscapeVisualizer.__init__() in `src/aps/viz/visualizer.py`
- [ ] T023 [US1] Implement compute_landscape() method in `src/aps/viz/visualizer.py`
- [ ] T024 [US1] Add grid generation logic to `src/aps/viz/utils.py` (create_grid function)
- [ ] T025 [US1] Add bounds inference logic to `src/aps/viz/utils.py` (infer_bounds function)
- [ ] T026 [US1] Add energy interpolation helper to `src/aps/viz/utils.py` (interpolate_energy function)
- [ ] T027 [US1] Create Plotly backend base class in `src/aps/viz/backends/plotly_backend.py`
- [ ] T028 [US1] Implement plot_heatmap() for Plotly in `src/aps/viz/backends/plotly_backend.py`
- [ ] T029 [US1] Add memory pattern marker overlay to Plotly backend
- [ ] T030 [US1] Implement plot_heatmap() wrapper in `src/aps/viz/visualizer.py` that delegates to backend
- [ ] T031 [US1] Add exports to `src/aps/viz/__init__.py` (EnergyLandscapeVisualizer, VisualizationConfig)
- [ ] T032 [US1] Verify all US1 tests pass and performance meets SC-001 (<5s)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - researchers can generate static heatmap visualizations

---

## Phase 4: User Story 2 - Interactive Point Exploration (Priority: P2)

**Goal**: Enable researchers to click/hover on points to see energy, nearest memory pattern, and basin membership in <100ms

**Independent Test**: Click various points in a visualization and verify tooltips show correct energy values, memory pattern IDs, and basin IDs with <100ms response time

### Tests for User Story 2

- [ ] T033 [P] [US2] Contract test for InteractionHandler in `tests/test_viz/test_interactions.py`
- [ ] T034 [P] [US2] Performance test for hover response (<100ms) in `tests/test_viz/test_interactions.py`
- [ ] T035 [P] [US2] Performance test for click response (<100ms) in `tests/test_viz/test_interactions.py`
- [ ] T036 [P] [US2] Unit test for basin depth calculation in `tests/test_viz/test_utils.py`

### Implementation for User Story 2

- [ ] T037 [P] [US2] Create InteractionHandler class in `src/aps/viz/interactions.py`
- [ ] T038 [US2] Implement on_hover() method in `src/aps/viz/interactions.py`
- [ ] T039 [US2] Implement on_click() method in `src/aps/viz/interactions.py`
- [ ] T040 [US2] Implement identify_basins() method in `src/aps/viz/visualizer.py`
- [ ] T041 [US2] Add basin identification logic using gradient descent in `src/aps/viz/utils.py`
- [ ] T042 [US2] Add basin_depth() calculation function to `src/aps/viz/utils.py`
- [ ] T043 [US2] Add hover tooltips to Plotly backend in `src/aps/viz/backends/plotly_backend.py`
- [ ] T044 [US2] Add click event handling to Plotly backend in `src/aps/viz/backends/plotly_backend.py`
- [ ] T045 [US2] Implement compare mode for multiple clicked points in `src/aps/viz/interactions.py`
- [ ] T046 [US2] Update `src/aps/viz/__init__.py` exports to include InteractionHandler
- [ ] T047 [US2] Verify all US2 tests pass and performance meets SC-002 (<100ms)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - researchers can explore landscapes interactively

---

## Phase 5: User Story 3 - Trajectory Visualization (Priority: P3)

**Goal**: Enable researchers to visualize gradient descent trajectories to understand basin dynamics in <2 seconds per trajectory

**Independent Test**: Select starting points, run add_trajectory(), verify trajectories converge to memory patterns and complete in <2 seconds

### Tests for User Story 3

- [ ] T048 [P] [US3] Contract test for add_trajectory() method in `tests/test_viz/test_visualizer.py`
- [ ] T049 [P] [US3] Performance test for trajectory computation (<2s) in `tests/test_viz/test_visualizer.py`
- [ ] T050 [P] [US3] Unit test for trajectory convergence validation in `tests/test_viz/test_utils.py`
- [ ] T051 [P] [US3] Integration test for multiple trajectories in `tests/test_viz/test_visualizer.py`

### Implementation for User Story 3

- [ ] T052 [P] [US3] Implement gradient descent simulation in `src/aps/viz/utils.py` (simulate_trajectory function)
- [ ] T053 [US3] Implement add_trajectory() method in `src/aps/viz/visualizer.py`
- [ ] T054 [US3] Add trajectory rendering to Plotly backend in `src/aps/viz/backends/plotly_backend.py`
- [ ] T055 [US3] Implement color-coding by destination basin in Plotly backend
- [ ] T056 [US3] Add trajectory hover interactions showing energy and gradient magnitude
- [ ] T057 [US3] Implement on_trajectory_request() in `src/aps/viz/interactions.py`
- [ ] T058 [US3] Add convergence detection logic to trajectory simulation
- [ ] T059 [US3] Add support for rendering multiple trajectories without obscuring landscape
- [ ] T060 [US3] Verify all US3 tests pass and performance meets SC-003 (<2s)

**Checkpoint**: All three user stories (1, 2, 3) should now work independently - researchers can visualize trajectories and basin dynamics

---

## Phase 6: User Story 4 - Cross-Sectional Views (Priority: P4)

**Goal**: Enable researchers to generate 1D energy profiles along arbitrary lines for detailed basin analysis

**Independent Test**: Draw a line across the visualization, call plot_cross_section(), verify 1D plot shows correct energy profile and basin crossings

### Tests for User Story 4

- [ ] T061 [P] [US4] Contract test for plot_cross_section() method in `tests/test_viz/test_visualizer.py`
- [ ] T062 [P] [US4] Unit test for line sampling logic in `tests/test_viz/test_utils.py`
- [ ] T063 [P] [US4] Unit test for basin crossing detection in `tests/test_viz/test_utils.py`
- [ ] T064 [P] [US4] Integration test for linked 1D-2D highlighting in `tests/test_viz/test_visualizer.py`

### Implementation for User Story 4

- [ ] T065 [P] [US4] Implement line sampling function in `src/aps/viz/utils.py` (sample_line function)
- [ ] T066 [P] [US4] Implement basin crossing detection in `src/aps/viz/utils.py`
- [ ] T067 [US4] Implement plot_cross_section() method in `src/aps/viz/visualizer.py`
- [ ] T068 [US4] Create 1D plot rendering for cross-sections in Plotly backend
- [ ] T069 [US4] Add linked highlighting between 1D plot and 2D landscape
- [ ] T070 [US4] Add CrossSection.to_dataframe() method in `src/aps/viz/data_structures.py`
- [ ] T071 [US4] Add cross-section export functionality
- [ ] T072 [US4] Verify all US4 tests pass

**Checkpoint**: All four user stories should now be independently functional - complete feature set for energy landscape exploration

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Production-ready features and optimizations affecting multiple user stories

- [ ] T073 [P] Implement export() method in `src/aps/viz/visualizer.py` for HTML format (SC-007)
- [ ] T074 [P] Add PNG export support to `src/aps/viz/visualizer.py`
- [ ] T075 [P] Add SVG export support to `src/aps/viz/visualizer.py`
- [ ] T076 [P] Create Matplotlib backend in `src/aps/viz/backends/mpl_backend.py` for static plots
- [ ] T077 [P] Implement high-dimensional support with PCA projection in `src/aps/viz/utils.py` (SC-004)
- [ ] T078 [P] Implement high-dimensional support with t-SNE projection in `src/aps/viz/utils.py` (SC-004)
- [ ] T079 [P] Add logarithmic color scale option to VisualizationConfig
- [ ] T080 [P] Implement adaptive grid resolution in `src/aps/viz/visualizer.py`
- [ ] T081 [P] Add EnergyLandscape serialization (.npz format) to `src/aps/viz/data_structures.py`
- [ ] T082 [P] Add basin clustering for >20 memory patterns
- [ ] T083 [P] Write integration tests for export functionality in `tests/test_viz/test_backends.py`
- [ ] T084 [P] Performance optimization: vectorize energy computation in compute_landscape()
- [ ] T085 [P] Memory optimization: implement chunked computation for large grids
- [ ] T086 [P] Add usage examples to docstrings matching quickstart.md patterns
- [ ] T087 Create example script in `scripts/demo_energy_viz.py` demonstrating all user stories
- [ ] T088 Update main README.md with visualization module documentation
- [ ] T089 Run complete quickstart.md validation workflow
- [ ] T090 Final performance validation: verify all success criteria (SC-001 through SC-007)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 (uses compute_landscape) but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Uses US1 landscape and US2 interactions but independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Uses US1 landscape but independently testable

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data structures before utilities
- Utilities before visualizer methods
- Core implementation before backend integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T004, T005)
- All Foundational data structure tasks marked [P] can run in parallel (T008-T012, T013, T014, T016)
- Once Foundational phase completes, all user stories (US1-US4) can start in parallel if team capacity allows
- All tests within a user story marked [P] can run in parallel
- Polish tasks marked [P] can run in parallel (T073-T086)
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task T017: "Contract test for EnergyLandscapeVisualizer initialization"
Task T018: "Contract test for compute_landscape() method"
Task T019: "Unit test for grid bounds inference"
Task T020: "Unit test for energy computation accuracy"
Task T021: "Performance test for 100x100 grid"

# After tests written, launch independent implementation tasks:
Task T024: "Add grid generation logic" (utils.py)
Task T025: "Add bounds inference logic" (utils.py)
Task T026: "Add energy interpolation helper" (utils.py)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently with quickstart.md examples
5. Demo capability to researchers

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP! - Static visualization)
3. Add User Story 2 → Test independently → Deploy/Demo (Interactivity added)
4. Add User Story 3 → Test independently → Deploy/Demo (Trajectories added)
5. Add User Story 4 → Test independently → Deploy/Demo (Cross-sections added)
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (T017-T032)
   - Developer B: User Story 2 (T033-T047)
   - Developer C: User Story 3 (T048-T060)
   - Developer D: User Story 4 (T061-T072)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability (US1=P1, US2=P2, US3=P3, US4=P4)
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD approach per plan.md)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All tasks include specific file paths as required
- Performance contracts from visualizer_api.md must be validated at story completion
