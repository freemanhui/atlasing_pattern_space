"""
Pytest configuration and shared fixtures for viz tests.
"""

# Import all fixtures to make them discoverable
from .fixtures import (
    mock_energy_2d,
    mock_energy_highdim,
    sample_grid_data,
    sample_landscape
)
