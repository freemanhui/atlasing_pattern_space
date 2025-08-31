import numpy as np

def basin_depth(energies: np.ndarray) -> float:
    """Gap between mean energy and minimum energy."""
    return float(np.mean(energies) - float(np.min(energies)))
