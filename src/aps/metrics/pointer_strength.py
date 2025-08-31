def pointer_strength(topology_score: float, causal_score: float, basin_score: float,
                     w_t: float = 1.0, w_c: float = 1.0, w_e: float = 1.0) -> float:
    return float(w_t*topology_score + w_c*causal_score + w_e*basin_score) / (w_t + w_c + w_e)
