# Atlasing Pattern Space (APS)

A reference code framework for **pattern-space embeddings** that integrate **Topology (T)**,
**Causality (C)**, and **Energy/Attractors (E)** — a practical instantiation of the
*Atlasing Pattern Space* idea (beyond token-space).

## Modules
- `aps.topology` — kNN-based topology-preserving loss and a minimal Topological Autoencoder.
- `aps.causality` — IRM-style invariance penalty and HSIC independence utility.
- `aps.energy` — Memory-based energy (log-sum-exp) to shape basins/attractors in latent space.
- `aps.metrics` — kNN preservation, trustworthiness, basin depth, pointer strength.
- `aps.utils` — toy corpus, PPMI co-occurrence, SVD embeddings, quick plotting.

## Quickstart
```bash
pip install -e .
python scripts/run_topo_ae.py --latent 2 --epochs 100 --topo-weight 1.0
python scripts/run_energy_demo.py
```

## Combined objective (sketch)
\[
\min_\theta\; L_{task} + \lambda_T L_{topo} + \lambda_C L_{causal} + \lambda_G L_{energy}.
\]

Install optional extras for PH or causal discovery:
```bash
pip install -e ".[topology,causality,dev]"
```

MIT License.
