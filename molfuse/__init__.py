"""
molfuse v4.0 (scaffold)

Core invariants:
- StandardScaler fitted on MF+ZINC only; actives projected with that scaler.
- UMAP random_state=None to enable parallelism (non-deterministic but fast).
- Exact 1-NN scoring in embedded space; score = -min_distance.
- Affinity cutoff applies to MF cloud for scoring only; actives are not filtered.
- Spearman's rho reported (pActivity vs score) for actives.
"""

__version__ = "4.0.0-dev"
