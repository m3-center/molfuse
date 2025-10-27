# ARCHIVE — Deprecated Features, Scripts, and Policies

Purpose: Preserve context for removed or retired approaches so their rationale and impact remain discoverable.

Last Updated: October 26, 2025

## Deprecated in v4 (molfuse)

- Scoring via batched cdist (scipy.spatial.distance.cdist)
  - Status: Replaced by exact 1-NN using scikit-learn NearestNeighbors (KDTree/BallTree) in embedded space.
  - Rationale: Identical nearest-neighbor distances within tolerance at 2–10D; lower memory and faster wall time.
  - Notes: cdist remains as a debugging fallback only.

- ID-based deduplication/aggregation of MF cloud
  - Status: Replaced by SMILES-only deduplication across MF, ZINC, and Actives.
  - Rationale: Avoids implicit weighting from replicated IDs across multiple targets; enforces strict molecule-level uniqueness.
  - Notes: MF–ZINC and Actives–(MF/ZINC) overlaps are removed strictly by SMILES to prevent trivial zero distances.

- Seeded UMAP (fixed random_state)
  - Status: Replaced by seedless UMAP (random_state=None) to enable multi-threaded execution on HPC.
  - Rationale: Parallel speedup outweighs exact reproducibility; replicated 5× per config to estimate variance.
  - Notes: Variance across replicates is reported; PCA remains deterministic.

## Archived scripts and notes

- archived_scripts/ (folder)
  - Contains legacy analysis and orchestration utilities retained for reference. New development occurs under molfuse/.

## Deletion Policy

- When removing code/behavior that affects results, add a short entry here with date, reason, and replacement.
- If assets are moved to archived_scripts/, link or summarize their original purpose.
