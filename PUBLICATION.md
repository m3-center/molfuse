# molfuse (UMMBAS v4) — Publication Synthesis

Last Updated: October 26, 2025

## Abstract

We present MolFuSE, an ultra-large, molecular function–guided virtual screening framework that learns a similarity space from a broad “MF cloud” (compounds active on proteins sharing the target’s molecular function) together with >1.29M ZINC decoys. The design follows a strict projection-only strategy: dimensionality reduction (PCA or UMAP) is fit on MF+ZINC, then held-out target ligands are projected and ranked by exact 1-NN distance to the MF cloud. UMAP runs are seedless (random_state=None) to enable parallelism; deduplication and overlap removal are enforced strictly by SMILES.

## v4 (molfuse) Invariants and Representation Details

- Projection-only: scaler/DR fit on MF+ZINC; held-out actives are only projected (no leakage).
- UMAP is seedless (random_state=None) for HPC parallel execution; PCA uses sklearn defaults.
- Scoring uses exact 1-NN in embedded space; distance-based score = -min_distance.
- Deduplication is SMILES-only; MF–ZINC and Actives–(MF/ZINC) overlaps are removed by SMILES.
- Affinity cutoff applies to MF cloud for scoring only; actives are never filtered by cutoff.
- Representations and distances:
  - Features: RDKit descriptors scaled with StandardScaler; PCA/UMAP with Euclidean distance.
  - Fingerprints: 2048-bit ECFP4; UMAP with Jaccard distance; PCA treated as a baseline.
- Replicates: 5 independent runs per configuration to quantify variability.

### RDKit Feature Descriptors Used (features representation)

The features representation uses the following descriptor columns (extracted from dataset headers without loading full matrices):

- DipoleMoment, ABC, nAcid, nBase, nAromAtom, nAtom, nH, nC, nN, nO, nS, nP, nX
- nBonds, nBondsO, nBondsS, nBondsD, nBondsT, nBondsA, nBondsM, nBondsKS, nBondsKD
- EState_VSA7, nHBAcc, nHBDon, Lipinski, apol, bpol
- nRing, n3Ring, n4Ring, n5Ring, n6Ring, n7Ring, n8Ring, nRot
- Diameter, TopoShapeIndex, Vabc, MW

Non-feature metadata columns (e.g., SMILES, accession, IDs) are excluded from modeling.

## Standing Research Questions and Primary Hypotheses

- RQ1: How does MF cloud composition (size, affinity cutoff) determine the relative performance of PCA vs UMAP?
  - H1: With small or no MF cloud, UMAP’s local structure preservation is advantageous; with large MF clouds, PCA’s global variance dominates.
- RQ2: What UMAP hyperparameter regimes are optimal for retrieval under clean (deduplicated) MF data?
  - H2: The optimal neighborhood size shifts to medium/large values (50–500) after deduplication; min_dist has a weak effect.
- RQ3: To what extent do results generalize across proteins sharing or not sharing the same MF?
  - H3: Same-MF transfer is stronger than cross-MF; dimensionality choice remains target- and MF-dependent.

## Notes for Methods

- Projection-only design to avoid data leakage: fit on MF+ZINC, project held-out actives.
- Features: 40 RDKit descriptors with StandardScaler; Fingerprints: 2048-bit ECFP4.
- Metrics: EF@1% primary; ROC-AUC and PR-AUC secondary; 5 replicates per config.

## Next Updates (Planned)

- Full Phase 1 rerun on deduplicated MF cloud, with expanded UMAP nn grid (10, 20, 50, 100, 500).
- Integrity assertion in analysis stage to ensure MF cloud row counts match deduplicated source.
- Phase 2–4 sequencing preserved (cutoff → MF ablation → generalization), using clean Phase 1 outputs.

### Descriptor-space evaluation (planned)

We will run an independent test comparing the current 40-feature subset against the full Mordred feature sets:

- Full 2D descriptors vs Full 2D+3D descriptors (3D via RDKit ETKDG + MMFF/UFF)
- Scoring by centroid proximity to actives in feature space; EF@1% as primary metric

Hypothesis: 2D+3D will improve EF@1% by capturing 3D shape/electronic effects. If confirmed, we will consider integrating a curated subset of high-signal 3D descriptors into the v4 features list to balance performance and compute cost.

Methods addendum (independent evaluation): To ensure fair comparisons and maximize usable data, we apply coverage-aware selection before imputation/scaling: chemistry-aware pre-pruning of rare-element E-state families, per-feature prevalence thresholds prioritized for targets (pf_target) and MF cloud (pf_mf), and per-row completeness thresholds with set-specific guards (targets protected, MF moderate, ZINC harshest). We also provide a threshold sweep to visualize Pareto trade-offs between row and column retention.

## Data generation note (Mordred features)

For reproducible feature generation at scale, we provide two dataset recreation paths that rebuild full Mordred 2D and 2D+3D descriptor CSVs (preserving metadata columns) and filter matching ECFP4 fingerprint CSVs:

- Baseline: single-process, chunked computation
- Alternative (HPC): multiprocessing with chunked IO, per-process 3D embedding (ETKDG), and stable schema across chunks

Both produce parallel directory trees: `datasets_2d_all/` and `datasets_2d3d_all/`. The HPC path is recommended for large end-to-end regenerations (>10^6 molecules) and was designed to keep memory bounded via chunk and batch sizing while exploiting process-level parallelism.
