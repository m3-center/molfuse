# UMMBAS v3.0 — Publication Synthesis

Last Updated: October 26, 2025

## Abstract

We present UMMBAS, an ultra-large, molecular function–guided virtual screening framework that learns a similarity space from a broad “MF cloud” (compounds active on proteins sharing the target’s molecular function) together with >1.29M ZINC decoys. The model follows a projection-only strategy: dimensionality reduction (PCA or UMAP) is fit on MF+ZINC, then held-out target ligands are projected and ranked by distance to the MF cloud. In Phase 1 (hyperparameter sweep), PCA consistently outperformed UMAP when trained on the full MF cloud. A critical data-integrity investigation revealed that source MF files contained duplicate Compound ChEMBL IDs (multiple targets per compound). After deduplicating the MF cloud by taking the minimum affinity per compound, UMAP performance dropped by 58.3% in a controlled comparison, indicating the original results were inflated by duplicate-driven density artifacts. An orchestration fix now enforces freshness-based recomputation of similarity spaces to prevent stale artifact reuse across reruns.

These findings reinforce the central role of MF cloud composition in determining optimal dimensionality reduction and establish a reproducible foundation for subsequent phases (cutoff sensitivity, MF cloud ablation, and cross-protein generalization). The present work emphasizes rigor over leaderboard performance: we document both successes and failure modes to map the problem space transparently.

## Standing Research Questions and Primary Hypotheses

- RQ1: How does MF cloud composition (size, affinity cutoff) determine the relative performance of PCA vs UMAP?
  - H1: With small or no MF cloud, UMAP’s local structure preservation is advantageous; with large MF clouds, PCA’s global variance dominates.
- RQ2: What UMAP hyperparameter regimes are optimal for retrieval under clean (deduplicated) MF data?
  - H2: The optimal neighborhood size shifts to medium/large values (50–500) after deduplication; min_dist has a weak effect.
- RQ3: To what extent do results generalize across proteins sharing or not sharing the same MF?
  - H3: Same-MF transfer is stronger than cross-MF; dimensionality choice remains target- and MF-dependent.

## Strongest Current Evidence

- Deduplication integrity: Source MF file contained 2.23× row duplication (430,794 rows → 193,244 unique compounds). Dedup by min affinity per compound.
- Controlled comparison (seed44, UMAP-5D, nn=10, md=0.1): EF@1% dropped from 43.78 → 18.27 (−58.3%) after deduplication.
- Mechanism: Duplicates created artificial high-density regions that small-nn UMAP exploited; larger nn values were more robust.
- Orchestration safeguard: main_orchestrator now recomputes similarity spaces when MF/ZINC inputs are newer than existing outputs; prevents stale PCA/UMAP divergence post-dedup.

## Notes for Methods

- Projection-only design to avoid data leakage: fit on MF+ZINC, project held-out actives.
- Features: 39–40 RDKit descriptors with StandardScaler; Fingerprints: 2048-bit ECFP4.
- Metrics: EF@1% primary; ROC-AUC and PR-AUC secondary; 5 replicates per config.

## Next Updates (Planned)

- Full Phase 1 rerun on deduplicated MF cloud, with expanded UMAP nn grid (10, 20, 50, 100, 500).
- Integrity assertion in analysis stage to ensure MF cloud row counts match deduplicated source.
- Phase 2–4 sequencing preserved (cutoff → MF ablation → generalization), using clean Phase 1 outputs.
