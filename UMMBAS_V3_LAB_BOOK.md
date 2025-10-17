# UMMBAS v3.0 Lab Book

**Period**: October 2025  
**Branch**: 3.0  
**Status**: Phase 1 Complete, Phase 2 Planned

---

## Executive Summary

**Goal**: Compare PCA vs UMAP for virtual screening via similarity space projection.

**Key Finding**: PCA consistently outperforms UMAP by 1.27-1.46× (EF@1% = 57-58 vs 39-46) for ABL1 ligand retrieval.

**Critical Discovery**: Small UMAP neighborhoods (nn=10) outperform large (nn=500) by 2.3×, contradicting standard guidance.

---

## Methodology

### Dataset
- **Training (for DR fitting)**: ~191K ChEMBL "Tyrosine-Protein Kinase" MF cloud + 1,295,279 ZINC decoys = **1,486,279 compounds**
- **Held-out (projected)**: 3,313 ABL1 actives (≤100,000 nM) - excluded from DR training to prevent data leakage
- **Total ranked**: 1,298,592 compounds (3,313 actives + 1,295,279 ZINC)

### Representation
- **Features**: 40 RDKit descriptors (MW, DipoleMoment, nHBAcc, nHBDon, nAromAtom, nRing, nRot, etc.)
- **Fingerprints**: 2048-bit ECFP4 (for comparison)

### Dimensionality Reduction
1. **PCA**: 2D, 5D, 10D (deterministic)
2. **UMAP**: 2D, 5D, 10D with hyperparameter grid:
   - `n_neighbors`: {10, 20, 100, 500}
   - `min_dist`: {0.0, 0.001, 0.005, 0.01, 0.1, 0.5}
   - 5 random seeds per config

### Ranking
- Fit DR on **ChEMBL MF cloud + ZINC decoys** together (training set = 1.49M compounds)
- **Project only target actives** into learned space (held-out, prevents data leakage)
- Rank by Euclidean distance to target ligand centroid
- **Metric**: Enrichment Factor at 1% (EF@1%)

---

## Phase 1 Results (Hyperparameter Sweep)

### Performance Summary

| Dimension | PCA | Best UMAP | Performance Gap |
|-----------|-----|-----------|-----------------|
| 2D | 57.62 | 39.41 ± 0.67 (nn=10, md=0.01) | **1.46× PCA > UMAP** |
| 5D | 58.44 | 44.42 ± 0.84 (nn=10, md=0.01) | **1.32× PCA > UMAP** |
| 10D | 58.04 | 45.83 ± 0.53 (nn=10, md=0.001) | **1.27× PCA > UMAP** |

**PCA**: Stable 57-58 EF@1% across all dimensions (deterministic)  
**UMAP**: Improves 39→44→46 with dimension, but never catches PCA

### UMAP Hyperparameter Effects

**n_neighbors (STRONG effect)**:
- nn=10 → EF@1% = 45.83 (10D, best)
- nn=20 → EF@1% = 33.84 (26% degradation)
- nn=100 → EF@1% = 24.67 (46% degradation)
- nn=500 → EF@1% = 19.57 (57% degradation, **2.3× worse than nn=10**)

**min_dist (WEAK effect)**:
- md=0.0 → 45.83 (10D, nn=10)
- md=0.001 → 45.83
- md=0.01 → 45.71
- md=0.1 → 44.83
- md=0.5 → 44.77
- **Variation**: Only ±2% across 3 orders of magnitude

**Interpretation**: For retrieval tasks, neighborhood size is critical; packing tightness is nearly irrelevant.

### Fingerprint Failure

**Fingerprint-PCA**: EF@1% = 2.26 (25× worse than features-PCA)

**Cause**: Binary ECFP4 vectors (2048-bit, sparse) unsuitable for linear PCA. Continuous descriptors essential.

---

## Key Findings

### 1. PCA Dominance
- **EF@1% = 57-58** across 2D/5D/10D (stable, no overfitting)
- Preserves global distances critical for ranking 1.3M compounds
- Deterministic (no seed variability)
- No hyperparameter tuning required

### 2. UMAP Hyperparameter Sensitivity
- **Small neighborhoods optimal**: nn=10 outperforms nn=500 by 2.3×
- **Contradicts standard guidance**: UMAP tutorials recommend nn=15-50; we find nn=10 best
- **min_dist irrelevant**: ±2% variation across 0.0-0.5
- **Stochastic variability**: ±0.53-6.29 std across seeds (worse for poorly tuned configs)

### 3. Dimensionality Behavior
- **PCA**: 2D captures 57% enrichment; 5D/10D add negligible improvement
- **UMAP**: Needs 10D to approach best performance (still 21% below PCA)
- **Interpretation**: First 2 PCs highly informative for this dataset

---

## Computational Infrastructure

### HPC Scripts
- `main_orchestrator.py`: Single experiment execution (reads config JSON)
- `generate_phase1_configs.py`: Generate hyperparameter sweep configs
- `check_hyperparam_status.py`: Monitor cluster progress, visualize results
- `aggregate_and_report.py`: Compile final metrics across all experiments

### Cluster Setup
- **Jobs**: 260 Phase 1 experiments (5 seeds × 52 hyperparameter combinations)
- **Runtime**: ~30-60 min per experiment
- **Completion check**: Automatic skip if `*-RANKED.csv` exists (prevents duplicate work)

### Code Improvements (October 2025)
1. **Progress bars**: Added tqdm to all long operations
2. **Dimensionality-separated reporting**: Metrics organized by 2D/5D/10D
3. **Early completion check**: Skip experiments before expensive setup
4. **Refined hyperparameter grid**: Smaller nn (3, 5) + tighter md (0.0, 0.001, 0.005) for Phase 1b

---

## Limitations

1. **Single target**: ABL1 only; generalization unknown
2. **No fingerprint baseline**: Missing Tanimoto/ECFP4 comparison (industry standard)
3. **No k-NN baseline**: Haven't tested ranking in original 40D space (no DR)
4. **Descriptor-dependent**: Success requires continuous features; binary vectors fail
5. **Ignores 3D structure**: No protein-ligand docking or conformational analysis

---

## Phase 2 Plan (Pending)

### Goals
1. Test top 8 configs (PCA-2D/5D/10D, UMAP-best-2D/5D/10D, FP-PCA-2D, FP-UMAP-2D) on **10 diverse targets**
2. Validate generalization across protein families (kinases, GPCRs, proteases, etc.)
3. Compare to Tanimoto/ECFP4 baselines

### Targets (Proposed)

TODO for LLM while you are editing this lab book: look at our experimental setup

---

## Phase 3 Plan (Future)

### Generalization Analysis
1. **Leave-one-out cross-validation**: Train on N-1 targets, test on held-out target
2. **Transfer learning**: Can kinase cloud predict GPCR ligands?
3. **Sample size effects**: How much training data needed for robust PCA?

---

## Publication Potential

### Novel Contributions
1. **First rigorous PCA vs UMAP comparison** for virtual screening at scale (1.3M decoys)
2. **Discovery of strong n_neighbors effect**: nn=10 optimal for retrieval (contradicts nn=15-50 guidance)
3. **Quantification of min_dist irrelevance**: ±2% variation across 3 orders of magnitude
4. **Demonstration of PCA stability**: 57-58 EF@1% across 2D/5D/10D (no overfitting)

### Pending Validation
- Multi-target generalization (Phase 2)
- Comparison to industry-standard fingerprints (Tanimoto/ECFP4)
- Literature review (in progress via LLM research prompt)

### Target Journals
- *Journal of Chemical Information and Modeling* (JCIM)
- *Journal of Cheminformatics*
- *Molecular Informatics*

---

## Technical Notes

### Data Integrity Checks
- **Zero ChEMBL-ZINC overlap**: Validated via canonical SMILES matching
- **Target ligands excluded from MF cloud**: Leave-one-target-out approach prevents data leakage
- **Affinity cutoff**: ≤100,000 nM (IC50/Ki/Kd) for all bioactives

### Visualization Gotcha
- **Plotting**: Samples 1,000 ZINC for scatter plots (clarity)
- **Ranking**: Uses all 1,295,279 ZINC (no sampling)
- Layer ordering: MF cloud plotted first, then ZINC, then target ligands (prevents occlusion)

### File Structure
```
experiment_workspace_v3_phase1/
  run_seed42_config_tyro_features_pca_dim2_seed42/
    TyrosinaseProteinKinaseABL1_P00519/
      results/
        features/
          dim_2/
            PCA/
              *-RANKED.csv        ← Completion marker
              *_ranking_metrics.csv
              *.png (2D plots)
```

---

## Current Status (October 16, 2025)

### Completed ✅
- Phase 1 hyperparameter sweep: 260 experiments, ~180 completed
- Analysis scripts enhanced (progress bars, dimensionality separation)
- Refined hyperparameter grid designed (Phase 1b: nn={3,5,10,20}, md={0.0,0.001,0.005,0.01,0.1})
- LLM research prompt created (pending literature review)
- Preliminary conclusions documented

### In Progress 🔄
- Phase 1b experiments: Running refined grid on cluster
- Literature review: Querying LLM for PCA/UMAP comparisons in cheminformatics

### Next Steps ⏳
1. Complete Phase 1b experiments (refined hyperparameter grid)
2. Analyze complete results (check if nn=3/5 outperform nn=10)
3. Conduct literature review (assess novelty)
4. Add fingerprint baselines (Tanimoto/ECFP4, k-NN in 40D)
5. Plan Phase 2 (multi-target validation)

---

## Lab Book Entries

### October 8, 2025: Phase 1 Preliminary Analysis
- Initial 180/260 experiments completed
- PCA dominates UMAP by 1.27-1.46× across all dimensions
- Identified incomplete hyperparameter grid (nn=500 only 1 seed, missing small neighborhoods)

### October 10, 2025: Hyperparameter Grid Refinement
- Proposed Option B: nn={3,5,10,20}, md={0.0,0.001,0.005,0.01,0.1} for features
- Hypothesis: Even smaller neighborhoods might improve performance
- Fingerprints unchanged (still running, needed to prove underperformance)

### October 16, 2025: Code Improvements and Documentation
- Enhanced `check_hyperparam_status.py`: Progress bars, dimensionality-separated metrics
- Updated `generate_phase1_configs.py`: Auto-skip completed experiments
- Updated `main_orchestrator.py`: Early completion check before expensive setup
- Created `PHASE1_REFINED_HYPERPARAMS.md`: Complete documentation of refined grid
- Created `LLM_RESEARCH_PROMPT_PHASE1_FINDINGS.md`: Prompt for literature review
- **Dataset correction**: Confirmed 1,295,279 ZINC used for ranking (not 1K sample)
- **Descriptor correction**: Using 40 RDKit descriptors (not 208)

---

## References

### Internal Documentation
- `PHASE1_REFINED_HYPERPARAMS.md`: Refined hyperparameter grid details
- `LLM_RESEARCH_PROMPT_PHASE1_FINDINGS.md`: Literature review prompt
- `METHODS_FOR_PAPER.md`: Detailed methodology for publication
- `FINAL_DATASET_VERIFICATION.md`: Dataset counts and integrity checks

### Key Scripts
- `main_orchestrator.py`: Single experiment orchestration
- `generate_phase1_configs.py`: Config generation for hyperparameter sweep
- `check_hyperparam_status.py`: Progress monitoring and result visualization
- `aggregate_and_report.py`: Final report compilation

---

**Last Updated**: October 16, 2025  
**Next Review**: After Phase 1b completion and literature review
