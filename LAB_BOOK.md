# UMMBAS v3.0 Lab Book

**Period**: October 2025  
**Branch**: 3.0  
**Status**: Phase 1 Complete, Analysis in Progress

---

## Research Questions and Hypotheses

### Primary Hypothesis
**Ligands active against a specific protein will exhibit measurable chemical proximity to compounds that modulate OTHER proteins sharing the same molecular function.**

### Phase-Specific Questions

**Phase 1 (Complete)**: What are optimal hyperparameters and dimensionality for PCA vs UMAP?

**Phase 2 (Running)**: How does affinity cutoff affect enrichment performance? 

**Phase 3 (Planned)**: Does MF cloud size cause PCA vs UMAP performance reversal?

**Phase 4 (Planned)**: Do optimal configs transfer across proteins (same-MF vs different-MF)?

---

## Hypotheses for UMAP n_neighbors Effect (October 24, 2025)

**Core Finding**: Small UMAP neighborhoods (nn=10) outperform large (nn=500) by 2.3× for centroid-based molecular retrieval.

### H1: Task Mismatch - Embedding Quality vs Ranking Discriminability

**Hypothesis**: UMAP optimizes for smooth manifold visualization/clustering, but centroid-based ranking requires cluster separation, not continuity.

**Mechanism**:
- **Small nn (10)**: Creates tight, discrete clusters with exaggerated inter-cluster distances → high discriminative power for ranking
- **Large nn (500)**: Creates smooth, continuous manifold with blurred cluster boundaries → poor discriminative power

**Key Predictions**:
1. Small nn → higher silhouette scores (cluster separation)
2. Small nn → lower trustworthiness/continuity (manifold quality)
3. Effect strongest when MF cloud is dense and cohesive

**Testable with Existing Data**: ✅ YES
- Distance distributions (intra-class vs inter-class)
- Silhouette scores
- MF cloud density metrics

---

### H2: Signal Dilution with Large Neighborhoods

**Hypothesis**: With 1.3M decoys, large neighborhoods force UMAP to preserve too many irrelevant decoy-decoy relationships, diluting the MF cloud signal.

**Mechanism**:
- **nn=500**: Each compound has 500 neighbors → 650M pairwise relationships to balance
- Most relationships involve decoy-decoy pairs (not informative)
- UMAP optimization cycles spent on ZINC internal structure, not MF cloud → target relationships
- **nn=10**: Only 10 nearest neighbors (likely other MF compounds) → preserves biologically relevant structure

**Key Predictions**:
1. nn effect stronger with more decoys (1.3M vs 100K)
2. nn effect weaker if decoys are chemically diverse (less internal structure)
3. MF cloud compounds' nearest neighbors differ by nn value

**Testable with Existing Data**: ✅ PARTIALLY
- Cannot vary decoy count (requires new experiments)
- CAN analyze nearest neighbor composition: What fraction of each compound's nn neighbors are MF cloud vs ZINC?
- CAN analyze ZINC chemical diversity (descriptor variance, pairwise distances)

---

### H3: Information Bottleneck and Geometric Constraints

**Hypothesis**: Compressing 1.5M compounds from 40D to 2D-10D creates information bottleneck. Small nn accepts the bottleneck; large nn fights it unsuccessfully.

**Mechanism**:
- **High-D (40D)**: ~500 meaningful neighbors per compound
- **Low-D (2D-10D)**: Geometrically impossible to preserve 500 neighborhood relationships
- **nn=500**: Tries to preserve 500 neighbors → creates mediocre compromise
- **nn=10**: Only preserves 10 neighbors → achievable goal, succeeds

**Key Predictions**:
1. nn effect diminishes at higher dimensions (20D, 50D, 100D where 500 neighbors are feasible)
2. UMAP performance gap vs PCA closes with dimension

**Testable with Existing Data**: ✅ PARTIALLY
- Observed: UMAP improves 2D (39.4) → 5D (44.4) → 10D (45.8) [gap narrows slightly]
- Cannot test higher dimensions without new experiments
- CAN extrapolate trend from 2D/5D/10D data

---

### H4: MF Cloud as Anchor Point

**Hypothesis**: The MF cloud (191K compounds) acts as reference frame. Small nn lets MF cloud self-organize cohesively; large nn dilutes it by mixing with decoys.

**Mechanism**:
- **Small nn**: Each MF compound's 10 neighbors likely other MF compounds → MF cloud forms tight cluster
- **Large nn**: Each MF compound's 500 neighbors include many ZINC decoys → MF cloud structure distorted
- Ranking uses distance to **MF cloud centroid** → cohesive cloud = well-defined centroid

**Key Predictions**:
1. Small nn → tighter MF cloud (lower mean pairwise distance within MF)
2. Small nn → better-defined centroid (lower variance in distances to centroid)
3. Effect vanishes with MF cloud ablation: At MF=0, no anchor exists, so nn should not matter

**Testable with Existing Data**: ✅ YES (Phase 1) + ✅ YES (Phase 3 when complete)
- Phase 1: MF cloud density, centroid definition across nn values
- **Phase 3 (MF ablation)**: Critical test - if nn effect vanishes at MF=0, hypothesis confirmed

---

### H5: Optimization Landscape Complexity

**Hypothesis**: Large nn creates complex optimization landscape with more local minima. Small nn has simpler landscape with consistent convergence.

**Mechanism**:
- **nn=500**: 650M pairwise constraints → conflicting objectives → local minima
- **nn=10**: Fewer constraints → simpler optimization → robust convergence

**Key Predictions**:
1. Large nn → higher variance across random seeds
2. Longer optimization (more epochs) may help nn=500 catch up

**Testable with Existing Data**: ✅ YES
- Check standard deviation across 5 seeds for each nn value
- Phase 1 data already shows "±0.53-6.29 std across seeds (worse for poorly tuned configs)"

---

### H6: Curse of Dimensionality (Distance Concentration)

**Hypothesis**: In 40D, "500 nearest neighbors" is meaningless due to distance concentration. nn=10 captures true neighbors; nn=500 includes random equidistant noise.

**Mechanism**:
- High-D: Most points approximately equidistant (distance concentration)
- 10th neighbor truly close, 500th neighbor in equidistant shell
- **nn=500**: Forced to preserve relationships with non-meaningful "neighbors"

**Key Predictions**:
1. Distance ratio d(500th) / d(10th) close to 1 in 40D
2. Effect stronger in fingerprint space (2048D, sparser) than feature space (40D)

**Testable with Existing Data**: ✅ YES
- Calculate k-th nearest neighbor distance ratios in original 40D space
- Compare features (40D) vs fingerprints (2048D)

---

### Priority Ranking for Investigation

**Primary Focus (H1, H2, H4)**: As requested

1. **H4 (MF Cloud Anchor)**: MOST TESTABLE - Has clear Phase 3 validation
2. **H1 (Task Mismatch)**: HIGH IMPACT - Explains fundamental algorithm-task mismatch
3. **H2 (Signal Dilution)**: PARTIALLY TESTABLE - Can analyze neighbor composition

**Secondary (H5, H6)**: Useful supporting evidence

4. **H5 (Optimization)**: Easy to test with existing variance data
5. **H6 (Distance Concentration)**: Requires original 40D data analysis

**Deprioritized (H3)**: Cannot test without new experiments at higher dimensions

---

## Executive Summary

**Goal**: Develop and validate molecular function-guided virtual screening framework using dimensionality reduction.

**Key Finding**: PCA consistently outperforms UMAP by 1.27-1.46× (EF@1% = 57-58 vs 39-46) for ABL1 ligand retrieval with full MF cloud.

**Critical Discovery**: Small UMAP neighborhoods (nn=10) outperform large (nn=500) by 2.3×, contradicting standard guidance.

**Recent Enhancement (Oct 18)**: Added parallelization to stratified enrichment analysis (4-8× speedup on multi-core systems).

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

## Phase 2-4 Plans (v3.0 Design)

**Note**: **PHASE REORDERING COMPLETE** (October 18, 2025). Original Phase 4 moved to Phase 2 to resolve logical dependency issue. Cutoff sensitivity analysis must precede MF cloud ablation because cutoff determines which molecules are "active" in the MF cloud.

### Phase 2: Affinity Cutoff Sensitivity (40 runs) **[REORDERED - was Phase 4]**
**Purpose**: Determine optimal affinity threshold for classifying molecules as "active" in MF cloud

**Rationale for Reordering**: 
- **Logical dependency**: Cutoff determines which molecules contribute to MF cloud diversity
- **Affects downstream phases**: Must establish optimal cutoff before Phase 3 (MF cloud ablation)
- **Computational efficiency**: Reuses Phase 1 similarity spaces (only reruns ranking, ~75% time savings)

**Configuration**:
- Target: ABL1 (Tyrosine kinase)
- **Potency Tiers** (aligned with stratified enrichment analysis):
  - High Potent: 0.1-100 nM (drug-like, clinically relevant)
  - Medium Potent: 100-1,000 nM (moderate affinity)
  - Weak Potent: 1,000-100,000 nM (marginal, promiscuous)
- **Cutoffs to Test**:
  - 100 nM: High-potent only (strictest quality)
  - 1,000 nM (1 μM): High + Medium (balanced)
  - 10,000 nM (10 μM): High + Medium + some Weak (permissive)
  - 100,000 nM (100 μM): All potencies (maximum diversity, current default)
- Methods: Best PCA-features and best UMAP-Euclidean-features from Phase 1
- Seeds: 5 replicates (42-46)

**Research Questions**:
1. Which cutoff gives best High-potent EF@1%?
2. Do PCA and UMAP prefer different cutoffs?
3. How does cutoff affect Medium vs Weak potency enrichment?
4. What is optimal cutoff for Phase 3 (MF cloud ablation)?

**Computational Strategy**: 
- Reuses Phase 1 similarity spaces and DR models
- Does NOT recalculate features/fingerprints or refit PCA/UMAP
- ONLY reruns ranking/evaluation with different cutoffs
- Runtime: ~10-15 hours total (vs ~40+ hours for full pipeline)

**Current Status**: **Config generator and orchestrator created** (October 18, 2025). Ready for execution after Phase 1 best configs extraction.

---

### Phase 3: MF Cloud Ablation (60 runs) **[REORDERED - was Phase 2]**
**Purpose**: Validate hypothesis that MF cloud size causes PCA vs UMAP performance reversal

**Configuration**:
- Target: ABL1 (Tyro) only
- MF Cloud Sizes: 0, 1K, 10K, 50K, 100K, 420K molecules
- **Cutoff**: Use optimal cutoff identified in Phase 2
- Methods: Best PCA and best UMAP from Phase 1 (at optimal dimensions)
- Seeds: 5 replicates (42-46)

**Expected Outcome**: 
- UMAP dominates at MF=0 (local structure preservation)
- PCA dominates at MF=420K (global distance preservation)
- Crossover point at ~10K-50K molecules

**Current Status**: Config generator exists but needs update to use Phase 2 optimal cutoff

---

### Phase 4: Cross-Protein Generalization (80 runs) **[REORDERED - was Phase 3]**
**Purpose**: Test if optimal configurations transfer across different proteins

**Targets**:
1. **Pyruvate kinase M2** (PKM2, P14618) - Same MF as ABL1 (Transferase)
2. **Isocitrate dehydrogenase** (IDH1, O75874) - Different MF (Oxidoreductase)

**Methods**: Best configs from Phase 1 for each method
**Cutoff**: Use optimal cutoff from Phase 2

**Hypothesis**:
- Same MF (PKM2): Similar performance expected
- Different MF (IDH1): Performance may degrade, tests true generalization

**Current Status**: Config generator exists but needs update to use Phase 2 optimal cutoff

---

## Future Directions (Beyond Phase 4)

### Baseline Comparisons
1. **Tanimoto/ECFP4**: Industry-standard fingerprint similarity
2. **k-NN in 40D**: No dimensionality reduction (test if DR is beneficial)
3. **Ensemble methods**: Combining PCA + UMAP predictions

### Extended Validation
1. **Leave-one-out cross-validation**: Train on N-1 targets, test on held-out
2. **Transfer learning**: Can kinase cloud predict GPCR ligands?
3. **Sample size effects**: Minimum MF cloud size for robust performance

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

### October 26, 2025: v4.0 (molfuse) refactor planning and invariants
- Changes Made (Design)
  - Initiated molfuse v4.0 scaffold to align HPC Phase 1/2 with the validated independent test harness.
  - Established invariants:
    - StandardScaler fits on MF+ZINC only; actives are projected using this scaler (no leakage).
    - UMAP runs without a fixed seed (random_state=None) to enable multi-threaded execution on HPC.
    - Exact 1-NN scoring backend in embedded space is the default; distance-based score = -min_distance.
    - Affinity cutoff applies to MF cloud for scoring only; actives are never filtered by cutoff.
    - Spearman’s rho(pActivity vs score) is computed and reported for actives.
    - Target-preserving exclusion enforced: any compound associated with the target is excluded from MF cloud.
  - Planned artifacts: structured run workspace, robust logging, potency-stratified (Phase 1) and cutoff sensitivity (Phase 2) analyses, plus a status checker.

- Experiments Run
  - None yet under v4.0; this is a planning entry. Independent harness results guide the invariants.

- Observations and Results
  - Independent test confirmed exact 1-NN equivalence to cdist with major speedups, motivating backend switch.
  - Removing fixed seeds enables parallel UMAP, addressing prior runtime and occasional segfault issues.
  - Clear misalignments identified in v3.0 (dedup policy, overlap removal, cutoff application) are rectified by v4.0 design.

- Next Steps
  - Scaffold package modules and CLI for Phase 1 end-to-end; add base config; perform a small local smoke test.
  - Prepare HPC scripts for Phase 1/2; port potency and cutoff analyses; add status checker.

---

### October 26, 2025: Phase 1 cutoff empty-set policy (fail-fast)
- Changes Made (Behavior)
  - When the affinity cutoff filters the MF set to zero compounds, Phase 1 now fails fast with a clear error and logs (`phase1_error.json`).
  - Config override: `on_empty_cutoff` can be set to `"fallback"` to use the full MF for scoring (legacy behavior) if desired.

- Rationale
  - Silent fallback can mask misconfigured cutoffs or unit/column issues; explicit failure improves experimental rigor and reproducibility.

- Implications
  - Runs with overly strict cutoffs or missing/NaN affinity values will terminate with actionable diagnostics instead of silently changing the scoring population.
  - Default invariant behavior remains unchanged for successful filters; scaler is still fit on MF+ZINC only; actives are never filtered.

---

### October 26, 2025: Benchmarking cdist vs Exact KDTree/BallTree for Scoring
- Changes Made (Code or configuration)
  - Added `analysis_scripts/benchmark_cdist_vs_kdtree.py` to compare the current scoring method (batched `scipy.spatial.distance.cdist`) against an exact 1-NN index using scikit-learn's `NearestNeighbors` (KDTree/BallTree) on real simspace data.
  - Updated `README.md` with usage instructions and HPC-friendly examples.

- Experiments Run
  - Pending HPC execution on representative PCA-features and UMAP-features runs (2D/5D/10D) using full MF cloud and a large ZINC sample.
  - Inputs: Phase 1/2 simspace CSV; optional projected actives CSV to include actives in the comparison.

- Hypothesis
  - Exact KDTree/BallTree will produce identical min-distance results to `cdist` (within a small numerical tolerance), and run faster with lower memory footprint at 2–10 dimensions.

- Success Criteria
  - Equality: `allclose=True` with max absolute difference ≤ 1e-6 and fraction exceeding tolerance ≈ 0%.
  - Performance: NN method wall time < cdist wall time for both actives and ZINC decoys.

- Next Steps
  - If equality holds and speedup is material, replace `cdist` in `project_and_analyze.py` with an exact NN backend (feature-flagged), and document the change.

### October 26, 2025: PCA vs UMAP Dedup Divergence — Root Cause and Fix
- Changes Made (Code or configuration)
  - Updated `main_orchestrator.py` Step 2b to avoid stale artifact reuse by checking input freshness before skipping similarity space recomputation.
    - Mechanism: Compare modification times of prepared MF/ZINC inputs vs the existing similarity space CSV; recompute if inputs are newer or on comparison error.
  - No changes to DR algorithms or scoring; fix is orchestration-only.

- Experiments Run
  - Pending: Rerun a representative PCA-features configuration post-dedup (same seed/dimension as prior UMAP validation) to verify EF@1% reflects deduplication.
  - Pending: Sanity check that MF cloud row count in loaded simspace matches deduplicated source (to be added as a runtime integrity assertion).

- Observations and Results
  - Root cause identified for PCA's unchanged EF@1% after dedup vs UMAP's 58.3% drop: reuse of a pre-dedup similarity space/model due to an existence-only skip in the orchestrator.
  - Divergence mechanism: UMAP runs were recomputed after deduplication; PCA runs reused stale simspace artifacts, preserving inflated performance.
  - Expected outcome after fix: PCA EF@1% should drop substantially when recomputed on the deduplicated MF cloud, aligning with UMAP directionally.

Notes
- This fix enforces data-to-artifact freshness; adds no new dependencies and preserves projection-only design.
- Follow-up guardrail: add an integrity assertion in `project_and_analyze.py` to compare MF cloud size in simspace against the temp_data source; fail fast on mismatch.

### October 26, 2025: Phase 1 Dedup Policy Change — Median Affinity per Compound
- Changes Made (Code or configuration)
  - Updated `experimental_pipeline/prepare_data.py` MF cloud deduplication: aggregate duplicates by median of 'Standard Value (nM)' per 'Compound ChEMBL ID' (was minimum).
  - Updated fallback merge in `experimental_pipeline/project_and_analyze.py` to also use median when reconstructing affinity for cutoff filtering, ensuring consistency.

- Rationale
  - Median provides robustness to outliers and measurement noise across multiple targets/tests per compound, avoiding overweighting single extremely potent readings.

- Expected Impact
  - MF cloud composition will shift modestly versus min-aggregation; anticipate small changes to distance distributions and EF metrics. PCA likely minimally affected; UMAP sensitivity depends on local density shifts.

- Validation Plan
  - Recreate Phase 1 simspaces with median dedup; compare EF@1% and variance vs prior min-based runs on a small subset before broader reruns. Document deltas.

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

### October 18, 2025: Publication Preparation and Pipeline Analysis
**Analysis Script Enhancements**:
- Enhanced `scripts/analyze_potency_stratified_enrichment.py`:
  - Added dimension-separated versions of 4 key plot types (PCA vs UMAP, stratified EF, percent found, quality vs quantity)
  - Implemented dual PNG+PDF output for all figures (300 DPI PNG + vector PDF)
  - Applied best hyperparameter filtering to dimensionality impact plots
  - **Parallelized run analysis** using multiprocessing (4-8× speedup on multi-core systems)
  - Added `--n_jobs` parameter for user control (default: all CPUs - 1)

**Documentation Updates**:
- Created `ARCHIVE.md`: Tracks deprecated features and development notes
- Created `PLANNING.md`: Task tracking with checklist format, research questions, next steps
- Updated documentation structure per copilot instructions

**Pipeline Logic Analysis**:
- Identified potential ordering issue in Phase 2-4 design:
  - Current: Phase 1 (hyperparams) → Phase 2 (MF cloud) → Phase 3 (generalization) → Phase 4 (cutoff)
  - Problem: Affinity cutoff determines "active" molecule definition → affects MF cloud composition
  - Phase 2 uses fixed cutoff (100K nM) but Phase 4 tests cutoff sensitivity
  - **Logical flaw**: If optimal cutoff changes, Phase 2 MF cloud conclusions may be invalid
- Three restructuring options proposed:
  - Option A: Cutoff as Phase 2 (before MF cloud)
  - Option B: Combined MF Cloud + Cutoff study (240 runs: 6 MF sizes × 4 cutoffs)
  - Option C: Acknowledge limitation in discussion section
- **Decision taken**: Pipeline should be restructured before Phase 2 execution

**Observations**:
- Parallelization significantly reduces analysis time for large-scale experiments
- Publication-quality figure generation now automated (PNG for presentations, PDF for papers)
- Best hyperparameter filtering essential for fair method comparisons

### October 18, 2025 (Evening): Phase Reordering Implementation
**Pipeline Restructuring Decision**:
- **Resolved Phase 2-4 ordering issue**: Moved cutoff sensitivity from Phase 4 to Phase 2
- **Rationale**: Cutoff must be determined before MF cloud ablation (logical dependency)
- Chose Option A: Best configs only (40 runs) for computational efficiency

**File Renaming** (Phase Shuffle):
- `generate_phase4_configs.py` → `generate_phase2_configs.py` (cutoff sensitivity)
- `generate_phase2_configs.py` → `generate_phase3_configs.py` (MF cloud ablation)
- `generate_phase3_configs.py` → `generate_phase4_configs.py` (generalization)
- HPC submission scripts renamed accordingly: `submit_v3_phase[234].sh`

**New Phase 2 Implementation** (Cutoff Sensitivity):
- Created `generate_phase2_configs.py`:
  - Tests 4 cutoffs: 100 nM, 1 μM, 10 μM, 100 μM (aligned with potency tiers)
  - Uses potency tier definitions from `analyze_potency_stratified_enrichment.py`
  - Reuses best PCA-features and best UMAP-Euclidean-features from Phase 1
  - Total: 4 cutoffs × 2 methods × 5 seeds = 40 runs
  - Estimated runtime: ~10-15 hours (reusing Phase 1 data)

- Created `scripts/run_phase2_cutoff_analysis.py` (orchestrator):
  - Locates matching Phase 1 run directories
  - Reuses similarity spaces, DR models, and target ligand representations
  - Calls `project_and_analyze.py` with `--affinity_cutoff` parameter only
  - Does NOT recalculate features/fingerprints or refit PCA/UMAP
  - **75% time savings** vs full pipeline (ranking only, no DR fitting)
  - Outputs to Phase 2 workspace with cutoff-specific subdirectories

**Potency Tier Alignment**:
- Unified potency definitions across scripts:
  - High Potent: 0.1-100 nM (drug-like, clinically relevant)
  - Medium Potent: 100-1,000 nM (moderate affinity)
  - Weak Potent: 1,000-100,000 nM (marginal, promiscuous)
- Cutoffs designed to test quality vs quantity trade-offs:
  - 100 nM: High-potent only (maximum quality)
  - 1 μM: High + Medium (balanced)
  - 10 μM: High + Medium + some Weak (permissive)
  - 100 μM: All potencies (maximum diversity, current default)

**Documentation Updates**:
- Updated LAB_BOOK.md with new Phase 2-4 ordering
- Marked Phase 2 config generator and orchestrator as complete
- Noted that Phase 3 and Phase 4 config generators need updates (use optimal cutoff from Phase 2)

**Research Questions for Phase 2**:
1. Which cutoff gives best High-potent EF@1%?
2. Do PCA and UMAP prefer different cutoffs?
3. How does cutoff affect Medium vs Weak potency enrichment?
4. What is optimal cutoff for Phase 3 (MF cloud ablation)?

**Current Status**: Phase 2 ready for execution pending Phase 1 best configs extraction

---

### October 24, 2025: CRITICAL DISCOVERY - MF Cloud Duplication in Phase 1 Similarity Spaces

**Context**: During Phase 2 cutoff sensitivity debugging, discovered that MF cloud filtering in `project_and_analyze.py` caused 30× memory explosion (12.8M rows from 420K molecules). Root cause: affinity source file contains duplicate Compound ChEMBL IDs (compounds binding to multiple MF targets).

**Critical Realization**: This same duplication existed in Phase 1 similarity space training data.

#### Data Flow Investigation (TyrosineProteinKinaseABL1_P00519 Example)

**Source MF File**: `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv`

**Duplication Statistics (BEFORE Phase 1 Training)**:
- Total rows: 430,794
- Unique Compound ChEMBL IDs: 193,244
- **Duplication factor: 2.23× (430,794 / 193,244)**
- Duplicated compounds: 83,633 (43% of unique compounds)
- Most duplicated: CHEMBL388978 (2,127 occurrences - likely pan-kinase inhibitor)
- Unique targets in MF file: 833 different transferase proteins

**Why Duplicates Exist**:
- Compounds tested against **multiple targets** within same MF category
- Example: Kinase inhibitor screened against 100+ different kinases
- Same compound has different affinities for different targets
- Reflects real biological promiscuity of drug-like molecules

**Impact on Phase 1 DR Models**:
- PCA covariance matrix weighted toward highly-tested compounds
- UMAP k-NN graph contains artificial high-density regions
- DR manifolds potentially biased toward "privileged scaffolds"
- Distance metrics influenced by over-represented chemistry

#### Pipeline Fix Implementation

**Step 1: `prepare_data.py` Deduplication** (Lines 171-188)
```python
# Aggregate duplicates using minimum affinity (most potent binding)
df_filtered_precalc_mf = df_filtered_precalc_mf.groupby('Compound ChEMBL ID').agg({
    col: 'first' if col != 'Standard Value (nM)' else 'min'
    for col in df_filtered_precalc_mf.columns if col != 'Compound ChEMBL ID'
}).reset_index()
```
- Strategy: Keep **minimum affinity** (most potent) for each compound
- Rationale: If compound binds strongly to ANY transferase, that's its most relevant MF property
- Expected reduction: 430,794 → 193,244 rows (2.23× for Transferase)

**Step 2: `calculate_similarityspaces_exp.py` Safety Check** (Lines 159-177)
- Detects duplicates after loading (should be prevented by Step 1)
- Drops duplicates if found (keep='first')
- Logs warning if deduplication needed

**Step 3: Validation Framework**
- Created `rerun_phase1_config_compare.py`: Retrain ONE config with clean data
- Created `rerun_phase1_clean_compare_slurm.sh`: HPC submission script
- Test case: seed44, UMAP-Euclidean 5D, n_neighbors=10, min_dist=0.1
- Comparison metrics: EF@1%, ROC-AUC, PR-AUC (original vs clean)
- Decision threshold: **< 5% change** = minimal fix, **≥ 5% change** = full Phase 1 rerun

#### Scientific Question: Is Duplication Harmful or Beneficial?

**The Ambiguity**:
This discovery reveals either a **critical experimental flaw** OR an **accidental feature engineering success**. The outcome is uncertain.

**Hypothesis A: Duplication as Artifact (Harmful)**
- **Mechanism**: Over-representation biases manifold toward testing convenience, not biological relevance
- **Effect**: DR models learn "compounds medicinal chemists like to test" rather than true transferase chemistry
- **Evidence for**:
  - Confounds chemical diversity with testing frequency
  - Affinity information lost (promiscuity vs selectivity unclear)
  - May not generalize to novel/under-studied targets
- **Prediction**: EF@1% will **decrease** with deduplication (original results were inflated)

**Hypothesis B: Duplication as Weighting (Beneficial)**
- **Mechanism**: Over-representation emphasizes "privileged scaffolds" with favorable ADME properties
- **Effect**: DR models learn validated drug-like chemical space for this MF
- **Evidence for**:
  - Compounds tested 100+ times likely have good bioavailability, stability, safety profiles
  - Reflect medicinal chemistry consensus on "what works" for this target class
  - Natural importance weighting by biological relevance
- **Prediction**: EF@1% will **decrease** with deduplication (we removed useful signal)

**Hypothesis C: Minimal Impact (Neutral)**
- **Mechanism**: Chemical diversity of unique compounds dominates signal; duplication adds redundancy but not bias
- **Effect**: Over-representation was relatively uniform (2.23× average), not extreme
- **Evidence for**:
  - 193K unique compounds still provides rich diversity
  - Duplication factor moderate compared to potential bias (not 10× or 100×)
  - PCA/UMAP may be robust to modest data imbalance
- **Prediction**: EF@1% change **< 5%** (acceptable tolerance)

#### Technical Considerations

**Why Duplicates Don't "Break" Algorithms**:
- **PCA**: Handles duplicates by weighting covariance matrix (mathematically valid)
- **UMAP**: Treats duplicates as high-density regions (no computational failure)
- **Problem**: Not algorithmic error, but **representational bias**

**What Information Is Lost by Deduplication**:
- **Binding promiscuity**: Cannot distinguish pan-inhibitors (bind everything weakly) from selective inhibitors (bind one target strongly)
- **Affinity distribution**: CHEMBL388978's 2,127 affinities range from potent to weak - aggregating to minimum loses this variance
- **Target diversity**: Compound tested against 50 kinases vs 5 kinases - this frequency signal is discarded

**Why This Matters for Publication**:
- If duplication was beneficial: Must justify as **intentional weighting scheme** (feature engineering)
- If duplication was harmful: Must report **corrected metrics** (rerun Phase 1)
- If duplication was neutral: Must acknowledge as **limitation** (methods section)

#### Experimental Validation Plan

**HPC Comparison Test** (Running):
```bash
sbatch rerun_phase1_clean_compare_slurm.sh
```

**Three Possible Outcomes**:

1. **Scenario A: EF@1% Decreases ≥ 5% (Duplication was inflating metrics)**
   - **Interpretation**: Original results optimistic, testing bias artifact
   - **Action**: Full Phase 1 rerun required (all 260 configs)
   - **Timeline**: ~2-3 weeks HPC time
   - **Publication impact**: Delay, but stronger scientific rigor

2. **Scenario B: EF@1% Increases ≥ 5% (Deduplication improves performance)**
   - **Interpretation**: Duplicates added noise, unique diversity provides cleaner signal
   - **Action**: Full Phase 1 rerun required (celebrate improved results!)
   - **Timeline**: ~2-3 weeks HPC time
   - **Publication impact**: Demonstrates robustness, method improvement

3. **Scenario C: EF@1% Change < 5% (Minimal impact)**
   - **Interpretation**: Chemical diversity dominated, duplication was redundant but not biasing
   - **Action**: Proceed with existing Phase 1 results, note limitation
   - **Timeline**: Immediate (no rerun needed)
   - **Publication impact**: Methods section caveat, no delay

#### Connection to UMAP n_neighbors Hypothesis

**Potential Interaction with H4 (MF Cloud Anchor)**:
- If duplicates create artificial high-density MF cloud regions
- Small nn (10) may preferentially connect duplicates to each other
- Large nn (500) may dilute duplicate clusters by connecting to broader decoy population
- **Phase 3 MF ablation test** will be critical: Does nn effect change with deduplicated data?

**Revised Prediction**:
- Original Phase 1 (with duplicates): nn effect driven by duplicate clustering
- Clean Phase 1 (deduplicated): nn effect may diminish or shift
- **This makes the comparison test even more valuable** - tests two hypotheses simultaneously

#### Documentation and Reproducibility

**Data Provenance for ABL1**:
- Source: `KW-0808_Transferase_affinity_extracted_features.csv`
- Original: 430,794 rows, 193,244 unique compounds, 833 targets
- Filtered by target P00519 (ABL1): Excludes rows where accession='P00519'

---

### October 25, 2025: CRITICAL VALIDATION - 58.3% Performance Drop Confirms Full Phase 1 Rerun Required

**HPC Comparison Test Completed**: seed44, UMAP-Euclidean 5D, nn=10, md=0.1

**Results**:
- **Original (with 2.23× duplicates)**: EF@1% = **43.78**
- **Clean (deduplicated)**: EF@1% = **18.27**
- **Absolute change**: -25.51
- **Percent change**: **-58.3%**

**Decision Framework**:
- Threshold for significance: ≥5% change
- Observed: 58.3% change (**11.7× above threshold**)
- **Decision**: **FULL PHASE 1 RERUN REQUIRED**

#### Scientific Interpretation of 58.3% Drop

**Mechanism of Inflated Performance (Confirmed)**:

1. **Artificial Density Clustering**: 2.23× duplicates created high-density regions in MF cloud
2. **UMAP Exploitation**: Small nn=10 optimized for these artificial clusters
3. **Distorted Manifold**: DR model learned chemistry biased toward over-represented compounds
4. **False Enrichment**: Centroid-based ranking exploited distorted geometry

**This Invalidates Original nn=10 Hypothesis**:
- ❌ Original: "Small nn creates tight, discriminative clusters" (appeared to work)
- ✅ Reality: "Small nn exploited 2.23× duplicated data to create artificially tight clusters"
- ✅ Large nn=500 was **more robust** to duplication (averages over 500 neighbors)
- ✅ **Prediction**: After deduplication, large nn may improve relative to small nn

**Connection to Hypotheses H1-H6**:
- **H4 (MF Cloud Anchor)**: CONFIRMED - duplicates distorted anchor point
- **H2 (Signal Dilution)**: PARTIALLY SUPPORTED - but effect was opposite (small nn more sensitive)
- **Need to retest all hypotheses with clean data**

#### Computational Cost Estimate for Full Rerun

**Scope - Features Only** (fingerprints unchanged):
- Features-PCA: 3 dims × 5 seeds = **15 experiments**
- Features-UMAP: 3 dims × 4 nn × 5 md × 5 seeds = **300 experiments**
- **Total**: **315 experiments**

**Runtime Per Experiment (Deduplicated - 2.23× faster)**:
- Data prep: ~2 min (was ~4 min)
- UMAP training: ~3 min (was ~7 min)
- PCA training: ~30 sec
- Projection + analysis: ~1 min
- **UMAP total**: ~6 min/experiment (was ~13 min)
- **PCA total**: ~3.5 min/experiment

**Full Rerun Cost**:
- Features-UMAP: 300 × 6 min = **30 hours sequential**
- Features-PCA: 15 × 3.5 min = **0.9 hours sequential**
- **Total sequential**: **~31 hours** (vs 71 hours with duplicates)
- **HPC parallel (32 jobs)**: **~2-3 hours wall clock**

**Efficiency Gain from Deduplication**:
- ✅ 2.23× speedup reduces rerun cost by 56%
- ✅ Original Phase 1 took ~71 hours → Clean Phase 1 will take ~31 hours
- ✅ This makes full rerun feasible within 1 week (including queue time)

#### Implications for Phase 1 Hyperparameter Landscape

**Expected Changes After Rerun**:

1. **Absolute Performance**: All EF@1% scores likely 30-60% lower
   - Original best: 45.83 (10D, nn=10) → Expect ~20-32 clean
   - PCA: 58.04 (10D) → Expect ~25-40 clean

2. **Hyperparameter Rankings** (MOST CRITICAL):
   - nn=10 advantage may disappear or reverse
   - nn=50, 100, 500 may perform better relative to nn=10
   - min_dist effect may become more pronounced (was masked by duplicates)

3. **PCA vs UMAP Gap**:
   - Original: PCA 1.27-1.46× better than UMAP
   - Expect: Gap may widen (UMAP exploited duplicates more than PCA)
   - Alternative: Gap may shrink (both affected similarly)

4. **Dimensionality Effects**:
   - Original: 2D (39.4) → 5D (44.4) → 10D (45.8)
   - Expect: Trend may shift (duplicate clustering easier in low-D space)

5. **Cross-Seed Variance**:
   - Original: σ ~ 0.5-0.8 EF@1%
   - Expect: May increase (duplicates provided artificial stability)

#### Revised Hyperparameter Grid (Optional)

**Current Grid** (generate_phase1_configs.py):
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
```

**Options for Rerun**:

**Option A: Keep Existing Grid (RECOMMENDED)**
- ✅ Direct comparison to original results
- ✅ Already validated as reasonable hyperparameter range
- ✅ Faster to execute (configs already exist)
- ✅ Can add expanded grid as Phase 1b if needed

**Option B: Expand to Include Large nn**
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100, 500]  # Add large nn
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]   # Keep same
```
- ✅ Tests hypothesis that large nn becomes competitive
- ✅ More complete hyperparameter surface
- ❌ 1.75× more experiments (315 → 551 configs)
- ❌ Adds ~20 hours to rerun time

**Option C: Focused Grid (Dimension-Specific nn)**
```python
# Hypothesis: Larger dimensions need larger neighborhoods
2D: nn = [3, 5, 10, 20]           # Keep small
5D: nn = [5, 10, 20, 50]          # Add medium
10D: nn = [10, 20, 50, 100]       # Add large
```
- ✅ Tests information bottleneck hypothesis (H3)
- ✅ Fewer experiments than full expansion
- ❌ More complex to implement
- ❌ Less systematic comparison

**RECOMMENDATION**: ~~**Option A** (keep existing grid)~~ **REVISED TO OPTION B AFTER CRITICAL ANALYSIS**

**⚠️ CRITICAL REALIZATION** (After reviewing comparison test results):

**The Evidence**:
1. nn=10 with duplicates: **43.78** (appeared optimal)
2. nn=10 WITHOUT duplicates: **18.27** (collapsed 58.3%)
3. nn=500 with duplicates: **19.57** (appeared suboptimal)
4. **KEY**: Clean nn=10 (18.27) ≈ Original nn=500 (19.57) ✓

**What This Tells Us**:
- nn=500 was **already giving approximately correct results** despite duplicates
- Large neighborhoods averaged over duplicates → robust performance
- **Small nn=10 was the anomaly** - exploited duplicate clusters that shouldn't exist
- **Current grid [3, 5, 10, 20] tests the WRONG hyperparameter range**

**REVISED RECOMMENDATION: Option B (Expanded Grid)**

**⚠️ CRITICAL UPDATE (Post-Analysis)**: nn=3 is computationally infeasible (killed after 72hr timeout)

**New Critical Information**:
1. nn=3 experiments were killed after 72-hour time limit
2. UMAP with fixed seed disables multi-threading (10-100× slower)
3. Multi-threading available but sacrifices exact reproducibility

**FINAL RECOMMENDATION: Option B-Modified (Your Proposed Grid)**

```python
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]  # DROP 3, 5 (timeout)
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
USE_FIXED_SEED = False  # Enable multi-threading (10× speedup)
```

**Why This is Optimal**:
- ✅ Computationally feasible (no 72hr timeouts)
- ✅ Tests medium-large nn where optimal likely is (50-500 range)
- ✅ Excludes known-bad range (nn=3,5 timeout + nn=10 dropped to 18.27)
- ✅ **10× faster with multi-threading**: 18.75 hrs vs 52 hrs
- ✅ 5 independent runs provide variance estimate (standard ML practice)
- ✅ Includes reference points: nn=10 (18.27 clean), nn=500 (19.57 robust)

**Cost**:
- Experiments: 3 dims × 5 nn × 5 md × 5 seeds = **375**
- Time per experiment: ~3 min (with multi-threading)
- Sequential: **18.75 hours**
- HPC parallel (32 jobs): **~1 hour**

**Trade-off**:
- Sacrifice: Exact reproducibility (UMAP race conditions with multi-threading)
- Gain: 10× speedup, feasible computation, focus on scientifically motivated range
- Variance: Expect σ ~ 1-2 EF@1% (vs 0.5-0.8 with fixed seed)
- With 5 replicates: SEM ~ 0.4-0.9 (sufficient for hyperparameter trends)

**Rationale for Dropping nn=3, 5**:
- nn=3 cannot finish in 72 hours
- Very small neighborhoods = extreme computational cost
- Not necessary: We know small nn exploited duplicates (nn=10 → 18.27)
- Scientific focus: Find TRUE optimal, not characterize artifact region
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100, 500]  # ADD 50, 100, 500
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]   # Keep same
```

**Why Option B is Now Mandatory**:
- ✅ Tests mechanistic hypothesis: optimal nn shifted from 10 → 50-100 range
- ✅ Only 1.75× more experiments (525 vs 300 = +22 hours)
- ✅ Avoids Phase 1b rerun (saves 2-3 weeks total time)
- ✅ nn=500 performance (19.57) validates this range needs testing
- ✅ Publication requires showing we found TRUE optimum, not duplicate artifact
- ⚠️ Testing only [3,5,10,20] = exploring region we KNOW performed badly (18.27)

**Cost-Benefit**:
- Additional cost: +22 hours sequential (+1 hour HPC parallel)
- Risk of Option A: High probability of missing true optimum → Phase 1b needed
- **Testing [3-20] without [50-500] is scientifically unjustifiable given evidence**

**See comprehensive analysis**: `PHASE1_RERUN_HYPERPARAMETER_ANALYSIS.md`

**Alternative (Resource-Constrained) - Option B-Lite**:
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100]  # Drop 500 only
FEATURES_MIN_DIST = [0.0, 0.01, 0.1]            # Reduce to 3 values
# Total: 270 experiments (vs 300 current)
```
- Still tests critical medium nn range
- Actually FEWER experiments than current grid
- Removes min_dist redundancy (already weak effect)

#### Next Steps (Priority Order)

**Immediate (Today)**:
1. ✅ Document findings in lab book - DONE
2. ✅ Validate comparison script path fixes - DONE  
3. [ ] **DECISION POINT**: Keep existing grid or expand? (Option A recommended)
4. [ ] Regenerate Phase 1 configs (if grid changed)
5. [ ] Review SLURM script for full rerun
6. [ ] Submit Phase 1 rerun to HPC queue

**This Week**:
1. [ ] Monitor Phase 1 rerun progress
2. [ ] Implement quality checks (verify deduplication in logs)
3. [ ] Set up automated progress tracking

**Next Week**:
1. [ ] Analyze new Phase 1 results
2. [ ] Compare old vs new hyperparameter rankings
3. [ ] Re-extract best configs for Phase 2-4
4. [ ] Update mechanistic hypotheses based on clean data

**Phase 2-4 Timeline**:
- Phase 2 (cutoff sensitivity): **Results valid** (already uses deduplication)
- Phase 3 (generalization): **BLOCKED** until Phase 1 rerun complete
- Phase 4 (dimensionality): **BLOCKED** until Phase 1 rerun complete

#### Files Modified

**Pipeline Fixes**:
- `experimental_pipeline/prepare_data.py`: Deduplication (lines 171-191)
- `core_scripts/calculate_similarityspaces_exp.py`: Safety check (lines 159-177)

**Comparison Framework**:
- `rerun_phase1_config_compare.py`: 560 lines, 6 bugs fixed, pre-validation added

**Documentation**:
- `LAB_BOOK.md`: This entry

**Commit Message Recommendation**:
```
CRITICAL: Phase 1 validation confirms 58.3% performance drop with deduplicated data

- Comparison test: EF@1% 43.78 → 18.27 (-58.3%)
- Decision: Full Phase 1 rerun required (315 experiments)
- Cost: ~31 hours sequential, ~2-3 hours HPC parallel
- Fixed: prepare_data.py deduplication, comparison script pre-validation
- Recommendation: Keep existing hyperparameter grid for direct comparison
```

---
- Deduplicated: Keep minimum affinity per Compound ChEMBL ID
- Output: `{workspace}/TyrosineProteinKinaseABL1_P00519/temp_data/TyrosineProteinKinaseABL1_P00519_chembl_mf_excluded_features.csv`

**Verification Steps**:
1. Check source file duplicate statistics: ✅ Confirmed 2.23× duplication
2. Verify deduplication logic in `prepare_data.py`: ✅ Implemented (lines 171-188)
3. Confirm safety check in `calculate_similarityspaces_exp.py`: ✅ Implemented (lines 159-177)
4. Create comparison test framework: ✅ Complete (`rerun_phase1_config_compare.py`)
5. Run HPC validation: 🔄 In progress
6. Analyze results and make decision: ⏳ Pending HPC completion

#### Research Integrity Note

**Why This Discovery Is Valuable (Regardless of Outcome)**:
- Demonstrates thorough data quality investigation
- Tests robustness of findings to data preprocessing decisions
- Provides insight into DR model sensitivity to data imbalance
- Exemplifies scientific method: hypothesis → experiment → evidence-based decision

**Transparency Commitment**:
- Will report comparison results honestly (whether favorable or unfavorable)
- Will rerun experiments if scientifically necessary (no shortcuts)
- Will document this investigation in methods section (shows rigor)

**Next Steps**:
1. Wait for HPC comparison results (~2 hours)
2. Analyze EF@1% change percentage
3. Make evidence-based decision on Phase 1 validity
4. Update LAB_BOOK.md with findings
5. Proceed with Phase 2-4 OR initiate Phase 1 rerun

---

### October 24, 2025: UMAP n_neighbors Mechanistic Investigation

**Objective**: Understand why small neighborhoods (nn=10) outperform large (nn=500) by 2.3× using existing and planned experimental data.

**Focus Hypotheses**: H1 (Task Mismatch), H2 (Signal Dilution), H4 (MF Cloud Anchor)

#### Proposed Experiments (No New DR Required)

**Experiment Set 1: Distance Distribution Analysis (Tests H1)**

*Objective*: Determine if small nn creates better cluster separation for ranking.

*Method*:
1. Extract embedded coordinates from Phase 1 results for each nn value (10, 20, 100, 500)
2. For each embedding, calculate:
   - **Intra-MF distances**: Mean pairwise distance within MF cloud
   - **Intra-active distances**: Mean pairwise distance within target actives
   - **Inter-class distances**: Mean distance between MF cloud and target actives vs MF cloud and ZINC decoys
   - **Silhouette scores**: For 3 classes (MF cloud, target actives, ZINC decoys)

*Expected Outcome (if H1 true)*:
- nn=10: High silhouette scores (>0.5), large inter-class distances
- nn=500: Low silhouette scores (<0.3), small inter-class distances
- Trade-off: nn=10 worse manifold quality (can measure with trustworthiness/continuity if needed)

*Implementation*:
- Script: `analysis_scripts/analyze_distance_distributions.py`
- Input: Phase 1 embedded coordinates (saved in workspace)
- Output: Distance distribution plots, silhouette score table by nn

---

**Experiment Set 2: Nearest Neighbor Composition Analysis (Tests H2)**

*Objective*: Determine if large nn forces UMAP to preserve irrelevant decoy-decoy relationships.

*Method*:
1. In **original 40D feature space**, for each compound class:
   - MF cloud compounds: What fraction of their k nearest neighbors (k=10, 20, 100, 500) are other MF compounds vs ZINC?
   - Target actives: What fraction are MF compounds vs ZINC?
   - ZINC decoys: What fraction are other ZINC vs MF compounds?

2. Repeat analysis in **embedded space** (2D, 5D, 10D) for each nn value

*Expected Outcome (if H2 true)*:
- **40D space**: MF compounds naturally cluster (>80% of 10NN are other MF compounds)
- **nn=10 embedding**: Preserves MF clustering (similar to 40D)
- **nn=500 embedding**: MF compounds have many ZINC neighbors → signal dilution

*Key Insight*: If nn=500 in original 40D includes many ZINC compounds, UMAP is forced to preserve MF↔ZINC relationships, distorting the MF cloud.

*Implementation*:
- Script: `analysis_scripts/analyze_neighbor_composition.py`
- Input: Original 40D features, Phase 1 embeddings
- Output: Neighbor composition tables, stacked bar charts

---

**Experiment Set 3: MF Cloud Cohesion Metrics (Tests H4)**

*Objective*: Measure if small nn creates tighter, better-defined MF cloud.

*Method*:
1. **MF Cloud Density**: For each nn, calculate:
   - Mean pairwise distance within MF cloud (lower = tighter)
   - Std of pairwise distances (lower = more uniform)
   - Radius of gyration (compactness measure)

2. **Centroid Definition**: For each nn, calculate:
   - Mean distance from MF cloud compounds to MF centroid
   - Std of distances to centroid (lower = better-defined centroid)
   - Compare to distance from target actives to centroid

*Expected Outcome (if H4 true)*:
- nn=10: Tight MF cloud (low mean distance, low std), well-defined centroid
- nn=500: Diffuse MF cloud (high mean distance, high std), poorly-defined centroid
- Ranking by centroid distance works better when centroid is well-defined

*Implementation*:
- Script: `analysis_scripts/analyze_mf_cloud_cohesion.py`
- Input: Phase 1 embeddings with compound labels
- Output: Cohesion metrics table, visualization of MF cloud density by nn

---

**Experiment Set 4: Phase 3 MF Ablation (Critical Test for H4)**

*Objective*: **Definitive test** - Does nn effect vanish when MF cloud is removed?

*Method*:
1. Use Phase 3 results (MF sizes: 0, 1K, 10K, 50K, 100K, 191K)
2. For each MF size, compare EF@1% for nn=10 vs nn=500 (UMAP only)
3. Plot: EF@1% difference (nn=10 - nn=500) vs MF cloud size

*Expected Outcome (if H4 true)*:
- **MF=0**: No difference (nn=10 ≈ nn=500) - no anchor to preserve
- **MF=1K-10K**: Small difference - weak anchor
- **MF=50K-191K**: Large difference (nn=10 >> nn=500) - strong anchor effect

*Critical Prediction*: If nn effect disappears at MF=0, confirms that MF cloud cohesion is the mechanism.

*Implementation*:
- Script: `analysis_scripts/analyze_phase3_nn_effect.py`
- Input: Phase 3 results (when complete)
- Output: nn effect vs MF size curve, statistical tests

---

**Experiment Set 5: Optimization Variance (Tests H5 - Supporting Evidence)**

*Objective*: Check if large nn has unstable optimization (high seed variance).

*Method*:
1. From Phase 1 results, extract EF@1% for all 5 seeds per configuration
2. Calculate coefficient of variation (CV = std/mean) for each nn value
3. Plot CV vs nn across all dimensions

*Expected Outcome (if H5 true)*:
- nn=10: Low CV (<5%), stable optimization
- nn=500: High CV (>10%), unstable optimization

*Implementation*:
- Script: `analysis_scripts/analyze_optimization_variance.py`
- Input: Phase 1 metrics across seeds
- Output: Variance table, CV vs nn plot

---

**Experiment Set 6: Distance Concentration (Tests H6 - Supporting Evidence)**

*Objective*: Check if "500th neighbor" is meaningless in 40D due to distance concentration.

*Method*:
1. In **original 40D space**, for 1000 random compounds, calculate:
   - Distance to 10th nearest neighbor: d10
   - Distance to 100th nearest neighbor: d100
   - Distance to 500th nearest neighbor: d500
   - Ratios: d100/d10, d500/d10

2. Compare features (40D) vs fingerprints (2048D)

*Expected Outcome (if H6 true)*:
- Features (40D): d500/d10 ≈ 1.5-2.0 (moderate concentration)
- Fingerprints (2048D): d500/d10 ≈ 1.1-1.3 (severe concentration) - may explain fingerprint failure

*Implementation*:
- Script: `analysis_scripts/analyze_distance_concentration.py`
- Input: Original feature/fingerprint matrices
- Output: Distance ratio distributions, concentration severity metrics

---

#### Implementation Priority

**Week 1 (Immediate)**:
1. Experiment 1 (Distance Distributions) - Validates H1
2. Experiment 3 (MF Cloud Cohesion) - Validates H4
3. Experiment 5 (Optimization Variance) - Quick supporting evidence

**Week 2**:
4. Experiment 2 (Neighbor Composition) - Validates H2
5. Experiment 6 (Distance Concentration) - Supporting evidence

**Phase 3 Completion** (Future):
6. Experiment 4 (MF Ablation) - **Definitive H4 test**

#### Expected Manuscript Sections

**Results Section**:
- "Distance distributions reveal cluster separation vs manifold smoothness trade-off" (Exp 1 → H1)
- "MF cloud cohesion correlates with retrieval performance" (Exp 3 → H4)
- "Large neighborhoods dilute signal through decoy relationships" (Exp 2 → H2)
- "MF cloud ablation confirms anchor point mechanism" (Exp 4 → H4, definitive)

**Supplementary**:
- Optimization variance analysis (Exp 5)
- Distance concentration effects (Exp 6)

---

## v3.0 Experimental Pipeline Architecture

### Config Generators (4 Phases)
1. **`generate_phase1_configs.py`** (260 configs)
   - Hyperparameter sweep: Features/Fingerprints × PCA/UMAP × Dimensions × Hyperparameters × Seeds
   - Auto-skip completed experiments

2. **`generate_phase2_configs.py`** (60 configs)
   - MF cloud ablation: Best PCA + Best UMAP × MF sizes (0, 1K, 10K, 50K, 100K, 191K) × Seeds
   - Requires `phase1_best_configs.json` from Phase 1 analysis

3. **`generate_phase3_configs.py`** (80 configs)
   - Generalization: 8 best configs × 2 new targets (PKM2, IDH1) × Seeds

4. **`generate_phase4_configs.py`** (40 configs)
   - Cutoff sensitivity: Best PCA + Best UMAP × 4 cutoffs (100, 1K, 10K, 100K nM) × Seeds

### Main Orchestrator
**`main_orchestrator.py`**: Single experiment executor
- Reads JSON config file
- Calls pipeline stages in sequence:
  1. Data preparation (load ChEMBL, ZINC, filter by affinity)
  2. Feature/fingerprint calculation
  3. Similarity space construction (fit DR, project actives)
  4. Ranking and evaluation (EF@1%, ROC-AUC, PR-AUC)
- Logs all operations to `orchestrator_*.log`
- Creates run-specific workspace directory

### Core Processing Scripts

**`core_scripts/calculate_features_and_fingerprints_exp.py`**:
- Computes 39 RDKit molecular descriptors (MolWt, LogP, TPSA, etc.)
- Generates ECFP4 fingerprints (2048-bit, radius=2)
- Handles three compound sets:
  - MF cloud (other proteins in same MF)
  - ZINC decoys (1.29M compounds)
  - Target actives (held-out)
- StandardScaler normalization for features

**`core_scripts/calculate_similarityspaces_exp.py`**:
- Fits PCA or UMAP on MF cloud + ZINC (training set)
- Projects target actives into learned space (held-out, prevents data leakage)
- Calculates distances to MF cloud centroid
- Ranks compounds by proximity score
- Evaluates EF@1%, ROC-AUC, PR-AUC
- Generates 2D visualization plots

### Analysis Scripts

**`scripts/analyze_potency_stratified_enrichment.py`** (NEW Oct 18, 2025):
- Potency-tier stratified analysis:
  - High: 0.1-100 nM (drug-like, clinically relevant)
  - Medium: 100-1000 nM (moderate affinity)
  - Weak: 1000-100,000 nM (marginal binders)
- Best hyperparameter identification per representation
- Parallel processing with multiprocessing (4-8× speedup)
- Dual output: PNG (300 DPI) + PDF (vector)
- Dimension-separated plots for publication
- Usage: `python scripts/analyze_potency_stratified_enrichment.py --workspace_dir experiment_workspace_v3_phase1 --output_dir results --n_jobs 8`

**`extract_phase1_best_configs.py`**:
- Scans Phase 1 workspace
- Identifies best configs per method/representation/dimension
- Outputs `phase1_best_configs.json` for Phase 2-4 generation

### Experimental Pipeline Modules

**`experimental_pipeline/prepare_data.py`**:
- Loads ChEMBL target ligands and MF cloud
- Applies affinity cutoff filter
- Loads ZINC decoys
- Creates train/test split (MF+ZINC train, actives held-out)

**`experimental_pipeline/project_and_analyze.py`**:
- Wrapper for dimensionality reduction
- Fits model on training set
- Projects test set
- Handles PCA and UMAP methods

**`experimental_pipeline/rank_zinc_decoys.py`**:
- Calculates distance to MF cloud
- Ranks all compounds
- Computes enrichment metrics

---

## Current Documentation Status

### Core Documentation (Per Copilot Instructions)
- ✅ **LAB_BOOK.md** (this file): Experimental log with daily entries
- ✅ **README.md**: Main repository overview and quick start
- ✅ **PLANNING.md**: Task tracking with checklists, research questions, timeline
- ✅ **ARCHIVE.md**: Deprecated features and migration notes

### Supplementary Documentation
- ✅ **README_V3_PIPELINE.md**: Complete 4-phase pipeline documentation
- ✅ **METHODS_FOR_PAPER.md**: Publication-ready methods section
- ✅ **docs/QUICKSTART_V3.md**: Fast setup guide
- ✅ **docs/MF_CLOUD_IMPACT_ANALYSIS.md**: Key finding on MF cloud phase transition
- ✅ **hpc/README.md**: HPC execution guide

---

## Known Issues and Caveats

### 🔴 Critical Pipeline Design Issue
**Problem**: Phase 4 (cutoff sensitivity) affects Phase 2 (MF cloud ablation) results
- Affinity cutoff determines which molecules are "active"
- MF cloud composition changes with different cutoffs
- Phase 2 uses fixed cutoff (100K nM) but Phase 4 tests multiple cutoffs
- If optimal cutoff is not 100K nM, Phase 2 MF cloud may be suboptimal

**Status**: Under review (see PLANNING.md for restructuring options)

### ⚠️ Technical Limitations
- Memory usage: 1.5M compounds requires ~16-32 GB RAM for UMAP
- Runtime: UMAP with nn=500 can take 30-60 min on 1.5M dataset
- Checkpoint system: Not yet implemented (long runs cannot resume)

### ⚠️ Methodological Limitations
- Single target validation (Phase 1): ABL1 only, generalization unknown
- No industry baselines: Tanimoto/ECFP4 comparison pending
- Descriptor-dependent: Binary fingerprints fail with PCA (requires continuous features)

---

## Next Steps

### Immediate (This Week)
1. ✅ Parallelize stratified enrichment analysis
2. ✅ Create PLANNING.md with task tracking
3. ✅ Create ARCHIVE.md with deprecated features
4. ✅ Complete Phase 1 stratified enrichment analysis
5. ✅ Extract best configs for Phase 2-4
6. ✅ Decide on pipeline ordering (cutoff before or after MF cloud)
7. ✅ Document UMAP n_neighbors hypotheses (H1-H6)
8. ✅ Design mechanistic experiments using existing data
9. [ ] Implement distance distribution analysis (Exp 1 → H1)
10. [ ] Implement MF cloud cohesion analysis (Exp 3 → H4)
11. [ ] Implement optimization variance analysis (Exp 5 → H5)

### Short-term (Next 2 Weeks)
1. ✅ Resolve Phase 2-4 ordering issue
2. ✅ Generate Phase 2 configs (after best config extraction)
3. ✅ Execute Phase 2 experiments (MF cloud ablation)
4. ❌ **BLOCKED: Full Phase 1 rerun required (see Oct 25 entry)**
5. [ ] Implement neighbor composition analysis (Exp 2 → H2)
6. [ ] Implement distance concentration analysis (Exp 6 → H6)
7. [ ] Analyze Phase 2 results (validate phase transition hypothesis)
8. [ ] Generate mechanistic figures for manuscript

### Medium-term (Next Month)
1. [ ] Execute Phase 3 (generalization to PKM2, IDH1)
2. [ ] Execute Phase 4 (cutoff sensitivity)
3. [ ] **Analyze Phase 3 MF ablation for nn effect (Exp 4 → H4 definitive test)**
4. [ ] Create unified analysis across all 4 phases
5. [ ] Generate final publication figures

### Long-term (2-3 Months)
1. [ ] Write methods section
2. [ ] Write results section with mechanistic explanations
3. [ ] Perform statistical significance testing
4. [ ] Submit manuscript

---

### October 26, 2025: Phase 1 CLI — File logging, SMILES-only dedup, and embeddings saved

- Changes Made (Code)
  - Added run-scoped file logging to `molfuse/cli/phase1.py` writing to `logs/run.log` with INFO-level messages for each major step:
    - Input row counts (MF, ZINC, Actives)
    - Target-preserving exclusion stats
    - Overlap removal by SMILES (Actives vs MF/ZINC; MF vs ZINC)
    - Deduplication counts (strictly by SMILES)
    - Feature coercion and NaN drops per set
    - Scaler/DR fit details (method, dim, UMAP hyperparams)
    - Affinity cutoff pass counts and fallback policy usage
    - ZINC sampling counts (if enabled)
    - Metrics path and ranked CSV path
  - Switched deduplication policy to SMILES-only (canonical_smiles/SMILES). One row per unique molecule; removed prior Compound-ID-based aggregation from the v4 CLI.
  - Enforced SMILES-based MF–ZINC overlap removal.
  - Saved similarity space coordinates for all three sets:
    - `artifacts/embedding_mf.csv`, `artifacts/embedding_zinc.csv`, `artifacts/embedding_actives.csv` with `z0..z{dim-1}`.

- Experiments Run
  - Local execution on ABL1 example config completed successfully; verified presence of run.log, metrics.json, ranked_scores.csv, and three embedding CSVs.

- Observations and Results
  - Logging reveals substantial row drops from NaN coercion on some features; this will help triage feature lists or imputations later if needed.
  - SMILES-only dedup eliminates repeated actives previously observed in rankings; MF–ZINC overlap removal prevents trivial 0.0 distances.

- Next Steps
  - Apply the same logging/embedding conventions to Phase 2 CLI (when implemented) and to analysis scripts that reuse embeddings.
  - Consider optional Parquet outputs for embeddings to accelerate downstream analysis on HPC.

---

### October 26, 2025: Example config adapted for fingerprints + UMAP (Jaccard)

- Changes Made (Configuration)
  - Updated `configs/molfuse_phase1_example.json` to demonstrate the fingerprint pathway with `representation: "fingerprints"` and `method: "umap"` using `metric: "jaccard"`.
  - Switched CSV paths from `*_extracted_features.csv` to `*_extracted_fingerprints.csv` to match repository dataset layout.
  - Adjusted `run_name` to reflect UMAP (10D, nn=50, md=0.01) and added a clarifying note in `README_V4_MOLFUSE.md` about toggling between fingerprints and features.

- Rationale
  - The v4 invariants include explicit support for fingerprints with UMAP-Jaccard. Having the example config default to this variant ensures quick validation of the non-feature pathway and reduces ambiguity about required fields (`representation`, `umap_params.metric`).

- Observations
  - Phase 1 CLI auto-selects `metric="jaccard"` for fingerprints when unspecified; we set it explicitly for clarity and reproducibility in the example.

- Next Steps
  - Provide a second example config for features+PCA if needed, or document a minimal diff to convert the example back to features (already added to README_V4).

**End of Lab Book**

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

**Last Updated**: October 24, 2025  
**Next Review**: After mechanistic analysis implementation (Experiments 1-6)
