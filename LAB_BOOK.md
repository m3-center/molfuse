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

**Phase 2 (Planned)**: Does MF cloud size cause PCA vs UMAP performance reversal?

**Phase 3 (Planned)**: Do optimal configs transfer across proteins (same-MF vs different-MF)?

**Phase 4 (Under Review)**: How does affinity cutoff affect enrichment performance? (See pipeline ordering issue below)

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
- **Decision pending**: Need to determine if pipeline should be restructured before Phase 2 execution

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
4. [ ] Complete Phase 1 stratified enrichment analysis
5. [ ] Extract best configs for Phase 2-4
6. [ ] Decide on pipeline ordering (cutoff before or after MF cloud)

### Short-term (Next 2 Weeks)
1. [ ] Resolve Phase 2-4 ordering issue
2. [ ] Generate Phase 2 configs (after best config extraction)
3. [ ] Execute Phase 2 experiments (MF cloud ablation)
4. [ ] Analyze Phase 2 results (validate phase transition hypothesis)

### Medium-term (Next Month)
1. [ ] Execute Phase 3 (generalization to PKM2, IDH1)
2. [ ] Execute Phase 4 (cutoff sensitivity, if ordering resolved)
3. [ ] Create unified analysis across all 4 phases
4. [ ] Generate final publication figures

### Long-term (2-3 Months)
1. [ ] Write methods section
2. [ ] Write results section
3. [ ] Perform statistical significance testing
4. [ ] Submit manuscript

---

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

**Last Updated**: October 18, 2025  
**Next Review**: After Phase 1b completion and pipeline ordering decision
