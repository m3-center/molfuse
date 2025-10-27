# molfuse v4.0 Planning & Task Tracking (formerly UMMBAS)

**Last Updated**: October 26, 2025  
**Branch**: 4.0  
**Status**: v4.0 refactor planning; Phase 1/2 pipelines to be re-implemented

---

## v4.0 Refactor Overview (Scope and Objectives)

We are refactoring the repository into “molfuse” (new name) with a clean, modular pipeline that matches the validated independent workflow while enabling independent execution of Phase 1 and Phase 2 on HPC. Key invariants:

- StandardScaler fits on MF cloud + ZINC only; held-out actives are transformed with the same scaler (no leakage).
- No fixed seed: UMAP runs with random_state=None to enable parallelism; PCA uses sklearn defaults.
- Exact 1-NN (NearestNeighbors) scoring in embedded space; cdist available as fallback only for debugging.
- Affinity cutoff applies to MF cloud for scoring only; actives are never filtered by affinity.
- Report Spearman’s rho between pActivity and score on actives.
- Enforce target-preserving exclusion: compounds linked to the target are held-out, never in MF cloud.
- Strong logging and deterministic workspace layout; analysis scripts for Phase 1 and Phase 2 included; status checker provided.

---

## v4.0 Architecture Plan

### Package layout (new)

```
molfuse/
   data/
      readers.py              # Chunked CSV IO; schema validation
      features.py             # Feature selection (rdkit list or fallback detector)
      splits.py               # Build held-out set, target-preserving exclusion, overlap removal
   dr/
      scaling.py              # StandardScaler/PassthroughScaler helpers
      pca.py                  # Projection-only PCA fit/transform
      umap.py                 # Projection-only UMAP fit/transform (random_state=None)
   scoring/
      neighbors.py            # Exact 1-NN scoring + detailed k-NN + centroid distance
   metrics/
      enrichment.py           # ROC-AUC, PR-AUC, EF@1/5/10
      correlations.py         # Spearman rho vs Standard Value (nM) → pActivity
   phases/
      phase1.py               # Hyperparameter/dim sweeps; projection-only
      phase2.py               # Cutoff sweeps; reuse Phase 1 models/simspaces
      common.py               # Shared orchestration utilities
   io/
      artifacts.py            # Save/load scalers/models/simspaces (compress_pickle)
      layout.py               # Run/workspace layout resolution
      logging.py              # Run-scoped logging setup
   analysis/
      phase1_potency.py       # Port of potency-stratified analysis (v4 API)
      phase2_cutoffs.py       # Port of Phase 2 cutoff analysis/plots (v4 API)
      status_check.py         # Live status checker (progress across runs)
   cli/
      molfuse_phase1.py       # CLI for Phase 1 execution
      molfuse_phase2.py       # CLI for Phase 2 execution
      molfuse_prepare.py      # Optional: prepare filtered MF/ZINC files
configs/
   base.json                 # Paths, features list, MF KW CSV, defaults
   phase1_grid.json          # Dims and UMAP grids (features/fingerprints)
   phase2_cutoffs.json       # Cutoff list for Phase 2
hpc/
   molfuse_phase1_cpu.sh     # SLURM single-job runner
   molfuse_phase2_cpu.sh     # SLURM single-job runner
   submit_phase1.sh          # Grid submit helper
   submit_phase2.sh          # Grid submit helper
workspace_v4/
docs/
   README.md                 # Main usage + HPC guides
   README_PHASE1.md          # Phase 1 usage and outputs
   README_PHASE2.md          # Phase 2 usage and outputs
```

Notes:
- All UMAP runs are parallel (no fixed seed). Logs explicitly state non-determinism.
- Fingerprints: UMAP-Jaccard only, no scaling; PCA allowed but documented as baseline.
- Features: UMAP-Euclidean; StandardScaler required.

### Config schema (unified)

```
{
   "project_name": "molfuse",
   "workspace_root": "workspace_v4/",
   "targets": [
      {"id_name": "TyrosineProteinKinaseABL1_P00519", "display_name": "ABL1", "uniprot_id": "P00519",
       "mf_canonical_name": "Transferase", "mf_filename_segment": "Transferase", "kw_code": "KW-0808"}
   ],
   "data_paths": {
      "chembl_affinity_full_csv": "...",
      "chembl_target_mapping_csv": "...",
      "mf_features_dir": "datasets/molecular_function_features_fingerprints/",
      "zinc_features_csv": "datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv",
      "molecular_function_keywords_csv": "datasets/chembl/molecular_function_keywords.csv"
   },
   "rdkit_features_list": ["MolWt", "TPSA", "NumAromaticRings", ...],
   "umap": {"init": "spectral", "low_memory": false},
   "scoring": {"backend": "nn", "k_for_knn": [1,5,10]},
   "phase_defaults": {
      "simspace_dims_features": [2,5,10],
      "simspace_dims_fingerprints": [2],
      "features_umap_grid": {"n_neighbors": [10,20,50,100,500], "min_dist": [0.0,0.001,0.005,0.01,0.1]},
      "fingerprints_umap_grid": {"n_neighbors": [20,50,100], "min_dist": [0.0,0.001,0.01,0.1]}
   },
   "phase2": {"affinity_cutoff_nM_list": [100,1000,10000,100000]},
   "logging": {"level": "INFO"}
}
```

Fallback feature detection mirrors the independent test (accept_ratio configurable) when rdkit_features_list is empty.

### Phase 1 (projection-only, hyperparam sweep)
- Build held-out actives from base ChEMBL via target mapping; enforce target-preserving exclusion.
- Filter MF file by accession != target, then drop any Compound IDs in the held-out set (safety).
- Remove ZINC overlaps by SMILES against MF cloud and actives.
- Feature selection: prefer curated list; else fallback intersection of mostly-numeric columns; coerce to numeric; drop rows with any NaN across selected features; log losses.
- Fit StandardScaler on MF+ZINC only (features); Passthrough for fingerprints.
- Train DR on MF+ZINC only; UMAP random_state=None for parallelism; save models.
- Project held-out actives with the saved scaler+model.
- Apply affinity cutoff to MF cloud for scoring only; actives never filtered.
- Score by exact 1-NN to MF cloud; compute ROC-AUC, PR-AUC, EF@1/5/10, Spearman rho; save detailed distances.
- Artifacts: models/, simspaces/, results/ with consistent filenames; logs per run.

### Phase 2 (cutoff sweeps)
- Reuse Phase 1 models and MF+ZINC simspaces.
- For each cutoff: filter MF cloud, rescore actives+ZINC, recompute metrics and Spearman rho; save outputs.

### Analyses (included in v4.0)
- Phase 1 potency-stratified enrichment (port of `analyze_potency_stratified_enrichment.py`) under `molfuse/analysis/phase1_potency.py` with CLI wrapper; PNG+PDF outputs; multiprocessing.
- Phase 2 cutoff sensitivity analysis (port of `analyze_phase2_cutoff_sensitivity.py`) under `molfuse/analysis/phase2_cutoffs.py` with the same figures (curves, heatmap, quality–quantity, tier-specific EF curves); CLI wrapper.
- Status checker (port of `check_hyperparam_status.py`) under `molfuse/analysis/status_check.py`: scans workspace_v4, reports completed/running/failed, extracts metrics, and groups by repr/method/dim.

### HPC integration
- `hpc/molfuse_phase1_cpu.sh` and `hpc/molfuse_phase2_cpu.sh` for single-job runs; no seeds passed to UMAP; logs to slurm_logs/ and run logs.
- `hpc/submit_phase1.sh` and `hpc/submit_phase2.sh` enumerate grids from configs and submit arrays.

---

## v4.0 Task Checklist

### 🚧 Refactor scaffolding
- [ ] Create molfuse/ package structure and empty modules
- [ ] Add cli/ entry points for Phase 1 and Phase 2
- [ ] Add configs/base.json, phase1_grid.json, phase2_cutoffs.json (skeletons)
- [ ] Add hpc scripts and submit helpers
- [ ] Update README.md and docs/ for v4.0 usage

### 🧪 Phase 1 (first target: ABL1)
- [ ] Implement data.splits with target-preserving exclusion and overlap removal
- [ ] Implement dr.scaling/pca/umap (projection-only; UMAP parallel)
- [ ] Implement scoring.neighbors and metrics (EFs, AUCs, Spearman)
- [ ] Wire phases.phase1 to run the sweep grid; log artifacts
 - [x] Add run-scoped file logging and save embeddings (MF/ZINC/Actives)
 - [x] Persist scaler/model artifacts (joblib) in run artifacts
 - [x] Create Phase 1 config generator and SLURM scripts

### 🧪 Phase 2 (reuse Phase 1 models)
- [ ] Implement phases.phase2 cutoff sweeps; reuse simspaces/models
- [ ] Verify MF cutoff filtering logic and metrics
 - [ ] Mirror Phase 1 logging and embedding outputs in Phase 2

### 📊 Analyses
- [ ] Port potency-stratified analysis (Phase 1) to molfuse/analysis/phase1_potency.py
- [ ] Port cutoff sensitivity analysis (Phase 2) to molfuse/analysis/phase2_cutoffs.py
- [ ] Port status checker to molfuse/analysis/status_check.py
 - [x] Phase 1 post-analysis plotting overhaul in `scripts/phase1_post_analysis.py` (combined bars, shared-scale heatmaps, seed-variability grid, fixed-range distance plots; CLI flags + plots manifest)

### 🔧 Quality gates
- [ ] Add end-to-end dry-run on small ZINC sample (local)
- [ ] Validate logging, artifacts, and metrics on a single config
- [ ] HPC smoke test: 1-2 jobs per phase

---

### 📚 Documentation
- [x] PUBLICATION.md: Update to v4 (molfuse) and list RDKit feature descriptors (via header-only read)
- [ ] README.md: Add v4 invariants quick link and PUBLICATION reference
 - [x] README.md: Document post-analysis usage and new flags
- [ ] ARCHIVE.md: Initialize and document deprecated items (cdist brute-force path; ID-based dedup)

---

## Task Checklist (legacy v3.0 — archived)

### ✅ Completed Tasks

- [x] Design and implement 4-phase experimental pipeline (Phase 1-4)
- [x] Create config generators for all 4 phases (260+60+80+40 = 440 experiments)
- [x] Implement main orchestrator for automated experiment execution
- [x] Create feature/fingerprint calculation pipeline
- [x] Create similarity space calculation pipeline
- [x] Implement potency-stratified enrichment analysis
- [x] Add parallelization to stratified enrichment analysis (multiprocessing)
- [x] Generate PNG + PDF outputs for publication-quality figures
- [x] Add dimension-separated plots for key analyses
- [x] Apply best hyperparameter filtering to dimensionality plots
- [x] Complete Phase 1 experiments (260 runs on ABL1 kinase)
- [x] Identify best UMAP hyperparameters (nn, min_dist) per representation
- [x] Document MF cloud phase transition hypothesis
- [x] Create PLANNING.md with task tracking and checklists
- [x] Create ARCHIVE.md documenting deprecated features
- [x] Move deprecated scripts to archived_scripts/ directory

### 🔄 In Progress

- [ ] Analyze Phase 1 results with potency stratification
- [ ] Extract best configurations for Phase 2-4 (`extract_phase1_best_configs.py`)
- [ ] Validate orchestrator recomputation safeguard (freshness check) by rerunning a representative PCA-features config post-dedup; record EF@1% in LAB_BOOK.md
- [ ] Add integrity assertion: verify MF cloud row count in simspace equals dedup source; abort with actionable message on mismatch

### ⚙️ Scoring Optimization & Validation
- [ ] Run benchmark: `analysis_scripts/benchmark_cdist_vs_kdtree.py` on representative PCA/UMAP configs (2D/5D/10D) with full MF cloud and large ZINC sample
- [ ] Decide adoption: If exact NN matches `cdist` within tolerance and is faster, switch `project_and_analyze.py` to use exact NN (feature-flagged), retain `cdist` as fallback
- [ ] Document change in `README.md` and `LAB_BOOK.md`; add a brief note in `ARCHIVE.md` if the brute-force path becomes deprecated

### 📋 Pending Tasks

#### Phase 2: Affinity Cutoff Sensitivity (40 runs) **[REORDERED - was Phase 4]**
- [x] **RESOLVED**: Phase 4 moved to Phase 2 (cutoff must precede MF cloud)
- [x] Created `generate_phase2_configs.py` (cutoff sensitivity, 40 runs)
- [x] Created `scripts/run_phase2_cutoff_analysis.py` (orchestrator with data reuse)
- [x] Aligned cutoffs with potency tiers (100 nM, 1 μM, 10 μM, 100 μM)
- [ ] Extract Phase 1 best configs (`python extract_phase1_best_configs.py`)
- [ ] Generate Phase 2 configs (`python generate_phase2_configs.py`)
- [ ] Execute Phase 2 experiments (~10-15 hours, reusing Phase 1 data)
- [ ] Analyze Phase 2 results with potency stratification
- [ ] Determine optimal cutoff for Phase 3/4

#### Phase 3: MF Cloud Ablation (60 runs) **[REORDERED - was Phase 2]**
- [ ] Update `generate_phase3_configs.py` to use optimal cutoff from Phase 2
- [ ] Generate Phase 3 configs using best hyperparameters
- [ ] Execute Phase 3 experiments (MF sizes: 0, 1K, 10K, 50K, 100K, 420K)
- [ ] Analyze MF cloud size impact on PCA vs UMAP performance
- [ ] Validate phase transition hypothesis (UMAP→PCA crossover)

#### Phase 4: Cross-Protein Generalization (80 runs) **[REORDERED - was Phase 3]**
- [ ] Update `generate_phase4_configs.py` to use optimal cutoff from Phase 2
- [ ] Generate Phase 4 configs for Pyruvate Kinase M2 (same MF)
- [ ] Generate Phase 4 configs for Isocitrate Dehydrogenase (different MF)
- [ ] Execute Phase 4 experiments
- [ ] Analyze same-MF vs different-MF generalization
- [ ] Test dimension transferability across proteins

#### Analysis & Reporting
- [ ] Create unified analysis pipeline for all 4 phases
- [ ] Generate comparative plots across phases
- [ ] Perform statistical significance testing (Wilcoxon, t-tests)
- [ ] Create final figures for publication
- [ ] Write methods section for paper
- [ ] Write results section for paper

#### Technical Improvements
- [ ] Add checkpoint/resume functionality to main orchestrator
- [ ] Implement progress tracking across multi-day HPC runs
- [ ] Add automated error detection and recovery
- [ ] Create validation scripts for output files
- [ ] Optimize memory usage for large-scale experiments
- [ ] Unit test: simulate input mtime update and assert simspace recomputation is triggered (orchestrator)

---

## Research Questions & Hypotheses

### Primary Research Question
**Does a molecular function (MF) cloud improve virtual screening performance for held-out proteins within the same MF category?**

**Hypothesis**: Ligands active against a target protein will be chemically proximal to compounds that modulate other proteins sharing the same molecular function.

### Phase-Specific Questions

#### Phase 1: Hyperparameter Optimization
- **Q1**: What are optimal UMAP hyperparameters (n_neighbors, min_dist) for features vs fingerprints?
- **Q2**: What dimensionality (2D, 5D, 10D) gives best enrichment for PCA vs UMAP?
- **Q3**: Do physicochemical features or structural fingerprints perform better?

**Status**: ✅ Complete - Results show PCA features-5D and UMAP features (nn=3-10, md=0.0-0.01) perform best

#### Phase 2: Affinity Cutoff Sensitivity **[REORDERED - was Phase 4]**
- **Q1**: Which cutoff gives best High-potent EF@1% (0.1-100 nM ligands)?
- **Q2**: Do PCA and UMAP prefer different cutoffs?
- **Q3**: How does cutoff affect Medium (100-1000 nM) vs Weak (1000-100K nM) enrichment?
- **Q4**: What is optimal cutoff for Phase 3 (MF cloud ablation)?

**Hypothesis**: Stricter cutoffs (100 nM) will enrich high-potent ligands but reduce overall diversity; permissive cutoffs (100 μM) maximize diversity but may include non-specific binders.

**Status**: 📋 Ready for execution - Config generator and orchestrator created (Oct 18, 2025)

#### Phase 3: MF Cloud Phase Transition **[REORDERED - was Phase 2]**
- **Q1**: Does MF cloud size affect PCA vs UMAP relative performance?
- **Q2**: Is there a critical MF cloud mass where PCA overtakes UMAP?
- **Q3**: What is the mechanism behind the phase transition?

**Hypothesis**: Small MF clouds favor UMAP (local structure); large MF clouds favor PCA (global variance)

**Status**: 📋 Pending - Config generator needs update to use optimal cutoff from Phase 2

#### Phase 4: Cross-Protein Generalization **[REORDERED - was Phase 3]**
- **Q1**: Do optimal hyperparameters transfer across proteins in same MF?
- **Q2**: Does performance degrade when transferring to different MF?
- **Q3**: Is dimensionality choice protein-dependent?

**Status**: 📋 Pending - Config generator needs update to use optimal cutoff from Phase 2

---

## Experimental Pipeline Architecture

### Config Generators (4 phases)
1. **`generate_phase1_configs.py`**: Hyperparameter sweep (260 configs)
   - Features/Fingerprints × PCA/UMAP × Dimensions × Hyperparameters × Seeds
   
2. **`generate_phase2_configs.py`**: MF cloud ablation (60 configs)
   - Best PCA + Best UMAP × MF sizes (0, 1K, 10K, 50K, 100K, 191K) × Seeds
   
3. **`generate_phase3_configs.py`**: Generalization (80 configs)
   - 8 best configs × 2 new targets (Pyru, Iso) × Seeds
   
4. **`generate_phase4_configs.py`**: Cutoff sensitivity (40 configs)
   - Best PCA + Best UMAP × 4 cutoffs (100, 1K, 10K, 100K nM) × Seeds

### Execution Pipeline
1. **`main_orchestrator.py`**: End-to-end experiment automation
   - Reads config JSON
   - Prepares datasets
   - Calculates features/fingerprints
   - Builds similarity spaces
   - Ranks compounds
   - Evaluates metrics

### Core Processing Scripts
1. **`core_scripts/calculate_features_and_fingerprints_exp.py`**
   - Computes 39 RDKit descriptors
   - Generates ECFP4 fingerprints (2048-bit)
   - Handles MF cloud + ZINC decoys + target actives

2. **`core_scripts/calculate_similarityspaces_exp.py`**
   - Fits PCA/UMAP on MF cloud + ZINC (training set)
   - Projects target actives (held-out)
   - Ranks by distance to MF cloud centroid
   - Calculates EF@1%, ROC-AUC, PR-AUC

### Analysis Scripts
1. **`scripts/analyze_potency_stratified_enrichment.py`**
   - Potency-tier stratified analysis (High/Medium/Weak)
   - Best hyperparameter identification
   - Parallel processing (multiprocessing)
   - PNG + PDF publication-quality outputs

2. **`analysis_scripts/`**: Various specialized analyses
   - Hyperparameter analysis
   - Dimensionality analysis
   - Generalization analysis
   - Cutoff analysis

---

## Known Issues & Decisions Needed

### 🔴 Critical: Phase Ordering Logic
**Issue**: Phase 4 (cutoff sensitivity) tests different affinity cutoffs, but Phase 2 (MF cloud ablation) uses a fixed cutoff (100,000 nM). Since cutoff determines which molecules are "active," it affects MF cloud composition and could change which method is optimal.

**Dependency Chain**:
```
Current:  Hyperparams → MF Cloud → Generalization → Cutoff
Logical:  Hyperparams → Cutoff → MF Cloud → Generalization
```

**Options**:
1. **Option A**: Restructure phases - run cutoff analysis before MF cloud ablation
2. **Option B**: Run combined MF Cloud × Cutoff study (6 × 4 = 24 conditions)
3. **Option C**: Acknowledge limitation and document in discussion

**Decision Required**: User input needed on experimental design restructure

### ⚠️ Medium Priority Issues
- Memory usage on large-scale experiments (1.5M compounds)
- Runtime optimization for UMAP (slow with large datasets)
- Checkpoint/resume functionality for multi-day HPC runs

---

## Timeline & Milestones

### October 2025
- [x] Week 1-2: Phase 1 execution complete
- [x] Week 3: Parallelization and plotting enhancements
- [ ] Week 3-4: Phase 1 analysis and best config extraction

### November 2025 (Projected)
- [ ] Week 1: Resolve phase ordering issue
- [ ] Week 1-2: Phase 2 execution (MF cloud ablation)
- [ ] Week 2-3: Phase 2 analysis
- [ ] Week 3-4: Phase 3 execution (generalization)

### December 2025 (Projected)
- [ ] Week 1-2: Phase 3 analysis
- [ ] Week 2-3: Phase 4 execution (cutoff sensitivity)
- [ ] Week 3-4: Phase 4 analysis
- [ ] Week 4: Unified analysis and paper figures

---

## Notes & Observations

### Recent Improvements (October 18, 2025)
- Added multiprocessing to `analyze_potency_stratified_enrichment.py`
  - Speedup: 4-8× on multi-core systems
  - Uses `cpu_count() - 1` workers by default
  - Maintains progress bar with `tqdm`
  
- All figures now output PNG (300 DPI) + PDF (vector) formats
  - Publication-ready quality
  - Helper function: `save_figure(fig, output_dir, basename)`

### Key Findings from Phase 1
- PCA-features-5D: EF@1% = 57.8 ± 0.9
- Best UMAP-features (nn=3-10, md=0.0-0.01): EF@1% = 46-48
- Small n_neighbors (3-10) >> Large n_neighbors (100-500)
- Fingerprints underperform features across all methods

### Outstanding Questions
1. Why does PCA outperform UMAP when literature suggests otherwise?
2. What is the mechanism behind n_neighbors sensitivity?
3. Will MF cloud ablation resolve the PCA vs UMAP puzzle?
