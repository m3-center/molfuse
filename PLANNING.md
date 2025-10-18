# UMMBAS v3.0 Planning & Task Tracking

**Last Updated**: October 18, 2025  
**Branch**: 3.0  
**Status**: Phase 1 analysis in progress

---

## Task Checklist

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
