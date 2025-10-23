# UMMBAS v3.0 - Comprehensive Experimental Pipeline

**Version:** 3.0  
**Date:** October 2025 (Updated October 23, 2025)  
**Branch:** `3.0`

## Overview

UMMBAS v3.0 is a rigorous experimental pipeline for evaluating dimensionality reduction methods in molecular screening. The pipeline tests **PCA** vs **UMAP** across multiple dimensions, representations, and proteins to identify optimal configurations for similarity-based virtual screening.

### Key Improvements over v2.0

1. **Optimized dimensionality testing** (2D, 5D, 10D vs 2D-20D)
2. **Refined UMAP hyperparameters** (nn=10,20,100,500 × md=0.01,0.1,0.5)
3. **MF cloud ablation study** to validate phase transition hypothesis
4. **Cross-protein generalization** testing
5. **Affinity cutoff sensitivity** analysis
6. **PCA dominance investigation** module

---

## Experimental Design

**Note:** Phase ordering was revised October 18, 2025. Affinity cutoff analysis (Phase 2) now precedes MF cloud ablation (Phase 3) because the cutoff determines which molecules are included in the MF cloud, making it a logical prerequisite.

### Phase 1: Hyperparameter Sweep (260 runs)

**Target:** TyrosineProteinKinaseABL1_P00519 (Tyro)

**Methods Tested:**
- **Features-PCA:** 2D, 5D, 10D
- **Features-UMAP-Euclidean:** 2D, 5D, 10D with hyperparameters
  - n_neighbors: 10, 20, 100, 500
  - min_dist: 0.01, 0.1, 0.5
- **Fingerprints-PCA:** 2D only
- **Fingerprints-UMAP-Jaccard:** 2D only with hyperparameters

**Seeds:** 5 (42, 43, 44, 45, 46)

**Purpose:** Identify best dimensionality and hyperparameters for each method

---

### Phase 2: Affinity Cutoff Sensitivity [REORDERED - was Phase 4] (60 runs)

**Target:** Tyro only

**Cutoffs:** 100 nM, 1 μM (1,000 nM), 10 μM (10,000 nM), 100 μM (100,000 nM)

**Rationale:** Aligned with potency tiers from stratified enrichment analysis:
- 100 nM: High-potent only (drug-like, 0.1-100 nM)
- 1 μM: High + Medium potent (100-1,000 nM)
- 10 μM: High + Medium + Weak (1,000-10,000 nM)
- 100 μM: All potencies (most permissive)

**Configurations Tested:** 3 specific 5D configurations from Phase 1:
1. **PCA/features/5D** (best overall PCA)
2. **UMAP/features/5D** - overall-EF optimized (quantity-focused)
3. **UMAP/features/5D** - high-potency-EF optimized (quality-focused)

**Computational Strategy:**
- **REUSES** Phase 1 similarity spaces (features/fingerprints)
- **REUSES** Phase 1 DR models (fitted PCA/UMAP)
- **ONLY** reruns ranking with different affinity cutoffs
- Expected runtime: ~10-15 min/run vs hours for full pipeline

**Purpose:** 
- Determine optimal affinity cutoff for Phase 3 and Phase 4
- Compare quantity-optimized vs quality-optimized UMAP hyperparameters
- Understand cutoff impact on potency-stratified enrichment

---

### Phase 3: MF Cloud Ablation [REORDERED - was Phase 2] (60 runs)

**Target:** Tyro only

**MF Cloud Sizes:** 0, 1K, 10K, 50K, 100K, 420K molecules

**Methods:** 
- Best PCA-features (at best dimension from Phase 1)
- Best UMAP-features (at best dimension from Phase 1)

**Affinity Cutoff:** Uses optimal cutoff determined from Phase 2

**Purpose:** Validate phase transition hypothesis:
- PCA expected to dominate at high MF counts (gradient formation)
- UMAP expected to dominate at low MF counts (isolated clusters)
- Crossover predicted at ~10K-50K molecules

---

### Phase 4: Generalization [REORDERED - was Phase 3] (80 runs)

**Targets:**
- **PyruvateKinaseM2_P14618** (Pyru) - Same function (Transferase)
- **IsocitrateDehydrogenaseNADP_O75874** (Iso) - Different function (Oxidoreductase)

**Methods:** Best 8 configurations from Phase 1:
- PCA-features: 2D, 5D, 10D
- UMAP-Euclidean-features: 2D, 5D, 10D
- PCA-fingerprints: 2D
- UMAP-Jaccard-fingerprints: 2D

**Affinity Cutoff:** Uses optimal cutoff determined from Phase 2

**Purpose:** 
- Test generalization across proteins
- Compare same-function vs different-function transferability
- Validate dimension-wise performance consistency

---

## Total Experiment Count: 460 runs

---

## Pipeline Execution

### Step 1: Generate Phase 1 Configs

```bash
python generate_phase1_configs.py
```

Output: `hyperparam_configs_v3_phase1/` (260 config files)

### Step 2: Run Phase 1 Experiments

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase1 \
  --workspace experiment_workspace_v3_phase1 \
  --n_jobs 10
```

### Step 3: Extract Best Configs from Phase 1

Run potency-stratified enrichment analysis (includes best config extraction):

```bash
python scripts/analyze_potency_stratified_enrichment.py \
  --workspace_dir experiment_workspace_v3_phase1 \
  --output_dir potency_analysis_results
```

Output: 
- `potency_analysis_results/phase1_best_configs_by_overall_ef.json` (quantity-focused)
- `potency_analysis_results/phase1_best_configs_by_high_potency_ef.json` (quality-focused)
- Potency-stratified enrichment analysis (CSV, plots, report)

### Step 4: Generate Phase 2 Configs (Cutoff Sensitivity)

```bash
python generate_phase2_configs.py
```

Requires: `phase1_best_configs_by_overall_ef.json` and `phase1_best_configs_by_high_potency_ef.json`  
Output: `hyperparam_configs_v3_phase2_cutoff/` (60 config files)

**Note:** Configs are for PCA/features/5D + 2 UMAP/features/5D variants (overall-EF and high-potency-EF optimized)

### Step 5: Run Phase 2 Experiments (Cutoff)

**Option A - Using orchestrator script:**
```bash
python scripts/run_phase2_cutoff_analysis.py \
  --config_dir hyperparam_configs_v3_phase2_cutoff \
  --workspace experiment_workspace_v3_phase2
```

**Option B - Using HPC:**
```bash
sbatch hpc/submit_v3_phase2.sh
```

### Step 6: Generate Phase 3 Configs (MF Cloud Ablation)

```bash
python generate_phase3_configs.py
```

Requires: Best configs JSONs + Phase 2 results (for optimal cutoff)  
Output: `hyperparam_configs_v3_phase3_ablation/` (60 config files)

### Step 7: Run Phase 3 Experiments (Ablation)

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase3_ablation \
  --workspace experiment_workspace_v3_phase3 \
  --n_jobs 10
```

### Step 8: Generate Phase 4 Configs (Generalization)

```bash
python generate_phase4_configs.py
```

Requires: Best configs JSONs + Phase 2 results (for optimal cutoff)  
Output: `hyperparam_configs_v3_phase4_generalization/` (80 config files)

### Step 9: Run Phase 4 Experiments (Generalization)

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase4_generalization \
  --workspace experiment_workspace_v3_phase4 \
  --n_jobs 10
```

### Step 10: Check Status

Monitor experiment progress:

```bash
python check_hyperparam_status.py --workspace experiment_workspace_v3_phase1
```

Generates:
- Status report CSV
- Debug logs
- Visualization figures (PCA vs UMAP scatter plots)

---

## Analysis Modules

### 1. PCA Dominance Analysis

**Module:** `analysis_scripts/analyze_pca_dominance.py`

**Analyses:**
1. **Variance Explained** - PCA eigenvalue spectrum, cumulative variance
2. **Manifold Quality** - UMAP trustworthiness, continuity, neighborhood preservation
3. **Gradient Linearity** - Measure linearity of MF cloud → TARGETS gradient
4. **Distance Distributions** - Intra/inter-class distances, silhouette scores
5. **Compression Ratio Impact** - Information loss quantification
6. **MF Cloud Structure** - Density, convexity, overlap analysis

**Usage:**
```bash
python analysis_scripts/analyze_pca_dominance.py \
  --workspace_phase1 experiment_workspace_v3_phase1 \
  --workspace_phase2 experiment_workspace_v3_phase2 \
  --output pca_dominance_analysis.pdf
```

### 2. Aggregated Reporting

```bash
python aggregate_and_report.py --workspace_v3
```

Generates comprehensive PDF report with:
- EF@1%, ROC/AUC, PR/AUC metrics
- Enrichment plots
- Distance distributions
- Cross-phase comparisons
- PNG and PDF figures

---

## Key Scientific Questions

1. **What is the optimal dimensionality for PCA and UMAP?**
   - Hypothesis: 2D favors PCA due to compression, higher D favors UMAP

2. **What is the optimal affinity cutoff for each method?** (Phase 2)
   - Test: 100 nM, 1 μM, 10 μM, 100 μM cutoffs
   - Compare: Quantity-focused vs quality-focused hyperparameters

3. **Why does PCA dominate at 2D with large MF clouds?** (Phase 3)
   - Hypothesis: MF cloud creates linear gradient, PCA captures it efficiently
   - Test: Ablation study at varying MF sizes

4. **Do optimal configs generalize across proteins?** (Phase 4)
   - Test: Same function (Tyro → Pyru) vs different function (Tyro → Iso)

---

## Expected Outcomes

### Phase 1
- Best dimension per method identified
- Optimal UMAP hyperparameters per dimension
- Compression ratio effects quantified
- **Two sets of best configs:** quantity-focused (overall EF) and quality-focused (high-potency EF)

### Phase 2
- Optimal affinity cutoff identified
- Cutoff sensitivity curves per method
- Comparison: quantity vs quality optimization strategies
- Potency-stratified enrichment patterns

### Phase 3
- MF cloud phase transition curve
- Crossover point identified (~10K-50K molecules)
- Mechanistic understanding of PCA 2D advantage

### Phase 4
- Generalization performance metrics
- Same-function vs different-function comparison
- Dimension transferability validated

---

## File Structure

```
UMMBAS_screening_experiments/
├── experiment_config.json                    # Base configuration
├── generate_phase1_configs.py                # Phase 1 config generator
├── generate_phase2_configs.py                # Phase 2 cutoff config generator  
├── generate_phase3_configs.py                # Phase 3 ablation config generator
├── generate_phase4_configs.py                # Phase 4 generalization config generator
├── orchestrate_v3_pipeline.py                # Master orchestrator
├── check_hyperparam_status.py                # Status checker with viz
├── hyperparam_configs_v3_phase1/             # Phase 1 configs (260)
├── hyperparam_configs_v3_phase2_cutoff/      # Phase 2 configs (60)
├── hyperparam_configs_v3_phase3_ablation/    # Phase 3 configs (60)
├── hyperparam_configs_v3_phase4_generalization/  # Phase 4 configs (80)
├── experiment_workspace_v3_phase1/           # Phase 1 results
├── experiment_workspace_v3_phase2/           # Phase 2 results
├── experiment_workspace_v3_phase3/           # Phase 3 results
├── experiment_workspace_v3_phase4/           # Phase 4 results
├── potency_analysis_results/                 # Phase 1 analysis + best configs
│   ├── phase1_best_configs_by_overall_ef.json        # Quantity-focused
│   ├── phase1_best_configs_by_high_potency_ef.json   # Quality-focused
│   ├── stratified_enrichment_detailed.csv
│   └── plots/
├── scripts/
│   ├── analyze_potency_stratified_enrichment.py      # Phase 1 analysis + extraction
│   └── run_phase2_cutoff_analysis.py                 # Phase 2 orchestrator
├── analysis_scripts/
│   └── analyze_pca_dominance.py              # PCA dominance analysis
└── README_V3_PIPELINE.md                     # This file
```

---

## Computational Requirements

- **Phase 1:** ~260 runs × ~2-4 hours = ~520-1040 hours
- **Phase 2:** ~60 runs × ~10-15 min = ~10-15 hours (reuses Phase 1 data!)
- **Phase 3:** ~60 runs × ~2-4 hours = ~120-240 hours  
- **Phase 4:** ~80 runs × ~2-4 hours = ~160-320 hours

**Total:** ~810-1615 CPU hours (~34-67 days on single core, ~3-7 days on 10-core cluster)

**Note:** Phase 2 is dramatically faster due to data reuse strategy (similarity spaces and DR models from Phase 1).

---

## Publication Deliverables

1. **Manuscript Figures** (PNG + PDF)
   - PCA vs UMAP performance across dimensions
   - Potency-stratified enrichment analysis (Phase 1)
   - Affinity cutoff sensitivity curves (Phase 2)
   - MF cloud phase transition curve (Phase 3)
   - Generalization heatmaps (Phase 4)
   - PCA dominance mechanistic diagrams

2. **Supplementary Materials**
   - Full hyperparameter screening results
   - Quantity vs quality optimization comparison
   - Distance distribution analyses
   - Manifold quality metrics
   - Per-seed variability assessments

3. **Code/Data Repository**
   - Configuration files
   - Analysis scripts (including potency stratification)
   - Best configurations JSONs (both optimization criteria)
   - Reproducibility instructions

---

## Recent Updates (October 2025)

### October 23, 2025
- Integrated best config extraction into potency stratification analysis
- Two optimization criteria: overall EF (quantity) and high-potency EF (quality)
- Phase 2 now tests 3 specific 5D configurations (60 runs)
- Updated config paths and script names

### October 18, 2025
- **Phase reordering:** Moved cutoff analysis from Phase 4 → Phase 2
  - Rationale: Cutoff determines MF cloud composition (logical prerequisite)
  - Phase 2: Cutoff sensitivity (was Phase 4)
  - Phase 3: MF cloud ablation (was Phase 2)
  - Phase 4: Generalization (was Phase 3)

---

## Contact

For questions or issues, contact the UMMBAS development team.

**Version:** 3.0  
**Last Updated:** October 23, 2025
