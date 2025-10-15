# UMMBAS v3.0 - Comprehensive Experimental Pipeline

**Version:** 3.0  
**Date:** October 2025  
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

### Phase 2: MF Cloud Ablation (60 runs)

**Target:** Tyro only

**MF Cloud Sizes:** 0, 1K, 10K, 50K, 100K, 420K molecules

**Methods:** 
- Best PCA-features (at best dimension from Phase 1)
- Best UMAP-features (at best dimension from Phase 1)

**Purpose:** Validate phase transition hypothesis:
- PCA expected to dominate at high MF counts (gradient formation)
- UMAP expected to dominate at low MF counts (isolated clusters)
- Crossover predicted at ~10K-50K molecules

---

### Phase 3: Generalization (80 runs)

**Targets:**
- **PyruvateKinaseM2_P14618** (Pyru) - Same function (Transferase)
- **IsocitrateDehydrogenaseNADP_O75874** (Iso) - Different function (Oxidoreductase)

**Methods:** Best 8 configurations from Phase 1:
- PCA-features: 2D, 5D, 10D
- UMAP-Euclidean-features: 2D, 5D, 10D
- PCA-fingerprints: 2D
- UMAP-Jaccard-fingerprints: 2D

**Purpose:** 
- Test generalization across proteins
- Compare same-function vs different-function transferability
- Validate dimension-wise performance consistency

---

### Phase 4: Cutoff Analysis (40 runs)

**Target:** Tyro only

**Cutoffs:** 100,000 / 10,000 / 1,000 / 100 nM

**Methods:**
- Best PCA overall (at best dimension)
- Best UMAP overall (at best dimension)

**Purpose:** Determine optimal affinity cutoff threshold per method

---

## Total Experiment Count: 440 runs

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

```bash
python extract_phase1_best_configs.py \
  --workspace experiment_workspace_v3_phase1 \
  --output phase1_best_configs.json
```

Output: `phase1_best_configs.json` (8 best configurations)

### Step 4: Generate Phase 2 Configs (Ablation)

```bash
python generate_phase2_configs.py
```

Requires: `phase1_best_configs.json`  
Output: `hyperparam_configs_v3_phase2_ablation/` (60 config files)

### Step 5: Run Phase 2 Experiments

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase2_ablation \
  --workspace experiment_workspace_v3_phase2 \
  --n_jobs 10
```

### Step 6: Generate Phase 3 Configs (Generalization)

```bash
python generate_phase3_configs.py
```

Requires: `phase1_best_configs.json`  
Output: `hyperparam_configs_v3_phase3_generalization/` (80 config files)

### Step 7: Run Phase 3 Experiments

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase3_generalization \
  --workspace experiment_workspace_v3_phase3 \
  --n_jobs 10
```

### Step 8: Generate Phase 4 Configs (Cutoff)

```bash
python generate_phase4_configs.py
```

Requires: `phase1_best_configs.json`  
Output: `hyperparam_configs_v3_phase4_cutoff/` (40 config files)

### Step 9: Run Phase 4 Experiments

```bash
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase4_cutoff \
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

2. **Why does PCA dominate at 2D with large MF clouds?**
   - Hypothesis: MF cloud creates linear gradient, PCA captures it efficiently
   - Test: Ablation study at varying MF sizes

3. **Do optimal configs generalize across proteins?**
   - Test: Same function (Tyro → Pyru) vs different function (Tyro → Iso)

4. **What is the optimal affinity cutoff?**
   - Test: 100K, 10K, 1K, 100 nM cutoffs

---

## Expected Outcomes

### Phase 1
- Best dimension per method identified
- Optimal UMAP hyperparameters per dimension
- Compression ratio effects quantified

### Phase 2
- MF cloud phase transition curve
- Crossover point identified (~10K-50K molecules)
- Mechanistic understanding of PCA 2D advantage

### Phase 3
- Generalization performance metrics
- Same-function vs different-function comparison
- Dimension transferability validated

### Phase 4
- Optimal cutoff per method
- Cutoff sensitivity curves
- Method-specific cutoff recommendations

---

## File Structure

```
UMMBAS_screening_experiments/
├── experiment_config.json                    # Base configuration
├── generate_phase1_configs.py                # Phase 1 config generator
├── generate_phase2_configs.py                # Phase 2 config generator  
├── generate_phase3_configs.py                # Phase 3 config generator
├── generate_phase4_configs.py                # Phase 4 config generator
├── extract_phase1_best_configs.py            # Best config extractor
├── orchestrate_v3_pipeline.py                # Master orchestrator
├── check_hyperparam_status.py                # Status checker with viz
├── hyperparam_configs_v3_phase1/             # Phase 1 configs (260)
├── hyperparam_configs_v3_phase2_ablation/    # Phase 2 configs (60)
├── hyperparam_configs_v3_phase3_generalization/  # Phase 3 configs (80)
├── hyperparam_configs_v3_phase4_cutoff/      # Phase 4 configs (40)
├── experiment_workspace_v3_phase1/           # Phase 1 results
├── experiment_workspace_v3_phase2/           # Phase 2 results
├── experiment_workspace_v3_phase3/           # Phase 3 results
├── experiment_workspace_v3_phase4/           # Phase 4 results
├── phase1_best_configs.json                  # Best configs for Phase 2-4
├── analysis_scripts/
│   └── analyze_pca_dominance.py              # PCA dominance analysis
└── README_V3_PIPELINE.md                     # This file
```

---

## Computational Requirements

- **Phase 1:** ~260 runs × ~2-4 hours = ~520-1040 hours
- **Phase 2:** ~60 runs × ~2-4 hours = ~120-240 hours  
- **Phase 3:** ~80 runs × ~2-4 hours = ~160-320 hours
- **Phase 4:** ~40 runs × ~2-4 hours = ~80-160 hours

**Total:** ~880-1760 CPU hours (~37-73 days on single core, ~4-7 days on 10-core cluster)

---

## Publication Deliverables

1. **Manuscript Figures** (PNG + PDF)
   - PCA vs UMAP performance across dimensions
   - MF cloud phase transition curve
   - Generalization heatmaps
   - Cutoff sensitivity plots
   - PCA dominance mechanistic diagrams

2. **Supplementary Materials**
   - Full hyperparameter screening results
   - Distance distribution analyses
   - Manifold quality metrics
   - Per-seed variability assessments

3. **Code/Data Repository**
   - Configuration files
   - Analysis scripts
   - Best configurations JSON
   - Reproducibility instructions

---

## Contact

For questions or issues, contact the UMMBAS development team.

**Version:** 3.0  
**Last Updated:** October 15, 2025
