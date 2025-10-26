# UMMBAS Screening Experiments - Complete Analysis Pipeline

## Overview

This repository contains three complementary analyses for the UMMBAS (Universal Molecular-space for Molecular Bioactivity Assessment and Screening) project:

1. **Hyperparameter Sweep** - Find optimal parameters for each method (ABL1)
2. **Generalization Analysis** - Test if methods work on new proteins
3. **Cutoff Analysis** - Test affinity threshold sensitivity (across proteins)

---

## Current Status

| Analysis | Proteins | Status | Location |
|----------|----------|--------|----------|
| 1. Hyperparameter Sweep | ABL1 | ✅ Complete | `experiment_workspace_rerun_hyperparam_sweep/` |
| 2. Generalization Baseline | PyruvateKinaseM2, IsocitrateDehydrogenase | ✅ Complete | `experiment_workspace_generalization/` |
| 3. Generalization Cutoff | PyruvateKinaseM2, IsocitrateDehydrogenase | ⏳ Run next | `experiment_workspace_generalization_cutoff/` |

---

## Analysis 1: Hyperparameter Sweep (ABL1) ✅ COMPLETE

**Scientific Question:** What are the optimal hyperparameters for each dimensionality reduction method?

**Details:**
- **Target:** TyrosineProteinKinaseABL1_P00519 only
- **Methods:** t-SNE (6 perplexities), UMAP (12 param combinations), PCA
- **Scale:** 720 configs × 5 seeds = 3,600 experiments
- **Duration:** ~2 weeks on HPC

**Key Results:**
- Best t-SNE: perplexity = 1000
- Best UMAP: n_neighbors = 500, min_dist = 0.01
- Heatmaps available showing performance landscape

---

## Analysis 2: Generalization Experiments

### 2a. Baseline Generalization ✅ COMPLETE

**Scientific Question:** Do the optimal methods generalize to different proteins?

**Details:**
- **Targets:** PyruvateKinaseM2_P14618, IsocitrateDehydrogenaseNADP_O75874
- **Methods:** Best hyperparameters from Analysis 1
- **Scale:** 10 configs × 5 seeds = 50 experiments
- **Affinity Cutoff:** 100,000 nM (same as hyperparameter sweep)

**Scripts:**
- `generate_generalization_configs.py` - Generate configuration files
- `hpc/submit_generalization_jobs.sh` - Submit to HPC
- Results in: `experiment_workspace_generalization/`

### 2b. Cutoff Analysis on Generalization ⏳ RUN NEXT

**Scientific Question:** Does the cutoff-performance relationship generalize?

Specifically: Does the negative correlation between affinity cutoff and EF@1% 
(observed in ABL1) also hold for PyruvateKinaseM2 and IsocitrateDehydrogenase?

**Details:**
- **Input:** Results from 2a (baseline generalization)
- **Cutoffs:** 100, 1000, 10000, 100000 nM
- **Scale:** 50 existing runs × 4 cutoffs (re-analysis, no new experiments)

**How to Run:**
```bash
# Step 1: Run cutoff analysis on generalization results
python run_cutoff_analysis_generalization.py \
  --original_workspace experiment_workspace_generalization/ \
  --output_workspace experiment_workspace_generalization_cutoff/ \
  --cutoffs 100,1000,10000,100000

# Step 2: After completion, aggregate ALL results (baseline + cutoff)
python aggregate_generalization_analysis.py \
  --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --generalization_workspace experiment_workspace_generalization/ \
  --generalization_cutoff_workspace experiment_workspace_generalization_cutoff/ \
  --output_dir final_report_generalization/
```

---

## Directory Structure

```
experiment_workspace_rerun_hyperparam_sweep/    # Analysis 1: ABL1 hyperparams
├── run_seed42_config_features_tsne_perplexity1000/
├── run_seed42_config_features_umap_euclidean_projection_nn500_md0.01/
└── ... (3,600 directories)

experiment_workspace_generalization/            # Analysis 2a: Generalization baseline
├── run_seed42_config_PyruvateKinaseM2_P14618_features_tsne/
├── run_seed42_config_IsocitrateDehydrogenaseNADP_O75874_features_tsne/
└── ... (50 directories)

experiment_workspace_generalization_cutoff/     # Analysis 2b: Generalization cutoffs
├── run_seed42_config_PyruvateKinaseM2_P14618_features_tsne/
│   ├── results_cutoff_100/
│   ├── results_cutoff_1000/
│   ├── results_cutoff_10000/
│   └── results_cutoff_100000/
└── ... (50 directories with 4 cutoffs each)

```

---

## Key Scripts

### Configuration Generation
- `generate_hyperparam_configs.py` - Generate hyperparameter sweep configs
- `generate_generalization_configs.py` - Generate generalization configs

### HPC Execution
- `hpc/submit_hyperparameter_jobs.sh` - Submit hyperparameter jobs
- `hpc/submit_generalization_jobs.sh` - Submit generalization jobs
- `hpc/ummbas_*_cpu.sh` - SLURM execution scripts

### Cutoff Analysis
- `run_cutoff_analysis.py` - Run cutoff analysis on ABL1 hyperparameter results
- `run_cutoff_analysis_generalization.py` - **NEW** Run cutoff analysis on generalization results
- `aggregate_cutoff_analysis.py` - Aggregate ABL1 cutoff results

### Aggregation & Reporting
- `analyze_hyperparams.py` - Analyze hyperparameter sweep
- `aggregate_generalization_analysis.py` - **UPDATED** Aggregate generalization results (with optional cutoff data)

### Testing & Utilities
- `test_generalization_compatibility.py` - Test script compatibility
- `run_generalization_experiment.sh` - Interactive quick-start script

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ANALYSIS 1: HYPERPARAMETER SWEEP (ABL1)                     │
│ Find optimal hyperparameters                                │
│ ✅ COMPLETE                                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
┌────────────────────┐    ┌────────────────────┐
│ ANALYSIS 2a:       │    │ ANALYSIS 3:        │
│ GENERALIZATION     │    │ ABL1 CUTOFF        │
│ (Baseline)         │    │ (Optional)         │
│ ✅ COMPLETE         │    │ ⏳ Not run          │
└─────────┬──────────┘    └────────────────────┘
          │
          ▼
┌────────────────────┐
│ ANALYSIS 2b:       │
│ GENERALIZATION     │
│ CUTOFF             │
│ ⏳ RUN NEXT         │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ FINAL AGGREGATION  │
│ Compare all 3      │
│ proteins with      │
│ cutoff analysis    │
└────────────────────┘
```

---

## Next Steps (Current Task)

### 1. Run Generalization Cutoff Analysis

```bash
python run_cutoff_analysis_generalization.py
```

This will:
- Take existing generalization results
- Re-analyze with cutoffs: 100, 1000, 10000, 100000 nM
- Save to: `experiment_workspace_generalization_cutoff/`
- Duration: Quick (re-analysis only, no new similarity spaces)

### 2. Aggregate All Results

```bash
python aggregate_generalization_analysis.py \
  --generalization_cutoff_workspace experiment_workspace_generalization_cutoff/
```

This will create:
- Comparative visualizations across all 3 proteins
- Cutoff vs. performance plots for each protein
- Tables showing if cutoff-performance relationship generalizes
- Assessment of method robustness

**Output:** `final_report_generalization/generalization_report_TIMESTAMP/`

---

## Scientific Questions Answered

### By Hyperparameter Sweep:
✅ What are the optimal hyperparameters for each method on ABL1?

### By Generalization (Baseline):
✅ Do methods maintain their ranking on different proteins?  
✅ Which method is most robust across proteins?  
✅ Does molecular function class affect performance?

### By Generalization (Cutoff):
⏳ Does the cutoff-performance relationship generalize?  
⏳ Are stricter cutoffs (lower nM) always better across proteins?  
⏳ Do proteins differ in optimal cutoff threshold?

---

## Important Notes

1. **All analyses use separate workspaces** - no data conflicts
2. **Generalization uses BEST hyperparameters** from ABL1 analysis
3. **Cutoff analysis is cheap** - re-analysis only, no new similarity spaces
4. **Features only** - fingerprints showed poor performance (<1% EF@1%)
5. **2D spaces only** - focused analysis on 2D similarity spaces

---

## Documentation Files

- `README.md` - This file (comprehensive overview)
- `GENERALIZATION_EXPERIMENT_README.md` - Detailed generalization setup
- `CUTOFF_ANALYSIS_UPDATES.md` - Updates to cutoff analysis scripts
- `GENERALIZATION_QUICKSTART.txt` - Quick reference guide
- `DEBUGGING.md` - Debugging information

---

## Contact & Support

For questions about the analysis pipeline:
1. Review this README
2. Check specific documentation files
3. Review log files in workspace directories
4. Contact: UMMBAS development team

---

## Updates Log

**2025-10-09:**
- ✅ Fixed config filename issue (now includes target protein name)
- ✅ Created `run_cutoff_analysis_generalization.py` for generalization cutoff analysis
- ✅ Updated `aggregate_generalization_analysis.py` to handle cutoff data
- ✅ Removed deprecated documentation (NEXT_STEPS_ANALYSIS.md, etc.)
- ✅ Created comprehensive README

**2025-10-08:**
- ✅ Created generalization experiment framework
- ✅ Updated cutoff analysis for new directory structure
- ✅ Updated aggregate_cutoff_analysis.py for multiple UMAP hyperparameters

---

## Quick Command Reference

```bash
# Generate generalization configs
python generate_generalization_configs.py

# Run generalization cutoff analysis
python run_cutoff_analysis_generalization.py

# Aggregate all generalization results (with cutoff data)
python aggregate_generalization_analysis.py \
  --generalization_cutoff_workspace experiment_workspace_generalization_cutoff/

# Optional: Run ABL1 cutoff analysis
python run_cutoff_analysis.py
python aggregate_cutoff_analysis.py

# Test script compatibility
python test_generalization_compatibility.py
```

---

**Current Focus:** Run generalization cutoff analysis to test if cutoff-performance relationship generalizes across proteins.
