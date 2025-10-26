# UMMBAS Screening Experiments - Complete Analysis Pipeline Overview

## Four Complementary Analyses

### 📊 Analysis 1: Hyperparameter Sweep (COMPLETED)
**Purpose**: Find optimal hyperparameters for dimensionality reduction methods on ABL1

**Location**: `experiment_workspace_rerun_hyperparam_sweep/`

**What was done**:
- Target: TyrosineProteinKinaseABL1_P00519 only
- Methods: t-SNE (6 perplexities), UMAP (3 n_neighbors × 4 min_dist), PCA
- Total: 720 configs × 5 seeds = 3,600 experiments
- Result: Identified best hyperparameters for each method

**Key Scripts**:
- `generate_hyperparam_configs.py` - Generate configs
- `hpc/submit_hyperparameter_jobs.sh` - Submit to HPC
- `analyze_hyperparams.py` - Analyze results

**Output**: Heatmaps showing optimal hyperparameters (attached images you showed)

---

### 🎯 Analysis 2: Affinity Cutoff Analysis (CAN BE RUN NOW)
**Purpose**: Test how affinity cutoff threshold affects performance of optimal methods

**Location**: `experiment_workspace_cutoff_analysis/`

**What to do**:
- Uses results from hyperparameter sweep (ABL1 only)
- Re-analyzes with different affinity cutoffs: 100, 1000, 10000, 100000 nM
- Uses ONLY the optimal hyperparameters identified in Analysis 1
- Target: Still just ABL1 (same protein, different cutoffs)

**Key Scripts**:
- ✅ `run_cutoff_analysis.py` - Re-run analysis with different cutoffs (UPDATED for new dir structure)
- ✅ `aggregate_cutoff_analysis.py` - Aggregate cutoff results (UPDATED for multiple hyperparams)

**How to run**:
```bash
# On HPC (after hyperparameter sweep completed)
python run_cutoff_analysis.py \
  --original_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --output_workspace experiment_workspace_cutoff_analysis/ \
  --cutoffs 100,1000,10000,100000
```

**Then aggregate**:
```bash
python aggregate_cutoff_analysis.py \
  --base_experiment_dir experiment_workspace_cutoff_analysis/ \
  --original_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --output_report_dir final_report_cutoff_analysis/
```

**Status**: ✅ Scripts updated and ready to run

---

### 🌍 Analysis 3: Generalization Experiment (COMPLETED!)
**Purpose**: Test if optimal methods generalize to NEW target proteins

**Location**: `experiment_workspace_generalization/`

**What was done**:
- Targets: PyruvateKinaseM2 + IsocitrateDehydrogenase (NOT ABL1)
- Methods: Best hyperparameters from Analysis 1
- Total: 10 configs × 5 seeds = 50 experiments
- Result: Shows if methods work on different proteins

**Key Scripts**:
- ✅ `generate_generalization_configs.py` - Generate configs (FIXED filename issue)
- ✅ `hpc/submit_generalization_jobs.sh` - Submit to HPC
- ✅ `aggregate_generalization_analysis.py` - Compare across proteins

**How to analyze** (jobs completed):
```bash
python aggregate_generalization_analysis.py \
  --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --generalization_workspace experiment_workspace_generalization/ \
  --output_dir final_report_generalization/
```

**New Feature: Active Distribution Analysis**
- Analyzes how actives are distributed across rank ranges (1-10K, 10K-50K, etc.)
- Compares enrichment patterns across methods and targets
- Generates stacked bar charts and comparison plots
- See `docs/ACTIVE_DISTRIBUTION_ANALYSIS.md` for details

**Status**: ✅ Jobs completed, ready to analyze

---

### 📐 Analysis 4: Dimensionality Experiment (READY TO RUN!)
**Purpose**: Test how similarity space dimensionality affects screening performance

**Location**: `experiment_workspace_dimensionality/`

**What to do**:
- Uses results from hyperparameter sweep (ABL1 only)
- Tests multiple dimensions: 2, 3, 5, 10, 20
- Uses ONLY the optimal hyperparameters identified in Analysis 1
- Target: ABL1 (same as Analysis 1)
- Strategy: Co-embedding only (as specified)
- Methods: PCA, UMAP, t-SNE

**Key Scripts**:
- ✅ `generate_dimensionality_configs.py` - Generate configs
- ✅ `hpc/submit_dimensionality_jobs.sh` - Submit to HPC
- ✅ `aggregate_dimensionality_analysis.py` - Analyze and visualize results

**How to run**:
```bash
# Generate configs
python generate_dimensionality_configs.py

# Submit to HPC
bash hpc/submit_dimensionality_jobs.sh

# After completion, aggregate
python aggregate_dimensionality_analysis.py \
  --workspace experiment_workspace_dimensionality/ \
  --output_dir final_report_dimensionality/
```

**Expected Output**:
- Line plots: EF@1% vs Dimensionality (with 90% CI error bars)
- Summary tables: Performance metrics across all dimensions
- LaTeX report: Comprehensive analysis

**Status**: ✅ Scripts ready, can be submitted to HPC

---

## Analysis Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ ANALYSIS 1: HYPERPARAMETER SWEEP (ABL1)                         │
│ ────────────────────────────────────────────────────────────────│
│ Find optimal hyperparameters for each method                    │
│                                                                  │
│ Input:  ABL1 data                                               │
│ Output: • Best perplexity for t-SNE: 1000                       │
│         • Best UMAP params: n_neighbors=500, min_dist=0.01      │
│         • Heatmaps, performance metrics                         │
└─────────────────────────────────────────────────────────────────┘
                    │                │              │
                    │                │              │
        ┌───────────┴──────┐  ┌──────┴────┐  ┌─────┴──────┐
        │                  │  │           │  │            │
        ▼                  │  ▼           │  ▼            │
┌────────────────┐         │ ┌───────────┐│ ┌────────────┐│
│ ANALYSIS 2:    │         │ │ANALYSIS 3:││ │ANALYSIS 4: ││
│ CUTOFF         │         │ │GENERAL-   ││ │DIMENSION-  ││
│ ANALYSIS       │         │ │IZATION    ││ │ALITY       ││
│ ───────────────│         │ │───────────││ │────────────││
│ Same protein   │         │ │New targets││ │Same protein││
│ (ABL1)         │         │ │Different  ││ │(ABL1)      ││
│ Varying cutoffs│         │ │proteins   ││ │Varying dims││
│                │         │ │           ││ │            ││
│ Uses optimal   │◄────────┘ │Uses       ││ │Uses        ││
│ hyperparameters│           │optimal    ││ │optimal     ││
│                │           │hyperparams││ │hyperparams ││
│                │           │           ││ │            ││
│ Questions:     │           │Questions: ││ │Questions:  ││
│ • Does cutoff  │           │• Do       ││ │• Does dim  ││
│   affect rank? │           │  methods  ││ │  affect    ││
│ • Optimal      │           │  work on  ││ │  perform?  ││
│   cutoff?      │           │  new      ││ │• Optimal   ││
│                │           │  proteins?││ │  dimension?││
└────────────────┘           └───────────┘│ └────────────┘
                                          │              │
                            All use optimal hyperparams  │
                            from Analysis 1 ◄────────────┘
```

---

## Current Status Summary

| Analysis | Target Proteins | Status | Next Action |
|----------|----------------|---------|-------------|
| 1. Hyperparameter Sweep | ABL1 | ✅ Complete | Review heatmaps |
| 2. Cutoff Analysis | ABL1 | ⏳ Ready to run | Run `run_cutoff_analysis.py` on HPC |
| 3. Generalization | Pyruvate Kinase + Isocitrate Dehydrogenase | ✅ Jobs complete | Run `aggregate_generalization_analysis.py` |
| 4. Dimensionality | ABL1 | ⏳ Ready to run | Run `generate_dimensionality_configs.py` then submit to HPC |

---

## Why Four Separate Analyses?

### Analysis 1 (Hyperparameter) → Analysis 2 (Cutoff)
- **Question**: "We found optimal hyperparameters, but does the AFFINITY CUTOFF matter?"
- **Same protein** (ABL1), same optimal hyperparameters, different cutoffs
- **Cheap**: Only tests optimal configs, not full grid

### Analysis 1 (Hyperparameter) → Analysis 3 (Generalization)
- **Question**: "Do these optimal hyperparameters work on OTHER PROTEINS?"
- **Different proteins**, same optimal hyperparameters
- **Cheap**: Only tests optimal configs, not full grid

### Analysis 1 (Hyperparameter) → Analysis 4 (Dimensionality)
- **Question**: "Do these optimal hyperparameters work in DIFFERENT DIMENSIONS?"
- **Same protein** (ABL1), same optimal hyperparameters, different dimensions
- **Relatively cheap**: Only co-embedding strategy, testing 5 dimensions

### Analyses 2, 3, 4 (Independent)
- **Cutoff Analysis**: ABL1 with varying cutoffs
- **Generalization**: New proteins with fixed cutoff and dimension
- **Dimensionality**: ABL1 with varying dimensions
- These can be run independently in parallel

---

## Workspace Directory Structure

```
experiment_workspace_rerun_hyperparam_sweep/  ← Analysis 1 (ABL1, all hyperparams)
├── run_seed42_config_features_tsne_perplexity15/
├── run_seed42_config_features_tsne_perplexity1000/
├── run_seed42_config_features_umap_euclidean_projection_nn500_md0.01/
└── ... (3,600 run directories)

experiment_workspace_cutoff_analysis/  ← Analysis 2 (ABL1, optimal hyperparams, varying cutoffs)
├── run_seed42_config_features_tsne_perplexity1000/
│   ├── results_cutoff_100/
│   ├── results_cutoff_1000/
│   ├── results_cutoff_10000/
│   └── results_cutoff_100000/
└── ... (fewer runs, multiple cutoffs per run)

experiment_workspace_generalization/  ← Analysis 3 (New proteins, optimal hyperparams)
├── run_seed42_config_PyruvateKinaseM2_P14618_features_tsne/
├── run_seed42_config_IsocitrateDehydrogenaseNADP_O75874_features_tsne/
├── run_seed42_config_PyruvateKinaseM2_P14618_features_umap_euclidean_coembedding/
└── ... (50 run directories)

experiment_workspace_dimensionality/  ← Analysis 4 (ABL1, optimal hyperparams, varying dimensions)
├── run_seed42_config_ABL1_features_pca_coembedding_TIMESTAMP/
│   └── TyrosineProteinKinaseABL1_P00519/
│       └── results/
│           └── features/
│               ├── dim_2/
│               ├── dim_3/
│               ├── dim_5/
│               ├── dim_10/
│               └── dim_20/
└── ... (15 run directories, 5 dimensions each)
```

---

## Next Steps

### For Analysis 2 (Cutoff Analysis) - TO DO
Since generalization experiments are done, you can now focus on the cutoff analysis:

1. **Run cutoff analysis on HPC**:
   ```bash
   python run_cutoff_analysis.py \
     --original_workspace experiment_workspace_rerun_hyperparam_sweep/ \
     --output_workspace experiment_workspace_cutoff_analysis/
   ```

2. **After cutoff jobs complete, aggregate results**:
   ```bash
   python aggregate_cutoff_analysis.py \
     --base_experiment_dir experiment_workspace_cutoff_analysis/ \
     --original_workspace experiment_workspace_rerun_hyperparam_sweep/
   ```

### For Analysis 3 (Generalization) - READY TO ANALYZE
Since you said the generalization experiments are done:

```bash
python aggregate_generalization_analysis.py \
  --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --generalization_workspace experiment_workspace_generalization/ \
  --output_dir final_report_generalization/
```

This will create:
- Comparative bar charts (all 3 proteins)
- Performance heatmaps
- Generalization assessment table
- Method ranking consistency analysis

---

## Important Notes

1. **All four analyses are INDEPENDENT** - they don't interfere with each other
2. **All use separate workspaces** - no data conflicts
3. **Analyses 2, 3, and 4** all depend on Analysis 1 (hyperparameter sweep) for determining optimal settings
4. **You can run Analyses 2, 3, and 4 in parallel** - they don't depend on each other

---

## Script Status Checklist

✅ **Hyperparameter Scripts** (Analysis 1)
- generate_hyperparam_configs.py
- hpc/submit_hyperparameter_jobs.sh
- analyze_hyperparams.py

✅ **Cutoff Analysis Scripts** (Analysis 2) - UPDATED
- run_cutoff_analysis.py (fixed glob pattern)
- aggregate_cutoff_analysis.py (handles multiple UMAP hyperparameters)

✅ **Generalization Scripts** (Analysis 3) - UPDATED
- generate_generalization_configs.py (fixed filename overwrites)
- hpc/submit_generalization_jobs.sh
- hpc/ummbas_generalization_cpu.sh
- aggregate_generalization_analysis.py
- test_generalization_compatibility.py

✅ **Dimensionality Scripts** (Analysis 4) - NEW
- generate_dimensionality_configs.py
- hpc/submit_dimensionality_jobs.sh
- hpc/ummbas_dimensionality_cpu.sh
- aggregate_dimensionality_analysis.py
- DIMENSIONALITY_EXPERIMENT_README.md
- DIMENSIONALITY_QUICKSTART.txt

---

## TL;DR

**Four independent but complementary analyses:**

- **Analysis 1 (Hyperparameter)**: "What are the best hyperparameters?" → COMPLETED
- **Analysis 2 (Cutoff)**: "How does affinity threshold affect ABL1 performance?" → READY
- **Analysis 3 (Generalization)**: "Do methods work on different proteins?" → COMPLETED
- **Analysis 4 (Dimensionality)**: "How does similarity space dimension affect performance?" → READY

All analyses 2-4 use optimal hyperparameters from Analysis 1, but answer different questions.

### Quick Start for Analysis 4 (Dimensionality)

```bash
# Generate configs
python generate_dimensionality_configs.py

# Submit to HPC
bash hpc/submit_dimensionality_jobs.sh

# After completion, analyze
python aggregate_dimensionality_analysis.py

# For detailed documentation
cat DIMENSIONALITY_QUICKSTART.txt
```

### What Makes Analysis 4 Different?

- **Tests multiple dimensions**: 2, 3, 5, 10, 20 (all other experiments used 2D only)
- **Same target as Analysis 1**: ABL1 (allows direct comparison)
- **Co-embedding only**: Simplified from projection vs co-embedding
- **Primary output**: Line plot showing EF@1% vs dimensionality with error bars
- **Research question**: Does higher dimensionality help or hurt performance?
