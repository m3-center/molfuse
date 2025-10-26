# HPC Execution Guide - UMMBAS v2.0

## Overview

This guide provides the exact order of operations for running the complete UMMBAS v2.0 pipeline on HPC infrastructure.

---

## Prerequisites

### 1. Data Preparation (One-Time Setup)
Ensure these datasets are available:
- ✅ `datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv` (425,088 records)
- ✅ `datasets/molecular_function_affinity_data/KW-0560_Oxidoreductase_affinity.csv` (97,729 records)
- ✅ `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_features.csv`
- ✅ `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_fingerprints.csv`
- ✅ `datasets/molecular_function_features_fingerprints/KW-0560_Oxidoreductase_features.csv`
- ✅ `datasets/molecular_function_features_fingerprints/KW-0560_Oxidoreductase_fingerprints.csv`
- ✅ `datasets/zinc_data.csv` (1.2M+ decoy compounds)

### 2. Configuration Generation
Generate experiment configurations based on your workflow:

```bash
# For v2.0 generalization experiments (recommended)
python config_generators/generate_generalization_configs.py

# For test run (small dataset)
python config_generators/generate_generalization_configs.py --test-run
```

This creates configuration files in `generalization_configs/` directory.

---

## Execution Workflows

### **Recommended Workflow Order:**
1. **WORKFLOW B** (Test Run) - Validate pipeline locally (~1 hour)
2. **WORKFLOW C** (Hyperparameter Optimization) - Find optimal parameters (~48-72 hours)
3. **WORKFLOW A** (Generalization Experiment) - Final evaluation (~24-48 hours)

---

### **WORKFLOW A: Full v2.0 Generalization Experiment** (After Hyperparameter Optimization)

This is the main v2.0 workflow testing projection-only strategy on held-out targets using optimal hyperparameters.

#### Step 1: Generate Configurations
```bash
cd /path/to/UMMBAS_screening_experiments
python config_generators/generate_generalization_configs.py
```

**Output:** 4 configuration files in `generalization_configs/`:
- `config_PyruvateKinaseM2_P14618_features_pca_projection.json`
- `config_PyruvateKinaseM2_P14618_features_umap_euclidean_projection.json`
- `config_IsocitrateDehydrogenaseNADP_O75874_features_pca_projection.json`
- `config_IsocitrateDehydrogenaseNADP_O75874_features_umap_euclidean_projection.json`

#### Step 2: Submit HPC Jobs
```bash
cd hpc/
bash submit_generalization_jobs.sh
```

**What This Does:**
- Submits 20 SLURM jobs (4 configs × 5 seeds: 42, 43, 44, 45, 46)
- Each job runs `ummbas_generalization_cpu.sh` which calls `main_orchestrator.py`
- Jobs run in parallel on HPC nodes
- Expected runtime: **~24-48 hours per job** (depending on MF cloud size)

**Resources per Job:**
- 64 CPU cores
- 350 GB RAM
- 24-hour time limit

#### Step 3: Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Check logs (in real-time)
tail -f slurm_logs/GEN_config_PyruvateKinaseM2_P14618_features_pca_projection_seed42_*.out

# Check for errors
grep -i error slurm_logs/*.err
```

#### Step 4: Run Baselines (After Jobs Complete)
```bash
# For Pyruvate Kinase M2 (Transferase)
python baselines/baseline_random_ranking.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_feature_space_distance.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_tanimoto_similarity.py PyruvateKinaseM2 P14618 KW-0808
python baselines/analyze_chemical_novelty.py PyruvateKinaseM2 P14618 KW-0808

# For Isocitrate Dehydrogenase (Oxidoreductase)
python baselines/baseline_random_ranking.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_feature_space_distance.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_tanimoto_similarity.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/analyze_chemical_novelty.py IsocitrateDehydrogenaseNADP O75874 KW-0560
```

**Note:** Baselines can run locally or on HPC. Random baseline (1000 iterations) takes ~10-30 minutes.

#### Step 5: Aggregate Results
```bash
# On HPC or locally after downloading workspace
python analysis_scripts/aggregate_generalization_analysis.py

# Generate comprehensive report
python analysis_scripts/aggregate_and_report.py
```

**Output Locations:**
- Individual results: `experiment_workspace_generalization/`
- Aggregated reports: `final_report_generalization/`
- Logs: `slurm_logs/`

---

### **WORKFLOW B: Test Run** (Small Dataset - Fast Validation)

Use this workflow to validate the pipeline on a small dataset before full runs.

#### Step 1: Generate Test Configurations
```bash
python config_generators/generate_generalization_configs.py --test-run
```

**What Changes:**
- Uses `datasets_debugging/` (small subset of data)
- Only 1 target: Pyruvate Kinase M2
- Limited dimensionality: 2D only
- Faster workspace: `test_experiment_workspace/`
- Test report directory: `test_final_report/`

**Output:** 2 test configuration files in `generalization_configs/`:
- `config_PyruvateKinaseM2_P14618_features_pca_projection_TEST.json`
- `config_PyruvateKinaseM2_P14618_features_umap_euclidean_projection_TEST.json`

#### Step 2: Submit Test Jobs
```bash
cd hpc/
bash submit_generalization_jobs.sh --test-run
```

**What This Does:**
- Submits 6 jobs (2 configs × 3 seeds: 42, 43, 44)
- Uses reduced resources (faster queue)
- Expected runtime: **~2-4 hours per job**

**Resources per Test Job:**
- 32 CPU cores
- 128 GB RAM  
- 4-hour time limit

#### Step 3: Validate Results
```bash
# Check test results
ls -lh test_experiment_workspace/

# Quick aggregation
python analysis_scripts/aggregate_generalization_analysis.py --test-run

# Verify metrics look reasonable
head test_final_report/*/aggregated_metrics.csv
```

If test run succeeds, proceed with full WORKFLOW A.

---

### **WORKFLOW C: Hyperparameter Optimization** (Required for Best Results)

Find optimal UMAP hyperparameters for ABL1 with corrected v2.0 Transferase MF (191k compounds).

#### Step 1: Generate Hyperparameter Configs
```bash
python config_generators/generate_hyperparam_configs.py
```

**What This Creates:**
- 42 configuration files in `hyperparam_configs/`:
  - 2 PCA baseline configs (features + fingerprints)
  - 20 UMAP-Euclidean configs (features only)
  - 20 UMAP-Jaccard configs (fingerprints only)
- Hyperparameter grid:
  - `n_neighbors`: [15, 50, 100, 200, 500]
  - `min_dist`: [0.0, 0.01, 0.1, 0.5]

#### Step 2: Submit Hyperparameter Jobs
```bash
cd hpc/
bash submit_all_replicates_hyperparameterization.sh
```

**What This Does:**
- Submits 210 SLURM jobs (42 configs × 5 seeds: 42, 43, 44, 45, 46)
- Each job tests one hyperparameter combination
- Jobs run in parallel on HPC nodes
- Expected runtime: **~48-72 hours total** (24-48 hours per job, parallel)

**Resources per Job:**
- 64 CPU cores
- 350 GB RAM
- 48-hour time limit

#### Step 3: Monitor Jobs
```bash
# Check job status
squeue -u $USER | grep HYPERPARAM

# Check logs
tail -f slurm_logs/HYPERPARAM_*.out

# Count completed jobs
ls hyperparam_configs/ -1 | wc -l  # Should be 42
ls experiment_workspace_hyperparam_sweep_v2/ -1 | wc -l  # Should grow to 210
```

#### Step 4: Analyze Hyperparameters
```bash
sbatch hpc/ummbas_hyperparameterization_analysis.sh
```

**Output:** Analysis identifies optimal hyperparameters for:
- Features: Best `n_neighbors` and `min_dist` for UMAP-Euclidean
- Fingerprints: Best `n_neighbors` and `min_dist` for UMAP-Jaccard

#### Step 5: Update Generalization Configs
```bash
# Edit config_generators/generate_generalization_configs.py
# Update the optimal hyperparameters in DR_METHODS:
#   - For features UMAP: Update n_neighbors, min_dist
#   - For fingerprints UMAP: Update n_neighbors, min_dist

# Then regenerate generalization configs
python config_generators/generate_generalization_configs.py
```

#### Step 6: Proceed to WORKFLOW A
With optimal hyperparameters identified, run the full generalization experiment (WORKFLOW A).

---

## Pipeline Stages (What Each Job Does)

Each submitted job (`ummbas_generalization_cpu.sh` → `main_orchestrator.py`) runs these stages:

### Stage 1: Feature/Fingerprint Calculation
- **Script:** `core_scripts/calculate_features_and_fingerprints_exp.py`
- **Input:** Affinity CSV files
- **Output:** Features and fingerprints for target actives, MF cloud, ZINC decoys
- **Time:** ~10-30 minutes (uses pre-calculated data if available)

### Stage 2: Similarity Space Construction
- **Script:** `core_scripts/calculate_similarityspaces_exp.py`
- **Input:** Features/fingerprints from Stage 1
- **Output:** Low-dimensional embeddings (PCA/UMAP), fitted models
- **Time:** ~2-8 hours (depends on MF cloud size, UMAP parameters)
- **v2.0:** Projection-only strategy (no co-embedding, no t-SNE)

### Stage 3: Projection & Analysis
- **Script:** `experimental_pipeline/project_and_analyze.py`
- **Input:** Fitted models from Stage 2, held-out actives
- **Output:** Projected coordinates, ranking metrics (ROC-AUC, PR-AUC, EF@1%)
- **Time:** ~10-30 minutes

### Stage 4: ZINC Ranking
- **Script:** `experimental_pipeline/rank_zinc_decoys.py`  
- **Input:** Fitted models, ZINC compounds
- **Output:** Ranked ZINC decoys for virtual screening
- **Time:** ~30-60 minutes

### Stage 5: Reporting
- **Script:** `reporting/generate_latex_report.py`
- **Output:** Per-run LaTeX report and figures
- **Time:** ~5-10 minutes

---

## Troubleshooting

### Jobs Failing with Memory Errors
- Increase `--mem` in SLURM script (e.g., `--mem=500G`)
- Reduce `n_jobs_molcalcs` in config to limit parallelism

### Jobs Timing Out
- Increase `--time` in SLURM script (e.g., `--time=2-00:00:00` for 48 hours)
- UMAP with large MF clouds takes longest

### Missing Features/Fingerprints
- Ensure pre-calculated files exist in `datasets/molecular_function_features_fingerprints/`
- If missing, Stage 1 will calculate them (adds ~2-4 hours)

### Config Not Found Errors
```bash
# Regenerate configs
python config_generators/generate_generalization_configs.py

# Check configs exist
ls -lh generalization_configs/
```

### SLURM Submission Fails
```bash
# Check SLURM is available
squeue

# Verify partition exists
sinfo -p hpc

# Check script paths
ls -lh hpc/ummbas_generalization_cpu.sh
ls -lh main_orchestrator.py
```

---

## Expected Outputs

### Directory Structure After Completion

```
experiment_workspace_generalization/
├── PyruvateKinaseM2_P14618_features_pca_projection_seed42/
│   ├── features_fingerprints/
│   ├── similarity_spaces/
│   ├── rankings/
│   ├── plots/
│   └── orchestrator_run_*.log
├── PyruvateKinaseM2_P14618_features_pca_projection_seed43/
│   └── ...
└── ...

final_report_generalization/
├── generalization_report_YYYYMMDD_HHMMSS/
│   ├── aggregated_metrics.csv
│   ├── aggregated_metrics_summary.csv
│   ├── per_seed_metrics.csv
│   ├── figures/
│   └── generalization_report.pdf

slurm_logs/
├── GEN_config_PyruvateKinaseM2_P14618_features_pca_projection_seed42_123456.out
├── GEN_config_PyruvateKinaseM2_P14618_features_pca_projection_seed42_123456.err
└── ...
```

### Key Metrics Files

- **`rankings/held_out_actives_and_decoys_ranking_metrics.csv`**: Per-run metrics
  - Columns: ROC-AUC, PR-AUC, EF@1%, EF@5%, Mean Distance, etc.
  
- **`aggregated_metrics.csv`**: Aggregated across seeds
  - Mean ± SD for each target × DR method × strategy

- **`per_seed_metrics.csv`**: Detailed per-seed breakdown

---

## Performance Expectations (v2.0)

Based on v1.0 results and v2.0 improvements:

### ABL1 (3,331 actives, 191k Transferase MF cloud):
- **Random Baseline:** EF@1% ≈ 1.0
- **PCA Projection:** EF@1% ≈ 3-5 (better than random, limited by linear DR)
- **UMAP Projection:** EF@1% ≈ 8-15 (best expected, non-linear manifold learning)
- **Feature Distance (no DR):** EF@1% ≈ 2-4 (curse of dimensionality in 39D)

### Pyruvate Kinase M2 (1,019 actives, same Transferase MF):
- Similar to ABL1 (same MF cloud)
- Tests generalization across kinase types

### Isocitrate Dehydrogenase (374 actives, 56k Oxidoreductase MF):
- Smaller MF cloud may impact performance
- Tests generalization to different MF

**v2.0 Improvements:**
- No data leakage (projection-only)
- Corrected MF assignments (191k vs 20 compounds)
- Proper fingerprint handling (Jaccard, no scaling)

---

## Quick Reference Commands

### Monitor All Jobs
```bash
watch -n 10 'squeue -u $USER'
```

### Cancel All Jobs
```bash
scancel -u $USER
```

### Check Disk Usage
```bash
du -sh experiment_workspace_generalization/
```

### Download Results to Local Machine
```bash
# From local machine
rsync -avz --progress \
  user@hpc:/path/to/UMMBAS_screening_experiments/final_report_generalization/ \
  ./local_results/
```

### Re-run Aggregation Locally
```bash
# After downloading experiment_workspace_generalization/
python analysis_scripts/aggregate_generalization_analysis.py
```

---

## Timeline Estimates

### Test Run (WORKFLOW B):
- Config generation: 1 minute
- Job submission: 1 minute
- Execution: **2-4 hours** (6 jobs × 2-4 hours each, parallel)
- Aggregation: 5 minutes
- **Total: 2-4 hours**

### Full Run (WORKFLOW A):
- Config generation: 1 minute  
- Job submission: 1 minute
- Execution: **24-48 hours** (20 jobs × 24-48 hours each, parallel)
- Baselines: 1-2 hours (can run in parallel)
- Aggregation: 10-20 minutes
- **Total: 24-48 hours**

### Hyperparameter Optimization (WORKFLOW C):
- Config generation: 1 minute
- Job submission: 2 minutes
- Execution: **48-72 hours** (210 jobs, parallel)
- Analysis: 1-2 hours
- Update configs: 5 minutes
- **Total: 48-72 hours**

---

## Contact & Support

For issues with:
- **SLURM/HPC:** Contact your HPC administrator
- **Pipeline errors:** Check `slurm_logs/*.err` and `orchestrator_*.log` files
- **Scientific questions:** See `docs/REFACTORING_PLAN_v2.0.md` for methodology

---

**Last Updated:** October 14, 2025  
**Version:** 2.0
