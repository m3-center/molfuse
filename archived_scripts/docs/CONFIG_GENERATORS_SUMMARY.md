# UMMBAS v2.0 Configuration Generators - Summary

## Overview

This document summarizes the v2.0 configuration generators and the complete test-to-production workflow.

---

## Configuration Generators

### 1. **generate_hyperparam_configs.py** - Hyperparameter Optimization
**Purpose:** Find optimal UMAP parameters for ABL1 with corrected v2.0 Transferase MF.

**Location:** `config_generators/generate_hyperparam_configs.py`

**v2.0 Corrections:**
- ✅ ABL1 MF: "Transferase (KW-0808)" (was "Protein kinase inhibitor")
- ✅ MF cloud: 191,648 compounds (was 20)
- ✅ Projection-only strategy (no co-embedding, no data leakage)
- ✅ Removed t-SNE (co-embedding only)

**Configurations Generated:** 42 total
- **PCA baseline:** 2 configs (features + fingerprints)
- **UMAP-Euclidean:** 20 configs (features only)
  - n_neighbors: [15, 50, 100, 200, 500]
  - min_dist: [0.0, 0.01, 0.1, 0.5]
- **UMAP-Jaccard:** 20 configs (fingerprints only)
  - n_neighbors: [15, 50, 100, 200, 500]
  - min_dist: [0.0, 0.01, 0.1, 0.5]

**Key Decision:** UMAP-Euclidean for features, UMAP-Jaccard for fingerprints (proper distance metrics for data types)

**Usage:**
```bash
python config_generators/generate_hyperparam_configs.py
# Creates: hyperparam_configs/ with 42 JSON files
```

**HPC Execution:**
```bash
bash hpc/submit_all_replicates_hyperparameterization.sh
# Submits: 42 configs × 5 seeds = 210 jobs
# Runtime: ~48-72 hours (parallel)
```

---

### 2. **generate_generalization_configs.py** - Generalization Experiments
**Purpose:** Test optimal parameters on held-out targets (Pyruvate Kinase M2, Isocitrate Dehydrogenase).

**Location:** `config_generators/generate_generalization_configs.py`

**v2.0 Corrections:**
- ✅ All targets use correct UniProt MF keywords
- ✅ Projection-only strategy
- ✅ Removed t-SNE and co-embedding
- ✅ Uses optimal UMAP hyperparameters from Phase 2

**Configurations Generated:** 4 configs (full run) or 2 configs (test run)

**Full Run:**
- Pyruvate Kinase M2 (Transferase KW-0808):
  - PCA projection
  - UMAP-Euclidean projection
- Isocitrate Dehydrogenase (Oxidoreductase KW-0560):
  - PCA projection
  - UMAP-Euclidean projection

**Test Run (--test-run flag):**
- Only Pyruvate Kinase M2:
  - PCA projection
  - UMAP-Euclidean projection
- Uses small debugging dataset

**Usage:**
```bash
# Full run
python config_generators/generate_generalization_configs.py
# Creates: generalization_configs/ with 4 JSON files

# Test run
python config_generators/generate_generalization_configs.py --test-run
# Creates: generalization_configs/ with 2 *_TEST.json files
```

**HPC Execution:**
```bash
# Full run
bash hpc/submit_generalization_jobs.sh
# Submits: 4 configs × 5 seeds = 20 jobs

# Test run
bash hpc/submit_generalization_jobs.sh --test-run
# Submits: 2 configs × 3 seeds = 6 jobs
```

---

### 3. **generate_dimensionality_configs.py** - Dimensionality Analysis
**Purpose:** Test how similarity space dimensionality [2, 3, 5, 10, 20] affects performance.

**Location:** `config_generators/generate_dimensionality_configs.py`

**v2.0 Corrections:**
- ✅ ABL1 MF: "Transferase (KW-0808)"
- ✅ Projection-only strategy
- ✅ Removed t-SNE
- ✅ Uses optimal UMAP hyperparameters

**Configurations Generated:** 2 configs
- PCA projection (tests all 5 dimensions)
- UMAP-Euclidean projection (tests all 5 dimensions)

**Note:** Update UMAP hyperparameters after hyperparameter sweep before running.

**Usage:**
```bash
python config_generators/generate_dimensionality_configs.py
# Creates: dimensionality_configs/ with 2 JSON files
```

**HPC Execution:**
```bash
bash hpc/submit_dimensionality_jobs.sh
# Submits: 2 configs × 5 seeds = 10 jobs
# Each job tests 5 dimensions = 50 total analyses
```

---

## Quick Test Script

### **quick_test_run.sh** - Fast Pipeline Validation

**Purpose:** Validate complete pipeline locally before HPC submission.

**Two Modes:**

#### Mode 1: Hyperparameter Test Only (15-30 minutes)
```bash
bash quick_test_run.sh --hyperparam-only
```

**What it does:**
1. Generates 2 minimal hyperparameter configs (1 features, 1 fingerprints)
2. Runs both with seed 42
3. Validates outputs
4. **Tests:** UMAP-Euclidean (features) + UMAP-Jaccard (fingerprints)

**Output:**
- `test_hyperparam_workspace/` - Results
- `test_hyperparam_*.log` - Execution logs

#### Mode 2: Complete Pipeline Test (30-45 minutes)
```bash
bash quick_test_run.sh
```

**What it does:**
1. **Part 1:** Hyperparameter test (as above)
2. **Part 2:** Generalization test with small dataset
3. Validates all outputs

**Output:**
- `test_hyperparam_workspace/` - Hyperparameter results
- `test_experiment_workspace/` - Generalization results
- `test_hyperparam_*.log` - Hyperparameter logs
- `test_generalization.log` - Generalization log

**Key Features:**
- ✅ Minimal UMAP grid (n_neighbors=50, min_dist=0.01)
- ✅ Uses debugging dataset for generalization test
- ✅ Tests both representations (features + fingerprints)
- ✅ Fast validation before expensive HPC runs

---

## Complete Workflow: Test to Production

### Phase 0: Local Validation
```bash
# Quick test (recommended first)
bash quick_test_run.sh --hyperparam-only

# If successful, test full pipeline
bash quick_test_run.sh
```

**Time:** 15-45 minutes  
**Purpose:** Validate pipeline before HPC

---

### Phase 1: Hyperparameter Optimization (HPC)

**Step 1:** Generate full hyperparameter configs
```bash
python config_generators/generate_hyperparam_configs.py
```

**Step 2:** Submit to HPC
```bash
cd hpc/
bash submit_all_replicates_hyperparameterization.sh
```

**Step 3:** Monitor jobs
```bash
squeue -u $USER | grep HYPERPARAM
```

**Step 4:** Analyze results (after completion)
```bash
sbatch hpc/ummbas_hyperparameterization_analysis.sh
```

**Time:** 48-72 hours (parallel)  
**Output:** Optimal n_neighbors and min_dist for each representation

---

### Phase 2: Update Generalization Configs

**Step 1:** Edit `config_generators/generate_generalization_configs.py`

Find optimal hyperparameters from Phase 1 analysis:
```python
DR_METHODS = {
    "umap_euclidean_projection": {
        "method_key": "umap_euclidean",
        "config": {
            "short_name": "UMAP-Euclidean",
            "metric": "euclidean",
            "n_neighbors": 500,  # UPDATE from results
            "min_dist": 0.01     # UPDATE from results
        },
        "strategy": "projection"
    }
}
```

**Step 2:** Regenerate generalization configs
```bash
python config_generators/generate_generalization_configs.py
```

---

### Phase 3: Generalization Experiment (HPC)

**Step 1:** Submit generalization jobs
```bash
cd hpc/
bash submit_generalization_jobs.sh
```

**Step 2:** Monitor jobs
```bash
squeue -u $USER
```

**Step 3:** Run baselines (after completion)
```bash
# Pyruvate Kinase M2
python baselines/baseline_random_ranking.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_feature_space_distance.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_tanimoto_similarity.py PyruvateKinaseM2 P14618 KW-0808

# Isocitrate Dehydrogenase
python baselines/baseline_random_ranking.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_feature_space_distance.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_tanimoto_similarity.py IsocitrateDehydrogenaseNADP O75874 KW-0560
```

**Step 4:** Aggregate results
```bash
python analysis_scripts/aggregate_generalization_analysis.py
python analysis_scripts/aggregate_and_report.py
```

**Time:** 24-48 hours per job (parallel)

---

### Optional: Dimensionality Analysis (HPC)

**When:** After Phase 1 (hyperparameter optimization)

**Step 1:** Update optimal parameters in `generate_dimensionality_configs.py`

**Step 2:** Generate configs
```bash
python config_generators/generate_dimensionality_configs.py
```

**Step 3:** Submit jobs
```bash
bash hpc/submit_dimensionality_jobs.sh
```

**Step 4:** Aggregate
```bash
python analysis_scripts/aggregate_dimensionality_analysis.py
```

**Time:** 24-48 hours (tests 5 dimensions per config)

---

## Timeline Summary

| Phase | Task | Duration | Cumulative |
|-------|------|----------|------------|
| 0 | Local test (`quick_test_run.sh`) | 15-45 min | 1 hour |
| 1 | Hyperparameter optimization (HPC) | 48-72 hours | 3 days |
| 1 | Hyperparameter analysis | 1-2 hours | 3 days |
| 2 | Update configs | 5-10 min | 3 days |
| 3 | Generalization experiment (HPC) | 24-48 hours | 5 days |
| 3 | Run baselines | 1-2 hours | 5 days |
| 3 | Aggregate & report | 30 min | 5 days |

**Total: ~5 days** (with parallel HPC execution)

---

## Key Files Reference

### Configuration Generators
- `config_generators/generate_hyperparam_configs.py` - 42 hyperparameter configs
- `config_generators/generate_generalization_configs.py` - 4 generalization configs
- `config_generators/generate_dimensionality_configs.py` - 2 dimensionality configs

### Test Scripts
- `quick_test_run.sh` - Local validation (15-45 min)

### HPC Submission Scripts
- `hpc/submit_all_replicates_hyperparameterization.sh` - Submit 210 hyperparam jobs
- `hpc/submit_generalization_jobs.sh` - Submit 20 generalization jobs
- `hpc/submit_dimensionality_jobs.sh` - Submit 10 dimensionality jobs

### Analysis Scripts
- `analysis_scripts/aggregate_generalization_analysis.py` - Aggregate generalization results
- `analysis_scripts/aggregate_dimensionality_analysis.py` - Aggregate dimensionality results
- `analysis_scripts/aggregate_and_report.py` - Generate comprehensive report
- `hpc/ummbas_hyperparameterization_analysis.sh` - Analyze hyperparameter sweep

### Documentation
- `docs/HPC_EXECUTION_GUIDE.md` - Complete HPC workflow guide
- `docs/EXECUTION_SUMMARY.md` - Quick reference summary
- `docs/REPOSITORY_STRUCTURE.md` - Repository organization
- `docs/CONFIG_GENERATORS_SUMMARY.md` - This file

---

## v2.0 Key Corrections Summary

| Aspect | v1.0 (Wrong) | v2.0 (Correct) |
|--------|--------------|----------------|
| ABL1 MF | "Protein kinase inhibitor" | "Transferase (KW-0808)" |
| MF Cloud Size | 20 compounds | 191,648 compounds |
| Strategy | Co-embedding | Projection-only |
| t-SNE | Included | Removed (co-embedding only) |
| Fingerprint UMAP | UMAP-Euclidean | UMAP-Jaccard |
| Fingerprint Scaling | StandardScaler | PassthroughScaler |
| Data Leakage | Present | Eliminated |
| Test Script | Single run only | Includes hyperparameter test |

---

**Last Updated:** October 14, 2025  
**Version:** 2.0
