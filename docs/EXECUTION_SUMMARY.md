# Summary: HPC Execution Order & Test Run Setup

## Issue 1: Data Format Correction ✅
**Problem:** Code references `.pkl` files, but actual data is in `.csv` format.

**Files to check:**
- `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv`
- `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv`
- `datasets/molecular_function_features_fingerprints/KW-0560_Oxidoreductase_affinity_extracted_features.csv`
- `datasets/molecular_function_features_fingerprints/KW-0560_Oxidoreductase_affinity_extracted_fingerprints_ECFP4.csv`

**Action Required:** Update config generators and scripts to use `.csv` extension.

---

## Issue 2: Hyperparameter Config Generator Found ✅

**Location:** `config_generators/generate_dimensionality_configs.py`

**Current State (v1.0 - OUTDATED):**
- Target: ABL1 (P00519)
- MF: "Protein kinase inhibitor" ❌ (WRONG - should be "Transferase KW-0808")
- Representations: Features only
- Methods: PCA, t-SNE, UMAP-Euclidean (all co-embedding) ❌
- Missing: Fingerprints evaluation

**What it needs for v2.0:**
1. Fix MF: ABL1 → Transferase (KW-0808)
2. Remove t-SNE (co-embedding only, data leakage)
3. Remove co-embedding (projection-only strategy)
4. Add fingerprints representation (UMAP-Jaccard)
5. Test hyperparameters: n_neighbors, min_dist for UMAP

---

## HPC Execution Order (Step-by-Step)

### **PHASE 1: Test Run (Local Validation)**
**Purpose:** Validate pipeline before large HPC jobs

```bash
# Run quick test (30-60 min)
bash quick_test_run.sh
```

**What it does:**
1. Generates test config (1 target, small dataset)
2. Runs locally with seed 42
3. Validates output structure

**Success criteria:**
- ✓ Workspace created
- ✓ Rankings file exists
- ✓ Metrics look reasonable (ROC-AUC > 0.5)

---

### **PHASE 2: Hyperparameter Optimization on ABL1**
**Purpose:** Find optimal UMAP parameters with corrected v2.0 data

#### Step 1: Create Hyperparameter Config Generator (v2.0)
**File:** `config_generators/generate_hyperparameter_configs.py` (TO BE CREATED)

**Requirements:**
- Target: ABL1 (Transferase KW-0808) - corrected from v1.0
- Representations: Features + Fingerprints
- Methods:
  - PCA (projection) - baseline for both representations
  - UMAP-Euclidean (projection, features only) - sweep n_neighbors, min_dist
  - UMAP-Jaccard (projection, fingerprints only) - sweep n_neighbors, min_dist
- Hyperparameter grid:
  - n_neighbors: [15, 50, 100, 200, 500]
  - min_dist: [0.0, 0.01, 0.1, 0.5]
  - Combinations: 5 × 4 = 20 configs per UMAP variant
- Seeds: 5 (42-46)
- Total jobs: 42 configs × 5 seeds = 210 jobs

#### Step 2: Generate Configs
```bash
python config_generators/generate_hyperparameter_configs.py
```

#### Step 3: Submit HPC Jobs
```bash
cd hpc/
bash submit_all_replicates_hyperparameterization.sh
```

**Expected runtime:** 48-72 hours (parallel execution)

#### Step 4: Analyze Results
```bash
sbatch hpc/ummbas_hyperparameterization_analysis.sh
```

**Output:** Optimal hyperparameters for features and fingerprints

---

### **PHASE 3: Generalization Experiment**
**Purpose:** Test optimal parameters on held-out targets

#### Step 1: Update Generalization Configs
Edit `config_generators/generate_generalization_configs.py`:
- Use optimal hyperparameters from Phase 2
- Update to use `.csv` file paths (not `.pkl`)

#### Step 2: Generate Configs
```bash
python config_generators/generate_generalization_configs.py
```

**Creates:** 4 configs (2 targets × 2 DR methods)
- Pyruvate Kinase M2 (Transferase): PCA + UMAP-Euclidean
- Isocitrate Dehydrogenase (Oxidoreductase): PCA + UMAP-Euclidean

#### Step 3: Submit Jobs
```bash
cd hpc/
bash submit_generalization_jobs.sh
```

**Jobs:** 4 configs × 5 seeds = 20 jobs  
**Expected runtime:** 24-48 hours per job (parallel)

#### Step 4: Run Baselines
```bash
# Pyruvate Kinase M2 (after main jobs complete)
python baselines/baseline_random_ranking.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_feature_space_distance.py PyruvateKinaseM2 P14618 KW-0808
python baselines/baseline_tanimoto_similarity.py PyruvateKinaseM2 P14618 KW-0808
python baselines/analyze_chemical_novelty.py PyruvateKinaseM2 P14618 KW-0808

# Isocitrate Dehydrogenase
python baselines/baseline_random_ranking.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_feature_space_distance.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/baseline_tanimoto_similarity.py IsocitrateDehydrogenaseNADP O75874 KW-0560
python baselines/analyze_chemical_novelty.py IsocitrateDehydrogenaseNADP O75874 KW-0560
```

**Runtime:** 1-2 hours total (can run in parallel)

#### Step 5: Aggregate Results
```bash
python analysis_scripts/aggregate_generalization_analysis.py
python analysis_scripts/aggregate_and_report.py
```

---

### **PHASE 4: Optional - Add Fingerprints to Generalization**
If Phase 2 shows fingerprints perform well:

1. Add fingerprints configs to generalization
2. Re-run Phase 3 with fingerprints included

---

## Complete Timeline (Recommended Workflow)

| Phase | Task | Duration | Total Time |
|-------|------|----------|------------|
| 1 | Local test run | 30-60 min | 1 hour |
| 2 | Generate hyperparameter configs | 1 min | 1 hour |
| 3 | Hyperparameter optimization (HPC) | 48-72 hours | 3 days |
| 4 | Hyperparameter analysis | 1-2 hours | 3 days |
| 5 | Update generalization configs | 5 min | 3 days |
| 6 | Generate generalization configs | 2 min | 3 days |
| 7 | Generalization experiment (HPC) | 24-48 hours | 5 days |
| 8 | Run baselines | 1-2 hours | 5 days |
| 9 | Aggregate & report | 30 min | 5 days |

**Total: ~5-6 days** (with parallel HPC execution)

**Note:** Hyperparameter optimization (Phase 3) can be skipped if using default parameters, but is highly recommended for best results with the corrected v2.0 Transferase MF cloud (191k compounds vs 20 in v1.0).

---

## Critical Files to Create/Fix

### TO CREATE:
1. ✅ `docs/HPC_EXECUTION_GUIDE.md` - Comprehensive HPC guide
2. ✅ `quick_test_run.sh` - Local test script
3. ❌ `config_generators/generate_hyperparameter_configs.py` - v2.0 hyperparameter sweep

### TO FIX:
1. ❌ `config_generators/generate_generalization_configs.py` - Change `.pkl` to `.csv`
2. ❌ `config_generators/generate_dimensionality_configs.py` - Update ABL1 MF to Transferase
3. ❌ `experiment_config.json` - Verify file paths use `.csv`
4. ❌ `core_scripts/calculate_features_and_fingerprints_exp.py` - Check CSV loading logic
5. ❌ `core_scripts/calculate_similarityspaces_exp.py` - Check CSV loading logic

---

## Next Immediate Steps

1. **Create `config_generators/generate_hyperparameter_configs.py`** for ABL1 v2.0
2. **Fix all file path references** from `.pkl` to `.csv`
3. **Run test:** `bash quick_test_run.sh`
4. **If test passes:** Proceed to Phase 2 (hyperparameter optimization)

---

## Key v2.0 Corrections

| Item | v1.0 (Wrong) | v2.0 (Correct) |
|------|--------------|----------------|
| ABL1 MF | "Protein kinase inhibitor" | "Transferase (KW-0808)" |
| MF Cloud Size | 20 compounds | 191,648 compounds |
| Data Leakage | Co-embedding included actives | Projection-only (no leakage) |
| t-SNE | Used with co-embedding | Removed completely |
| Fingerprint Processing | StandardScaler, PCA pre-reduction | PassthroughScaler, Jaccard-only |
| File Format | Expecting `.pkl` | Actually `.csv` |

---

**Status:** Ready to proceed with file path fixes and hyperparameter config creation.
