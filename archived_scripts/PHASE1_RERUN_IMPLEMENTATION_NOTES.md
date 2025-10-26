# Phase 1 Rerun Implementation Notes

**Date**: October 25, 2025  
**Decision**: Adopt revised hyperparameter grid with multi-threading enabled

---

## Approved Hyperparameter Grid

```python
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
FEATURE_DIMS = [2, 5, 10]
SEEDS = [42, 43, 44, 45, 46]
USE_FIXED_SEED = False  # Enable UMAP multi-threading
```

**Total Experiments**: 3 × 5 × 5 × 5 = **375 configs**

**Changes from Original**:
- ✅ Removed: nn=3, 5 (72-hour timeout - computationally infeasible)
- ✅ Added: nn=50, 100, 500 (hypothesis: optimal shifted to medium-large range)
- ✅ Disabled fixed seed for 10× speedup via UMAP multi-threading

---

## Rationale

### Why Remove nn=3, 5?
1. **Computational**: nn=3 killed after 72-hour timeout
2. **Scientific**: Small nn exploited duplicate clusters (artifact, not signal)
3. **Evidence**: nn=10 dropped 58.3% with clean data (43.78 → 18.27)

### Why Add nn=50, 100, 500?
1. **Hypothesis**: Optimal nn shifted upward after removing 2.23× duplicates
2. **Evidence**: nn=500 showed robust 19.57 despite duplicates (≈ clean nn=10's 18.27)
3. **Scientific**: Must test medium-large nn range to find TRUE optimum

### Why Disable Fixed Seed?
1. **Speed**: 10-100× faster with multi-threading enabled
2. **Feasibility**: Makes nn=500 tractable
3. **Trade-off**: Increased variance (~2× higher) acceptable with 5 replicates
4. **Standard Practice**: ML papers report mean ± std across multiple runs

---

## Implementation Checklist

### 1. Config Generation (DONE)
- [x] Update `generate_phase1_configs.py`
  - Changed `FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]`
  - Added `USE_FIXED_SEED = False` flag
  - Updated output dir to `hyperparam_configs_v3_phase1_rerun`
  - Updated workspace dir to `experiment_workspace_v3_phase1_rerun`

### 2. UMAP Code Modification (TODO)
**File**: `core_scripts/calculate_similarityspaces_exp.py`

**Current code** (slow, reproducible):
```python
umap_model = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=simspace_dim,
    metric=metric,
    random_state=config['random_seed']  # ← REMOVE THIS
)
```

**Modified code** (fast, multi-threaded):
```python
umap_model = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=simspace_dim,
    metric=metric
    # random_state REMOVED to enable multi-threading
)
```

**Location to modify**: Search for `random_state` parameter in UMAP initialization

### 3. Generate Configs
```bash
cd /path/to/UMMBAS_screening_experiments
python generate_phase1_configs.py
```

**Expected output**:
- Directory: `hyperparam_configs_v3_phase1_rerun/`
- Files: 375 JSON config files (+ 15 PCA configs)

### 4. SLURM Script Update
**File**: `hpc/submit_v3_phase1_rerun.sh` (create new)

**Changes from original**:
- Point to new config dir: `hyperparam_configs_v3_phase1_rerun/`
- Point to new workspace: `experiment_workspace_v3_phase1_rerun/`
- Update time limit: 4 hours (was 2 hours)
- Update job array size: 0-389 (375 UMAP + 15 PCA)

### 5. Quality Control During Run
Monitor for:
- Deduplication logs in prepare_data.py output
- UMAP training time (should be ~3 min with multi-threading, not 6+ min)
- Cross-seed variance for each (nn, md, dim) combination
- Flag configs with σ > 3 EF@1% for inspection

---

## Expected Runtime

**Per experiment** (with multi-threading):
- Data prep: ~2 min
- UMAP training: ~3 min (was 6 min with fixed seed)
- PCA training: ~30 sec
- Projection + analysis: ~1 min

**Total**:
- UMAP: 375 × 6 min = 37.5 hours sequential
- PCA: 15 × 3.5 min = 0.9 hours sequential
- **Combined**: ~38 hours sequential
- **HPC parallel (32 jobs)**: ~1.5-2 hours wall clock

**Comparison**:
- Original plan (with nn=3,5): Would timeout
- Option B (all with seed): 52 hours
- **This approach**: 38 hours ✓

---

## Variance Analysis Plan

### Expected Variance
- **With fixed seed** (original): σ ~ 0.5-0.8 EF@1%
- **Multi-threaded** (new): σ ~ 1-2 EF@1% (2-3× higher)
- **SEM with 5 replicates**: 0.4-0.9

### Quality Checks
1. Calculate σ for each (nn, md, dim) combination
2. Flag combinations with σ > 3 EF@1%
3. If flagged, consider:
   - Re-run specific config with fixed seed
   - Increase replicate count (add seed 47, 48)
   - Accept higher variance if trend is clear

### Publication Reporting
- Report: Mean ± SEM across 5 independent runs
- Note: "UMAP trained with multi-threading for computational efficiency"
- Compare variance to original Phase 1: "Cross-run variance increased 2× but within acceptable range for hyperparameter selection"

---

## Hypothesis Testing Framework

### Primary Hypothesis
**H1**: Optimal nn shifted from 10 → 50-100 after duplicate removal

**Test**: Compare mean EF@1% across nn values
- **Expected**: nn=50 or nn=100 > nn=10, nn=500
- **Effect size**: >5 EF@1% (exceeds variance ~1-2)

### Secondary Hypotheses
**H2**: Dimensionality effect remains (2D < 5D < 10D)
**H3**: min_dist effect remains weak (<5% variation)

### Statistical Analysis
- ANOVA across nn values (primary factor)
- Tukey HSD for pairwise comparisons
- Effect size: Cohen's d for nn comparisons
- Report: "nn=X significantly outperformed nn=Y (p<0.05, d=Z)"

---

## Next Steps

**Immediate**:
1. [ ] Modify `calculate_similarityspaces_exp.py` to remove `random_state`
2. [ ] Generate 375 configs with updated `generate_phase1_configs.py`
3. [ ] Create `hpc/submit_v3_phase1_rerun.sh`
4. [ ] Test single config locally to verify multi-threading works
5. [ ] Submit to HPC queue

**During Run**:
1. [ ] Monitor progress (expected ~2 hours)
2. [ ] Check logs for deduplication confirmation
3. [ ] Verify UMAP speed (~3 min, not 6+ min)

**After Completion**:
1. [ ] Calculate cross-seed variance for each config
2. [ ] Identify optimal (nn, md, dim) combination
3. [ ] Compare to original Phase 1 rankings
4. [ ] Extract best configs for Phase 2-4
5. [ ] Update mechanistic hypotheses

---

## Files to Modify/Create

**Modified**:
- ✅ `generate_phase1_configs.py` - Updated hyperparameters
- ✅ `LAB_BOOK.md` - Documented decision
- ✅ `PHASE1_RERUN_HYPERPARAMETER_ANALYSIS.md` - Comprehensive analysis
- [ ] `core_scripts/calculate_similarityspaces_exp.py` - Remove random_state

**Created**:
- ✅ `PHASE1_RERUN_IMPLEMENTATION_NOTES.md` - This file
- [ ] `hpc/submit_v3_phase1_rerun.sh` - SLURM submission script
- [ ] `hyperparam_configs_v3_phase1_rerun/` - Config directory (375 files)

---

## Commit Message

```
Phase 1 Rerun: Revised hyperparameters after 58.3% performance drop

- Comparison test confirmed: nn=10 dropped from 43.78 → 18.27 with clean data
- Removed nn=3, 5 (72hr timeout - computationally infeasible)
- Added nn=50, 100, 500 (hypothesis: optimal shifted to medium-large range)
- Disabled fixed seed for 10× speedup via UMAP multi-threading
- Total: 375 experiments, ~38 hrs sequential, ~2 hrs HPC parallel

Evidence: nn=500 showed robust 19.57 despite duplicates (≈ clean nn=10's 18.27)
→ Suggests optimal nn in 50-500 range, not 3-20 range

See: PHASE1_RERUN_HYPERPARAMETER_ANALYSIS.md, LAB_BOOK.md
```
