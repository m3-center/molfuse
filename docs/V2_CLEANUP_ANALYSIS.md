# v2.0 Codebase Cleanup Analysis

## Current Status: Partially Clean ⚠️

The critical HPC execution error has been **fixed** ✅, but there are still **legacy artifacts** that should be cleaned up for code clarity and maintainability.

---

## ✅ What's Already Fixed (Critical)

### 1. `main_orchestrator.py` - **FIXED**
- ✅ Removed `--run_coembedding_for_pca_umap` argument
- ✅ Removed `--dr_method_tsne` argument  
- ✅ Removed `run_coembedding_pca_umap_flag` variable
- ✅ Removed co-embedding analysis block (~30 lines)
- ✅ Removed t-SNE analysis block (~23 lines)
- ✅ Simplified projection strategy loop

**Impact:** HPC jobs now run successfully without argument errors.

---

## ⚠️ Legacy Artifacts Remaining (Non-Critical but Confusing)

### 1. `main_orchestrator.py` - Line 217
**Issue:** Argument name still references t-SNE/co-embedding

```python
"--target_ligands_unscaled_path_for_tsne_and_coembed", os.path.abspath(...)
```

**Should be:**
```python
"--target_ligands_path_for_projection", os.path.abspath(...)
```

**Impact:** 
- ❌ Confusing naming (implies t-SNE/co-embedding still used)
- ✅ Functionally correct (argument still accepted by script)
- 📝 Should rename for clarity

---

### 2. `core_scripts/calculate_similarityspaces_exp.py` - Line 263
**Issue:** Argument name still references t-SNE/co-embedding

```python
parser.add_argument(
    "--target_ligands_unscaled_path_for_tsne_and_coembed", default=None)
```

**Should be:**
```python
parser.add_argument(
    "--target_ligands_path_for_projection", default=None,
    help="Path to held-out target ligands for projection (v2.0 projection-only)")
```

**Impact:**
- ❌ Confusing naming
- ✅ Functionally correct
- 📝 Should rename for clarity

---

### 3. `core_scripts/calculate_similarityspaces_exp.py` - Line 320
**Issue:** Variable name references old argument

```python
df_target, X_target_original, X_target_scaled = prepare_target_ligands(
    args.target_ligands_unscaled_path_for_tsne_and_coembed,
    ...
)
```

**Should be:**
```python
df_target, X_target_original, X_target_scaled = prepare_target_ligands(
    args.target_ligands_path_for_projection,
    ...
)
```

---

### 4. `debugging_config.json` - **ENTIRE FILE OUTDATED** ⚠️
**Issue:** Contains v1.0 settings

```json
{
  "global_settings": {
    "tsne_pca_components": 50,  // ❌ t-SNE removed
    "run_coembedding_for_pca_umap": true  // ❌ Co-embedding removed
  },
  "dimensionality_reduction_methods": {
    "pca": {"allow_coembedding": true},  // ❌ No co-embedding in v2.0
    "umap_euclidean": {"allow_coembedding": true},  // ❌
    "tsne": {"perplexity": 30, "allow_coembedding": true}  // ❌ t-SNE removed
  }
}
```

**Should be:** Updated to match `experiment_config.json` structure (v2.0)

**Impact:**
- ⚠️ Will cause errors if used with fixed `main_orchestrator.py`
- 📝 Should update to v2.0 format
- ℹ️ This is a **testing config** - may not be actively used

---

### 5. Function `prepare_target_ligands()` - Line 164
**Status:** Already has good v2.0 docstring ✅

```python
def prepare_target_ligands(path, repr_type, features_list, scaler):
    """
    Load and prepare held-out target ligands for PROJECTION ONLY.
    These ligands will be transformed using a pre-fitted DR model.
    Co-embedding removed in v2.0 to prevent data leakage.
    """
```

**Impact:** Documentation is correct and clear ✅

---

## 🎯 Recommended Cleanup Actions

### Priority 1: High (Improves Code Clarity) 🟨

#### Action 1.1: Rename Argument in Both Files
**Files to change:**
1. `main_orchestrator.py` (line 217)
2. `core_scripts/calculate_similarityspaces_exp.py` (line 263, 320)

**Change:**
```bash
# Find and replace across both files
target_ligands_unscaled_path_for_tsne_and_coembed 
→ target_ligands_path_for_projection
```

**Benefits:**
- ✅ Clear naming reflects v2.0 projection-only strategy
- ✅ No confusion about removed features
- ✅ Better code documentation

**Risk:** Low (simple rename, no logic change)

---

#### Action 1.2: Update `debugging_config.json` to v2.0
**File:** `debugging_config.json`

**Changes needed:**
1. Remove `"tsne_pca_components": 50`
2. Remove `"run_coembedding_for_pca_umap": true`
3. Remove `"allow_coembedding": true` from all DR methods
4. Remove `"tsne"` method entirely
5. Add proper v2.0 structure matching `experiment_config.json`

**Template:**
```json
{
  "global_settings": {
    "experiment_name": "UMMBAS_Debugging",
    "base_workspace_dir": "test_experiment_workspace",
    "simspace_dims_to_test": [2],
    "k_for_knn_distance": [5, 10, 20],
    "rdkit_features_list_target": [...],
    "fingerprint_pca_components": 50
  },
  "targets": [...],
  "dimensionality_reduction_methods": {
    "pca": {
      "short_name": "PCA"
    },
    "umap_euclidean": {
      "short_name": "UMAP-Euclidean",
      "metric": "euclidean"
    }
  }
}
```

**Benefits:**
- ✅ Debugging tests work with v2.0 orchestrator
- ✅ Consistent with production configs
- ✅ No confusion about what's supported

**Risk:** Low (only affects debugging/testing)

---

### Priority 2: Optional (Documentation) 🟦

#### Action 2.1: Add v2.0 Header to `calculate_similarityspaces_exp.py`
**File:** `core_scripts/calculate_similarityspaces_exp.py` (top of file)

**Add:**
```python
"""
UMMBAS v2.0 Similarity Space Calculator

v2.0 Changes:
- PROJECTION-ONLY strategy (no co-embedding)
- Removed t-SNE (co-embedding-only method)
- Target ligands projected using pre-fitted models
- Prevents data leakage from co-embedding

Strategy:
1. Fit DR models on MF cloud only
2. Transform MF cloud compounds
3. Project held-out target ligands using fitted models
4. Never mix target ligands with MF cloud during fitting
"""
```

---

## 📊 Summary Table

| Component | Status | Priority | Impact if Not Fixed |
|-----------|--------|----------|---------------------|
| `main_orchestrator.py` argument passing | ✅ Fixed | Critical | ~~Jobs fail~~ ✅ |
| `main_orchestrator.py` co-embedding block | ✅ Removed | Critical | ~~Jobs fail~~ ✅ |
| `main_orchestrator.py` t-SNE block | ✅ Removed | Critical | ~~Jobs fail~~ ✅ |
| Argument name (both files) | ⚠️ Confusing | High | Code confusion |
| `debugging_config.json` | ⚠️ Outdated | High | Debug tests fail |
| File header documentation | 📝 Missing | Low | Minor confusion |

---

## 🚦 Recommendation

### Immediate Action: **NO** ❌
The HPC error is **fixed** and jobs can run now. No immediate action needed.

### Recommended Follow-up: **YES** ✅
For code quality and maintainability:

1. **Rename the argument** (10 minutes)
   - Clearer code, better documentation
   - Simple find/replace operation
   - Low risk

2. **Update debugging_config.json** (15 minutes)
   - Ensures debugging/testing works
   - Matches production config structure
   - Only affects local testing

**Total time:** ~25 minutes
**Risk level:** Low
**Benefit:** Improved code clarity and consistency

---

## 🔍 Verification Commands

After cleanup (if done):

```bash
# 1. Check no references to old naming remain
grep -r "target_ligands_unscaled_path_for_tsne_and_coembed" --include="*.py"
# Should return: No matches

# 2. Verify new naming is consistent
grep -r "target_ligands_path_for_projection" --include="*.py"
# Should find: 3 matches (main_orchestrator.py and calculate_similarityspaces_exp.py)

# 3. Check debugging config is v2.0 compliant
grep -E "tsne|allow_coembedding|run_coembedding" debugging_config.json
# Should return: No matches

# 4. Test with debugging config
bash quick_test_run.sh --hyperparam-only
```

---

## 💡 My Recommendation

**Option A: Clean now** (25 minutes)
- Better code quality
- No technical debt
- Easier for future maintenance
- Good practice

**Option B: Clean later**
- HPC jobs work now ✅
- Can do cleanup during next maintenance window
- Focus on running experiments first

**My vote:** Option A (clean now) - it's quick and prevents confusion later.

What's your preference?

---

**Last Updated:** October 14, 2025  
**Version:** 2.0 Post-Fix Analysis
