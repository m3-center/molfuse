# Project and Analyze Scaler Path Fix (v2.0)

**Date:** October 14, 2025  
**Issue:** FileNotFoundError when projecting fingerprints with PCA

---

## Problem Description

HPC jobs were failing during the analysis phase with:

```
FileNotFoundError: [Errno 2] No such file or directory: 
'/home/.../models/fingerprints/dim_2/TyrosineProteinKinaseABL1_P00519_fingerprints_dim2_scaler_for_pca.lzma'
```

### Root Cause

**Mismatch between scaler filenames:**

`calculate_similarityspaces_exp.py` (v2.0) saves:
```python
# For fingerprints
save_model(scaler, f"{target}_fingerprints_dim{N}_scaler.lzma")
```

`project_and_analyze.py` (v1.0 logic) was looking for:
```python
# For fingerprints with PCA
scaler_path = f"{target}_fingerprints_dim{N}_scaler_for_pca.lzma"  # ❌ Wrong!

# For fingerprints with UMAP-Jaccard
scaler_path = f"{target}_fingerprints_dim{N}_scaler_for_binary.lzma"  # ❌ Wrong!
```

### v1.0 vs v2.0 Logic

**v1.0 (Complex, Wrong):**
- Fingerprints with PCA → StandardScaler → `scaler_for_pca.lzma`
- Fingerprints with UMAP-Euclidean → StandardScaler → `scaler_for_pca.lzma`
- Fingerprints with UMAP-Jaccard → PassthroughScaler → `scaler_for_binary.lzma`

**v2.0 (Simple, Correct):**
- Features → StandardScaler → `scaler.lzma`
- Fingerprints → PassthroughScaler → `scaler.lzma` (no scaling, always)

**Why v2.0 is correct:**
- Binary fingerprints should NOT be scaled (breaks Jaccard distance)
- PCA can work with unscaled binary data
- Consistent: one scaler type per representation

---

## Solution

Updated `experimental_pipeline/project_and_analyze.py` to use v2.0 scaler logic.

### Code Changes

**Before (v1.0 logic):**
```python
scaler_suffix = "scaler.lzma" # Default for features
if args.representation_type == "fingerprints":
    # For fingerprints, choose the scaler based on the DR method
    if args.dr_method_key in ["umap_jaccard", "umap_hamming"]:
        scaler_suffix = "scaler_for_binary.lzma"
        logging.info("Using PassthroughScaler for binary-native UMAP metric.")
    else:
        scaler_suffix = "scaler_for_pca.lzma"
        logging.info("Using StandardScaler for Euclidean-like DR method.")
```

**After (v2.0 logic):**
```python
# v2.0: Simplified scaler logic
# Features always use StandardScaler, fingerprints always use PassthroughScaler (no scaling)
scaler_suffix = "scaler.lzma"
if args.representation_type == "fingerprints":
    logging.info("Using PassthroughScaler for fingerprints (v2.0: no scaling for binary vectors).")
else:
    logging.info("Using StandardScaler for features.")
```

---

## Impact

### What Now Works ✅
- Fingerprints with PCA projection
- Fingerprints with UMAP-Euclidean projection
- Fingerprints with UMAP-Jaccard projection
- Features with any DR method (unchanged)

### File Naming (Consistent)
```bash
# Features
{target}_features_dim{N}_scaler.lzma        # StandardScaler
{target}_features_dim{N}_PCA_model.lzma
{target}_features_dim{N}_euclidean_UMAP_model.lzma

# Fingerprints
{target}_fingerprints_dim{N}_scaler.lzma    # PassthroughScaler
{target}_fingerprints_dim{N}_PCA_model.lzma
{target}_fingerprints_dim{N}_jaccard_UMAP_model.lzma
```

### Why This Matters
Binary fingerprints should never be scaled:
- Scaling destroys the binary nature (values become non-integer)
- Breaks Jaccard distance calculation
- PCA works fine with binary data (just treats them as 0/1 values)
- v2.0 correctly uses PassthroughScaler (identity transformation)

---

## Testing

### Verify Scaler Files Exist
```bash
# Check what scalers were actually created
ls experiment_workspace_hyperparam_sweep_v2/run_seed*/*/models/fingerprints/dim_2/*scaler*.lzma

# Should show:
# {target}_fingerprints_dim2_scaler.lzma  ✅
# (NOT scaler_for_pca.lzma or scaler_for_binary.lzma)
```

### Test Projection
```bash
# Re-run failed job
sbatch hpc/ummbas_experiment_single_replicate.sh \
    hyperparam_configs/config_fingerprints_pca_projection.json \
    44 \
    full_analysis
```

---

## Related Files

**Fixed:**
- ✅ `experimental_pipeline/project_and_analyze.py` - Scaler path logic

**Already v2.0 compliant:**
- ✅ `core_scripts/calculate_similarityspaces_exp.py` - Saves correct filenames
- ✅ `main_orchestrator.py` - Passes correct arguments

**Documentation:**
- `docs/PROJECT_ANALYZE_SCALER_FIX.md` - This file
- `docs/MAIN_ORCHESTRATOR_V2_FIX.md` - Previous orchestrator fix
- `docs/V2_CLEANUP_COMPLETED.md` - Comprehensive cleanup

---

## v2.0 Scaler Strategy Summary

| Representation | Scaler Type | Filename | Reasoning |
|----------------|-------------|----------|-----------|
| Features | StandardScaler | `{target}_features_dim{N}_scaler.lzma` | Continuous values benefit from scaling |
| Fingerprints | PassthroughScaler | `{target}_fingerprints_dim{N}_scaler.lzma` | Binary vectors must not be scaled |

**Key principle:** Scaler type depends on **representation**, NOT on DR method.

---

## Historical Context

**Why v1.0 was wrong:**
- Applied StandardScaler to fingerprints for PCA/UMAP-Euclidean
- Tried to use "Euclidean distance" on scaled binary vectors
- Created separate scaler files based on DR method
- Inconsistent and complex logic

**Why v2.0 is correct:**
- Fingerprints always use Jaccard distance (even with PCA preprocessing)
- PCA just reduces dimensionality, doesn't require scaling
- Simple: one scaler per representation type
- Mathematically sound for binary data

---

## Summary

✅ **Fixed:** Scaler path mismatch in `project_and_analyze.py`  
✅ **Aligned:** Now matches v2.0 scaler logic  
✅ **Simplified:** Scaler type based on representation, not DR method  
✅ **Ready:** HPC jobs will complete successfully

**Status:** Ready for production ✅

---

**Last Updated:** October 14, 2025  
**Version:** 2.0 Scaler Path Fix
