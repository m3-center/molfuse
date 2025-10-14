# Main Orchestrator v2.0 Compatibility Fix

## Issue Description

When running jobs on the HPC cluster, `main_orchestrator.py` was passing deprecated v1.0 command-line arguments to `calculate_similarityspaces_exp.py`, causing the following error:

```
calculate_similarityspaces_exp.py: error: unrecognized arguments: --run_coembedding_for_pca_umap=False --dr_method_tsne=False
```

## Root Cause

`main_orchestrator.py` still contained v1.0 logic for:
1. **Co-embedding strategy** - passing `--run_coembedding_for_pca_umap` argument
2. **t-SNE method** - passing `--dr_method_tsne` and related arguments
3. **Analysis blocks** - separate code blocks for co-embedding and t-SNE analysis

These features were removed in v2.0 because:
- **Co-embedding causes data leakage** (target ligands embedded with MF cloud)
- **t-SNE is a co-embedding-only method** (cannot project new points)
- **v2.0 uses projection-only strategy** (fit on MF cloud, transform target ligands)

## Changes Made

### 1. Removed Legacy Command-Line Arguments

**File:** `main_orchestrator.py` (lines ~215-240)

**Before:**
```python
cmd_calc_simspace = [ "python", "core_scripts/calculate_similarityspaces_exp.py",
    # ... other args ...
    f"--run_coembedding_for_pca_umap={str(run_coembedding_pca_umap_flag)}",
    "--dr_method_configs_json_str", dr_method_configs_json_str,
    # ... other args ...
]
# ... later ...
cmd_calc_simspace.append(f"--dr_method_tsne={str('tsne' in active_dr_methods_cfg)}")
if 'tsne' in active_dr_methods_cfg:
    cmd_calc_simspace.extend([f"--tsne_perplexity={...}", f"--tsne_pca_components={...}"])
```

**After:**
```python
cmd_calc_simspace = [ "python", "core_scripts/calculate_similarityspaces_exp.py",
    # ... other args ...
    "--dr_method_configs_json_str", dr_method_configs_json_str,
    # ... other args ...
]
# v2.0: t-SNE removed (co-embedding only, causes data leakage)
```

### 2. Removed Co-embedding Flag Variable

**File:** `main_orchestrator.py` (line ~145)

**Before:**
```python
run_coembedding_pca_umap_flag = gs.get('run_coembedding_for_pca_umap', False)
```

**After:**
```python
# v2.0: Co-embedding removed (projection-only strategy)
```

### 3. Removed Co-embedding Analysis Block

**File:** `main_orchestrator.py` (lines ~280-310)

**Before:**
```python
# --- CO-EMBEDDING STRATEGY RUN (for PCA/UMAP) ---
if run_coembedding_pca_umap_flag:
    for dr_key, dr_params_cfg in active_dr_methods_cfg.items():
        # ... 30 lines of co-embedding analysis logic ...
```

**After:**
```python
# v2.0: Co-embedding and t-SNE blocks removed (projection-only strategy)
```

### 4. Removed t-SNE Analysis Block

**File:** `main_orchestrator.py` (lines ~312-335)

**Before:**
```python
# --- DEDICATED T-SNE BLOCK (Always Co-Embedded) ---
if 'tsne' in active_dr_methods_cfg:
    dr_key = 'tsne'
    # ... 23 lines of t-SNE analysis logic ...
```

**After:**
```python
# v2.0: Co-embedding and t-SNE blocks removed (projection-only strategy)
```

### 5. Simplified Projection Strategy Logic

**File:** `main_orchestrator.py` (lines ~249-253)

**Before:**
```python
for dr_key, dr_params_cfg in active_dr_methods_cfg.items():
    if dr_key == "tsne": continue # t-SNE is handled separately
    # Skip methods that will be handled by co-embedding block
    if run_coembedding_pca_umap_flag and dr_params_cfg.get("allow_coembedding", False) and (dr_key.startswith("pca") or dr_key.startswith("umap")):
        continue # This method will run in co-embedding block
```

**After:**
```python
# --- PROJECTION STRATEGY RUN (PCA / UMAP Only) - v2.0 ---
for dr_key, dr_params_cfg in active_dr_methods_cfg.items():
    # v2.0: t-SNE removed (co-embedding only method)
    if dr_key == "tsne": continue
```

## Impact

### What Now Works ✅
- Jobs run successfully on HPC without argument errors
- Projection-only strategy for all DR methods (PCA, UMAP)
- No data leakage from co-embedding
- Consistent with v2.0 pipeline design

### What's Removed ❌
- Co-embedding functionality (was causing data leakage)
- t-SNE method (cannot project new points)
- Separate analysis blocks for co-embedded spaces

### Backward Compatibility ⚠️
- **v1.0 configs will not work** if they specify:
  - `"run_coembedding_for_pca_umap": true`
  - `"tsne"` in `dimensionality_reduction_methods`
- All generated v2.0 configs already comply with these changes

## Testing

To verify the fix works:

```bash
# Test locally with quick test
bash quick_test_run.sh --hyperparam-only

# Test on HPC with single config
sbatch hpc/ummbas_experiment_single_replicate.sh \
    hyperparam_configs/config_features_umap_euclidean_projection_nn50_md0.01.json \
    42 \
    full_analysis
```

## Related Files

**Config Generators (already v2.0 compliant):**
- `config_generators/generate_hyperparam_configs.py` ✅
- `config_generators/generate_generalization_configs.py` ✅
- `config_generators/generate_dimensionality_configs.py` ✅

**Core Scripts (already v2.0 compliant):**
- `core_scripts/calculate_similarityspaces_exp.py` ✅
- `core_scripts/calculate_features_and_fingerprints_exp.py` ✅

**Documentation:**
- `docs/HPC_EXECUTION_GUIDE.md` - Updated with v2.0 workflows
- `docs/EXECUTION_SUMMARY.md` - v2.0 timeline and requirements
- `docs/CONFIG_GENERATORS_SUMMARY.md` - v2.0 config generation

## Summary

This fix aligns `main_orchestrator.py` with the v2.0 architecture:
- **Projection-only strategy** (no data leakage)
- **Removed co-embedding** (was flawed methodology)
- **Removed t-SNE** (co-embedding only method)
- **Clean argument passing** (only valid v2.0 arguments)

**Status:** ✅ Ready for HPC execution

---

**Last Updated:** October 14, 2025  
**Version:** 2.0
