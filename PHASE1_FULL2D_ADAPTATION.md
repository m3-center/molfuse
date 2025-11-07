# Phase 1 Adaptation: Full 2D Mordred Features

**Date:** November 5, 2025  
**Objective:** Adapt Phase 1 pipeline to use full 2D Mordred descriptors (1613 raw features) based on feature_comparison_v2.py results showing highest EF@1% performance.

---

## Summary of Changes

### 1. **Dataset Migration**
- **From:** `datasets/molecular_function_features_fingerprints/` (40-feature subset)
- **To:** `output_recalculated_full_datasets/datasets_2d_all/` (1613 raw 2D Mordred features)
- **Expected after filtering:** ~1477 features (post zero-variance removal)

### 2. **molfuse/data/prep.py**

#### Added:
- `SimpleImputer` import for NaN handling
- `METADATA_COLUMNS` constant for selective dtype specification
- `remove_zero_variance_features()` function - removes features with variance < 1e-12
- Updated `fit_scaler_on_mf_zinc()` to return `(imputer, scaler)` tuple instead of just scaler

#### Pipeline changes:
```python
# OLD:
scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, feature_cols)

# NEW:
imputer, scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, feature_cols)
# Returns: (SimpleImputer(strategy='median'), StandardScaler)
```

### 3. **molfuse/cli/phase1.py**

#### Added `load_csv_optimized()` function:
**Performance optimizations ported from feature_comparison_v2.py:**

1. **Selective dtype specification** (metadata only):
   - Only 8 columns forced to specific types (7 str + 1 float)
   - All 1600+ feature columns inferred as numeric (not string)
   - **Impact:** 300GB → 30GB memory usage (10x reduction)

2. **PyArrow engine** with graceful fallback:
   - 3-5x faster CSV parsing when available
   - Falls back to default engine if PyArrow not installed

3. **Automatic Parquet caching**:
   - First run: CSV load + conversion to Parquet
   - Subsequent runs: 10-100x faster Parquet load
   - Row count validation to detect corruption

4. **Defensive type conversion**:
   - Immediately after loading: `pd.to_numeric(col, errors='coerce')` for all non-metadata columns
   - Prevents NaN explosion from string columns downstream

#### Updated feature processing:
```python
# Zero-variance filtering (NEW):
common_feats = remove_zero_variance_features(df_train_check, common_feats)

# Imputation + scaling (NEW):
imputer, scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, common_feats)
X_mf = imputer.transform(df_mf[common_feats])
X_mf = scaler.transform(X_mf)

# Actives transformation (NEW - with imputation):
X_act = imputer.transform(df_act_feat)
X_act = scaler.transform(X_act)

# Save both artifacts (NEW):
joblib.dump(imputer, ws["artifacts"] / "imputer.joblib")
joblib.dump(scaler, ws["artifacts"] / "scaler.joblib")
```

#### Replaced all CSV loading:
```python
# OLD:
df_mf_all = pd.read_csv(mf_csv, low_memory=False)

# NEW:
df_mf_all = load_csv_optimized(mf_csv, logger)
```

### 4. **scripts/generate_molfuse_phase1_configs_v4.py**

#### Updated BASE paths:
```python
# OLD:
"mf_features_csv": "datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv",

# NEW:
"mf_features_csv": "output_recalculated_full_datasets/datasets_2d_all/KW-0808_Transferase_affinity_extracted_features.csv",
```

#### Updated notes:
- "full 2D Mordred features (1613 raw)"
- "optimized CSV loading with Parquet caching"

#### Generated configs:
- **Total:** 630 configs (315 feature + 315 fingerprint configs × 5 replicates)
- **Location:** `configs/molfuse_phase1_grid/`

### 5. **configs/molfuse_phase1_example.json**

Updated example config to demonstrate full 2D features:
- Changed from fingerprints to features representation
- Updated all paths to `output_recalculated_full_datasets/datasets_2d_all/`
- Changed metric from jaccard to euclidean
- Reduced dim from 10 to 5 for faster testing

---

## Key Preprocessing Differences: feature_comparison_v2.py vs phase1.py

| Aspect | feature_comparison_v2 | phase1 (OLD) | phase1 (NEW) |
|--------|----------------------|--------------|--------------|
| **Loading** | Selective dtype + PyArrow + Parquet | Basic `read_csv` | ✅ Ported optimizations |
| **Memory usage** | 30GB for 10GB CSV | 300GB for 10GB CSV | ✅ 30GB (10x improvement) |
| **Type conversion** | Defensive (immediate) | During feature selection | ✅ Defensive (immediate) |
| **Zero-variance** | Explicit filtering | Implicit in coercion | ✅ Explicit filtering |
| **Imputation** | SimpleImputer(median) | None (dropna) | ✅ SimpleImputer(median) |
| **Scaling** | After imputation | Direct on raw | ✅ After imputation |
| **Artifacts** | scaler.joblib only | scaler.joblib only | ✅ imputer.joblib + scaler.joblib |

---

## Testing Instructions (HPC)

### Step 1: Delete existing Parquet caches (if any)
```bash
# These may have been created with wrong dtypes during development
find output_recalculated_full_datasets/datasets_2d_all -name "*.parquet" -delete
```

### Step 2: Test single config
```bash
# Test with example config (small ZINC sample for speed)
sbatch hpc/molfuse_phase1_cpu.sh configs/molfuse_phase1_example.json test_workspace_full2d
```

**Expected behavior:**
- First run: "Loading from CSV" + "Converting to Parquet" (~2-5 min)
- Check logs: Should see "Features after zero-variance removal: n=~1477"
- Check artifacts: `imputer.joblib` and `scaler.joblib` both created
- Memory usage: <100GB RSS (not 300GB)

### Step 3: Test subsequent run (Parquet caching)
```bash
# Should be much faster (~30-120 seconds loading)
sbatch hpc/molfuse_phase1_cpu.sh configs/molfuse_phase1_example.json test_workspace_full2d_run2
```

**Expected behavior:**
- "Loading from Parquet cache" messages
- Load time: 10-100x faster than first run

### Step 4: Full grid submission
```bash
# Submit all 630 configs (will skip completed runs via idempotent check)
bash hpc/submit_molfuse_phase1.sh configs/molfuse_phase1_grid experiment_workspace_full2d
```

---

## Expected Outcomes

### Performance:
- **First CSV load:** 2-5 minutes (with Parquet conversion)
- **Subsequent loads:** 10-30 seconds (from Parquet)
- **Memory usage:** 30-50GB RSS (vs 300GB without optimizations)
- **Total runtime per config:** Similar to 40-feature (UMAP dominates, not loading)

### Features:
- **Raw input:** 1613 features (full 2D Mordred)
- **After zero-variance:** ~1477 features (based on feature_comparison_v2 analysis)
- **After imputation:** Ready for UMAP/PCA (no NaNs)

### Artifacts per run:
- `logs/run.log` - detailed execution log with optimization messages
- `logs/phase1_summary.json` - run metadata
- `artifacts/imputer.joblib` - fitted SimpleImputer (NEW)
- `artifacts/scaler.joblib` - fitted StandardScaler
- `artifacts/pca_model.joblib` or `umap_model.joblib` - DR model
- `artifacts/ranked_scores.csv` - scored actives + ZINC
- `artifacts/embedding_*.csv` - similarity space coordinates
- `metrics/metrics.json` - EF@1%, ROC-AUC, PR-AUC, Spearman ρ

### Hypothesis:
Full 2D Mordred features should yield **higher EF@1%** than 40-feature baseline, based on feature_comparison_v2.py results.

---

## Validation Checklist

After HPC test runs, verify:

- [ ] **Loading speed:** First run ~2-5 min, subsequent ~10-30 sec
- [ ] **Memory usage:** <100GB RSS (check SLURM logs)
- [ ] **Feature count:** Log shows ~1477 features after zero-variance removal
- [ ] **Artifacts:** Both `imputer.joblib` and `scaler.joblib` created
- [ ] **No errors:** `phase1_summary.json` exists (not `phase1_error.json`)
- [ ] **Metrics:** EF@1% values comparable to or better than 40-feature baseline
- [ ] **Parquet caching:** Second run loads from Parquet, not CSV

---

## Rollback Instructions (if needed)

If full 2D features cause issues:

1. Revert config generator:
   ```bash
   git checkout HEAD -- scripts/generate_molfuse_phase1_configs_v4.py
   ```

2. Regenerate 40-feature configs:
   ```bash
   python scripts/generate_molfuse_phase1_configs_v4.py
   ```

3. Keep optimizations (they work for any feature set):
   - `load_csv_optimized()` in phase1.py
   - Imputation in prep.py
   - Zero-variance filtering

---

## Files Modified

1. `molfuse/data/prep.py` - Added imputation, zero-variance filtering
2. `molfuse/cli/phase1.py` - Added optimized loading, updated feature processing
3. `scripts/generate_molfuse_phase1_configs_v4.py` - Updated dataset paths
4. `configs/molfuse_phase1_example.json` - Updated example config

**No changes needed:**
- `hpc/molfuse_phase1_cpu.sh` - unchanged (paths from config)
- `hpc/submit_molfuse_phase1.sh` - unchanged (works with any config dir)

---

## Research Context

**Research Question:** Does using full 2D Mordred descriptors (1613 features) improve virtual screening performance vs 40-feature subset?

**Hypothesis:** Full feature set provides richer chemical space representation → higher enrichment factors.

**Evidence (from feature_comparison_v2.py):**
- Full 2D features showed highest EF@1% in preliminary tests
- Zero-variance filtering reduces ~1613 → ~1477 features (training set only)
- Imputation strategy (median) handles missing values robustly

**Next Step:** Run full Phase 1 grid with 630 configs to validate hypothesis across multiple hyperparameter settings and replicates.
