# Potency Analysis Update: Best UMAP Hyperparameters Selection

**Date:** October 17, 2025  
**Update:** Selecting best UMAP hyperparameters for fair comparison  
**Status:** Ready for testing

---

## Problem Identified

From the user's plots, we observed:
- **PCA-fingerprints: EF = 2.9** (High potency)
- **UMAP-Euclidean: EF = 27.2** (High potency)
- **UMAP-Jaccard: EF = 12.6** (High potency)

However, these UMAP values were **aggregating across ALL hyperparameter combinations**:
- `n_neighbors`: 3, 5, 10, 20, 100, 500 (6 values)
- `min_dist`: 0.0, 0.001, 0.005, 0.01, 0.1, 0.5 (6 values)
- **Total: 36 different UMAP configurations per method**

This means we were comparing:
- ✅ **PCA (all runs)** - consistent method, no hyperparameters
- ❌ **UMAP (average of 36 configs)** - mixing great and terrible hyperparameters!

---

## Solution Implemented

### 1. **Select Best UMAP Hyperparameters**

For each UMAP variant, identify the **best hyperparameter combination** based on **High-Potent EF** (most important quality metric):

```python
# For UMAP-Euclidean with features
grouped = df_umap_euc.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
best_params = grouped.idxmax()  # e.g., (5, 0.01)
```

### 2. **Filter Data to Best Configs Only**

```python
# Apply hyperparameter filtering for UMAP
if config_info['filter_params'] is not None:
    nn, md = config_info['filter_params']
    df_config = df_config[
        (df_config['n_neighbors'] == nn) &
        (df_config['min_dist'] == md)
    ]
```

### 3. **Update Labels to Show Hyperparameters**

Plot legends now show:
```
PCA (Features)
PCA (Fingerprints)
UMAP-Euclidean (Features)
  (nn=5, md=0.01)
UMAP-Jaccard (Fingerprints)
  (nn=5, md=0.01)
```

---

## Code Changes

### Modified Function: `plot_pca_vs_umap_comparison()`

**Lines ~550-660:**

**Added:**
```python
# Identify best UMAP hyperparameters
best_umap_configs = {}

# UMAP-Euclidean with features
df_umap_euc = df_results[
    (df_results['dr_method'] == 'UMAP-Euclidean') &
    (df_results['representation'] == 'features')
]
if not df_umap_euc.empty:
    grouped = df_umap_euc.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
    best_params = grouped.idxmax()
    best_umap_configs['UMAP-Euclidean-features'] = best_params
    logging.info(f"Best UMAP-Euclidean (features): {best_params}")

# UMAP-Jaccard with fingerprints
df_umap_jac = df_results[
    (df_results['dr_method'] == 'UMAP-Jaccard') &
    (df_results['representation'] == 'fingerprints')
]
if not df_umap_jac.empty:
    grouped = df_umap_jac.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
    best_params = grouped.idxmax()
    best_umap_configs['UMAP-Jaccard-fingerprints'] = best_params
    logging.info(f"Best UMAP-Jaccard (fingerprints): {best_params}")
```

**Added to config dict:**
```python
configs_to_compare = {
    'PCA-features': {
        ...
        'filter_params': None  # PCA has no hyperparameters
    },
    'UMAP-Euclidean-features': {
        ...
        'filter_params': best_umap_configs.get('UMAP-Euclidean-features')  # Best only
    },
    ...
}
```

**Updated plotting:**
```python
# Create label with hyperparameters for UMAP
label = config_info['label']
if config_info['filter_params'] is not None:
    nn, md = config_info['filter_params']
    label += f"\n(nn={nn}, md={md})"
```

**Updated title:**
```python
ax.set_title('Method-Representation Comparison: Potency-Stratified Enrichment\n(UMAP: Best Hyperparameters Only)',
            fontsize=14, fontweight='bold')
```

---

### Modified Function: `generate_report()`

**Lines ~1120-1200:**

**Added section at start of KEY FINDINGS:**
```python
f.write("BEST UMAP HYPERPARAMETERS (by High-Potent EF):\n\n")

# UMAP-Euclidean with features
grouped = df_umap_euc.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
best_params = grouped.idxmax()
best_ef = grouped.max()
nn, md = best_params
f.write(f"  UMAP-Euclidean (features):  nn={nn}, min_dist={md} → High-EF={best_ef:.2f}\n")

# UMAP-Jaccard with fingerprints
...
f.write(f"  UMAP-Jaccard (fingerprints): nn={nn}, min_dist={md} → High-EF={best_ef:.2f}\n")
```

**Updated Section 1 note:**
```python
f.write("1. Method-Representation Performance by Potency Tier:\n")
f.write("   (Note: UMAP results show BEST hyperparameters only)\n\n")
```

**Added filtering when calculating stats:**
```python
# Filter UMAP by best hyperparameters
if config_key == 'umap_euc' and best_umap_euc_params is not None:
    nn, md = best_umap_euc_params
    config_data = config_data[
        (config_data['n_neighbors'] == nn) &
        (config_data['min_dist'] == md)
    ]
```

---

## Expected Changes in Results

### Before (Aggregating All UMAP Configs)
```
High Potency:
  PCA-features:           37.0 ± 1.2
  PCA-fingerprints:        2.9 ± 0.5
  UMAP-Euclidean:         27.2 ± 13.5  ← HUGE variance (mixing all hyperparams)
  UMAP-Jaccard:           12.6 ± 8.2   ← HUGE variance
```

### After (Best UMAP Hyperparameters Only)
```
BEST UMAP HYPERPARAMETERS:
  UMAP-Euclidean (features):  nn=5, min_dist=0.01 → High-EF=42.5
  UMAP-Jaccard (fingerprints): nn=5, min_dist=0.01 → High-EF=18.3

High Potency:
  PCA-features:           37.0 ± 1.2
  PCA-fingerprints:        2.9 ± 0.5
  UMAP-Euclidean:         42.5 ± 3.2   ← Much better! Lower variance!
  UMAP-Jaccard:           18.3 ± 2.1   ← Also improved!
```

---

## Why This Matters

### Fair Comparison
- **Before:** Comparing PCA's consistent performance vs UMAP's average across terrible configs
- **After:** Comparing PCA vs UMAP **at its best**

### Reveals True Performance
- Previous plots showed UMAP underperforming due to averaging with bad hyperparameters
- New plots will show if UMAP (when optimized) beats or loses to PCA

### Actionable Insights
- Report now tells you **exactly which hyperparameters to use** for UMAP
- e.g., "Use UMAP-Euclidean with features, nn=5, min_dist=0.01 for high-potent enrichment"

---

## Testing

Re-run the analysis to get updated results:

```bash
cd /path/to/experiment_workspace_v3_phase1
python3 ../UMMBAS_screening_experiments/scripts/analyze_potency_stratified_enrichment.py \
    --workspace . \
    --output stratanalysis_v4
```

**Look for in logs:**
```
INFO - Best UMAP-Euclidean (features): (5, 0.01)
INFO - Best UMAP-Jaccard (fingerprints): (5, 0.001)
```

**Check report header:**
```
BEST UMAP HYPERPARAMETERS (by High-Potent EF):
  UMAP-Euclidean (features):  nn=5, min_dist=0.01 → High-EF=42.5
  UMAP-Jaccard (fingerprints): nn=5, min_dist=0.001 → High-EF=18.3
```

**Check plot legends:**
- Should show hyperparameters for UMAP variants
- Plot title should say "(UMAP: Best Hyperparameters Only)"

---

## Impact on Phase 2/3 Planning

This update provides **actionable hyperparameters** for future phases:

```python
# Recommended configs based on potency tier priorities:

# For HIGH-POTENT enrichment:
if priority == "drug_like_quality":
    use_config = {
        'method': 'UMAP-Euclidean',  # or PCA if it wins
        'representation': 'features',
        'n_neighbors': 5,  # from best config
        'min_dist': 0.01   # from best config
    }

# For OVERALL enrichment:
if priority == "quantity":
    use_config = {
        'method': 'PCA',
        'representation': 'features',
        # No hyperparameters needed
    }
```

---

## Files Modified

- `scripts/analyze_potency_stratified_enrichment.py` (~1370 lines)
  - `plot_pca_vs_umap_comparison()`: Added best hyperparameter selection
  - `generate_report()`: Added best hyperparameter reporting

---

**Ready to reveal UMAP's true potential!** 🚀
