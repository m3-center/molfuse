# Cutoff Analysis Script Updates for Multiple Hyperparameters

## Date: October 8, 2025

## Summary
Updated both `run_cutoff_analysis.py` and `aggregate_cutoff_analysis.py` to properly handle the new hyperparameter sweep structure, especially UMAP configurations with multiple hyperparameters (n_neighbors and min_dist).

---

## Changes to `run_cutoff_analysis.py`

### Issue
The glob pattern was looking for directories named `run_seed*_reprfeatures_*` but the new structure uses `run_seed*_config_features_*`.

### Fix
**Line 59:** Updated glob pattern
```python
# OLD:
original_run_dirs = glob.glob(os.path.join(args.original_workspace, f"run_seed*_repr{REPR_TYPE_TO_PROCESS}_*"))

# NEW:
original_run_dirs = glob.glob(os.path.join(args.original_workspace, f"run_seed*_config_{REPR_TYPE_TO_PROCESS}_*"))
```

This now correctly matches directory names like:
- `run_seed42_config_features_tsne_perplexity100`
- `run_seed42_config_features_umap_euclidean_projection_nn100_md0.01`
- etc.

---

## Changes to `aggregate_cutoff_analysis.py`

### Issue
The script was only extracting a single hyperparameter (either n_neighbors OR perplexity), but UMAP has TWO hyperparameters (n_neighbors AND min_dist) that need to be tracked together.

### Fix 1: Enhanced Hyperparameter Extraction (Lines ~150-175)

**OLD CODE:**
```python
hyperparam_name, hyperparam_value = "N/A", "N/A"
if 'n_neighbors' in dr_params:
    hyperparam_name = 'n_neighbors'; hyperparam_value = dr_params['n_neighbors']
elif 'perplexity' in dr_params:
    hyperparam_name = 'perplexity'; hyperparam_value = dr_params['perplexity']
```

**NEW CODE:**
```python
# Extract hyperparameters - handle UMAP's multiple hyperparameters
hyperparam_names = []
hyperparam_values = []
hyperparam_dict = {}  # Store as dict for later filtering

if 'n_neighbors' in dr_params:
    hyperparam_names.append('n_neighbors')
    hyperparam_values.append(dr_params['n_neighbors'])
    hyperparam_dict['n_neighbors'] = dr_params['n_neighbors']
if 'min_dist' in dr_params:
    hyperparam_names.append('min_dist')
    hyperparam_values.append(dr_params['min_dist'])
    hyperparam_dict['min_dist'] = dr_params['min_dist']
if 'perplexity' in dr_params:
    hyperparam_names.append('perplexity')
    hyperparam_values.append(dr_params['perplexity'])
    hyperparam_dict['perplexity'] = dr_params['perplexity']

# Create combined string representations
if hyperparam_names:
    hyperparam_name = ', '.join(hyperparam_names)
    hyperparam_value = ', '.join(str(v) for v in hyperparam_values)
else:
    hyperparam_name = "N/A"
    hyperparam_value = "N/A"
```

**Key improvements:**
- Now extracts ALL relevant hyperparameters (not just the first one found)
- Stores them as a dictionary for exact matching later
- Creates human-readable combined strings for display (e.g., "n_neighbors, min_dist" with values "100, 0.1")

### Fix 2: Store Full Hyperparameter Dictionary (Line ~185)

**OLD CODE:**
```python
metric_row = {'Seed': seed, 'Affinity_Cutoff': cutoff_val, 'Representation': repr_type.capitalize(),
              'DR_Method': dr_params['short_name'], 'Embedding_Strategy': embedding_strategy,
              'Hyperparameter': hyperparam_name, 'Hyperparameter_Value': hyperparam_value}
```

**NEW CODE:**
```python
metric_row = {'Seed': seed, 'Affinity_Cutoff': cutoff_val, 'Representation': repr_type.capitalize(),
              'DR_Method': dr_params['short_name'], 'Embedding_Strategy': embedding_strategy,
              'Hyperparameter': hyperparam_name, 'Hyperparameter_Value': hyperparam_value,
              'Hyperparam_Dict': json.dumps(hyperparam_dict)}  # Store full dict for exact matching
```

**Why:** The JSON-encoded dictionary allows exact matching of hyperparameter combinations, crucial for UMAP where we need to identify configurations like `{n_neighbors: 100, min_dist: 0.1}`.

### Fix 3: Optimal Hyperparameter Selection with Exact Matching (Lines ~275-295)

**OLD CODE:**
```python
optimal_value = "N/A"
if hyperparam_name != 'N/A':
    mean_perf_table = df_exp.groupby('Hyperparameter_Value')['EF_1Perc'].mean().reset_index()
    if not mean_perf_table.empty and not mean_perf_table['EF_1Perc'].isnull().all():
        optimal_value = mean_perf_table.loc[mean_perf_table['EF_1Perc'].idxmax()]['Hyperparameter_Value']

df_best_hyperparams_list.append({'Method_Repr': method, 'Embedding_Strategy': strategy, 'Hyperparameter': hyperparam_name, 'Optimal_Value': optimal_value})

df_optimal_replicates = df_agg[
    (df_agg['Method_Repr'] == method) &
    (df_agg['Embedding_Strategy'] == strategy) &
    (df_agg['Hyperparameter_Value'] == optimal_value)
].copy()
```

**NEW CODE:**
```python
optimal_value = "N/A"
optimal_dict_str = "{}"

if hyperparam_name != 'N/A':
    # Group by the full hyperparam dictionary (stored as JSON string)
    mean_perf_table = df_exp.groupby('Hyperparam_Dict')['EF_1Perc'].mean().reset_index()
    if not mean_perf_table.empty and not mean_perf_table['EF_1Perc'].isnull().all():
        optimal_dict_str = mean_perf_table.loc[mean_perf_table['EF_1Perc'].idxmax()]['Hyperparam_Dict']
        # Also get the human-readable string value for display
        matching_row = df_exp[df_exp['Hyperparam_Dict'] == optimal_dict_str].iloc[0]
        optimal_value = matching_row['Hyperparameter_Value']

df_best_hyperparams_list.append({'Method_Repr': method, 'Embedding_Strategy': strategy, 'Hyperparameter': hyperparam_name, 'Optimal_Value': optimal_value})

# Filter using the exact hyperparameter dictionary match
df_optimal_replicates = df_agg[
    (df_agg['Method_Repr'] == method) &
    (df_agg['Embedding_Strategy'] == strategy) &
    (df_agg['Hyperparam_Dict'] == optimal_dict_str)
].copy()
```

**Why:** 
- Groups by the complete hyperparameter dictionary (not just a string representation)
- Ensures exact matching when filtering for optimal configurations
- Prevents issues where string matching might fail with multiple hyperparameters

---

## Example Data Flow

### For t-SNE (single hyperparameter):
```
Extracted: {perplexity: 100}
Display: "perplexity" = "100"
Stored: '{"perplexity": 100}'
```

### For UMAP (multiple hyperparameters):
```
Extracted: {n_neighbors: 100, min_dist: 0.1}
Display: "n_neighbors, min_dist" = "100, 0.1"
Stored: '{"n_neighbors": 100, "min_dist": 0.1}'
```

### For PCA (no hyperparameters):
```
Extracted: {}
Display: "N/A" = "N/A"
Stored: '{}'
```

---

## Testing Recommendations

1. **Test with t-SNE runs:** Single hyperparameter (perplexity) should work correctly
2. **Test with UMAP runs:** Multiple hyperparameters (n_neighbors, min_dist) should be properly combined
3. **Test with PCA runs:** No hyperparameters should be handled gracefully
4. **Verify optimal selection:** The script should correctly identify the best hyperparameter combination for each method
5. **Check filtering:** The filtered dataset for trend plots should only include rows matching the optimal hyperparameter configuration

---

## Directory Structure Compatibility

The scripts now work with the new directory naming convention:
```
experiment_workspace_rerun_hyperparam_sweep/
├── run_seed42_config_features_pca_coembedding/
├── run_seed42_config_features_tsne_perplexity100/
├── run_seed42_config_features_umap_euclidean_projection_nn100_md0.01/
├── run_seed42_config_features_umap_euclidean_coembedding_nn100_md0.1/
└── ...
```

Previously expected (OLD):
```
experiment_workspace_hyperparam_sweep/
├── run_seed42_reprfeatures_pca_coembedding/
└── ...
```
