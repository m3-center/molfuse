# Potency Analysis V3: Method-Representation Separation Update

**Date:** 2025-01-17  
**Status:** Ready for testing  
**Location:** `scripts/analyze_potency_stratified_enrichment.py`

## Overview

Updated the potency-stratified enrichment analysis to properly separate and compare specific **method-representation pairs** rather than treating all PCA or UMAP runs as identical.

## Problem Addressed

The previous analysis aggregated:
- **All PCA runs** (both features and fingerprints representations)
- **All UMAP runs** (both Euclidean/features and Jaccard/fingerprints)

This masked important performance differences between:
- `PCA-features` vs `PCA-fingerprints`
- `UMAP-Euclidean-features` vs `UMAP-Jaccard-fingerprints`

## Solution

### 1. Four Specific Configurations Compared

```python
configs_to_compare = {
    'PCA-features': {
        'method': 'PCA', 
        'repr': 'features', 
        'label': 'PCA (Features)',
        'short_label': 'PCA-feat'
    },
    'PCA-fingerprints': {
        'method': 'PCA', 
        'repr': 'fingerprints', 
        'label': 'PCA (Fingerprints)',
        'short_label': 'PCA-fing'
    },
    'UMAP-Euclidean-features': {
        'method': 'UMAP-Euclidean', 
        'repr': 'features', 
        'label': 'UMAP-Euclidean (Features)',
        'short_label': 'UMAP-Euc'
    },
    'UMAP-Jaccard-fingerprints': {
        'method': 'UMAP-Jaccard', 
        'repr': 'fingerprints', 
        'label': 'UMAP-Jaccard (Fingerprints)',
        'short_label': 'UMAP-Jac'
    }
}
```

### 2. Updated Plot Functions

#### `plot_pca_vs_umap_comparison()`
**Changed:**
- Now plots **4 bars** per tier (instead of 2)
- Each bar represents a specific method-representation pair
- Color-coded: Blue (PCA-feat), Green (PCA-fing), Red (UMAP-Euc), Orange (UMAP-Jac)

**Outputs:**
- `pca_vs_umap_by_tier.png`: Bar chart with 4 configs × 3 tiers
- `pca_vs_umap_distributions.png`: 3 boxplots showing distribution for each tier

#### `generate_report()`
**Changed:**
- Section "1. Method-Representation Performance by Potency Tier"
- Shows all 4 configs per tier with mean ± std
- Identifies the winner for each potency tier

**Example output:**
```
  High Potency (0.1-100.0 nM):
    PCA-features                 :  56.80 ± 12.30
    PCA-fingerprints             :  48.20 ±  9.50
    UMAP-Euclidean-features      :  33.50 ±  7.20
    UMAP-Jaccard-fingerprints    :  41.20 ±  8.90
    → Winner: PCA-features (EF = 56.80)

  Medium Potency (100.0-1000.0 nM):
    PCA-features                 :  42.10 ± 10.20
    PCA-fingerprints             :  39.50 ±  8.70
    UMAP-Euclidean-features      :  28.30 ±  6.10
    UMAP-Jaccard-fingerprints    :  35.80 ±  7.50
    → Winner: PCA-features (EF = 42.10)

  Weak Potency (1000.0-100000.0 nM):
    PCA-features                 :  18.50 ±  5.30
    PCA-fingerprints             :  22.10 ±  6.80
    UMAP-Euclidean-features      :  15.20 ±  4.20
    UMAP-Jaccard-fingerprints    :  19.80 ±  5.90
    → Winner: PCA-fingerprints (EF = 22.10)
```

## Key Changes to Code

### File: `scripts/analyze_potency_stratified_enrichment.py`

**Lines ~560-590:** Added `configs_to_compare` dictionary with 4-way configuration definition

**Lines ~590-630:** Updated bar chart plotting
- Changed from 2 bars (PCA/UMAP) to 4 bars (specific configs)
- Added proper filtering by both `dr_method` AND `representation`
- Updated colors and labels

**Lines ~630-680:** Updated boxplot plotting
- Changed from 2 boxes per tier to 4 boxes per tier
- Added `short_label` for compact x-axis labels
- Improved label rotation for readability

**Lines ~1040-1080:** Updated report generation
- Added method-representation filtering in findings section
- Shows all 4 configs per potency tier
- Identifies winner based on specific method-repr pairs

## Testing Instructions

### 1. Update Script on HPC
```bash
cd /path/to/UMMBAS_screening_experiments
git pull origin main
```

### 2. Re-run Analysis
```bash
cd /path/to/experiment_workspace_v3_phase1
sbatch ../UMMBAS_screening_experiments/hpc/submit_potency_analysis.sh
```

Or run interactively:
```bash
python3 ../UMMBAS_screening_experiments/scripts/analyze_potency_stratified_enrichment.py \
    --workspace . \
    --output potency_analysis_v3
```

### 3. Check Outputs

**Updated plots:**
- `potency_analysis_v3/pca_vs_umap_by_tier.png`: Should show 4 bars per tier
- `potency_analysis_v3/pca_vs_umap_distributions.png`: Should show 4 boxes per tier

**Updated report:**
- `potency_analysis_v3/potency_stratified_report.txt`: Section 1 should list all 4 configs

## Expected Insights

This separation will reveal:

1. **Representation Impact on PCA:**
   - Does PCA prefer features or fingerprints?
   - Does the preference change by potency tier?

2. **Method-Distance Alignment:**
   - Is UMAP-Euclidean (features) better than UMAP-Jaccard (fingerprints)?
   - Or does the distance metric matter less than the representation?

3. **Optimal Configuration per Tier:**
   - High-potent: Best method-representation pair?
   - Medium-potent: Different optimal configuration?
   - Weak-potent: Does fingerprint similarity help?

4. **Dimensionality Preferences:**
   - Do different method-repr pairs prefer different dimensions?
   - (Next update will enhance `plot_dimensionality_impact()` similarly)

## Next Steps

1. **Test updated script** on HPC with full Phase 1 data
2. **Review new plots** to confirm proper separation
3. **Analyze findings** in updated report
4. **Consider enhancing** `plot_dimensionality_impact()` to also separate by representation
5. **Document best configurations** for Phase 2/3 planning

## Notes

- All analysis logic unchanged (same EF calculations)
- Only visualization and reporting updated
- Backward compatible with existing data structure
- No changes to data loading or preprocessing

## Files Modified

- `scripts/analyze_potency_stratified_enrichment.py` (~1208 lines)
  - `plot_pca_vs_umap_comparison()` function
  - `generate_report()` function (section 1)

## Files NOT Modified (Future Work)

- `plot_dimensionality_impact()`: Still aggregates by method only (could be enhanced)
- `plot_stratified_comparison()`: Still focuses on UMAP hyperparameters (separate analysis)

---

**Ready for deployment and testing!** 🚀
