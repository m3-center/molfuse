# Potency-Stratified Analysis - Updates (Oct 17, 2025)

## Issues Found and Fixed

### Issue 1: Ranked files not found
**Problem:** Script was looking for files ending in `-RANKED.csv` but actual files are named:
- `TYROSINEPROTEINKINASEABL1_P00519-PCA-2D-FEATURES.csv`
- `TYROSINEPROTEINKINASEABL1_P00519-UMAPEUCLIDEAN-2D-FEATURES.csv`
- etc.

**Fix:** Updated `find_ranked_file()` to try multiple patterns:
- `*-RANKED.csv` (original pattern)
- `*-FEATURES.csv` (for features representation)
- `*-FINGERPRINTS.csv` (for fingerprints representation)
- `TYR*-*-*D-*.csv` (uppercase target patterns)

### Issue 2: Affinity data not in expected locations
**Problem:** Affinity data (`Standard Value (nM)`) not present in:
- Ranked files
- Detailed distances files

**Fix:** Added new data loading strategies in priority order:
1. **temp_data raw files** (most likely to have affinity):
   - `*/temp_data/*_target_ligands_for_feature_calc_raw.csv`
2. **target_ligands_calculated files**:
   - `*/target_ligands_calculated/*/*_target_ligands_for_calc_*.csv`
3. Detailed distances files (original)
4. Prepared data files (original)

Also added support for multiple column name conventions:
- `Standard Value (nM)` (preferred)
- `standard_value` (converted to nM)
- `pchembl_value` (converted to nM using: nM = 10^(9-pChEMBL))

## Updated Files

1. **scripts/analyze_potency_stratified_enrichment.py**
   - Fixed `find_ranked_file()` function
   - Enhanced `load_affinity_data()` function with 5 strategies
   - Better column name handling

2. **scripts/debug_run_structure.py**
   - Checks multiple ranked file patterns
   - Checks raw temp data files
   - Reports which affinity columns are available

## Now Run This on HPC

```bash
cd /home/ahagg2s/UMMBAS_screening_experiments

# Test with updated scripts
bash scripts/test_potency_analysis.sh experiment_workspace_v3_phase1
```

## What Should Happen Now

### If temp_data files have affinity:
```
✅ Raw temp data files found: 1
  Has 'Standard Value (nM)': True
  
✅ Ranked files found: 1
  Has 'RANKING': True
  Has 'TYPE': True

✅ Successfully analyzed N runs
```

### If still no affinity data:
The debug output will show exactly which files exist and what columns they contain, helping you identify where the affinity data is stored in your specific file structure.

## Manual Check (if needed)

If the script still can't find affinity data, manually check one of these files on your HPC:

```bash
# Check the raw temp data file
head -n 2 experiment_workspace_v3_phase1/run_seed42_config_tyro_features_pca_dim2_seed42/TyrosineProteinKinaseABL1_P00519/temp_data/TyrosineProteinKinaseABL1_P00519_target_ligands_for_feature_calc_raw.csv

# Or the calculated file
head -n 2 experiment_workspace_v3_phase1/run_seed42_config_tyro_features_pca_dim2_seed42/TyrosineProteinKinaseABL1_P00519/target_ligands_calculated/features/TyrosineProteinKinaseABL1_P00519_target_ligands_for_calc_features.csv
```

Look for columns containing:
- Compound ChEMBL ID (molecule identifier)
- Standard Value (nM), standard_value, or pchembl_value (affinity measure)

If you find them, let me know the exact column names and I'll adjust the script!

## Performance Note

The script now tries multiple file locations, which is slightly slower but much more robust. For 492 runs:
- Should complete in 2-5 minutes
- Will report progress with tqdm bar
- First run analyzed verbosely for debugging

## Questions?

If it still doesn't work after this update:
1. Run the debug script and share the output
2. Share the output of one of the head commands above
3. I can adjust the patterns/column names accordingly
