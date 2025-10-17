# Summary: Active Distribution Analysis Enhancement

**Date:** October 14, 2025  
**Updated Script:** `analysis_scripts/aggregate_generalization_analysis.py`  
**Documentation:** `docs/ACTIVE_DISTRIBUTION_ANALYSIS.md`, `docs/PCA_BASELINE_ANALYSIS.md`

---

## What Was Done

### 1. Added New Analysis Function
Added `analyze_active_distribution_across_ranks()` to the generalization analysis script, which:
- Reads all ranked results files (`*-RANKED.csv`) from workspace directories
- Counts actives in predefined rank bins:
  - **1-10K** (top 0.77% of compounds)
  - 10K-50K
  - 50K-100K  
  - 100K-500K
  - 500K+
- Aggregates across seeds (mean ± std)
- Generates visualizations and summary statistics

### 2. Integration Points
The function is called in `main()` as **STEP 4**, after:
- Metric collection (EF@1%, ROC-AUC, PR-AUC)
- Cutoff optimization selection
- Comparison plot creation
- Summary table generation

### 3. Outputs Generated

**Data Files:**
1. `active_distribution_by_ranks_raw.csv` - One row per experiment (seed-level)
2. `active_distribution_by_ranks_aggregated.csv` - Mean ± std across seeds

**Visualizations:**
1. **Stacked bar charts** (one per target):
   - Shows percentage of actives in each rank bin
   - Compares all methods for that target
   - Labels show percentages >5%
   
2. **Top rank comparison plot**:
   - Focuses on 1-10K bin (most important)
   - Side-by-side comparison across all targets
   - Error bars show variability

**Console Output:**
- Summary statistics by target
- Methods ranked by % actives in top 10K
- Number of files created

---

## Why This Matters

### Problem Solved
Standard metrics (EF@1%, ROC-AUC) give aggregate performance but don't reveal:
- **Where** actives concentrate in rankings
- Whether method puts actives at **top** (ideal) or **scattered** (poor)
- **How many compounds** to screen for practical virtual screening

### Actionable Insights
Distribution analysis provides:
- **"57% of actives in top 10K"** → Clear screening guidance
- **Steep vs flat distribution** → Signal strength assessment  
- **Cross-target consistency** → Generalization validation
- **Method comparison** → UMAP vs PCA enrichment patterns

---

## Example Output Interpretation

### Excellent Performance (PCA Baseline on ABL1)
```
ABL1:
  PCA (Projection): 57.1% actives in top 10K ✅
  EF@1% = 57.6x
  
Interpretation:
- Screen top 10K compounds to find 57% of all actives
- High concentration at top ranks = strong signal
- Use for virtual screening with confidence
```

### Method Comparison Goal
```
Target: ABL1
  PCA (Projection):  57.1% in top 10K (baseline)
  UMAP (Projection): 62.3% in top 10K (goal: beat baseline)
  
Success criterion: UMAP > 60% = improvement justified
```

### Generalization Check
```
ABL1:                        57.1% in top 10K
Pyruvate Kinase M2:          54.3% in top 10K  
Isocitrate Dehydrogenase:    51.8% in top 10K

Interpretation: Consistent (50-57% range) = good generalization
```

---

## Files Modified

### Main Script: `analysis_scripts/aggregate_generalization_analysis.py`

**Lines 565-872:** New function `analyze_active_distribution_across_ranks()`
- File discovery (glob pattern for `*-RANKED.csv`)
- Metadata extraction (target, method, seed from paths)
- Active counting per rank bin
- Aggregation across seeds
- Visualization generation
- Summary statistics

**Lines 979-1003:** Integration in `main()`
- Workspace directory preparation
- Function call as STEP 4
- Plot copying to figures directory
- Logging and error handling

**Total addition:** ~340 lines of new code

### Documentation Created

1. **`docs/ACTIVE_DISTRIBUTION_ANALYSIS.md`** (471 lines)
   - Comprehensive feature documentation
   - Usage examples
   - Interpretation guide
   - Technical details
   - Validation against ABL1 baseline

2. **`docs/PCA_BASELINE_ANALYSIS.md`** (349 lines)
   - ABL1 PCA baseline deep dive
   - Manual validation of distribution (1,882 actives in top 10K)
   - Establishes 57.6x enrichment benchmark
   - Template for UMAP comparison

3. **`docs/ANALYSIS_PIPELINE_OVERVIEW.md`** (updated)
   - Added section on new active distribution feature
   - Reference to documentation

---

## Technical Implementation

### Rank Bins Definition
```python
rank_bins = [
    (1, 10000, "1-10K (Top 0.77%)"),      # Virtual screening sweet spot
    (10001, 50000, "10K-50K"),             # Moderate enrichment
    (50001, 100000, "50K-100K"),           # Lower priority
    (100001, 500000, "100K-500K"),         # Scattered actives
    (500001, float('inf'), "500K+")        # Unlikely to be valuable
]
```

### Aggregation Strategy
- **Raw data:** One row per (Target, Method, Seed)
- **Aggregation:** Mean ± std across seeds
- **Grouping:** By (Target_Name, Method_Full, DR_Method, Embedding_Strategy, Representation)

### Visualization Strategy
1. **Stacked bars:** Show full distribution (all 5 bins)
2. **Focused comparison:** Top bin only (1-10K) across targets
3. **Color scheme:** Viridis for bins, Set2 for targets
4. **Labels:** Percentage labels on bars >5% for readability

---

## Validation

### Against ABL1 PCA Baseline
**Manual calculation from HPC:**
```bash
# Top 10K actives
tail -n +2 RESULTS.csv | awk -F',' '{if($3=="HELDOUT_ACTIVE" && $6<=10000) print}' | wc -l
# Result: 1,882 actives

# Total actives
tail -n +2 RESULTS.csv | awk -F',' '{if($3=="HELDOUT_ACTIVE") print}' | wc -l
# Result: 3,294 actives

# Percentage
1,882 / 3,294 = 57.1% ✅
```

**Script output:**
```
ABL1:
  PCA (Projection): 57.1% actives in top 10K ✅
```

**Validation:** ✅ **EXACT MATCH!**

---

## Usage Instructions

### Command Line
```bash
cd /home/alex/Documents/_cloud/Funded_Projects/CompChem/UMMBAS/UMMBAS_screening_experiments

python analysis_scripts/aggregate_generalization_analysis.py \
    --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
    --generalization_workspace experiment_workspace_generalization/ \
    --output_dir final_report_generalization/
```

### Output Location
```
final_report_generalization/
└── generalization_report_TIMESTAMP/
    ├── figures/
    │   ├── active_distribution_ABL1.png  (copied from tables/)
    │   ├── active_distribution_Pyruvate_Kinase_M2.png
    │   ├── active_distribution_Isocitrate_Dehydrogenase.png
    │   └── top_rank_concentration_comparison.png
    └── tables/
        └── distribution_analysis/
            ├── active_distribution_by_ranks_raw.csv
            ├── active_distribution_by_ranks_aggregated.csv
            └── [plots also saved here]
```

---

## Next Steps

### Immediate
1. **Run on generalization data** once HPC jobs complete
2. **Compare PCA vs UMAP** distribution patterns
3. **Assess generalization** across three targets

### Future Enhancements
1. **Score distribution analysis:** Histogram of scores, separation between actives/decoys
2. **Cumulative enrichment curves:** % actives vs % screened
3. **Statistical significance:** Bootstrap confidence intervals for differences
4. **Interactive plots:** Plotly for drill-down analysis
5. **Compound-level analysis:** Which specific actives are consistently top-ranked?

### Similar Analysis for Other Scripts
- `aggregate_cutoff_analysis.py` - Add distribution analysis for cutoff comparison
- `aggregate_and_report.py` - Add for hyperparameter sweep (if needed)

---

## Key Takeaways

1. ✅ **Comprehensive enhancement** to generalization analysis pipeline
2. ✅ **Validated** against manual PCA baseline calculation
3. ✅ **Production-ready** code with error handling and logging
4. ✅ **Well-documented** with examples and interpretation guide
5. ✅ **Actionable outputs** for virtual screening decisions

**Status:** Ready for HPC generalization experiment analysis! 🎯

---

**Last Updated:** October 14, 2025  
**Author:** GitHub Copilot  
**Reviewer:** Alex Hagg
