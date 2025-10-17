# Active Distribution Analysis - Quick Reference

## 🎯 What It Does

Analyzes **where actives concentrate** in virtual screening rankings:
- Top 10K (0.77% of compounds) - **Virtual screening sweet spot**
- 10K-50K, 50K-100K, 100K-500K, 500K+ - Lower priority bins

## 📊 Output Examples

### Stacked Bar Chart (per target)
```
     PCA (Projection)         UMAP (Projection)
     ┌─────────────┐         ┌─────────────┐
100% │             │         │             │
     │   57.1%     │         │   62.3%     │  ← Top 10K (green)
 80% ├─────────────┤         ├─────────────┤
     │             │         │             │
 60% │    4.7%     │         │    5.2%     │  ← 10K-50K (blue)
     ├─────────────┤         ├─────────────┤
 40% │    5.8%     │         │    4.1%     │  ← 50K-100K (yellow)
     ├─────────────┤         ├─────────────┤
 20% │   26.1%     │         │   22.9%     │  ← 100K-500K (orange)
     ├─────────────┤         ├─────────────┤
  0% │    6.3%     │         │    5.5%     │  ← 500K+ (red)
     └─────────────┘         └─────────────┘
```

### Top Rank Comparison (all targets)
```
% Actives in Top 10K
     ┌───┬───┬───┐
 60% │▓▓▓│   │   │ ABL1
     │▓▓▓│░░░│   │ Pyruvate Kinase M2  
 50% │▓▓▓│░░░│▒▒▒│ Isocitrate Dehydrogenase
     │▓▓▓│░░░│▒▒▒│
 40% └───┴───┴───┘
      PCA  UMAP Other
```

## 📁 Files Generated

```
tables/distribution_analysis/
├── active_distribution_by_ranks_raw.csv
│   → One row per experiment (seed-level data)
│   → Columns: Actives_1-10K, Pct_1-10K, etc.
│
├── active_distribution_by_ranks_aggregated.csv  
│   → Mean ± std across seeds
│   → Grouped by: Target, Method, Representation
│
└── [plots]
    ├── active_distribution_ABL1.png
    ├── active_distribution_Pyruvate_Kinase_M2.png
    ├── active_distribution_Isocitrate_Dehydrogenase.png
    └── top_rank_concentration_comparison.png
```

## 🔍 Interpretation Guide

| % in Top 10K | Interpretation | Action |
|--------------|----------------|---------|
| **>60%** | 🔥 Outstanding | Use for primary screening |
| **50-60%** | ✅ Excellent | PCA baseline level |
| **40-50%** | ⚠️ Good | Acceptable performance |
| **<40%** | ❌ Poor | Method needs tuning |

## 💻 Command to Run

```bash
python analysis_scripts/aggregate_generalization_analysis.py \
    --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
    --generalization_workspace experiment_workspace_generalization/ \
    --output_dir final_report_generalization/
```

## 📖 Full Documentation

- **Feature docs:** `docs/ACTIVE_DISTRIBUTION_ANALYSIS.md` (471 lines)
- **PCA baseline:** `docs/PCA_BASELINE_ANALYSIS.md` (349 lines)  
- **Enhancement summary:** `docs/ACTIVE_DISTRIBUTION_ENHANCEMENT_SUMMARY.md` (208 lines)

## ✅ Validation

**Manual count (ABL1 PCA baseline):**
```bash
# HPC command
awk -F',' '{if($3=="HELDOUT_ACTIVE" && $6<=10000) print}' RESULTS.csv | wc -l
# Result: 1,882 actives in top 10K

# Percentage
1,882 / 3,294 = 57.1% ✅
```

**Script output:**
```
ABL1:
  PCA (Projection): 57.1% actives in top 10K ✅
```

**Match:** EXACT! ✅

---

**Ready to use when generalization experiments complete!** 🚀
