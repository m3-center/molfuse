# Dimensionality Experiment - Quick Checklist

## Pre-Flight Checklist ✈️

- [x] Config generator created (`generate_dimensionality_configs.py`)
- [x] HPC scripts created (`hpc/submit_dimensionality_jobs.sh`, `hpc/ummbas_dimensionality_cpu.sh`)
- [x] Aggregation script created (`aggregate_dimensionality_analysis.py`)
- [x] Documentation written (README and QUICKSTART)
- [x] Configs generated (3 files in `dimensionality_configs/`)
- [x] Compatibility verified (uses existing `main_orchestrator.py`)

## Execution Checklist 🚀

### Before Submission
- [ ] Review config files in `dimensionality_configs/`
- [ ] Verify HPC account and quotas
- [ ] Check that datasets are accessible on HPC
- [ ] Ensure conda environment `ummbas-screening` is available

### Submit Jobs
```bash
cd /path/to/UMMBAS_screening_experiments
bash hpc/submit_dimensionality_jobs.sh
```

- [ ] Jobs submitted successfully (should see 15 job IDs)
- [ ] Check initial logs for errors

### During Execution
- [ ] Monitor job queue: `squeue -u $USER | grep DIM`
- [ ] Check logs periodically: `tail -f slurm_logs/DIM_*.out`
- [ ] Verify output files are being created

### Progress Tracking
```bash
# Check how many analyses have completed (should reach 75)
find experiment_workspace_dimensionality/ -name "*_ranking_metrics.csv" | wc -l

# Check completion by dimension
for d in 2 3 5 10 20; do
    echo "Dimension $d: $(find experiment_workspace_dimensionality/ -path "*/dim_$d/*" -name "*_ranking_metrics.csv" | wc -l) / 15"
done
```

### After Completion
- [ ] All 75 metrics files exist
- [ ] No error messages in SLURM logs
- [ ] All jobs completed successfully

### Aggregation
```bash
python aggregate_dimensionality_analysis.py \
  --workspace experiment_workspace_dimensionality/ \
  --output_dir final_report_dimensionality/
```

- [ ] Aggregation runs without errors
- [ ] Line plots created (3 plots: EF@1%, ROC-AUC, PR-AUC)
- [ ] Summary tables created (3 tables)
- [ ] LaTeX report generated
- [ ] PDF compiled (if pdflatex available)

## Expected Outputs ✅

### During Execution
```
experiment_workspace_dimensionality/
├── run_seed42_config_ABL1_features_pca_coembedding_TIMESTAMP/
├── run_seed42_config_ABL1_features_tsne_coembedding_TIMESTAMP/
├── run_seed42_config_ABL1_features_umap_euclidean_coembedding_TIMESTAMP/
├── run_seed43_config_ABL1_features_pca_coembedding_TIMESTAMP/
├── ... (15 run directories total)
└── Each with subdirectories: dim_2/, dim_3/, dim_5/, dim_10/, dim_20/
```

### After Aggregation
```
final_report_dimensionality/
├── dimensionality_all_metrics.csv
├── plots/
│   ├── dimensionality_ef_1perc_lineplot.png      ⭐ PRIMARY FIGURE
│   ├── dimensionality_roc_auc_lineplot.png
│   └── dimensionality_pr_auc_lineplot.png
├── tables/
│   ├── summary_table_ef_1perc.csv
│   ├── summary_table_roc_auc.csv
│   └── summary_table_pr_auc.csv
└── dimensionality_report_TIMESTAMP/
    ├── dimensionality_analysis_report.tex
    └── dimensionality_analysis_report.pdf
```

## Troubleshooting 🔧

### Issue: Configs not generated
```bash
python generate_dimensionality_configs.py
ls -l dimensionality_configs/
```

### Issue: Job submission fails
```bash
# Check script exists
ls -l hpc/submit_dimensionality_jobs.sh

# Test single job manually
sbatch --job-name=test_dim \
  hpc/ummbas_dimensionality_cpu.sh 42 \
  dimensionality_configs/config_ABL1_features_pca_coembedding.json
```

### Issue: Missing results
```bash
# Check for errors in logs
grep -i error slurm_logs/DIM_*.err

# Check specific dimension
find experiment_workspace_dimensionality/ -path "*/dim_10/*" -name "*_ranking_metrics.csv"
```

### Issue: Aggregation fails
```bash
# Check if workspace exists
ls -l experiment_workspace_dimensionality/

# Run with verbose output
python aggregate_dimensionality_analysis.py 2>&1 | tee aggregation.log
```

## Key Metrics to Track 📊

### Primary Metric
- **EF@1%** (Enrichment Factor at 1%) - How well does each method rank actives?

### Questions to Answer
1. Does performance increase or decrease with dimensionality?
2. Is there an optimal dimension for each method?
3. Do PCA, UMAP, and t-SNE respond differently to dimensionality?
4. Are differences statistically significant (check error bar overlap)?

## Time Estimates ⏱️

| Task | Time |
|------|------|
| Config generation | < 1 minute |
| Job submission | ~5 minutes |
| Job execution | 12-24 hours (wall time) |
| Aggregation | 5-10 minutes |
| Total | ~1 day |

## Storage Estimates 💾

- Per run: ~5-10 GB
- Total (15 runs): ~75-150 GB
- After aggregation: +~100 MB

## Contact & Support 📞

If you encounter issues:
1. Check the log files first
2. Review `DIMENSIONALITY_EXPERIMENT_README.md`
3. Check `ANALYSIS_PIPELINE_OVERVIEW.md`
4. Look for similar issues in other experiments

## Quick Commands Reference 📝

```bash
# Generate configs
python generate_dimensionality_configs.py

# Submit all jobs
bash hpc/submit_dimensionality_jobs.sh

# Check job status
squeue -u $USER | grep DIM

# Count completed analyses
find experiment_workspace_dimensionality/ -name "*_ranking_metrics.csv" | wc -l

# Aggregate results
python aggregate_dimensionality_analysis.py

# View primary plot
xdg-open final_report_dimensionality/plots/dimensionality_ef_1perc_lineplot.png
```

## Success Indicators ✨

✅ 15 jobs submitted  
✅ All jobs complete without errors  
✅ 75 metrics files created (3 methods × 5 seeds × 5 dimensions)  
✅ Line plots show clear trends  
✅ Error bars indicate statistical robustness  
✅ LaTeX report compiles successfully  

---

**Ready to run? Start with:** `bash hpc/submit_dimensionality_jobs.sh`
