# Generalization Experiment Setup - Complete

## Date: October 8, 2025

## Summary

A complete generalization experiment framework has been created to test whether the dimensionality reduction methods evaluated on TyrosineProteinKinaseABL1 (ABL1) generalize to other target proteins without expensive hyperparameter retuning.

---

## Files Created

### 1. Configuration Generator
**File**: `generate_generalization_configs.py`

Generates 10 configuration files for:
- 2 target proteins (PyruvateKinaseM2, IsocitrateDehydrogenase)
- 5 methods with reasonable default hyperparameters:
  - PCA Projection
  - PCA Co-embedding
  - t-SNE (perplexity=100)
  - UMAP-Euclidean Projection (n_neighbors=100, min_dist=0.1)
  - UMAP-Euclidean Co-embedding (n_neighbors=100, min_dist=0.1)

**Output**: `generalization_configs/` directory

---

### 2. HPC Execution Scripts

**File**: `hpc/ummbas_generalization_cpu.sh`
- SLURM batch script for running individual experiments
- Identical setup to hyperparameter sweep (64 cores, 350GB RAM, 24h)
- Activates conda environment and runs main_orchestrator.py

**File**: `hpc/submit_generalization_jobs.sh`
- Submits 50 jobs (10 configs × 5 seeds)
- Seeds: 42, 43, 44, 45, 46 (same as hyperparameter sweep)
- Automatically creates job names like `GEN_config_features_tsne_seed42`

---

### 3. Analysis Script

**File**: `aggregate_generalization_analysis.py`

Comprehensive analysis script that:
- Collects metrics from both ABL1 (hyperparameter sweep) and generalization experiments
- Creates comparative visualizations:
  - Method comparison bar charts across proteins
  - Performance heatmaps
  - Consistency analysis (performance vs. variation)
- Generates summary tables:
  - Performance summary by method and protein
  - Method rankings within each protein
  - Generalization assessment (ABL1 vs. others)

**Output**: `final_report_generalization/generalization_report_TIMESTAMP/`

---

### 4. Documentation

**File**: `GENERALIZATION_EXPERIMENT_README.md`
- Complete documentation of experimental design
- Step-by-step usage instructions
- Interpretation guidelines
- Troubleshooting section

---

### 5. Quick Start Script

**File**: `run_generalization_experiment.sh`

Interactive script with three steps:
```bash
bash run_generalization_experiment.sh 1      # Generate configs
bash run_generalization_experiment.sh 2      # Submit jobs
bash run_generalization_experiment.sh 3      # Analyze results
```

---

## Key Design Decisions

### 1. Workspace Separation
- **Hyperparameter sweep**: `experiment_workspace_rerun_hyperparam_sweep/`
- **Generalization**: `experiment_workspace_generalization/`
- **Reason**: Prevents any data conflicts or overwrites

### 2. Hyperparameter Choices
Mid-range values that showed reasonable performance on ABL1:
- **t-SNE**: perplexity=100 (not extreme values like 15 or 1000)
- **UMAP**: n_neighbors=100, min_dist=0.1 (middle of grid)
- **Rationale**: Test if "reasonable defaults" work without exhaustive tuning

### 3. Protein Selection
- **PyruvateKinaseM2_P14618**: Same molecular function as ABL1 (Protein kinase inhibitor)
- **IsocitrateDehydrogenaseNADP_O75874**: Different molecular function (Oxidoreductase)
- **Rationale**: Test generalization within and across molecular function classes

### 4. Statistical Rigor
- 5 random seeds (42-46) for each configuration
- Same as hyperparameter sweep for fair comparison
- Total: 50 experiments (vs. 3,600 in hyperparameter sweep)

### 5. Features Only
- Only use `features` representation (not fingerprints)
- **Reason**: Fingerprints showed poor EF@1% performance (<1%) in hyperparameter sweep
- Reduces computational cost by 50%

---

## Computational Efficiency

| Aspect | Hyperparameter Sweep | Generalization | Reduction |
|--------|---------------------|----------------|-----------|
| Configurations | 720 | 10 | 98.6% |
| Total Experiments | 3,600 | 50 | 98.6% |
| Estimated Time | ~2 weeks | ~1 day | 93% |

---

## How to Use

### On Local Machine

1. **Generate configurations:**
   ```bash
   python generate_generalization_configs.py
   ```

2. **Review configs:**
   ```bash
   ls generalization_configs/
   ```

3. **Transfer to HPC:**
   ```bash
   rsync -avz generalization_configs/ hpc:~/UMMBAS/generalization_configs/
   rsync -avz hpc/ hpc:~/UMMBAS/hpc/
   ```

### On HPC

1. **Submit jobs:**
   ```bash
   cd ~/UMMBAS
   bash hpc/submit_generalization_jobs.sh
   ```

2. **Monitor progress:**
   ```bash
   squeue -u $USER
   watch -n 60 'squeue -u $USER'
   tail -f slurm_logs/GEN_*.out
   ```

3. **After completion, transfer results:**
   ```bash
   rsync -avz hpc:~/UMMBAS/experiment_workspace_generalization/ ./experiment_workspace_generalization/
   ```

### Back on Local Machine

4. **Analyze results:**
   ```bash
   python aggregate_generalization_analysis.py \
     --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
     --generalization_workspace experiment_workspace_generalization/ \
     --output_dir final_report_generalization/
   ```

5. **Review outputs:**
   ```bash
   ls -lh final_report_generalization/generalization_report_*/
   ```

---

## Expected Outputs

### Visualizations
1. **Method comparison bar charts**: EF@1%, ROC-AUC, PR-AUC across proteins
2. **Performance heatmaps**: Method × Protein performance matrices
3. **Consistency plots**: Mean performance vs. coefficient of variation

### Tables
1. **performance_summary.csv**: Mean ± std for each method/protein combo
2. **method_rankings.csv**: Rank of each method within each protein
3. **generalization_assessment.csv**: Performance drop from ABL1 to other proteins
4. **master_generalization_metrics.csv**: All raw data

---

## Interpretation Guidelines

### Strong Generalization Indicators
✓ Methods maintain relative rankings across proteins  
✓ Performance drop from ABL1 < 20%  
✓ Low coefficient of variation (CV < 30%)  

### Weak Generalization Indicators
✗ Method rankings flip dramatically between proteins  
✗ Performance drop from ABL1 > 50%  
✗ High coefficient of variation (CV > 50%)  

---

## Integration with Existing Analysis

The generalization analysis script is designed to:
- Load ABL1 results from the hyperparameter sweep workspace
- Load new protein results from the generalization workspace
- Compare them on equal footing
- Generate unified visualizations and statistics

No modifications to existing files are required - everything uses separate workspaces.

---

## Next Steps After Analysis

Depending on results:

1. **If generalization is good:**
   - Document recommended default hyperparameters
   - Use these settings for future proteins
   - Publish generalization findings

2. **If generalization is moderate:**
   - Identify protein characteristics that affect performance
   - Consider protein-family-specific tuning
   - May need broader hyperparameter brackets

3. **If generalization is poor:**
   - Investigate why methods are protein-specific
   - Consider per-protein hyperparameter tuning
   - May need adaptive/learned hyperparameter selection

---

## Compatibility Notes

- **Main orchestrator**: No changes needed - reads workspace from config
- **Cutoff analysis**: Can be adapted if needed for generalization proteins
- **Hyperparameter configs**: Separate from generalization configs
- **Reports**: Independent output directories

---

## Validation Checklist

Before running on HPC:

- [ ] Configurations generated successfully (10 files)
- [ ] All scripts are executable (`chmod +x`)
- [ ] HPC scripts have correct conda environment path
- [ ] `slurm_logs/` directory exists or will be created
- [ ] No conflicting jobs running
- [ ] Sufficient HPC allocation for 50 jobs

After HPC completion:

- [ ] All 50 jobs completed successfully
- [ ] No error messages in log files
- [ ] All expected output directories exist
- [ ] Metrics files present for all configs
- [ ] Ready to run aggregation script

---

## Cost-Benefit Analysis

**Benefits:**
- 98.6% reduction in computational cost vs. full hyperparameter sweep
- Scientifically rigorous (5 replicates, proper controls)
- Tests both within-class and across-class generalization
- Provides actionable insights for future experiments

**Limitations:**
- Only tests 2 additional proteins (could test more if needed)
- Uses mid-range hyperparameters (not guaranteed optimal)
- Focuses on features only (fingerprints excluded)

**Recommendation:** This is an excellent trade-off for assessing generalization while minimizing computational burden.

---

## Contact & Support

For questions or issues:
1. Check `GENERALIZATION_EXPERIMENT_README.md`
2. Review log files in `slurm_logs/`
3. Contact: UMMBAS development team

---

## File Permissions

All scripts have been made executable:
```
✓ generate_generalization_configs.py
✓ aggregate_generalization_analysis.py
✓ run_generalization_experiment.sh
✓ hpc/submit_generalization_jobs.sh
✓ hpc/ummbas_generalization_cpu.sh
```

Ready to run!
