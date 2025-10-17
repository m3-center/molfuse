# Generalization Experiment

## Overview

This directory contains scripts and configurations for the **generalization experiment**, which tests whether the dimensionality reduction methods evaluated on TyrosineProteinKinaseABL1 (ABL1) generalize to other target proteins.

### Scientific Rationale

After extensive hyperparameter tuning on ABL1 (720 configurations × 5 seeds = 3,600 experiments), we want to assess:
1. **Do the methods generalize?** Do methods that work well on ABL1 also work on other proteins?
2. **Is hyperparameter tuning necessary?** Can we use reasonable defaults instead of exhaustive search?
3. **Which methods are most robust?** Which methods maintain performance across diverse targets?

### Experimental Design

- **Training Target**: TyrosineProteinKinaseABL1_P00519 (ABL1) - used for hyperparameter optimization
- **Generalization Targets**:
  - PyruvateKinaseM2_P14618 (same molecular function as ABL1: Protein kinase inhibitor)
  - IsocitrateDehydrogenaseNADP_O75874 (different molecular function: Oxidoreductase)

- **Methods tested** (with reasonable default hyperparameters):
  - PCA (Projection)
  - PCA (Co-embedding)
  - t-SNE (perplexity=100)
  - UMAP-Euclidean Projection (n_neighbors=100, min_dist=0.1)
  - UMAP-Euclidean Co-embedding (n_neighbors=100, min_dist=0.1)

- **Replicates**: 5 random seeds (42-46) for statistical significance
- **Total experiments**: 2 proteins × 5 methods × 5 seeds = 50 experiments
- **Comparison**: ~1.4% of the computational cost of full hyperparameter sweep

## Directory Structure

```
generalization_configs/          # Generated configuration files
experiment_workspace_generalization/  # Output workspace (created during runs)
final_report_generalization/     # Analysis results and reports
hpc/
  ├── ummbas_generalization_cpu.sh        # SLURM execution script
  └── submit_generalization_jobs.sh       # Job submission script
```

## Usage

### Step 1: Generate Configuration Files

```bash
python generate_generalization_configs.py
```

This creates 10 configuration files (2 targets × 5 methods) in `generalization_configs/`.

**Expected output:**
```
✓ Created: config_features_pca_projection.json
✓ Created: config_features_pca_coembedding.json
✓ Created: config_features_tsne.json
✓ Created: config_features_umap_euclidean_projection.json
✓ Created: config_features_umap_euclidean_coembedding.json
...
```

### Step 2: Submit Jobs to HPC

```bash
bash hpc/submit_generalization_jobs.sh
```

This submits 50 jobs (10 configs × 5 seeds) to the SLURM scheduler.

**Monitor progress:**
```bash
squeue -u $USER                           # Check job queue
tail -f slurm_logs/GEN_*_seed42_*.out    # Watch a specific job
```

### Step 3: Aggregate and Analyze Results

Once all jobs complete:

```bash
python aggregate_generalization_analysis.py \
  --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
  --generalization_workspace experiment_workspace_generalization/ \
  --output_dir final_report_generalization/
```

This generates:
- **Comparative visualizations**: Method performance across proteins
- **Summary tables**: Rankings, generalization metrics
- **Statistical analysis**: Performance consistency, variance across proteins

## Output Files

### Figures
- `EF_1Perc_method_comparison.png`: Bar chart comparing all methods across proteins
- `EF_1Perc_heatmap.png`: Heatmap showing method × protein performance
- `performance_consistency.png`: Scatter plot of mean performance vs. consistency
- Similar plots for ROC-AUC and PR-AUC

### Tables
- `master_generalization_metrics.csv`: Raw metrics from all experiments
- `performance_summary.csv`: Mean and std for each method × protein combination
- `method_rankings.csv`: Method rankings within each protein
- `generalization_assessment.csv`: Performance comparison ABL1 vs. other proteins

## Key Differences from Hyperparameter Sweep

| Aspect | Hyperparameter Sweep | Generalization Experiment |
|--------|---------------------|---------------------------|
| Target Proteins | ABL1 only | PyruvateKinaseM2, IsocitrateDehydrogenase |
| Goal | Find optimal hyperparameters | Test method generalization |
| Hyperparameters | Extensive grid (720 configs) | Single reasonable default per method |
| Total Experiments | 3,600 | 50 |
| Computational Cost | ~2 weeks on HPC | ~1 day on HPC |
| Workspace | `experiment_workspace_rerun_hyperparam_sweep/` | `experiment_workspace_generalization/` |
| Reports | `final_report_rerun_hyperparam_sweep/` | `final_report_generalization/` |

## Interpretation

### Good Generalization
If methods maintain relative ranking across proteins:
- **Implication**: Optimal hyperparameters from ABL1 are likely transferable
- **Practical value**: Users can apply these settings to new proteins without retuning

### Poor Generalization
If method rankings change dramatically:
- **Implication**: Protein-specific hyperparameter tuning may be necessary
- **Action**: Investigate protein characteristics that affect method performance

### Robustness Assessment
Methods with low coefficient of variation (CV) across proteins are more robust.

## Hyperparameter Choices

The default hyperparameters used in this experiment are mid-range values that showed reasonable performance in the ABL1 sweep:

- **t-SNE**: perplexity=100 (mid-range, showed good performance)
- **UMAP**: n_neighbors=100, min_dist=0.1 (middle values in the grid)
- **PCA**: No hyperparameters (deterministic given input dimensionality)

These are **not necessarily optimal** for the generalization targets, which is the point - we want to see how well "reasonable defaults" work without expensive tuning.

## Notes

- Only **features** representation is used (fingerprints showed poor EF@1% in hyperparameter sweep)
- Only **2D** similarity spaces (as in the focused hyperparameter analysis)
- The analysis script automatically handles missing results (partial runs)
- Results are independent of the hyperparameter sweep workspace

## Troubleshooting

**No configs generated?**
- Check that `generate_generalization_configs.py` ran successfully
- Verify `generalization_configs/` directory exists

**Jobs not submitting?**
- Ensure you're on the HPC login node
- Check SLURM is available: `which sbatch`
- Verify script has execute permissions: `chmod +x hpc/submit_generalization_jobs.sh`

**No results to aggregate?**
- Check job status: `squeue -u $USER`
- Review logs: `ls -lh slurm_logs/GEN_*`
- Verify workspace directory: `ls experiment_workspace_generalization/`

**Aggregation script fails?**
- Ensure at least some jobs have completed
- Check that workspace paths are correct
- Review the log file: `cat generalization_analysis_*.log`

## Citation

If you use this experimental design in your work, please cite:
- The main UMMBAS paper (when published)
- Note the generalization methodology in your methods section

## Contact

For questions about the generalization experiment setup, contact the UMMBAS development team.
