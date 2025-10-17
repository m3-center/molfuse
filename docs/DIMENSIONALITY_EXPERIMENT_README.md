# UMMBAS Dimensionality Experiment - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Motivation and Research Questions](#motivation-and-research-questions)
3. [Experimental Design](#experimental-design)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Expected Results](#expected-results)
7. [Analysis Workflow](#analysis-workflow)
8. [Compatibility Notes](#compatibility-notes)

---

## Overview

The **Dimensionality Experiment** (Experiment 4) is designed to investigate how the dimensionality of the similarity space affects the performance of molecular virtual screening methods. This experiment builds directly on the findings from Experiment 1 (Hyperparameter Sweep) by using the optimal hyperparameters identified there, but systematically varying the target dimensionality.

### Key Characteristics

- **Experiment Type:** Dimensionality sensitivity analysis
- **Target Protein:** ABL1 (Tyrosine-protein Kinase ABL1, P00519)
- **Representation:** Physicochemical features only
- **Dimensions Tested:** 2, 3, 5, 10, 20
- **Methods:** PCA, UMAP (Euclidean), t-SNE
- **Strategy:** Co-embedding only
- **Replicates:** 5 random seeds (42-46)

---

## Motivation and Research Questions

### Background

In dimensionality reduction for molecular virtual screening:
- **Low dimensions (2D, 3D):** Easy to visualize, interpretable, but may lose important information
- **High dimensions (10D, 20D):** May capture more complex relationships, but harder to interpret

Previous experiments (1-3) all used **2D similarity spaces** for consistency and visualization. However, the optimal dimensionality for screening performance is unknown.

### Research Questions

1. **Does higher dimensionality improve screening performance?**
   - Hypothesis: Higher dimensions may better capture molecular diversity
   
2. **Do different methods respond differently to dimensionality changes?**
   - PCA (linear) might benefit more from higher dimensions
   - UMAP/t-SNE (non-linear) might already capture relationships well in 2D
   
3. **Is there a "sweet spot" dimension that balances performance and interpretability?**
   - Too low: Information loss
   - Too high: Curse of dimensionality, overfitting

4. **How does dimensionality affect the distance-based ranking metric?**
   - In higher dimensions, distances may become less meaningful (distance concentration)

---

## Experimental Design

### Fixed Parameters (from Experiment 1)

These were identified as optimal in the hyperparameter sweep:

| Method | Hyperparameter | Optimal Value |
|--------|----------------|---------------|
| PCA | - | (No tunable hyperparameters) |
| UMAP | `n_neighbors` | 500 |
| UMAP | `min_dist` | 0.01 |
| t-SNE | `perplexity` | 1000 |

### Variable Parameter

**Similarity Space Dimensionality:** `d ∈ {2, 3, 5, 10, 20}`

### Controlled Variables

- **Target:** ABL1 only (same as Experiment 1)
- **Representation:** Features only (best performing)
- **Embedding Strategy:** Co-embedding only (as specified)
- **Data Split:** Same as Experiment 1 (leave-one-target-out)
- **Affinity Cutoff:** 100,000 nM (standard)
- **Random Seeds:** 42-46 (N=5 replicates)

### Experimental Matrix

```
3 methods × 5 dimensions × 5 seeds = 75 total analyses
(within 15 jobs: 3 methods × 5 seeds)
```

---

## Methodology

### Data Pipeline

Same as Experiments 1-3:

1. **MF Cloud Construction**
   - ChEMBL compounds with molecular function = "Protein kinase inhibitor"
   - Exclude ABL1 actives (held out)
   
2. **Decoy Set**
   - ZINC acquirable compounds
   
3. **Held-out Actives**
   - ABL1 active compounds (affinity ≤ 100,000 nM)
   - These are co-embedded with MF Cloud

### Dimensionality Reduction Process

For each dimension `d`:

1. **Input:** N × 39 feature matrix (N = MF Cloud + decoys + held-out actives)
2. **Scaling:** StandardScaler on features
3. **DR Application:** 
   - PCA: Project to d dimensions
   - UMAP: Reduce to d dimensions (co-embedding)
   - t-SNE: Reduce to d dimensions (co-embedding)
4. **Output:** N × d coordinate matrix

### Ranking and Evaluation

For each d-dimensional space:

1. **Distance Calculation:**
   ```
   For each compound (active or decoy):
       dist = min(euclidean_distance(compound, mf_cloud_member))
   ```
   
2. **Ranking:** Sort all compounds by distance (ascending)

3. **Metrics:**
   - **Primary:** EF@1% (Enrichment Factor at top 1%)
   - **Secondary:** ROC-AUC, PR-AUC

4. **Statistical Robustness:** Repeat across 5 seeds, report mean ± 90% CI

---

## Implementation Details

### Configuration Files

Each method gets one config file that specifies ALL dimensions:

```json
{
  "global_settings": {
    "simspace_dims_to_test": [2, 3, 5, 10, 20],
    "workspace_base_dir": "experiment_workspace_dimensionality/",
    ...
  },
  "dimensionality_reduction_methods": {
    "pca": {
      "short_name": "PCA",
      "allow_coembedding": true
    }
  }
}
```

### Pipeline Execution

The existing `main_orchestrator.py` handles this automatically:

```python
for simspace_dim_val in gs['simspace_dims_to_test']:  # Loops through [2, 3, 5, 10, 20]
    # Create similarity space in d dimensions
    # Run analysis
    # Save results to dim_X/ subdirectories
```

**No code changes needed!** The orchestrator already supports multiple dimensions.

### Directory Structure

```
experiment_workspace_dimensionality/
├── run_seed42_config_ABL1_features_pca_coembedding_TIMESTAMP/
│   ├── run_config.json
│   └── TyrosineProteinKinaseABL1_P00519/
│       ├── temp_data/
│       ├── target_ligands_calculated/
│       ├── models/
│       │   └── features/
│       │       ├── dim_2/
│       │       ├── dim_3/
│       │       ├── dim_5/
│       │       ├── dim_10/
│       │       └── dim_20/
│       ├── similarity_spaces/
│       │   └── features/
│       │       ├── dim_2/
│       │       │   └── *_PCA_similarity_space_COEMBED.csv
│       │       ├── dim_3/
│       │       ├── dim_5/
│       │       ├── dim_10/
│       │       └── dim_20/
│       └── results/
│           └── features/
│               ├── dim_2/
│               │   └── PCA_Coembed/
│               │       └── TyrosineProteinKinaseABL1_P00519_ranking_metrics.csv
│               ├── dim_3/
│               ├── dim_5/
│               ├── dim_10/
│               └── dim_20/
```

### File Naming Conventions

Similarity spaces:
```
{target_id}_{repr_type}_dim{d}_{method}_similarity_space_COEMBED.csv
```

Results:
```
{target_id}_ranking_metrics.csv
```

---

## Expected Results

### Hypotheses

Based on dimensionality reduction theory:

#### PCA (Linear Method)
- **Expected:** Performance may improve with higher dimensions
- **Reasoning:** More principal components capture more variance
- **Caveat:** Diminishing returns after capturing most variance

#### UMAP (Non-linear, Manifold Learning)
- **Expected:** Performance may plateau or decrease at very high dimensions
- **Reasoning:** Already captures non-linear structure well in 2-3D
- **Caveat:** May benefit moderately from 5-10D

#### t-SNE (Non-linear, Local Structure)
- **Expected:** Sensitive to dimensionality, may work best in 2D
- **Reasoning:** Designed for 2-3D visualization
- **Caveat:** Computationally expensive in high dimensions

### Possible Outcomes

**Scenario 1: Higher is Better**
```
EF@1% increases monotonically with dimension
→ Recommendation: Use 20D for best performance
```

**Scenario 2: Sweet Spot at Mid-Dimension**
```
EF@1% peaks at 5-10D
→ Recommendation: Use 5D or 10D (balance performance + interpretability)
```

**Scenario 3: 2D is Optimal**
```
EF@1% highest at 2D, decreases with dimension
→ Recommendation: Stick with 2D (also easiest to visualize)
```

**Scenario 4: Method-Dependent**
```
PCA benefits from high D, UMAP/t-SNE prefer low D
→ Recommendation: Choose dimension based on method
```

---

## Analysis Workflow

### Step 1: Generate Configs

```bash
python generate_dimensionality_configs.py
```

**Output:** 3 config files in `dimensionality_configs/`

### Step 2: Submit to HPC

```bash
bash hpc/submit_dimensionality_jobs.sh
```

**Jobs submitted:** 15 (3 configs × 5 seeds)

### Step 3: Monitor Progress

```bash
# Check job queue
squeue -u $USER | grep DIM

# Check specific log
tail -f slurm_logs/DIM_config_ABL1_features_pca_coembedding_seed42_*.out

# Count completed analyses
find experiment_workspace_dimensionality/ -name "*_ranking_metrics.csv" | wc -l
# Should eventually reach 75
```

### Step 4: Aggregate Results

```bash
python aggregate_dimensionality_analysis.py
```

**Outputs:**
- `dimensionality_all_metrics.csv` - Raw data
- Line plots (EF@1%, ROC-AUC, PR-AUC vs dimension)
- Summary tables (mean ± SD for each method × dimension)
- LaTeX report (comprehensive document)

### Step 5: Interpret Results

1. **Primary Plot:** EF@1% vs Dimension
   - Identify trends for each method
   - Note where error bars overlap (non-significant differences)
   - Find optimal dimension for each method

2. **Statistical Significance:**
   - Compare 90% CI error bars
   - If bars don't overlap → significant difference
   - If bars overlap → not significantly different

3. **Practical Recommendations:**
   - Balance performance with interpretability
   - Consider computational cost (higher D = more expensive)
   - Recommend optimal dimension per method

---

## Compatibility Notes

### With Existing Pipeline

✅ **Fully compatible** - No code changes needed

The existing `main_orchestrator.py` already:
- Loops through `simspace_dims_to_test`
- Creates dimension-specific subdirectories
- Handles co-embedding for PCA/UMAP/t-SNE
- Saves results per dimension

### With Other Experiments

| Aspect | Compatibility |
|--------|---------------|
| **Data sources** | ✅ Same datasets (ChEMBL, ZINC) |
| **Target protein** | ✅ ABL1 (same as Exp 1) |
| **Hyperparameters** | ✅ Uses optimal from Exp 1 |
| **Workspace** | ✅ Separate directory (no conflicts) |
| **Aggregation scripts** | ✅ Independent (new script) |
| **HPC scripts** | ✅ New scripts (no interference) |

### Version Control

This experiment is on the `dimensionality` branch (as indicated by repo context).

To integrate results:
```bash
# On dimensionality branch
git add generate_dimensionality_configs.py
git add hpc/submit_dimensionality_jobs.sh
git add hpc/ummbas_dimensionality_cpu.sh
git add aggregate_dimensionality_analysis.py
git add DIMENSIONALITY_*.txt
git commit -m "Add Experiment 4: Dimensionality analysis"
```

---

## Computational Requirements

### Per Job

- **Nodes:** 1
- **CPUs:** 64
- **Memory:** 350 GB
- **Time:** ~12-24 hours (depends on dimension)
- **Storage:** ~5-10 GB per job

### Total Requirements

- **Total jobs:** 15
- **Total CPU-hours:** ~300-500 hours
- **Total storage:** ~100-150 GB

### Scalability

Higher dimensions are more computationally expensive:
- **2D, 3D:** Fast (~2-4 hours)
- **5D:** Moderate (~4-8 hours)
- **10D, 20D:** Slow (~8-16 hours)

Most time spent on:
1. UMAP fitting (O(N^2) for high D)
2. t-SNE optimization (O(N^2 log N) for high D)
3. Distance calculations (O(N × M × d))

---

## Troubleshooting

### Common Issues

**Issue 1: t-SNE fails for dimension > 2**
- **Cause:** t-SNE is primarily designed for 2D/3D
- **Solution:** Check if t-SNE supports higher dimensions in your sklearn version
- **Workaround:** May need to adjust t-SNE parameters or skip high dimensions

**Issue 2: UMAP memory error for high dimensions**
- **Cause:** Large dataset + high dimension = high memory usage
- **Solution:** Increase job memory or reduce `n_neighbors`

**Issue 3: Distance concentration in high dimensions**
- **Cause:** In very high dimensions, distances become similar
- **Solution:** This is expected; analyze whether performance degrades

**Issue 4: Missing metrics files**
- **Cause:** Job crashed or incomplete
- **Solution:** Check SLURM error logs, resubmit failed jobs

### Validation Checks

```bash
# 1. Check config generation
ls -l dimensionality_configs/*.json
# Should see 3 files

# 2. Verify dimensions in config
grep "simspace_dims_to_test" dimensionality_configs/config_ABL1_features_pca_coembedding.json
# Should show: [2, 3, 5, 10, 20]

# 3. Check job submission
ls -l slurm_logs/DIM_*.out
# Should see 15 files after submission

# 4. Verify results structure
find experiment_workspace_dimensionality/ -type d -name "dim_*"
# Should show multiple dim_2, dim_3, dim_5, dim_10, dim_20 directories

# 5. Count metrics files per dimension
for d in 2 3 5 10 20; do
    echo "Dimension $d:"
    find experiment_workspace_dimensionality/ -path "*/dim_$d/*" -name "*_ranking_metrics.csv" | wc -l
    # Should show 15 for each (3 methods × 5 seeds)
done
```

---

## References

1. **Hyperparameter Sweep (Experiment 1):** Source of optimal hyperparameters
2. **Leave-one-target-out methodology:** Standard for this pipeline
3. **Dimensionality reduction theory:** PCA, UMAP, t-SNE papers
4. **Distance concentration:** Beyer et al. (1999) - "When is 'nearest neighbor' meaningful?"

---

## Authors and Contact

**UMMBAS Project Team**

For questions or issues with this experiment:
1. Check this README
2. Check `DIMENSIONALITY_QUICKSTART.txt` for quick reference
3. Check main `ANALYSIS_PIPELINE_OVERVIEW.md`
4. Review log files for detailed error messages

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-11 | 1.0 | Initial documentation for Experiment 4 |
