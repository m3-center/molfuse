# Potency-Stratified Enrichment Analysis

## Quick Start Guide

### On HPC (During/After Phase 1 Experiments)

```bash
# Load environment
conda activate ummbas-screening  # or your environment name

# Navigate to experiment directory
cd /home/ahagg2s/UMMBAS_screening_experiments

# Run analysis on all completed experiments
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir analysis_results/potency_stratified

# View results
cat analysis_results/potency_stratified/potency_stratified_report.txt
```

### Monitor Progress While Experiments Run

```bash
# Quick summary only (faster, no plots)
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir analysis_results/potency_quick \
    --summary_only

# Analyze specific seed
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --seed 42 \
    --output_dir analysis_results/potency_seed42
```

## What This Analysis Reveals

### The Problem with Traditional EF@1%

Traditional enrichment metrics treat all "actives" equally:
- 100 nM ligand (very potent, clinically relevant) = 1 active
- 100,000 nM ligand (1000× weaker, not useful) = 1 active

This can mislead hyperparameter optimization!

### Example Scenario

**Configuration A (nn=10):**
- Overall EF@1% = 57.6 (looks great!)
- But: Enriches many weak binders (1,000-100,000 nM)
- High-Potent EF@1% = 12.3

**Configuration B (nn=20):**
- Overall EF@1% = 33.8 (looks worse)
- But: Enriches fewer, highly potent binders (0.1-100 nM)
- High-Potent EF@1% = 45.7

**For drug discovery, Configuration B is superior!**

## Potency Tiers

The script classifies actives into three tiers based on affinity:

| Tier | Range (nM) | Significance |
|------|------------|--------------|
| **High** | 0.1 - 100 | Drug-like, clinically relevant, therapeutically useful |
| **Medium** | 100 - 1,000 | Moderate affinity, potential for optimization |
| **Weak** | 1,000 - 100,000 | Marginal activity, likely promiscuous binders |

## Output Files

### 1. `stratified_enrichment_detailed.csv`
Complete results for every run, including:
- Configuration details (method, dimension, hyperparameters)
- Enrichment factors for each potency tier
- Number of actives found per tier

### 2. `stratified_enrichment_summary.csv`
Aggregated statistics across seeds:
- Mean and standard deviation for each metric
- Sorted by overall performance

### 3. `potency_stratified_report.txt`
Human-readable summary highlighting:
- Top-performing configurations
- Quality vs. quantity trade-offs
- Recommendations for hyperparameter selection

### 4. Plots (in `plots/` subdirectory)

#### `stratified_ef_by_nn.png`
Comparison of enrichment factors across potency tiers for different n_neighbors values

#### `stratified_percent_found.png`
Recovery rates: what percentage of each tier is found in the top 1%?

#### `quality_vs_quantity_tradeoff.png`
Scatter plot showing High-Potent EF vs. Overall EF to identify trade-offs

## Key Metrics Explained

### Enrichment Factor (EF@1%) per Tier
- **High_EF**: How well the method enriches highly potent compounds
- **Medium_EF**: Enrichment of moderate affinity compounds
- **Weak_EF**: Enrichment of weak binders
- **Overall_EF**: Traditional metric (all actives equally weighted)

### Percent Found per Tier
- What fraction of each potency tier appears in the top 1% of rankings?
- More actionable than EF for understanding recovery

## Interpreting Results

### High-Potent EF >> Overall EF
✅ **Good**: Method is selective for potent compounds
- Ideal for drug discovery
- Fewer false positives

### High-Potent EF ≈ Overall EF
✓ **Neutral**: Method doesn't discriminate by potency
- Treats all actives equally

### High-Potent EF << Overall EF
⚠️ **Warning**: Method preferentially enriches weak binders
- May indicate artifacts or promiscuous binding
- High overall EF is misleading

## Advanced Usage

### Compare Specific Configurations

```bash
# Analyze only UMAP runs
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir analysis_results/umap_only

# Filter in the CSV afterwards
python -c "
import pandas as pd
df = pd.read_csv('analysis_results/potency_stratified/stratified_enrichment_detailed.csv')
umap_df = df[df['dr_method'].str.contains('UMAP')]
umap_df.to_csv('analysis_results/umap_stratified.csv', index=False)
"
```

### Integration with Existing Analysis

This script complements existing analysis scripts:
- `scripts/check_hyperparam_status.py` - Check completion status
- `analysis_scripts/aggregate_and_report.py` - Overall performance metrics
- **This script** - Potency-stratified deep dive

Use together for comprehensive hyperparameter evaluation!

## Expected Runtime

- **Summary only**: ~5-10 seconds per 100 runs
- **With plots**: ~30-60 seconds per 100 runs
- Scales linearly with number of runs

Safe to run on login node for quick checks, or submit as job for full analysis.

## Troubleshooting

### "No results found"
- Check that runs have completed (RANKED.csv files exist)
- Verify workspace path is correct
- Ensure affinity data is available in ranked files

### Missing affinity data
The script requires `Standard Value (nM)` column in ranked files. If not present:
1. Check that `project_and_analyze.py` preserved affinity data
2. May need to merge with original ChEMBL data

### Import errors
Ensure environment has required packages:
```bash
pip install pandas numpy matplotlib seaborn tqdm
```

## Contact

For questions or issues, check the main repository documentation or lab book:
- `UMMBAS_V3_LAB_BOOK.md`
- `README_V3_PIPELINE.md`
