# Potency-Stratified Enrichment Analysis for UMMBAS v3 Phase 1

## Overview

This analysis tool addresses a critical limitation in traditional enrichment metrics: **all actives are not created equal**. A 100 nM ligand (highly potent, therapeutically relevant) is vastly more valuable than a 100,000 nM ligand (weak, likely promiscuous), yet traditional EF@1% treats them identically.

This can mislead hyperparameter optimization, causing you to select configurations that enrich many weak binders over fewer, highly potent binders.

## The Problem

**Example Trade-off:**

| Config | Overall EF@1% | High-Potent EF@1% | Interpretation |
|--------|---------------|-------------------|----------------|
| nn=10  | **57.6** ✓    | 12.3              | Enriches many weak binders |
| nn=20  | 33.8          | **45.7** ✓        | Enriches fewer, potent binders |

Traditional metrics would select **nn=10** (higher overall EF), but **nn=20** is superior for drug discovery!

## The Solution

Stratify enrichment analysis by potency tiers:

- **High Potent** (0.1-100 nM): Drug-like, clinically relevant
- **Medium Potent** (100-1,000 nM): Moderate affinity, optimizable
- **Weak Potent** (1,000-100,000 nM): Marginal activity, promiscuous

## Quick Start

### 1. Validate Your Data

```bash
# Check if workspace has required data
python scripts/validate_potency_analysis.py \
    --workspace_dir /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1
```

### 2. Run Analysis

```bash
# Full analysis with plots
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1 \
    --output_dir potency_analysis_results

# Quick summary only (faster, good for monitoring)
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1 \
    --output_dir potency_analysis_quick \
    --summary_only

# Analyze specific seed
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1 \
    --seed 42 \
    --output_dir potency_analysis_seed42
```

### 3. Review Results

```bash
# Read the summary report
cat potency_analysis_results/potency_stratified_report.txt

# View plots
ls potency_analysis_results/plots/

# Analyze detailed CSV
head potency_analysis_results/stratified_enrichment_summary.csv
```

## Output Files

### Main Outputs

1. **`potency_stratified_report.txt`** - Human-readable summary with key findings and recommendations
2. **`stratified_enrichment_summary.csv`** - Aggregated statistics (mean ± std) per configuration
3. **`stratified_enrichment_detailed.csv`** - Complete results for every individual run

### Plots (in `plots/` subdirectory)

1. **`stratified_ef_by_nn.png`** - Compare enrichment across tiers for different n_neighbors
2. **`stratified_percent_found.png`** - Recovery rates for each potency tier
3. **`quality_vs_quantity_tradeoff.png`** - Scatter plot revealing trade-offs

## Key Metrics

### Per-Tier Metrics

- **High_EF**: Enrichment factor for highly potent compounds (0.1-100 nM)
- **Medium_EF**: Enrichment factor for moderate compounds (100-1,000 nM)
- **Weak_EF**: Enrichment factor for weak binders (1,000-100,000 nM)
- **Overall_EF**: Traditional metric (all actives treated equally)

### Recovery Metrics

- **High_percent_found**: What % of highly potent compounds appear in top 1%?
- Similar for Medium and Weak tiers

## Interpreting Results

### ✅ High_EF >> Overall_EF
**Good!** Method is selective for potent compounds
- Ideal for drug discovery
- Fewer false positives from promiscuous binders

### ⚠️ High_EF << Overall_EF  
**Warning!** Method enriches weak binders preferentially
- High overall EF is misleading
- May indicate artifacts or promiscuous binding patterns

### Example from Report

```
⚠️  QUALITY vs QUANTITY TRADE-OFF DETECTED:

Best for HIGH-POTENT enrichment:
  Config: nn=20, md=0.01
  High-Potent EF: 45.7
  Overall EF: 33.8

Best for OVERALL enrichment:
  Config: nn=10, md=0.1
  High-Potent EF: 12.3
  Overall EF: 57.6

RECOMMENDATION: For drug discovery, prioritize the configuration
with best High-Potent enrichment to find therapeutically relevant
compounds, even if overall EF is slightly lower.
```

## Integration with Existing Analysis

This tool complements existing UMMBAS analysis:

1. **`scripts/check_hyperparam_status.py`** - Check which runs completed
2. **`scripts/validate_potency_analysis.py`** - Validate data availability (NEW)
3. **`scripts/analyze_potency_stratified_enrichment.py`** - Potency-stratified analysis (NEW)
4. **`analysis_scripts/aggregate_and_report.py`** - Overall performance reporting

### Recommended Workflow

```bash
# 1. Check completion status
python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1

# 2. Validate data for potency analysis
python scripts/validate_potency_analysis.py --workspace_dir experiment_workspace_v3_phase1

# 3. Run potency-stratified analysis
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_analysis_results

# 4. Use results to inform Phase 2+ decisions
cat potency_analysis_results/potency_stratified_report.txt
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

These should already be in your `ummbas-screening` conda environment.

## Performance

- **Validation**: ~5 seconds
- **Summary only**: ~10 seconds per 100 runs
- **Full analysis**: ~1 minute per 100 runs

Safe to run on HPC login node for monitoring. For comprehensive analysis of all Phase 1 runs (~260), submit as a short job or run on login node (~3-5 minutes total).

## Troubleshooting

### "No results found"

- **Check**: Run `validate_potency_analysis.py` first
- **Cause**: Runs haven't completed or ranked files missing
- **Solution**: Wait for experiments to complete

### "No affinity data found"

- **Check**: Look at one of your ranked CSV files - does it have `Standard Value (nM)` column?
- **Cause**: Affinity data not preserved during analysis pipeline
- **Solution**: Script attempts to load from source files (detailed_active_distances.csv or prepared_data files)

### Import Errors

```bash
# Ensure environment is activated
conda activate ummbas-screening

# Install any missing packages
pip install tqdm seaborn
```

## Files

```
scripts/
├── analyze_potency_stratified_enrichment.py  # Main analysis script
├── validate_potency_analysis.py              # Data validation utility
└── check_hyperparam_status.py                # Existing completion checker

docs/
└── POTENCY_STRATIFIED_ANALYSIS.md            # Detailed documentation
```

## Citation / Background

This analysis approach is motivated by:

1. **Clinical relevance**: Drug candidates typically require IC50 < 100 nM
2. **Promiscuity**: Weak binders (>10 μM) often show non-specific binding
3. **Optimization potential**: Moderate binders offer better starting points than ultra-weak hits

Traditional EF metrics emerged from high-throughput screening where any "hit" was valuable. Modern virtual screening should prioritize quality over quantity.

## Contact

For questions or issues:
- Check `UMMBAS_V3_LAB_BOOK.md`
- Review `README_V3_PIPELINE.md`
- See main repository documentation

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Compatible with**: UMMBAS v3.0 Phase 1
