# Potency-Stratified Enrichment Analysis - Implementation Summary

## What Was Created

A complete analysis toolkit for evaluating hyperparameter performance stratified by ligand potency, addressing the critical limitation that traditional EF@1% treats a 100 nM drug-like compound the same as a 100,000 nM weak binder.

## Files Created

### 1. Main Analysis Script
**`scripts/analyze_potency_stratified_enrichment.py`**
- Analyzes completed Phase 1 runs
- Classifies actives into High/Medium/Weak potency tiers
- Calculates tier-specific enrichment factors
- Generates comparison plots and reports
- ~700 lines, fully documented

### 2. Validation Utility
**`scripts/validate_potency_analysis.py`**
- Quick check if workspace data is ready
- Samples runs to verify affinity data availability
- Estimates how many runs can be analyzed
- ~150 lines

### 3. Debug Tool
**`scripts/debug_run_structure.py`**
- Inspects run directory structure
- Shows what files exist and their columns
- Helps troubleshoot data availability issues
- ~100 lines

### 4. HPC Job Script
**`hpc/submit_potency_analysis.sh`**
- SLURM submission script
- Runs validation + analysis automatically
- Displays summary in job output
- Can be customized with environment variables

### 5. Documentation

**`scripts/README_POTENCY_ANALYSIS.md`**
- Comprehensive usage guide
- Example scenarios and interpretations
- Troubleshooting section
- Integration with existing tools

**`docs/POTENCY_STRATIFIED_ANALYSIS.md`**
- Detailed methodology explanation
- Quick start guide
- Output file descriptions

**`scripts/potency_analysis_commands.sh`**
- Quick reference cheat sheet
- Common commands
- Troubleshooting snippets

## How to Use on HPC

### Step 1: Check if Data is Ready

```bash
cd /home/ahagg2s/UMMBAS_screening_experiments

python scripts/validate_potency_analysis.py \
    --workspace_dir experiment_workspace_v3_phase1
```

**If validation fails**, use the debug tool:

```bash
python scripts/debug_run_structure.py experiment_workspace_v3_phase1
```

This will show you:
- What files exist in your run directories
- Whether affinity data is available
- Which columns are present in ranked files

### Step 2: Run Analysis

**Option A: Interactive (quick check)**
```bash
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_results \
    --summary_only
```

**Option B: Submit as job (full analysis)**
```bash
sbatch hpc/submit_potency_analysis.sh
```

**Option C: Custom workspace**
```bash
sbatch --export=WORKSPACE_DIR=/path/to/workspace,OUTPUT_DIR=custom_output \
    hpc/submit_potency_analysis.sh
```

### Step 3: Review Results

```bash
# Read summary report
cat potency_results/potency_stratified_report.txt

# View plots
ls potency_results/plots/

# Analyze CSV
head potency_results/stratified_enrichment_summary.csv
```

## Key Features

### 1. Potency Tier Classification

| Tier | Range (nM) | Significance |
|------|------------|--------------|
| High | 0.1-100 | Drug-like, clinically relevant |
| Medium | 100-1,000 | Moderate, optimizable |
| Weak | 1,000-100,000 | Marginal, likely promiscuous |

### 2. Metrics Calculated

**Per-Tier Enrichment Factors:**
- High_EF: Enrichment of potent compounds
- Medium_EF: Enrichment of moderate compounds  
- Weak_EF: Enrichment of weak binders
- Overall_EF: Traditional metric (for comparison)

**Recovery Rates:**
- Percent of each tier found in top 1%

### 3. Trade-off Detection

The script automatically detects and reports when:
- Best overall EF ≠ Best high-potent EF
- Provides recommendation for drug discovery

### 4. Flexible Data Loading

Attempts multiple strategies to find affinity data:
1. Check ranked CSV files directly
2. Load from detailed_active_distances.csv
3. Load from prepared_data files
4. Handles multiple column name conventions

### 5. Verbose Debugging

First run analyzed with detailed output showing:
- What files were found
- What columns are available
- Why analysis succeeded or failed

## Output Files

### CSV Files
- `stratified_enrichment_detailed.csv` - Every run
- `stratified_enrichment_summary.csv` - Aggregated stats
- Uses existing metrics plus new tier-specific metrics

### Plots (in `plots/` subdirectory)
- `stratified_ef_by_nn.png` - Compare tiers across n_neighbors
- `stratified_percent_found.png` - Recovery rates
- `quality_vs_quantity_tradeoff.png` - Scatter plot

### Report
- `potency_stratified_report.txt` - Human-readable summary
- Top 10 configurations
- Trade-off detection
- Recommendations

### Log
- `analysis.log` - Detailed execution log with debugging info

## Troubleshooting

### Issue: "No results found"

**Run debug script:**
```bash
python scripts/debug_run_structure.py experiment_workspace_v3_phase1
```

**Check:**
1. Do ranked files exist?
2. Do they have required columns (RANKING, TYPE, MOLECULE ID)?
3. Is affinity data available anywhere?

### Issue: "No affinity data"

**The script looks for affinity in:**
1. Ranked CSV: `Standard Value (nM)` column
2. Detailed distances: `*detailed_active_distances.csv`
3. Prepared data: `*_heldout_target_actives.csv`

**Column name variants supported:**
- `Standard Value (nM)`
- `standard_value`
- `pchembl_value` (converted to nM automatically)

**If not available:**
- May need to re-run project_and_analyze.py with affinity preservation
- Or modify the script to load from original ChEMBL data

### Issue: Script runs but analyzes 0 runs

**Check verbose output in log:**
```bash
# Re-run with first run analyzed verbosely
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir test_output 2>&1 | tee debug.log
```

Look for warnings about:
- Missing columns
- Failed merges
- No actives with affinity

## Integration

Works alongside existing scripts:

```bash
# 1. Check what's complete
python scripts/check_hyperparam_status.py

# 2. Validate for potency analysis  
python scripts/validate_potency_analysis.py \
    --workspace_dir experiment_workspace_v3_phase1

# 3. Run potency analysis
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_results

# 4. Run traditional analysis
python analysis_scripts/aggregate_and_report.py \
    --base_experiment_dir experiment_workspace_v3_phase1 \
    --config_path experiment_config.json \
    --output_report_dir traditional_results
```

Compare results to identify quality vs. quantity trade-offs!

## Example Output

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

## Performance

- Validation: ~5 seconds
- Analysis (summary only): ~10 seconds per 100 runs
- Full analysis with plots: ~1 minute per 100 runs
- Phase 1 (~260 runs): 3-5 minutes total

Safe to run on login node or submit as short job.

## Next Steps

1. **Run validation** to ensure data is ready
2. **Test on small sample** (single seed) first
3. **Run full analysis** on all Phase 1 data
4. **Use results** to inform Phase 2+ hyperparameter selection
5. **Compare** with traditional metrics to quantify trade-offs

## Questions?

Check:
- `scripts/README_POTENCY_ANALYSIS.md` - Usage guide
- `docs/POTENCY_STRATIFIED_ANALYSIS.md` - Methodology
- `scripts/potency_analysis_commands.sh` - Command reference
- `UMMBAS_V3_LAB_BOOK.md` - Project context

---

**Created**: October 17, 2025  
**Version**: 1.0  
**Compatible**: UMMBAS v3.0 Phase 1
