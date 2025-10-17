#!/bin/bash
# Potency-Stratified Enrichment Analysis - Quick Reference Commands
# =================================================================

# ON YOUR HPC (in your UMMBAS_screening_experiments directory)
# =============================================================

# 1. VALIDATE - Check if data is ready for analysis
python scripts/validate_potency_analysis.py \
    --workspace_dir experiment_workspace_v3_phase1

# 2. QUICK CHECK - Fast summary while experiments are running
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_quick \
    --summary_only

# 3. FULL ANALYSIS - Complete analysis with plots
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_analysis_results

# 4. SUBMIT AS JOB - For long-running analysis
sbatch hpc/submit_potency_analysis.sh

# 5. ANALYZE SPECIFIC SEED
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --seed 42 \
    --output_dir potency_seed42

# 6. VIEW RESULTS
cat potency_analysis_results/potency_stratified_report.txt
ls potency_analysis_results/plots/

# 7. COMPARE CONFIGURATIONS - Extract top performers
python -c "
import pandas as pd
df = pd.read_csv('potency_analysis_results/stratified_enrichment_summary.csv')
print('\nTop 5 by High-Potent EF:')
print(df.nlargest(5, 'High_EF_mean')[['dr_method', 'n_neighbors', 'min_dist', 'High_EF_mean', 'Overall_EF_mean']])
"

# COMMON TROUBLESHOOTING
# ======================

# Check if experiments are complete
python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1

# List completed runs
ls experiment_workspace_v3_phase1/*/results/*/dim_*/*/*-RANKED.csv | wc -l

# Check affinity data in a ranked file
head -n 2 experiment_workspace_v3_phase1/run_seed42_*/*/results/*/dim_*/*/*-RANKED.csv

# Monitor job progress
squeue -u $USER
tail -f logs/potency_analysis_*.out

# ENVIRONMENT SETUP
# =================

# Activate environment
conda activate ummbas-screening

# Check Python packages
python -c "import pandas, numpy, matplotlib, seaborn, tqdm; print('All packages available')"

# Install missing packages if needed
pip install tqdm seaborn

# FILE LOCATIONS
# ==============
# Main script:       scripts/analyze_potency_stratified_enrichment.py
# Validation:        scripts/validate_potency_analysis.py  
# HPC submission:    hpc/submit_potency_analysis.sh
# Documentation:     scripts/README_POTENCY_ANALYSIS.md
#                    docs/POTENCY_STRATIFIED_ANALYSIS.md

# OUTPUT FILES
# ============
# Summary report:        potency_stratified_report.txt
# Detailed results:      stratified_enrichment_detailed.csv
# Aggregated stats:      stratified_enrichment_summary.csv
# Plots:                 plots/*.png
# Log file:              analysis.log
