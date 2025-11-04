#!/bin/bash
#SBATCH --job-name=potency_analysis
#SBATCH --output=logs/potency_analysis_%j.out
#SBATCH --error=logs/potency_analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Potency-Stratified Enrichment Analysis - HPC Job Script
# 
# This script runs the potency-stratified enrichment analysis on completed
# Phase 1 experiments. Can be submitted while experiments are still running
# to get interim results.
#
# Usage:
#   sbatch hpc/submit_potency_analysis.sh
#
# Or with custom workspace:
#   sbatch --export=WORKSPACE_DIR=/path/to/workspace hpc/submit_potency_analysis.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "============================================================"
echo "Potency-Stratified Enrichment Analysis"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "============================================================"

# Configuration
WORKSPACE_DIR=${WORKSPACE_DIR:-"experiment_workspace_v3_phase1"}
OUTPUT_DIR=${OUTPUT_DIR:-"analysis_results/potency_stratified_$(date +%Y%m%d_%H%M%S)"}
SEED=${SEED:-""}  # Empty = analyze all seeds

# Create logs directory
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc  # or source /path/to/conda/etc/profile.d/conda.sh
conda activate ummbas-screening

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Step 1: Validate data
echo ""
echo "Step 1: Validating workspace data..."
echo "------------------------------------------------------------"
python scripts/validate_potency_analysis.py \
    --workspace_dir "$WORKSPACE_DIR" || {
    echo "⚠️  Validation warnings detected, but continuing..."
}

# Step 2: Run analysis
echo ""
echo "Step 2: Running potency-stratified enrichment analysis..."
echo "------------------------------------------------------------"
echo "Workspace: $WORKSPACE_DIR"
echo "Output: $OUTPUT_DIR"

if [ -n "$SEED" ]; then
    echo "Analyzing seed: $SEED"
    python scripts/analyze_potency_stratified_enrichment.py \
        --workspace_dir "$WORKSPACE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED"
else
    echo "Analyzing all seeds"
    python scripts/analyze_potency_stratified_enrichment.py \
        --workspace_dir "$WORKSPACE_DIR" \
        --output_dir "$OUTPUT_DIR"
fi

# Step 3: Report results
echo ""
echo "============================================================"
echo "Analysis Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

if [ -f "$OUTPUT_DIR/potency_stratified_report.txt" ]; then
    echo "Key Findings:"
    echo "------------------------------------------------------------"
    head -n 50 "$OUTPUT_DIR/potency_stratified_report.txt"
    echo ""
    echo "Full report: $OUTPUT_DIR/potency_stratified_report.txt"
fi

echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.txt 2>/dev/null || true

if [ -d "$OUTPUT_DIR/plots" ]; then
    echo ""
    echo "Plots:"
    ls -lh "$OUTPUT_DIR/plots"/*.png 2>/dev/null || true
fi

echo ""
echo "End time: $(date)"
echo "============================================================"
