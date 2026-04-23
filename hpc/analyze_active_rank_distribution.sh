#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=active_ranks
#SBATCH --output=slurm_logs/active_rank_dist_%j.out
#SBATCH --error=slurm_logs/active_rank_dist_%j.err

# ============================================================================
# Active Rank Distribution Analysis
# ============================================================================
# Diagnoses BEDROC vs EF1% discrepancy by analyzing where actives are ranked.
#
# Scientific Question:
#   If EF1% is high (20-50) but BEDROC is moderate (~0.55), we expect a
#   bimodal distribution: some actives ranked very early (driving EF1%),
#   others ranked poorly (dragging down BEDROC).
#
# Usage:
#   sbatch hpc/analyze_active_rank_distribution.sh [WORKSPACE_DIR] [OUTPUT_DIR]
#
# Example:
#   sbatch hpc/analyze_active_rank_distribution.sh experiment_workspace_v4 reporting/rank_analysis
#
# Output:
#   - active_rank_summary.csv: Statistics for each run
#   - rank_distribution_*.png: Histograms showing rank distribution
#   - bimodality_diagnostic.png: Scatter plot of top1% vs bottom50%
# ============================================================================

WORKSPACE_DIR="${1:-experiment_workspace_v4}"
OUTPUT_DIR="${2:-reporting/rank_analysis}"

# Ensure directories exist
mkdir -p slurm_logs "${OUTPUT_DIR}" || true

echo "=========================================="
echo "Active Rank Distribution Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Workspace: ${WORKSPACE_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Verify Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================================
# RUN PYTHON SCRIPT
# ============================================================================

python scripts/analyze_active_rank_distribution.py \
    --workspace_dir "${WORKSPACE_DIR}" \
    --phase phase1 \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "=========================================="
echo "Active Rank Distribution Analysis Complete"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

exit ${EXIT_CODE}
