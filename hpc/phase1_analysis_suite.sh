#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=phase1_analysis
#SBATCH --output=slurm_logs/phase1_analysis_%j.out
#SBATCH --error=slurm_logs/phase1_analysis_%j.err

# Phase 1 Analysis Suite
# Runs post-analysis and stratified scores for Phase 1 experiments
#
# Usage:
#   sbatch hpc/phase1_analysis_suite.sh [WORKSPACE_DIR] [OUTPUT_BASE_DIR]
#
# Example:
#   sbatch hpc/phase1_analysis_suite.sh experiment_workspace_v4 reporting
#
# Output structure:
#   <OUTPUT_BASE_DIR>/phase1_post_analysis/
#   <OUTPUT_BASE_DIR>/phase1_stratified/

WORKSPACE_DIR="${1:-experiment_workspace_v4}"
OUTPUT_BASE_DIR="${2:-reporting}"

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "Phase 1 Analysis Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Workspace: ${WORKSPACE_DIR}"
echo "Output Base Dir: ${OUTPUT_BASE_DIR}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 128G"
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
# STEP 1: Phase 1 Post-Analysis
# ============================================================================
echo "=========================================="
echo "STEP 1: Phase 1 Post-Analysis"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

POST_ANALYSIS_OUTPUT="${OUTPUT_BASE_DIR}/phase1_post_analysis"

python scripts/phase1_post_analysis.py \
    --workspace_dir "${WORKSPACE_DIR}" \
    --phase phase1 \
    --output_dir "${POST_ANALYSIS_OUTPUT}" \
    --metrics ef1

POST_ANALYSIS_EXIT=$?

echo ""
echo "Phase 1 Post-Analysis Exit Code: ${POST_ANALYSIS_EXIT}"
echo "End Time: $(date)"
echo ""

if [[ ${POST_ANALYSIS_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 1 Post-Analysis failed with exit code ${POST_ANALYSIS_EXIT}"
    echo "Aborting analysis suite."
    exit ${POST_ANALYSIS_EXIT}
fi

# ============================================================================
# STEP 2: Phase 1 Stratified Scores
# ============================================================================
echo "=========================================="
echo "STEP 2: Phase 1 Stratified Scores"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

STRATIFIED_OUTPUT="${OUTPUT_BASE_DIR}/phase1_stratified"

# Use conservative worker count to avoid OOM (max 8 workers for memory-intensive operations)
STRATIFIED_WORKERS=$(( SLURM_CPUS_PER_TASK < 8 ? SLURM_CPUS_PER_TASK : 8 ))

python scripts/phase1_stratified_scores.py \
    --workspace_dir "${WORKSPACE_DIR}" \
    --phase phase1 \
    --output_dir "${STRATIFIED_OUTPUT}" \
    --n_workers ${STRATIFIED_WORKERS}

STRATIFIED_EXIT=$?

echo ""
echo "Phase 1 Stratified Scores Exit Code: ${STRATIFIED_EXIT}"
echo "End Time: $(date)"
echo ""

if [[ ${STRATIFIED_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 1 Stratified Scores failed with exit code ${STRATIFIED_EXIT}"
    exit ${STRATIFIED_EXIT}
fi

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "=========================================="
echo "Analysis Suite Complete"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo ""
echo "Output Directories:"
echo "  - Post-Analysis: ${POST_ANALYSIS_OUTPUT}"
echo "  - Stratified Scores: ${STRATIFIED_OUTPUT}"
echo ""
echo "Exit Codes:"
echo "  - Post-Analysis: ${POST_ANALYSIS_EXIT}"
echo "  - Stratified Scores: ${STRATIFIED_EXIT}"
echo "=========================================="

exit 0
