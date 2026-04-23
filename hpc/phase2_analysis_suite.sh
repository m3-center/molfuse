#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=phase2_analysis
#SBATCH --output=slurm_logs/phase2_analysis_%j.out
#SBATCH --error=slurm_logs/phase2_analysis_%j.err

# Phase 2 Analysis Suite
# Runs post-analysis for Phase 2 experiments (Affinity Cutoff Sensitivity)
#
# Usage:
#   sbatch hpc/phase2_analysis_suite.sh [WORKSPACE_DIR] [OUTPUT_BASE_DIR] [PHASE2_RUN_NAME]
#
# Example:
#   sbatch hpc/phase2_analysis_suite.sh experiment_workspace_v4 reporting cutoff_sweep
#
# Output structure:
#   <OUTPUT_BASE_DIR>/phase2_post_analysis/

WORKSPACE_DIR="${1:-experiment_workspace_v4}"
OUTPUT_BASE_DIR="${2:-reporting}"
PHASE2_RUN_NAME="${3:-cutoff_sweep}"

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "Phase 2 Analysis Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Workspace: ${WORKSPACE_DIR}"
echo "Output Base Dir: ${OUTPUT_BASE_DIR}"
echo "Phase 2 Run Name: ${PHASE2_RUN_NAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 200G"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV:-molfuse}"
if [[ -z "${CONDA_ACTIVATE:-}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_ACTIVATE="$(dirname "$CONDA_EXE")/activate"
fi
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${CONDA_ENV:-molfuse}"

# Verify Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================================
# Phase 2 Post-Analysis (Cutoff Sensitivity)
# ============================================================================
echo "=========================================="
echo "Phase 2 Post-Analysis (Cutoff Sensitivity)"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

POST_ANALYSIS_OUTPUT="${OUTPUT_BASE_DIR}/phase2_post_analysis"

python scripts/phase2_post_analysis.py \
    --workspace_dir "${WORKSPACE_DIR}" \
    --phase2_run_name "${PHASE2_RUN_NAME}" \
    --output_dir "${POST_ANALYSIS_OUTPUT}"

POST_ANALYSIS_EXIT=$?

echo ""
echo "Phase 2 Post-Analysis Exit Code: ${POST_ANALYSIS_EXIT}"
echo "End Time: $(date)"
echo ""

if [[ ${POST_ANALYSIS_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 2 Post-Analysis failed with exit code ${POST_ANALYSIS_EXIT}"
    echo "Aborting analysis suite."
    exit ${POST_ANALYSIS_EXIT}
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
echo "Output Directory:"
echo "  - Post-Analysis: ${POST_ANALYSIS_OUTPUT}"
echo ""
echo "Exit Code:"
echo "  - Post-Analysis: ${POST_ANALYSIS_EXIT}"
echo ""
echo "Generated Files:"
ls -lh "${POST_ANALYSIS_OUTPUT}"/*.{png,pdf,csv,json} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo "=========================================="

exit 0
