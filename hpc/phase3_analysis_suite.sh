#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=phase3_analysis
#SBATCH --output=slurm_logs/phase3_analysis_%j.out
#SBATCH --error=slurm_logs/phase3_analysis_%j.err

# Phase 3 Analysis Suite
# Runs post-analysis for Phase 3 experiments (MF Cloud Ablation)
#
# Usage:
#   sbatch hpc/phase3_analysis_suite.sh [WORKSPACE_DIR] [OUTPUT_BASE_DIR] [PHASE3_RUN_NAME]
#
# Example:
#   sbatch hpc/phase3_analysis_suite.sh experiment_workspace_v4 reporting mf_ablation
#
# Output structure:
#   <OUTPUT_BASE_DIR>/phase3_post_analysis/

WORKSPACE_DIR="${1:-experiment_workspace_v4}"
OUTPUT_BASE_DIR="${2:-reporting}"
PHASE3_RUN_NAME="${3:-mf_ablation}"

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "Phase 3 Analysis Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Workspace: ${WORKSPACE_DIR}"
echo "Output Base Dir: ${OUTPUT_BASE_DIR}"
echo "Phase 3 Run Name: ${PHASE3_RUN_NAME}"
echo "Stratify by potency: enabled"
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
# Phase 3 Post-Analysis (MF Cloud Ablation)
# ============================================================================
echo "=========================================="
echo "Phase 3 Post-Analysis (MF Cloud Ablation)"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

POST_ANALYSIS_OUTPUT="${OUTPUT_BASE_DIR}/phase3_post_analysis"

python scripts/phase3_post_analysis.py \
    --workspace_dir "${WORKSPACE_DIR}" \
    --phase3_run_name "${PHASE3_RUN_NAME}" \
    --output_dir "${POST_ANALYSIS_OUTPUT}" \
    --stratify

POST_ANALYSIS_EXIT=$?

echo ""
echo "Phase 3 Post-Analysis Exit Code: ${POST_ANALYSIS_EXIT}"
echo "End Time: $(date)"
echo ""

if [[ ${POST_ANALYSIS_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 3 Post-Analysis failed with exit code ${POST_ANALYSIS_EXIT}"
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
