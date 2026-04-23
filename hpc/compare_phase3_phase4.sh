#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=compare_phase3_phase4
#SBATCH --output=slurm_logs/compare_phase3_phase4_%j.out
#SBATCH --error=slurm_logs/compare_phase3_phase4_%j.err

# Phase 3 vs Phase 4 Comparison
# Generates overlay plots comparing MF size effects across Phase 3 (single target ablation)
# and Phase 4 (cross-target generalization).
#
# Prerequisites:
#   - Phase 3 post-analysis must be complete (phase3_summary_aggregated.csv and phase3_summary_stratified.csv)
#   - Phase 4 post-analysis must be complete (phase4_summary_aggregated.csv and phase4_summary_stratified.csv)
#
# Usage:
#   sbatch hpc/compare_phase3_phase4.sh [PHASE3_OUTPUT_DIR] [PHASE4_OUTPUT_DIR] [COMPARISON_OUTPUT_DIR]
#
# Example:
#   sbatch hpc/compare_phase3_phase4.sh \
#       reporting/phase3_post_analysis \
#       reporting/phase4_post_analysis \
#       reporting/phase3_phase4_comparison
#
# Output structure:
#   <COMPARISON_OUTPUT_DIR>/
#       phase3_phase4_overlay_overall_ef1.{png,pdf}
#       phase3_phase4_overlay_high_potency_ef1.{png,pdf}
#       phase3_phase4_overlay_bedroc20.{png,pdf}
#       phase3_phase4_overlay_bedroc160.{png,pdf}
#       phase3_phase4_overlay_roc_pr.{png,pdf}
#       phase3_phase4_5panel_comparison.{png,pdf}

PHASE3_OUTPUT_DIR="${1:-reporting/phase3_post_analysis}"
PHASE4_OUTPUT_DIR="${2:-reporting/phase4_post_analysis}"
COMPARISON_OUTPUT_DIR="${3:-reporting/phase3_phase4_comparison}"

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "Phase 3 vs Phase 4 Comparison"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Phase 3 Output Dir: ${PHASE3_OUTPUT_DIR}"
echo "Phase 4 Output Dir: ${PHASE4_OUTPUT_DIR}"
echo "Comparison Output Dir: ${COMPARISON_OUTPUT_DIR}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 16G"
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
# Input Validation
# ============================================================================
echo "=========================================="
echo "Validating Input Files"
echo "=========================================="

PHASE3_AGG="${PHASE3_OUTPUT_DIR}/phase3_summary_aggregated.csv"
PHASE3_STRAT="${PHASE3_OUTPUT_DIR}/phase3_summary_stratified.csv"
PHASE4_AGG="${PHASE4_OUTPUT_DIR}/phase4_summary_aggregated.csv"
PHASE4_STRAT="${PHASE4_OUTPUT_DIR}/phase4_summary_stratified.csv"

MISSING_FILES=0

# Check Phase 3 files
if [[ ! -f "${PHASE3_AGG}" ]]; then
    echo "ERROR: Phase 3 aggregated file not found: ${PHASE3_AGG}"
    MISSING_FILES=1
else
    echo "✓ Phase 3 aggregated: ${PHASE3_AGG} ($(stat -c%s "${PHASE3_AGG}" | numfmt --to=iec-i --suffix=B))"
fi

if [[ ! -f "${PHASE3_STRAT}" ]]; then
    echo "ERROR: Phase 3 stratified file not found: ${PHASE3_STRAT}"
    MISSING_FILES=1
else
    echo "✓ Phase 3 stratified: ${PHASE3_STRAT} ($(stat -c%s "${PHASE3_STRAT}" | numfmt --to=iec-i --suffix=B))"
fi

# Check Phase 4 files
if [[ ! -f "${PHASE4_AGG}" ]]; then
    echo "ERROR: Phase 4 aggregated file not found: ${PHASE4_AGG}"
    MISSING_FILES=1
else
    echo "✓ Phase 4 aggregated: ${PHASE4_AGG} ($(stat -c%s "${PHASE4_AGG}" | numfmt --to=iec-i --suffix=B))"
fi

if [[ ! -f "${PHASE4_STRAT}" ]]; then
    echo "ERROR: Phase 4 stratified file not found: ${PHASE4_STRAT}"
    MISSING_FILES=1
else
    echo "✓ Phase 4 stratified: ${PHASE4_STRAT} ($(stat -c%s "${PHASE4_STRAT}" | numfmt --to=iec-i --suffix=B))"
fi

if [[ ${MISSING_FILES} -eq 1 ]]; then
    echo ""
    echo "ERROR: One or more required input files are missing."
    echo "Please ensure Phase 3 and Phase 4 post-analysis have completed successfully."
    echo ""
    echo "To run prerequisites:"
    echo "  sbatch hpc/phase3_analysis_suite.sh"
    echo "  sbatch hpc/phase4_post_analysis.sh --stratify"
    echo ""
    exit 1
fi

echo ""

# ============================================================================
# Run Comparison
# ============================================================================
echo "=========================================="
echo "Generating Phase 3 vs Phase 4 Comparison Plots"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

python scripts/compare_phase3_phase4.py \
    --phase3_aggregated "${PHASE3_AGG}" \
    --phase3_stratified "${PHASE3_STRAT}" \
    --phase4_aggregated "${PHASE4_AGG}" \
    --phase4_stratified "${PHASE4_STRAT}" \
    --output_dir "${COMPARISON_OUTPUT_DIR}"

COMPARISON_EXIT=$?

echo ""
echo "Comparison Exit Code: ${COMPARISON_EXIT}"
echo "End Time: $(date)"
echo ""

if [[ ${COMPARISON_EXIT} -ne 0 ]]; then
    echo "ERROR: Comparison failed with exit code ${COMPARISON_EXIT}"
    exit ${COMPARISON_EXIT}
fi

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
echo "=========================================="
echo "Comparison Complete"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo ""
echo "Output Directory: ${COMPARISON_OUTPUT_DIR}"
echo ""
echo "Generated Files:"
ls -lh "${COMPARISON_OUTPUT_DIR}"/*.{png,pdf} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Plot Summary:"
echo "  - Overall EF@1% overlay"
echo "  - High-potency EF@1% overlay"
echo "  - BEDROC (α=20) overlay"
echo "  - BEDROC (α=160) overlay"
echo "  - ROC-AUC & PR-AUC overlay"
echo "  - 5-panel comprehensive comparison"
echo "=========================================="

exit 0
