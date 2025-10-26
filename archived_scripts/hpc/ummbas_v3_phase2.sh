#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00

# ============================================================================
# UMMBAS v3.0 - Phase 2 Cutoff Analysis - HPC Execution Script
# ============================================================================
# Phase 2: Affinity Cutoff Sensitivity Analysis [REORDERED - was Phase 4]
# 
# This script runs a single Phase 2 cutoff configuration by REUSING Phase 1
# similarity spaces and DR models, only rerunning the ranking with different
# affinity cutoffs.
#
# Arguments:
#   $1: Path to the Phase 2 config JSON file
#   $2: Phase 1 workspace path (source data)
#   $3: Phase 2 workspace path (output)
#
# COMPUTATIONAL STRATEGY:
# - REUSES Phase 1 similarity spaces (features/fingerprints)
# - REUSES Phase 1 DR models (fitted PCA/UMAP)
# - ONLY reruns ranking with different affinity cutoffs
# - Expected runtime: ~10-15 min (vs ~2-3 hours for full pipeline)
#
# Note: Job name, output, and error files are set by the submission script
# ============================================================================

CONFIG_FILE=$1
PHASE1_WORKSPACE=$2
PHASE2_WORKSPACE=$3

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "UMMBAS v3.0 - PHASE 2 CUTOFF ANALYSIS"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "---"
echo "Configuration File: $CONFIG_FILE"
echo "Phase 1 Workspace: $PHASE1_WORKSPACE"
echo "Phase 2 Workspace: $PHASE2_WORKSPACE"
echo "========================================================================"

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# Define orchestrator script path
ORCHESTRATOR_SCRIPT="$(pwd)/scripts/run_phase2_cutoff_analysis.py"

# Validate files and directories exist
if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then 
    echo "ERROR: Orchestrator script not found at: ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then 
    echo "ERROR: Configuration file not found at: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "${PHASE1_WORKSPACE}" ]; then 
    echo "ERROR: Phase 1 workspace not found at: ${PHASE1_WORKSPACE}"
    exit 1
fi

# Create Phase 2 workspace if needed
mkdir -p "${PHASE2_WORKSPACE}"

# --- Run the Phase 2 Orchestrator ---
echo ""
echo "Starting Phase 2 cutoff analysis (data reuse mode)..."
echo "----------------------------------------------------------------------"

python -u "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --phase1_workspace "${PHASE1_WORKSPACE}" \
    --phase2_workspace "${PHASE2_WORKSPACE}"

EXIT_CODE=$?

echo "----------------------------------------------------------------------"
echo ""
echo "========================================================================"
echo "Phase 2 analysis finished with exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "========================================================================"

exit ${EXIT_CODE}
