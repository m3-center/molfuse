#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 2 Job Submission Script
# =============================================================================
# Phase 2: Affinity Cutoff Sensitivity Analysis [REORDERED - was Phase 4]
# - Target: TyrosineProteinKinaseABL1 (Tyro)
# - Cutoffs: 100 nM, 1 μM, 10 μM, 100 μM (4 cutoffs)
# - Configurations: PCA/features/5D + 2×UMAP/features/5D (3 configs)
# - Seeds: 42, 43, 44, 45, 46 (5 seeds)
# - Total: 60 runs (4 cutoffs × 3 configs × 5 seeds)
# 
# COMPUTATIONAL STRATEGY:
# - REUSES Phase 1 similarity spaces (features/fingerprints)
# - REUSES Phase 1 DR models (fitted PCA/UMAP)
# - ONLY reruns ranking with different affinity cutoffs
# - Expected runtime: ~10-15 min/run (vs hours for full pipeline)
#
# Usage: bash hpc/submit_v3_phase2.sh
# =============================================================================

echo "========================================================================"
echo "UMMBAS v3.0 - Phase 2 Cutoff Sensitivity - Job Submission"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase2_cutoff"
PHASE1_WORKSPACE="experiment_workspace_v3_phase1"
PHASE2_WORKSPACE="experiment_workspace_v3_phase2"
SLURM_SCRIPT="hpc/ummbas_v3_phase2.sh"

# --- Pre-submission Checks ---
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script '${SLURM_SCRIPT}' not found!"
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found!"
    echo "Please run: python generate_phase2_configs.py"
    exit 1
fi

if [ ! -d "${PHASE1_WORKSPACE}" ]; then
    echo "ERROR: Phase 1 workspace '${PHASE1_WORKSPACE}' not found!"
    echo "Phase 2 requires Phase 1 data to reuse!"
    exit 1
fi

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p "${PHASE2_WORKSPACE}"

# Count total configs
TOTAL_CONFIGS=$(find "${CONFIG_DIR}" -name "config_*.json" | wc -l)

echo "Configuration directory: ${CONFIG_DIR}"
echo "Phase 1 workspace: ${PHASE1_WORKSPACE}"
echo "Phase 2 workspace: ${PHASE2_WORKSPACE}"
echo "Number of configs found: ${TOTAL_CONFIGS}"
echo "Expected: 60 configs (4 cutoffs × 3 configs × 5 seeds)"
echo ""
echo "Note: Each config already contains seed and cutoff information."
echo "      Data reuse from Phase 1 reduces runtime to ~10-15 min per job."
echo ""
echo "========================================================================"
echo ""

# --- Submit All Jobs ---
# Counter for submitted jobs
SUBMITTED=0
FAILED=0

# Loop through all JSON config files
for CONFIG_FILE in "${CONFIG_DIR}"/config_*.json; do
    if [ ! -f "${CONFIG_FILE}" ]; then
        continue
    fi
    
    CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .json)
    JOB_NAME="UMMBAS_v3_phase2_${CONFIG_BASENAME}"
    
    # Submit the job
    SUBMIT_OUTPUT=$(sbatch \
        --job-name="${JOB_NAME}" \
        --output="slurm_logs/${JOB_NAME}_%j.out" \
        --error="slurm_logs/${JOB_NAME}_%j.err" \
        "${SLURM_SCRIPT}" "${CONFIG_FILE}" "${PHASE1_WORKSPACE}" "${PHASE2_WORKSPACE}" 2>&1)
    
    if [ $? -eq 0 ]; then
        JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'Submitted batch job \K\d+')
        echo "✓ ${CONFIG_BASENAME}: Job ${JOB_ID} submitted"
        ((SUBMITTED++))
    else
        echo "✗ ${CONFIG_BASENAME}: FAILED - ${SUBMIT_OUTPUT}"
        ((FAILED++))
    fi
done
echo ""
echo "========================================================================"
echo "Job Submission Summary"
echo "========================================================================"
echo "Successfully submitted: ${SUBMITTED}"
echo "Failed: ${FAILED}"
echo "Total: $((SUBMITTED + FAILED))"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: slurm_logs/"
echo ""
echo "After jobs complete, analyze Phase 2 results to determine optimal cutoff,"
echo "then proceed to Phase 3 (MF cloud ablation)."
echo ""
echo "End time: $(date)"
echo "========================================================================"
