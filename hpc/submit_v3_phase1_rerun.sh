#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 1 RERUN Job Submission Script
# =============================================================================
# Phase 1 RERUN: Hyperparameter Sweep with DEDUPLICATED MF Cloud Data
# - Target: TyrosineProteinKinaseABL1 (Tyro)
# - Dimensions: 2D, 5D, 10D (features), 2D (fingerprints)
# - Representations: features (PCA + UMAP), fingerprints (PCA + UMAP-Jaccard)
# 
# CRITICAL CHANGES FROM ORIGINAL PHASE 1:
# - MF cloud deduplicated (2.23× → 1.0× for ABL1 Transferase)
# - Comparison test: 58.3% performance drop with clean data (EF@1% 43.78 → 18.27)
# - REVISED features UMAP grid: [10, 20, 50, 100, 500] nn (removed 3, 5)
# - DISABLED fixed seed for 10× UMAP speedup via multi-threading
# - Fingerprints also rerun (same duplication issue affects both)
#
# Expected: 455 total runs (390 features + 65 fingerprints)
# - Features-PCA: 15 configs (3 dims × 5 seeds)
# - Features-UMAP: 375 configs (3 dims × 5 nn × 5 md × 5 seeds)
# - Fingerprints-PCA: 5 configs (1 dim × 5 seeds)
# - Fingerprints-UMAP: 60 configs (1 dim × 3 nn × 4 md × 5 seeds)
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase1"
SLURM_SCRIPT="hpc/ummbas_v3_cpu.sh"

# NOTE: Each config file already contains a specific seed.
# We do NOT loop over seeds here - that would create duplicate jobs!

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 1 RERUN Submission"
echo "============================================================"

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script '${SLURM_SCRIPT}' not found."
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    echo "Run: python generate_phase1_configs.py"
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    echo "Run: python generate_phase1_configs.py"
    exit 1
fi

# Count config files
NUM_CONFIGS=$(ls -1 ${CONFIG_DIR}/*.json | wc -l)
TOTAL_JOBS=${NUM_CONFIGS}

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Number of Configs: ${NUM_CONFIGS}"
echo "Total Jobs: ${TOTAL_JOBS} (expected: 455)"
echo ""
echo "CRITICAL CHANGES FROM ORIGINAL:"
echo "  - MF cloud deduplicated (2.23× → 1.0×)"
echo "  - Features UMAP nn: [10, 20, 50, 100, 500] (was [3, 5, 10, 20])"
echo "  - Fixed seed DISABLED (10× UMAP speedup)"
echo "  - Fingerprints also rerun (same duplication issue)"
echo "============================================================"

read -p "Proceed with submission? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

# --- Create logs directory ---
mkdir -p slurm_logs

# Create detailed submission log
SUBMISSION_LOG="submission_log_$(date +%Y%m%d_%H%M%S).txt"
echo "========================================" | tee -a "${SUBMISSION_LOG}"
echo "SLURM Submission Log" | tee -a "${SUBMISSION_LOG}"
echo "Started: $(date)" | tee -a "${SUBMISSION_LOG}"
echo "Config Dir: ${CONFIG_DIR}" | tee -a "${SUBMISSION_LOG}"
echo "SLURM Script: ${SLURM_SCRIPT}" | tee -a "${SUBMISSION_LOG}"
echo "========================================" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"

# --- Submit All Jobs ---
echo "Starting job submission..." | tee -a "${SUBMISSION_LOG}"
echo "------------------------------------------------------------" | tee -a "${SUBMISSION_LOG}"

JOB_COUNT=0
FAILED_COUNT=0

# Each config file already has a seed in it, so we just submit once per file
for config_file in "${CONFIG_DIR}"/*.json; do
    config_basename=$(basename "${config_file}" .json)
    job_name="UMMBAS_v3_phase1_rerun_${config_basename}"
    
    # Extract seed from config filename (format: ..._seed42.json)
    seed=$(echo "${config_basename}" | grep -oP 'seed\K\d+' || echo "unknown")
    
    # Log what we're about to submit
    echo "----------------------------------------" >> "${SUBMISSION_LOG}"
    echo "Config: ${config_basename}" >> "${SUBMISSION_LOG}"
    echo "Job Name: ${job_name}" >> "${SUBMISSION_LOG}"
    echo "Seed: ${seed}" >> "${SUBMISSION_LOG}"
    echo "Config File: ${config_file}" >> "${SUBMISSION_LOG}"
    echo "Command: sbatch --job-name=\"${job_name}\" \"${SLURM_SCRIPT}\" \"${seed}\" \"${config_file}\"" >> "${SUBMISSION_LOG}"
    
    # Submit the SLURM job with the seed from the config (EXACTLY like original Phase 1)
    submit_output=$(sbatch --job-name="${job_name}" "${SLURM_SCRIPT}" "${seed}" "${config_file}" 2>&1)
    submit_status=$?
    
    echo "Submit Output: ${submit_output}" >> "${SUBMISSION_LOG}"
    echo "Exit Code: ${submit_status}" >> "${SUBMISSION_LOG}"
    
    if [ ${submit_status} -eq 0 ]; then
        job_id=$(echo "${submit_output}" | grep -oP 'Submitted batch job \K\d+')
        echo "Status: SUCCESS (Job ID: ${job_id})" >> "${SUBMISSION_LOG}"
        echo "  ✓ ${config_basename} -> Job ${job_id}"
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "Status: FAILED" >> "${SUBMISSION_LOG}"
        echo "  ✗ ${config_basename} FAILED: ${submit_output}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo "" >> "${SUBMISSION_LOG}"
done

echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "Submission Complete" | tee -a "${SUBMISSION_LOG}"
echo "Finished: $(date)" | tee -a "${SUBMISSION_LOG}"
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "Jobs submitted: ${JOB_COUNT}" | tee -a "${SUBMISSION_LOG}"
echo "Jobs failed: ${FAILED_COUNT}" | tee -a "${SUBMISSION_LOG}"
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "DETAILED LOG SAVED TO: ${SUBMISSION_LOG}" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "Monitor progress with:" | tee -a "${SUBMISSION_LOG}"
echo "  squeue -u \$USER" | tee -a "${SUBMISSION_LOG}"
echo "  squeue -u \$USER | wc -l" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "Check submitted jobs:" | tee -a "${SUBMISSION_LOG}"
echo "  sacct -u \$USER -S $(date +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName%50,State" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
