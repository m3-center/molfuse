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
WORKSPACE_DIR="experiment_workspace_v3_phase1"

# Dry run mode: set to "true" to preview what would be submitted without actually submitting
DRY_RUN="${DRY_RUN:-false}"

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
echo "Workspace Directory: ${WORKSPACE_DIR}"
echo "Number of Configs: ${NUM_CONFIGS}"
echo "Total Jobs: ${TOTAL_JOBS} (expected: 455)"
echo ""
if [ "${DRY_RUN}" = "true" ]; then
    echo "***** DRY RUN MODE *****"
    echo "Will preview submissions without actually submitting to SLURM"
    echo ""
fi
echo "CRITICAL CHANGES FROM ORIGINAL:"
echo "  - MF cloud deduplicated (2.23× → 1.0×)"
echo "  - Features UMAP nn: [10, 20, 50, 100, 500] (was [3, 5, 10, 20])"
echo "  - Fixed seed DISABLED (10× UMAP speedup)"
echo "  - Fingerprints also rerun (same duplication issue)"
echo "============================================================"

if [ "${DRY_RUN}" != "true" ]; then
    read -p "Proceed with submission? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 0
    fi
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
SKIPPED_COUNT=0

# Each config file already has a seed in it, so we just submit once per file
for config_file in "${CONFIG_DIR}"/*.json; do
    config_basename=$(basename "${config_file}" .json)
    job_name="UMMBAS_v3_phase1_rerun_${config_basename}"
    
    # Extract seed from config filename (format: ..._seed42.json)
    seed=$(echo "${config_basename}" | grep -oP 'seed\K\d+' || echo "unknown")
    
    # Check if experiment already completed by looking for ranking metrics CSV
    # Pattern matches the specific workspace directory structure created by this config
    # Example: experiment_workspace_v3_phase1/run_seed46_config_tyro_features_pca_dim10_seed46/TyrosineProteinKinaseABL1_P00519/results/features/dim_10/PCA/*_ranking_metrics.csv
    workspace_pattern="${WORKSPACE_DIR}/run_seed${seed}_${config_basename}/*/*/*/*/*/*_ranking_metrics.csv"
    
    if ls ${workspace_pattern} 2>/dev/null | grep -q .; then
        echo "  ⊙ ${config_basename} -> SKIPPED (already completed)"
        echo "----------------------------------------" >> "${SUBMISSION_LOG}"
        echo "Config: ${config_basename}" >> "${SUBMISSION_LOG}"
        echo "Status: SKIPPED (ranking metrics found)" >> "${SUBMISSION_LOG}"
        echo "Matched: $(ls ${workspace_pattern} 2>/dev/null | head -1)" >> "${SUBMISSION_LOG}"
        echo "" >> "${SUBMISSION_LOG}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    
    # Log what we're about to submit
    echo "----------------------------------------" >> "${SUBMISSION_LOG}"
    echo "Config: ${config_basename}" >> "${SUBMISSION_LOG}"
    echo "Job Name: ${job_name}" >> "${SUBMISSION_LOG}"
    echo "Seed: ${seed}" >> "${SUBMISSION_LOG}"
    echo "Config File: ${config_file}" >> "${SUBMISSION_LOG}"
    echo "Command: sbatch --job-name=\"${job_name}\" \"${SLURM_SCRIPT}\" \"${seed}\" \"${config_file}\"" >> "${SUBMISSION_LOG}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        # Dry run: just log what would be submitted
        echo "DRY RUN - Would submit: ${config_basename}" >> "${SUBMISSION_LOG}"
        echo "Status: DRY RUN (not submitted)" >> "${SUBMISSION_LOG}"
        echo "  [DRY RUN] ${config_basename}"
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        # Real submission: Submit the SLURM job with the seed from the config
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
    fi
    echo "" >> "${SUBMISSION_LOG}"
done

echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "Submission Complete" | tee -a "${SUBMISSION_LOG}"
echo "Finished: $(date)" | tee -a "${SUBMISSION_LOG}"
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "Jobs submitted: ${JOB_COUNT}" | tee -a "${SUBMISSION_LOG}"
echo "Jobs skipped (already completed): ${SKIPPED_COUNT}" | tee -a "${SUBMISSION_LOG}"
echo "Jobs failed: ${FAILED_COUNT}" | tee -a "${SUBMISSION_LOG}"
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "DETAILED LOG SAVED TO: ${SUBMISSION_LOG}" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "Monitor progress with:" | tee -a "${SUBMISSION_LOG}"
echo "  squeue -u \$USER" | tee -a "${SUBMISSION_LOG}"
echo "  squeue -u \$USER | wc -l" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "Check completed experiments:" | tee -a "${SUBMISSION_LOG}"
echo "  ls ${WORKSPACE_DIR}/run_seed*_*/*/*/*/*/*_ranking_metrics.csv | wc -l" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
echo "Check submitted jobs:" | tee -a "${SUBMISSION_LOG}"
echo "  sacct -u \$USER -S $(date +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName%50,State" | tee -a "${SUBMISSION_LOG}"
echo "" | tee -a "${SUBMISSION_LOG}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "============================================================" | tee -a "${SUBMISSION_LOG}"
    echo "DRY RUN COMPLETED - No jobs were actually submitted" | tee -a "${SUBMISSION_LOG}"
    echo "To submit for real, run: bash ${0}" | tee -a "${SUBMISSION_LOG}"
    echo "============================================================" | tee -a "${SUBMISSION_LOG}"
fi
echo "============================================================" | tee -a "${SUBMISSION_LOG}"
