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

# --- Submit All Jobs ---
echo "Starting job submission..."
echo "------------------------------------------------------------"

JOB_COUNT=0
FAILED_COUNT=0

# Each config file already has a seed in it, so we just submit once per file
for config_file in "${CONFIG_DIR}"/*.json; do
    config_basename=$(basename "${config_file}" .json)
    job_name="UMMBAS_v3_phase1_rerun_${config_basename}"
    
    # Extract seed from config filename (format: ..._seed42.json)
    seed=$(echo "${config_basename}" | grep -oP 'seed\K\d+' || echo "unknown")
    
    # Submit the SLURM job with the seed from the config
    sbatch --job-name="${job_name}" \
           --output="slurm_logs/${job_name}_%j.out" \
           --error="slurm_logs/${job_name}_%j.err" \
           "${SLURM_SCRIPT}" "${seed}" "${config_file}"
    
    if [ $? -eq 0 ]; then
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "ERROR: Failed to submit job for ${config_file}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Jobs submitted: ${JOB_COUNT}"
echo "Jobs failed: ${FAILED_COUNT}"
echo "============================================================"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1"
echo ""
echo "After completion, extract best configs with:"
echo "  python extract_phase1_best_configs.py --workspace experiment_workspace_v3_phase1"
echo "============================================================"
