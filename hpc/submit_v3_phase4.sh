#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 3 Job Submission Script
# =============================================================================
# Phase 3: Cross-Protein Generalization Study
# - Targets: PyruvateKinaseM2 (Pyru), IsocitrateDehydrogenaseNADP (Iso)
# - Uses best configs from Phase 1 on new proteins
# - Total: 80 runs (16 configs × 5 seeds)
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase3_generalization"
SLURM_SCRIPT="hpc/ummbas_v3_cpu.sh"

# NOTE: Each config file already contains a specific seed.
# We do NOT loop over seeds here - that would create duplicate jobs!

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 3 Submission"
echo "============================================================"

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM script '${SLURM_SCRIPT}' not found."
    exit 1
fi

if [ ! -f "phase1_best_configs.json" ]; then
    echo "ERROR: phase1_best_configs.json not found."
    echo "Run: python extract_phase1_best_configs.py"
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    echo "Run: python generate_phase3_configs.py"
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    echo "Run: python generate_phase3_configs.py"
    exit 1
fi

# Count config files
NUM_CONFIGS=$(ls -1 ${CONFIG_DIR}/*.json | wc -l)
TOTAL_JOBS=${NUM_CONFIGS}

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Number of Configs: ${NUM_CONFIGS}"
echo "Total Jobs: ${TOTAL_JOBS} (expected: 80)"
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
    job_name="UMMBAS_v3_phase3_${config_basename}"
    
    # Extract seed from config filename (format: ..._seed42.json)
    seed=$(echo "${config_basename}" | grep -oP 'seed\K\d+' || echo "unknown")
    
    # Submit the SLURM job with the seed from the config
    sbatch --job-name="${job_name}" "${SLURM_SCRIPT}" "${seed}" "${config_file}"
    
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
echo "  python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase3"
echo "============================================================"
