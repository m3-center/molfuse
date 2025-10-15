#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 2 Job Submission Script
# =============================================================================
# Phase 2: MF Cloud Size Ablation Study
# - Target: TyrosineProteinKinaseABL1 (Tyro)
# - MF sizes: 0, 1K, 10K, 50K, 100K, 420K molecules
# - Uses best configs from Phase 1
# - Total: 60 runs (12 configs × 5 seeds)
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase2_ablation"
RANDOM_SEEDS=(42 43 44 45 46)
SLURM_SCRIPT="hpc/ummbas_v3_cpu.sh"

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 2 Submission"
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
    echo "Run: python generate_phase2_configs.py"
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    echo "Run: python generate_phase2_configs.py"
    exit 1
fi

# Count config files
NUM_CONFIGS=$(ls -1 ${CONFIG_DIR}/*.json | wc -l)
NUM_SEEDS=${#RANDOM_SEEDS[@]}
TOTAL_JOBS=$((NUM_CONFIGS * NUM_SEEDS))

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Number of Configs: ${NUM_CONFIGS}"
echo "Number of Seeds: ${NUM_SEEDS}"
echo "Total Jobs: ${TOTAL_JOBS} (expected: 60)"
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

for config_file in "${CONFIG_DIR}"/*.json; do
    for seed in "${RANDOM_SEEDS[@]}"; do
        config_basename=$(basename "${config_file}" .json)
        job_name="UMMBAS_v3_phase2_${config_basename}_s${seed}"
        
        # Submit the SLURM job
        sbatch --job-name="${job_name}" "${SLURM_SCRIPT}" "${seed}" "${config_file}"
        
        if [ $? -eq 0 ]; then
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            echo "ERROR: Failed to submit job for ${config_file}, seed=${seed}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done
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
echo "  python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase2"
echo "============================================================"
