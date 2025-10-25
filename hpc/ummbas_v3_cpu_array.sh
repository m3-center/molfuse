#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=170G
#SBATCH --time=0-24:00:00

# =============================================================================
# UMMBAS v3.0 - Job Array Worker Script
# =============================================================================
# This script is called by submit_v3_phase1_rerun_array.sh
# Each array task processes one config file
#
# Environment variables set by SLURM:
#   SLURM_ARRAY_TASK_ID: Index of this task (0 to N-1)
#   SLURM_ARRAY_JOB_ID: Parent job array ID
# =============================================================================

CONFIG_LIST_FILE=$1

# --- Validate Arguments ---
if [ -z "${CONFIG_LIST_FILE}" ]; then
    echo "ERROR: Missing config list file argument"
    echo "Usage: $0 <config_list_file>"
    exit 1
fi

if [ ! -f "${CONFIG_LIST_FILE}" ]; then
    echo "ERROR: Config list file not found: ${CONFIG_LIST_FILE}"
    exit 1
fi

# --- Get Config File for This Array Task ---
CONFIG_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")

if [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: No config file found for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file does not exist: ${CONFIG_FILE}"
    exit 1
fi

# --- Extract Seed from Config Filename ---
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .json)
RANDOM_SEED=$(echo "${CONFIG_BASENAME}" | grep -oP 'seed\K\d+' || echo "42")

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "UMMBAS v3.0 - Job Array Task"
echo "========================================================================"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "------------------------------------------------------------------------"
echo "Random Seed: ${RANDOM_SEED}"
echo "Configuration File: ${CONFIG_FILE}"
echo "Config Basename: ${CONFIG_BASENAME}"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Activate conda environment
# NOTE: Update this path to match your HPC environment
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# --- Validate Paths ---
ORCHESTRATOR_SCRIPT="$(pwd)/main_orchestrator.py"

if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then
    echo "ERROR: Orchestrator script not found at: ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi

# --- Run the Orchestrator ---
echo "Starting main_orchestrator.py..."
echo "------------------------------------------------------------------------"

python -u "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --random_seed "${RANDOM_SEED}"

EXIT_CODE=$?

echo "------------------------------------------------------------------------"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
