#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=170G
#SBATCH --time=0-24:00:00

# =============================================================================
# UMMBAS v3.0 - Single Experiment SLURM Script (CPU)
# =============================================================================
# This script runs a single experiment configuration with a given random seed.
# It is called by submit_v3_phase*.sh scripts for each experimental phase.
#
# Arguments:
#   $1: Random Seed (e.g., 42)
#   $2: Path to the specific JSON config file for this job
# =============================================================================

RANDOM_SEED=$1
CONFIG_FILE=$2

# --- Validate Arguments ---
if [ -z "${RANDOM_SEED}" ] || [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: $0 <random_seed> <config_file>"
    exit 1
fi

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "UMMBAS v3.0 - Single Experiment Job"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "------------------------------------------------------------------------"
echo "Random Seed: $RANDOM_SEED"
echo "Configuration File: $CONFIG_FILE"
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

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Configuration file not found at: ${CONFIG_FILE}"
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
