#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=350G
#SBATCH --time=0-24:00:00
#SBATCH --exclude=wr43

# ============================================================================
# UMMBAS Generalization Experiment - HPC Execution Script
# ============================================================================
# This script runs a single configuration with a specific random seed for
# the generalization experiment (testing on new target proteins).
#
# Arguments:
#   $1: Random Seed (e.g., 42)
#   $2: Path to the specific JSON config file for this job
#
# Note: Job name, output, and error files are set by the submission script
# ============================================================================

RANDOM_SEED=$1
CONFIG_FILE=$2

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "UMMBAS GENERALIZATION EXPERIMENT"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "---"
echo "Random Seed: $RANDOM_SEED"
echo "Configuration File: $CONFIG_FILE"
echo "========================================================================"

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# Define orchestrator script path
ORCHESTRATOR_SCRIPT="$(pwd)/main_orchestrator.py"

# Validate files exist
if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then 
    echo "ERROR: Orchestrator script not found at: ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then 
    echo "ERROR: Configuration file not found at: ${CONFIG_FILE}"
    exit 1
fi

# --- Run the Orchestrator ---
echo ""
echo "Starting main_orchestrator.py..."
echo "----------------------------------------------------------------------"

python -u "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --random_seed "${RANDOM_SEED}"

EXIT_CODE=$?

echo "----------------------------------------------------------------------"
echo ""
echo "========================================================================"
echo "Orchestrator finished with exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "========================================================================"

exit ${EXIT_CODE}
