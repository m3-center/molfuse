#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=180G
#SBATCH --time=0-08:00:00
#SBATCH --exclude=wr43

# --- Arguments passed from submit script ---
# $1: Random Seed (e.g., 42)
# $2: Path to the specific JSON config file for this job

RANDOM_SEED=$1
CONFIG_FILE=$2

# --- Define unique output/error filenames ---
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .json)
JOB_NAME="UMMBAS_${CONFIG_BASENAME}_seed${RANDOM_SEED}"

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "---"
echo "Random Seed: $RANDOM_SEED"
echo "Configuration File: $CONFIG_FILE"
echo "========================================================================"

mkdir -p slurm_logs
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

ORCHESTRATOR_SCRIPT="$(pwd)/main_orchestrator.py"

if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then echo "ERROR: Orchestrator script not found."; exit 1; fi
if [ ! -f "${CONFIG_FILE}" ]; then echo "ERROR: Configuration file not found."; exit 1; fi

# --- Run the Orchestrator ---
echo "Starting main_orchestrator.py..."
python -u "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --random_seed "${RANDOM_SEED}"

EXIT_CODE=$?
echo "========================================================================"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
