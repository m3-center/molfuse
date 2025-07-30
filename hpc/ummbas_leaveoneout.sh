#!/bin/bash
#SBATCH --partition=gpu4          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=90G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=3-00:00:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --gres=gpu:1

# Arguments to be passed to this script:
# $1: Representation Mode (features or fingerprints)
# $2: Random Seed (e.g., 42)

REPR_MODE=$1
RANDOM_SEED=$2

# --- Environment Setup ---
echo "------------------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $SLURMD_NODENAME"
echo "Submitted from: $SLURM_SUBMIT_HOST"
echo "Working directory: $(pwd)"
echo "Representation Mode: $REPR_MODE"
echo "Random Seed: $RANDOM_SEED"
echo "Start Time: $(date)"
echo "------------------------------------------------------------------------"


# --- Define unique output/error filenames ---
JOB_NAME="UMMBAS_${REPR_MODE}_seed${RANDOM_SEED}"
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}_%j.out    # Store logs in a subdirectory
#SBATCH --error=slurm_logs/${JOB_NAME}_%j.err     # Store logs in a subdirectory

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# --- Load Modules ---
module load nvidia-hpc/default # Or your specific HPC SDK version
module load cuda               # Or your specific CUDA version
echo "Modules loaded."

source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# --- Define Base Directory for Experiment (Important for consistency) ---
# Assuming main_orchestrator.py is in the same directory as this submission script
# or you are submitting from the project root.
# Adjust SCRIPT_DIR if your project structure is different.
SCRIPT_DIR=$(pwd) # Or specify absolute path to your project root
ORCHESTRATOR_SCRIPT="${SCRIPT_DIR}/main_orchestrator.py"
CONFIG_FILE="${SCRIPT_DIR}/experiment_config.json"

# --- Run the Orchestrator ---
echo "Starting main_orchestrator.py..."
python "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --representation_mode "${REPR_MODE}" \
    --random_seed "${RANDOM_SEED}"

EXIT_CODE=$?
echo "------------------------------------------------------------------------"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "------------------------------------------------------------------------"

exit $EXIT_CODE
