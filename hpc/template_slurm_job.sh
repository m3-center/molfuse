#!/bin/bash
#SBATCH --partition=gpu4          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32      # number of tasks per node (8 is often a good number for data loading/Python overhead for a single GPU job)
#SBATCH --mem=180G               # memory per node
#SBATCH --time=3-00:00:00        # total runtime of job allocation (3 days)
#SBATCH --gres=gpu:1             # Request 1 GPU

# --- Arguments passed from submit_all_replicates.sh ---
# $1: Representation Mode (e.g., features)
# $2: Random Seed (e.g., 42)
# $3: Path to the specific JSON config file for this job

REPR_MODE=$1
RANDOM_SEED=$2
CONFIG_FILE=$3

# --- Define unique output/error filenames ---
# Use a more descriptive name based on the config file's basename
CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .json)
JOB_NAME="UMMBAS_${CONFIG_BASENAME}_seed${RANDOM_SEED}"

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/%x_%j.out    # %x is SLURM_JOB_NAME, %j is SLURM_JOB_ID
#SBATCH --error=slurm_logs/%x_%j.err     # Store logs in a subdirectory

# --- Environment Setup and Logging ---
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Submitted from: $SLURM_SUBMIT_HOST"
echo "Working directory: $(pwd)"
echo "Start Time: $(date)"
echo "---"
echo "Representation Mode: $REPR_MODE"
echo "Random Seed: $RANDOM_SEED"
echo "Configuration File: $CONFIG_FILE"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# --- Load Modules ---
echo "Loading required modules..."
module purge # Start with a clean environment
module load nvidia-hpc/default # Or your specific HPC SDK version
module load cuda               # Or your specific CUDA version
echo "Modules loaded successfully."

# --- Activate Conda Environment ---
echo "Activating Conda environment..."
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# --- Define Script Paths ---
# Assumes this template is in the project root with the other scripts.
# Adjust SCRIPT_DIR if your project structure is different.
SCRIPT_DIR=$(pwd) 
ORCHESTRATOR_SCRIPT="${SCRIPT_DIR}/main_orchestrator.py"

# Check that required files exist
if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then
    echo "ERROR: Orchestrator script not found at ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Configuration file not found at ${CONFIG_FILE}"
    exit 1
fi

# --- Run the Orchestrator ---
echo "Starting main_orchestrator.py..."
python -u "${ORCHESTRATOR_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --representation_mode "${REPR_MODE}" \
    --random_seed "${RANDOM_SEED}"

EXIT_CODE=$?
echo "========================================================================"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE