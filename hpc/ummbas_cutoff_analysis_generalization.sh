#!/bin/bash
#SBATCH --partition=any          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32      # number of tasks per node (8 is often a good number for data loading/Python overhead for a single GPU job)
#SBATCH --mem=240G               # memory per node
#SBATCH --time=3-00:00:00        # total runtime of job allocation 

JOB_NAME="UMMBAS_Cutoff_Analysis_Generalization"

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
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# --- Activate Conda Environment ---
echo "Activating Conda environment..."
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# --- Define Script Paths ---
# Assumes this template is in the project root with the other scripts.
# Adjust SCRIPT_DIR if your project structure is different.
SCRIPT_DIR=$(pwd) 
ORCHESTRATOR_SCRIPT="${SCRIPT_DIR}/run_cutoff_analysis_generalization.py"

# Check that required files exist
if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then
    echo "ERROR: run_cutoff_analysis_generalization script not found at ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi

# --- Define Workspace Paths ---
ORIGINAL_WORKSPACE="${SCRIPT_DIR}/experiment_workspace_generalization/"
OUTPUT_WORKSPACE="${SCRIPT_DIR}/experiment_workspace_generalization_cutoff/"
CONFIG_PATH="${SCRIPT_DIR}/experiment_config.json"
CUTOFFS="100,1000,10000,100000"

echo "Configuration:"
echo "  Original workspace: ${ORIGINAL_WORKSPACE}"
echo "  Output workspace: ${OUTPUT_WORKSPACE}"
echo "  Config file: ${CONFIG_PATH}"
echo "  Affinity cutoffs (nM): ${CUTOFFS}"
echo "========================================================================"

# Create output workspace directory if it doesn't exist
mkdir -p "${OUTPUT_WORKSPACE}"

# --- Run the Cutoff Analysis ---
echo "Starting run_cutoff_analysis_generalization.py..."

python -u "${ORCHESTRATOR_SCRIPT}" \
    --original_workspace "${ORIGINAL_WORKSPACE}" \
    --output_workspace "${OUTPUT_WORKSPACE}" \
    --config_path "${CONFIG_PATH}" \
    --cutoffs "${CUTOFFS}"

EXIT_CODE=$?
echo "========================================================================"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
