#!/bin/bash
#SBATCH --partition=any          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32      # number of tasks per node (8 is often a good number for data loading/Python overhead for a single GPU job)
#SBATCH --mem=360G               # memory per node
#SBATCH --time=0-12:00:00        # total runtime of job allocation 


REPR_MODE=$1
RANDOM_SEED=$2
CONFIG_FILE=$3
JOB_NAME="UMMBAS_HYPERPARAMETERIZATION_ANALYSIS"

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
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analysis_scripts/analyze_hyperparams.py"

# --- Run the Orchestrator ---
mkdir -p hyperparameterization_report
echo "Starting main_orchestrator.py..."
python -u "${ANALYSIS_SCRIPT}" \
    --base_experiment_dir "experiment_workspace_rerun_hyperparam_sweep" \
    --output_report_dir "hyperparameterization_report"

EXIT_CODE=$?
echo "========================================================================"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
