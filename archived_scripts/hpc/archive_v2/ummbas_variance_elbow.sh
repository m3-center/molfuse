#!/bin/bash
#SBATCH --partition=any          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32      # number of tasks per node (8 is often a good number for data loading/Python overhead for a single GPU job)
#SBATCH --mem=350G               # memory per node
#SBATCH --time=0-05:00:00        # total runtime of job allocation 

JOB_NAME="UMMBAS_Variance_Elbow_Analysis"

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
ORCHESTRATOR_SCRIPT="${SCRIPT_DIR}/core_scripts/analyze_pca_variance.py"



# Check that required files exist
if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then
    echo "ERROR: analyze_pca_variance script not found at ${ORCHESTRATOR_SCRIPT}"
    exit 1
fi

# --- Run the Orchestrator ---
echo "Starting analyze_pca_variance.py..."

python -u "${ORCHESTRATOR_SCRIPT}" --chembl_mf_data_path /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_hyperparam_sweep/tyro/run_seed42_reprfingerprints_20250826_131440/TyrosineProteinKinaseABL1_P00519/temp_data/TyrosineProteinKinaseABL1_P00519_chembl_mf_excluded_fingerprints.csv   --zinc_data_path /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_hyperparam_sweep/tyro/run_seed42_reprfingerprints_20250826_131440/TyrosineProteinKinaseABL1_P00519/temp_data/TyrosineProteinKinaseABL1_P00519_zinc_excluded_fingerprints.csv --output_plot_path explained_variance_fingerprints.png --max_components 1000


EXIT_CODE=$?
echo "========================================================================"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
