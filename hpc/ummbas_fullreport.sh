#!/bin/bash
#SBATCH --partition=any          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=180G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=0-05:00:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR


# --- Define unique output/error filenames ---
JOB_NAME="UMMBAS_Aggregated_Report"
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm_logs/${JOB_NAME}_%j.out    # Store logs in a subdirectory
#SBATCH --error=slurm_logs/${JOB_NAME}_%j.err     # Store logs in a subdirectory

source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

SCRIPT_DIR=$(pwd) # Or specify absolute path to your project root
EXPERIMENT_DIR="${SCRIPT_DIR}/experiment_workspace"
REPORT_SCRIPT="${SCRIPT_DIR}/aggregate_and_report.py"
CONFIG_FILE="${SCRIPT_DIR}/experiment_config.json"

# --- Run the report generator ---
echo "Starting aggregate_and_report.py..."
python "${REPORT_SCRIPT}" \
    --base_experiment_dir "${EXPERIMENT_DIR}" \
    --config_path "${CONFIG_FILE}" \
    --output_report_dir "${EXPERIMENT_DIR}/final_report"

EXIT_CODE=$?
echo "------------------------------------------------------------------------"
echo "Python script finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "------------------------------------------------------------------------"

exit $EXIT_CODE
