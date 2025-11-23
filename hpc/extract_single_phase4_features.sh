#!/bin/bash
#SBATCH --job-name=extract_p4_feat
#SBATCH --output=slurm_logs/extract_phase4_features_%j.out
#SBATCH --error=slurm_logs/extract_phase4_features_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute

# Extract Phase 4 features for a single run
# Usage: sbatch hpc/extract_single_phase4_features.sh <phase4_run_dir>
# Example: sbatch hpc/extract_single_phase4_features.sh experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4

set -euo pipefail

# Get phase4 run directory from command line argument
PHASE4_RUN_DIR="${1:-}"

if [ -z "$PHASE4_RUN_DIR" ]; then
    echo "ERROR: Phase 4 run directory not provided"
    echo "Usage: sbatch hpc/extract_single_phase4_features.sh <phase4_run_dir>"
    exit 1
fi

# Convert to absolute path if needed
if [[ ! "$PHASE4_RUN_DIR" = /* ]]; then
    PHASE4_RUN_DIR="${SLURM_SUBMIT_DIR}/${PHASE4_RUN_DIR}"
fi

echo "================================================================================"
echo "EXTRACT PHASE 4 FEATURES (Single Run)"
echo "================================================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  ${SLURM_SUBMIT_DIR}"
echo "Phase 4 run:  ${PHASE4_RUN_DIR}"
echo "================================================================================"

# Navigate to workspace
cd "${SLURM_SUBMIT_DIR}"

# Load Python environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment (adjust name as needed)
source activate molfuse

# Run extraction script
echo ""
echo "Running extraction script..."
python scripts/extract_phase4_features.py "${PHASE4_RUN_DIR}"

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "EXTRACTION COMPLETED"
echo "================================================================================"
echo "Exit code:    ${EXIT_CODE}"
echo "End time:     $(date)"
echo "================================================================================"

exit ${EXIT_CODE}
