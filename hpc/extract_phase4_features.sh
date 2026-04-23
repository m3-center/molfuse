#!/bin/bash
#SBATCH --job-name=extract_p4_features
#SBATCH --output=slurm_logs/extract_phase4_features_%A_%a.out
#SBATCH --error=slurm_logs/extract_phase4_features_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=64
#SBATCH --partition=hpc

# Extract Phase 4 Features - Single Run
# This script extracts features_used.txt and imputer.joblib from a Phase 4 run
# Requires 300GB RAM to load full MF+ZINC datasets

set -e  # Exit on error

# Activate conda environment
if [[ -z "${CONDA_ACTIVATE:-}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_ACTIVATE="$(dirname "$CONDA_EXE")/activate"
fi
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${CONDA_ENV:-molfuse}"

# Get Phase 4 run directory from array
PHASE4_RUN_DIR=$1

if [ -z "$PHASE4_RUN_DIR" ]; then
    echo "ERROR: No Phase 4 run directory provided"
    echo "Usage: sbatch hpc/extract_phase4_features.sh <phase4_run_dir>"
    exit 1
fi

if [ ! -d "$PHASE4_RUN_DIR" ]; then
    echo "ERROR: Phase 4 run directory not found: $PHASE4_RUN_DIR"
    exit 1
fi

echo "=================================="
echo "Extract Phase 4 Features"
echo "=================================="
echo "Run Directory: $PHASE4_RUN_DIR"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=================================="

# Run extraction
python scripts/extract_phase4_features.py "$PHASE4_RUN_DIR" --force

EXIT_CODE=$?

echo "=================================="
echo "Extraction completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================="

exit $EXIT_CODE
