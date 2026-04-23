#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=0-72:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Phase 5: Validation & Baseline Experiments
# Single-run job script for SLURM submission
#
# Usage:
#   sbatch hpc/molfuse_phase5_cpu.sh <CONFIG_JSON> <WORKSPACE_DIR>
#
# Example:
#   sbatch hpc/molfuse_phase5_cpu.sh configs/molfuse_phase5_grid/phase5_negative_control_rep1.json experiment_workspace_v4

CONFIG_PATH="${1:-}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: CONFIG_JSON path is required as first argument" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

# Extract config basename for logging
CONFIG_BASENAME=$(basename "$CONFIG_PATH" .json)

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "SLURM Job: Phase 5 - ${CONFIG_BASENAME}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Config: ${CONFIG_PATH}"
echo "Workspace: ${WORKSPACE_DIR}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 200G"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV:-molfuse}"
if [[ -z "${CONDA_ACTIVATE:-}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_ACTIVATE="$(dirname "$CONDA_EXE")/activate"
fi
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${CONDA_ENV:-molfuse}"

# Verify Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run Phase 5 experiment
echo "Starting Phase 5 experiment..."
python -m molfuse.cli.phase5 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished with exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "Duration: ${SECONDS}s"
echo "=========================================="

exit ${EXIT_CODE}
