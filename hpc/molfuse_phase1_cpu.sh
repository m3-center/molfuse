#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=350G
#SBATCH --time=0-48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Usage:
# sbatch hpc/molfuse_phase1_cpu.sh <CONFIG_JSON> <WORKSPACE_DIR> 
# Example:
# sbatch hpc/molfuse_phase1_cpu.sh configs/molfuse_phase1_grid/ABL1_PCA_features_10d.json experiment_workspace_v4

CONFIG_PATH="${1:-}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "CONFIG_JSON path is required as first argument" >&2
  exit 1
fi

# Extract config basename for job name (set before SLURM reads it)
CONFIG_BASENAME=$(basename "$CONFIG_PATH" .json)

# Ensure slurm_logs directory exists
mkdir -p slurm_logs || true

# Log startup information
echo "=========================================="
echo "SLURM Job: Phase 1 - ${CONFIG_BASENAME}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Config: ${CONFIG_PATH}"
echo "Workspace: ${WORKSPACE_DIR}"
echo "=========================================="
echo ""

source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity
python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Job finished with exit code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "=========================================="
exit ${EXIT_CODE}
