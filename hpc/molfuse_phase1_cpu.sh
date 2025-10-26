#!/bin/bash
#SBATCH --job-name=molfuse_p1
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=0-12:00:00

# Usage:
# sbatch hpc/molfuse_phase1_cpu.sh <CONFIG_JSON> <WORKSPACE_DIR> 
# Example:
# sbatch hpc/molfuse_phase1_cpu.sh configs/molfuse_phase1_grid/ABL1_PCA_features_10d.json experiment_workspace_v4

set -euo pipefail
CONFIG_PATH="${1:-}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "CONFIG_JSON path is required as first argument" >&2
  exit 1
fi

mkdir -p slurm_logs || true

source /home/ahagg2s/miniforge3/bin/activate ummbas-screening
python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"
