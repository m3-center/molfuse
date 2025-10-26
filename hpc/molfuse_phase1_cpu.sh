#!/usr/bin/env bash
#SBATCH --job-name=molfuse_p1
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

# Usage:
# sbatch hpc/molfuse_phase1_cpu.sh <CONFIG_JSON> <WORKSPACE_DIR> [CONDA_ENV]
# Example:
# sbatch hpc/molfuse_phase1_cpu.sh configs/molfuse_phase1_grid/ABL1_PCA_features_10d.json experiment_workspace_v4 ummbas_screening

set -euo pipefail
CONFIG_PATH="${1:-}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"
CONDA_ENV="${3:-ummbas_screening}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "CONFIG_JSON path is required as first argument" >&2
  exit 1
fi

mkdir -p slurm_logs || true

mamba activate "$CONDA_ENV"
python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"
