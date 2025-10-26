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

# Run using the requested environment without activating shell rc files
if command -v mamba >/dev/null 2>&1; then
  mamba run -n "$CONDA_ENV" python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"
elif command -v conda >/dev/null 2>&1; then
  conda run -n "$CONDA_ENV" --no-capture-output python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"
else
  echo "Neither mamba nor conda found in PATH. Falling back to system python." >&2
  python -m molfuse.cli.phase1 --config "$CONFIG_PATH" --workspace "$WORKSPACE_DIR"
fi
