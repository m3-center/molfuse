#!/bin/bash
# Submit all Phase 1 v4 configs to SLURM
# Usage: bash hpc/submit_molfuse_phase1.sh [CONFIG_DIR] [WORKSPACE_DIR] [CONDA_ENV]
# Defaults: CONFIG_DIR=configs/molfuse_phase1_grid, WORKSPACE_DIR=experiment_workspace_v4, CONDA_ENV=ummbas_screening

CONFIG_DIR="${1:-configs/molfuse_phase1_grid}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

mkdir -p slurm_logs || true

COUNT=0
for cfg in "$CONFIG_DIR"/*.json; do
  if [[ ! -f "$cfg" ]]; then continue; fi
  
  # Extract config basename for job name
  CONFIG_BASENAME=$(basename "$cfg" .json)
  
  # Submit with unique job name
  sbatch --job-name="phase1_${CONFIG_BASENAME}" hpc/molfuse_phase1_cpu.sh "$cfg" "$WORKSPACE_DIR"
  COUNT=$((COUNT+1))
  sleep 0.1
done

echo "Submitted $COUNT Phase 1 jobs from $CONFIG_DIR"
