#!/bin/bash
#SBATCH --job-name=molfuse_phase2
#SBATCH --output=slurm_logs/phase2_%j.out
#SBATCH --error=slurm_logs/phase2_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=any

# molfuse Phase 2: Affinity Cutoff Sensitivity (Re-scoring Only)
# Single-job runner for Phase 2 execution
# Usage: sbatch hpc/molfuse_phase2_cpu.sh <config_path> <workspace_dir> <conda_env>

set -e
set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: sbatch $0 <config_path> <workspace_dir> <conda_env>"
    exit 1
fi

CONFIG_PATH=$1
WORKSPACE_DIR=$2
CONDA_ENV=$3

echo "========================================="
echo "molfuse Phase 2: Cutoff Sensitivity"
echo "========================================="
echo "Config: $CONFIG_PATH"
echo "Workspace: $WORKSPACE_DIR"
echo "Conda env: $CONDA_ENV"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening

# Create slurm_logs directory if needed
mkdir -p slurm_logs

# Run Phase 2
python -m molfuse.cli.phase2 \
    --config "$CONFIG_PATH" \
    --workspace "$WORKSPACE_DIR"

echo "========================================="
echo "Phase 2 job completed"
echo "========================================="
