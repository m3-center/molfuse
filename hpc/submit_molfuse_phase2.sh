#!/bin/bash
# Batch submit helper for molfuse Phase 2
# Usage: bash hpc/submit_molfuse_phase2.sh <config_dir> <workspace_dir> <conda_env> [--dry-run]

set -e
set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config_dir> <workspace_dir> <conda_env> [--dry-run]"
    echo "Example: bash hpc/submit_molfuse_phase2.sh configs/molfuse_phase2_grid experiment_workspace_v4 ummbas_screening"
    exit 1
fi

CONFIG_DIR=$1
WORKSPACE_DIR=$2
CONDA_ENV=$3
DRY_RUN=0

if [ "$#" -ge 4 ] && [ "$4" == "--dry-run" ]; then
    DRY_RUN=1
fi

echo "========================================="
echo "molfuse Phase 2 Batch Submit"
echo "========================================="
echo "Config dir: $CONFIG_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "Conda env: $CONDA_ENV"
echo "Dry run: $DRY_RUN"
echo "========================================="

# Create slurm_logs directory
mkdir -p slurm_logs

# Count configs
CONFIG_COUNT=$(find "$CONFIG_DIR" -name "*.json" -type f | wc -l)
echo "Found $CONFIG_COUNT Phase 2 config(s)"

if [ "$CONFIG_COUNT" -eq 0 ]; then
    echo "ERROR: No config files found in $CONFIG_DIR"
    exit 1
fi

# Submit jobs
SUBMITTED=0
SKIPPED=0

for CONFIG_PATH in "$CONFIG_DIR"/*.json; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .json)
    
    # Check if Phase 2 already completed (idempotent skip)
    # Phase 2 typically has a single run per config (cutoff_sweep)
    PHASE2_SUMMARY="$WORKSPACE_DIR/phase2/cutoff_sweep/logs/phase2_summary.json"
    
    if [ -f "$PHASE2_SUMMARY" ]; then
        echo "SKIP: $CONFIG_NAME (phase2_summary.json exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: Would submit $CONFIG_NAME"
        SUBMITTED=$((SUBMITTED + 1))
    else:
        echo "SUBMIT: $CONFIG_NAME"
        sbatch hpc/molfuse_phase2_cpu.sh "$CONFIG_PATH" "$WORKSPACE_DIR" "$CONDA_ENV"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo "========================================="
echo "Batch submit completed"
echo "Submitted: $SUBMITTED"
echo "Skipped: $SKIPPED"
echo "========================================="
