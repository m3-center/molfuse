#!/bin/bash
# Submit Phase 2 config to SLURM
# Usage: bash hpc/submit_molfuse_phase2.sh [--dry-run] [CONFIG_DIR] [WORKSPACE_DIR]
# Defaults: CONFIG_DIR=configs/molfuse_phase2_grid, WORKSPACE_DIR=experiment_workspace_v4
# 
# Examples:
#   bash hpc/submit_molfuse_phase2.sh --dry-run
#   bash hpc/submit_molfuse_phase2.sh
#   bash hpc/submit_molfuse_phase2.sh configs/molfuse_phase2_grid experiment_workspace_v4

# Parse dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

CONFIG_DIR="${1:-configs/molfuse_phase2_grid}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

mkdir -p slurm_logs || true

echo "========================================="
echo "molfuse Phase 2 Batch Submit"
echo "========================================="
echo "Config dir: $CONFIG_DIR"
echo "Workspace: $WORKSPACE_DIR"
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
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY-RUN: Would submit $CONFIG_NAME"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "SUBMIT: $CONFIG_NAME"
        sbatch hpc/molfuse_phase2_cpu.sh "$CONFIG_PATH" "$WORKSPACE_DIR"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo "========================================="
echo "Batch submit completed"
echo "Submitted: $SUBMITTED"
echo "Skipped: $SKIPPED"
echo "========================================="
