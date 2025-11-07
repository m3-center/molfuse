#!/bin/bash
#
# Batch submit Phase 3: MF Cloud Ablation Study
#
# Submits all Phase 3 configs as parallel HPC jobs with idempotent skip logic.
# Skips runs that already have phase3_summary.json.
#
# Usage:
#   bash hpc/submit_molfuse_phase3.sh [--dry-run] <config_dir> <workspace_dir>
#
# Example:
#   bash hpc/submit_molfuse_phase3.sh configs/molfuse_phase3_grid experiment_workspace_v4
#   bash hpc/submit_molfuse_phase3.sh --dry-run configs/molfuse_phase3_grid experiment_workspace_v4

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

if [ $# -lt 2 ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 [--dry-run] <config_dir> <workspace_dir>"
    exit 1
fi

CONFIG_DIR="$1"
WORKSPACE_DIR="$2"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# ============================================================================
# Count Configs and Check Existing Runs
# ============================================================================

echo "=========================================="
echo "PHASE 3 BATCH SUBMISSION: MF Cloud Ablation"
echo "=========================================="
echo "Config dir: $CONFIG_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "Dry run: $DRY_RUN"
echo "=========================================="

# Find all configs
CONFIGS=($(find "$CONFIG_DIR" -name "*.json" | sort))
N_TOTAL=${#CONFIGS[@]}

if [ $N_TOTAL -eq 0 ]; then
    echo "ERROR: No configs found in $CONFIG_DIR"
    exit 1
fi

echo "Found $N_TOTAL Phase 3 configs"
echo

# Check which runs already exist
N_COMPLETE=0
N_PENDING=0
PENDING_CONFIGS=()

for CONFIG in "${CONFIGS[@]}"; do
    RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run_name'])")
    PHASE3_RUN=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('phase3_run_name', 'mf_ablation'))")
    SUMMARY_PATH="$WORKSPACE_DIR/phase3/$PHASE3_RUN/$RUN_NAME/logs/phase3_summary.json"
    
    if [ -f "$SUMMARY_PATH" ]; then
        echo "  ✓ SKIP: $RUN_NAME (already complete)"
        N_COMPLETE=$((N_COMPLETE + 1))
    else
        echo "  → SUBMIT: $RUN_NAME"
        PENDING_CONFIGS+=("$CONFIG")
        N_PENDING=$((N_PENDING + 1))
    fi
done

echo
echo "=========================================="
echo "Summary:"
echo "  Total configs: $N_TOTAL"
echo "  Already complete: $N_COMPLETE"
echo "  Pending submission: $N_PENDING"
echo "=========================================="

if [ $N_PENDING -eq 0 ]; then
    echo
    echo "✓ All Phase 3 runs already complete. Nothing to submit."
    exit 0
fi

# ============================================================================
# Dry Run: Show What Would Be Submitted
# ============================================================================

if [ "$DRY_RUN" = true ]; then
    echo
    echo "DRY RUN: Would submit the following jobs:"
    echo
    for CONFIG in "${PENDING_CONFIGS[@]}"; do
        RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run_name'])")
        echo "  - $RUN_NAME"
    done
    echo
    echo "Total jobs: $N_PENDING"
    echo
    echo "To actually submit, run without --dry-run"
    exit 0
fi

# ============================================================================
# Create Slurm Logs Directory
# ============================================================================

mkdir -p logs/slurm

# ============================================================================
# Submit Jobs
# ============================================================================

echo
echo "Submitting $N_PENDING Phase 3 jobs..."
echo

SUBMITTED_JOBS=()

for CONFIG in "${PENDING_CONFIGS[@]}"; do
    RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run_name'])")
    
    JOB_ID=$(sbatch --parsable \
        --job-name="p3_$RUN_NAME" \
        hpc/molfuse_phase3_cpu.sh "$CONFIG" "$WORKSPACE_DIR")
    
    SUBMITTED_JOBS+=("$JOB_ID")
    echo "  ✓ Submitted: $RUN_NAME (Job ID: $JOB_ID)"
done

echo
echo "=========================================="
echo "✓ Submitted $N_PENDING Phase 3 jobs"
echo "=========================================="
echo
echo "Job IDs: ${SUBMITTED_JOBS[@]}"
echo
echo "Monitor status:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo
echo "Check logs:"
echo "  tail -f logs/slurm/phase3_*.out"
echo
echo "Aggregate results (after completion):"
echo "  python scripts/phase3_post_analysis.py --workspace_dir $WORKSPACE_DIR --output_dir reporting/phase3_post_analysis"
