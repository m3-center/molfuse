#!/bin/bash
#SBATCH --job-name=molfuse_phase3
#SBATCH --output=slurm_logs/phase3_%A_%a.out
#SBATCH --error=slurm_logs/phase3_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --partition=hpc

# Phase 3: MF Cloud Ablation - Single Run Script
# 
# Usage (single run):
#   sbatch hpc/molfuse_phase3_cpu.sh configs/molfuse_phase3_grid/pca_features_mf1000.json experiment_workspace_v4
#
# Usage (array job, called by submit script):
#   sbatch --array=1-24 hpc/molfuse_phase3_cpu.sh <config_dir> <workspace_dir>

# ============================================================================
# Arguments
# ============================================================================

if [ $# -lt 2 ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 <config_path_or_dir> <workspace_dir>"
    exit 1
fi

CONFIG_INPUT="$1"
WORKSPACE_DIR="$2"

# ============================================================================
# Determine Config Path
# ============================================================================

if [ -f "$CONFIG_INPUT" ]; then
    # Single config file provided
    CONFIG_PATH="$CONFIG_INPUT"
elif [ -d "$CONFIG_INPUT" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Array job: select config by task ID
    CONFIGS=($(find "$CONFIG_INPUT" -name "*.json" | sort))
    if [ ${#CONFIGS[@]} -eq 0 ]; then
        echo "ERROR: No configs found in $CONFIG_INPUT"
        exit 1
    fi
    
    TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    if [ $TASK_IDX -ge ${#CONFIGS[@]} ]; then
        echo "ERROR: Task ID $SLURM_ARRAY_TASK_ID exceeds number of configs (${#CONFIGS[@]})"
        exit 1
    fi
    
    CONFIG_PATH="${CONFIGS[$TASK_IDX]}"
else
    echo "ERROR: Invalid config input or missing SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ============================================================================
# Idempotent Skip Logic
# ============================================================================

# Extract run_name from config
RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['run_name'])")
PHASE3_RUN=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('phase3_run_name', 'mf_ablation'))")
SUMMARY_PATH="$WORKSPACE_DIR/phase3/$PHASE3_RUN/$RUN_NAME/logs/phase3_summary.json"

if [ -f "$SUMMARY_PATH" ]; then
    echo "✓ SKIP: $RUN_NAME (phase3_summary.json exists)"
    echo "  Summary: $SUMMARY_PATH"
    exit 0
fi

# ============================================================================
# Environment Setup
# ============================================================================

echo "=========================================="
echo "PHASE 3: MF Cloud Ablation"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Run: $RUN_NAME"
echo "Workspace: $WORKSPACE_DIR"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "=========================================="

# Activate mamba environment
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Verify Python environment
echo "Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Set parallel execution environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Parallel threads: $OMP_NUM_THREADS"
echo "=========================================="

# ============================================================================
# Run Phase 3
# ============================================================================

START_TIME=$(date +%s)

python3 -m molfuse.cli.phase3 \
    --config "$CONFIG_PATH" \
    --workspace "$WORKSPACE_DIR"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Phase 3 completed successfully"
    echo "  Run: $RUN_NAME"
    echo "  Elapsed: ${ELAPSED}s"
    echo "  Summary: $SUMMARY_PATH"
else
    echo "✗ Phase 3 failed (exit code: $EXIT_CODE)"
    echo "  Run: $RUN_NAME"
    echo "  Check logs for details"
fi
echo "=========================================="

exit $EXIT_CODE
