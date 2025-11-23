#!/bin/bash
# Submit all Phase 4 feature extraction jobs to SLURM
# Usage: bash hpc/submit_extract_phase4_features.sh [phase4_parent_dir]

set -e

PHASE4_PARENT_DIR="${1:-experiment_workspace_v4/phase4/cross_target}"

if [ ! -d "$PHASE4_PARENT_DIR" ]; then
    echo "ERROR: Phase 4 parent directory not found: $PHASE4_PARENT_DIR"
    exit 1
fi

echo "=================================="
echo "Submit Phase 4 Feature Extraction"
echo "=================================="
echo "Phase 4 Directory: $PHASE4_PARENT_DIR"
echo "=================================="

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# Find all Phase 4 run directories
PHASE4_RUNS=($(ls -d ${PHASE4_PARENT_DIR}/umap_features_* 2>/dev/null || true))

if [ ${#PHASE4_RUNS[@]} -eq 0 ]; then
    echo "ERROR: No Phase 4 runs found in $PHASE4_PARENT_DIR"
    echo "Expected directories with prefix: umap_features_"
    exit 1
fi

echo "Found ${#PHASE4_RUNS[@]} Phase 4 runs"
echo ""

# Submit jobs
SUBMITTED=0
SKIPPED=0
FAILED=0

for PHASE4_RUN_DIR in "${PHASE4_RUNS[@]}"; do
    RUN_NAME=$(basename "$PHASE4_RUN_DIR")
    
    # Check if artifacts already exist
    FEATURES_FILE="$PHASE4_RUN_DIR/artifacts/features_used.txt"
    IMPUTER_FILE="$PHASE4_RUN_DIR/artifacts/imputer.joblib"
    
    echo "DEBUG: Checking $RUN_NAME"
    echo "  Features file: $FEATURES_FILE"
    echo "  Exists: $([ -f "$FEATURES_FILE" ] && echo 'YES' || echo 'NO')"
    echo "  Imputer file: $IMPUTER_FILE"
    echo "  Exists: $([ -f "$IMPUTER_FILE" ] && echo 'YES' || echo 'NO')"
    
    if [ -f "$FEATURES_FILE" ] && [ -f "$IMPUTER_FILE" ]; then
        echo "SKIP: $RUN_NAME (artifacts exist)"
        ((SKIPPED++))
        continue
    fi
    
    # Submit job
    echo "  Submitting job for $RUN_NAME..."
    JOB_ID=$(sbatch --parsable hpc/extract_phase4_features.sh "$PHASE4_RUN_DIR" 2>&1)
    SUBMIT_EXIT=$?
    
    if [ $SUBMIT_EXIT -eq 0 ]; then
        echo "SUBMITTED: $RUN_NAME (Job ID: $JOB_ID)"
        ((SUBMITTED++))
    else
        echo "FAILED: $RUN_NAME (submission failed with exit code $SUBMIT_EXIT)"
        echo "  Error: $JOB_ID"
        ((FAILED++))
    fi
    echo ""
done

echo ""
echo "=================================="
echo "Submission Summary"
echo "=================================="
echo "Total runs: ${#PHASE4_RUNS[@]}"
echo "  Submitted: $SUBMITTED"
echo "  Skipped: $SKIPPED"
echo "  Failed: $FAILED"
echo "=================================="

if [ $SUBMITTED -gt 0 ]; then
    echo ""
    echo "Monitor jobs with: squeue -u $USER"
    echo "Check logs in: slurm_logs/extract_phase4_features_*.out"
fi
