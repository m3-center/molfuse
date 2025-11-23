#!/bin/bash
#
# Batch extract features and reconstruct imputers for all Phase 4 runs.
#
# This script processes all Phase 4 cross-target experiments and extracts:
# - features_used.txt: Exact feature list after zero-variance filtering
# - imputer.joblib: Reconstructed imputer (Phase 4 didn't save it)
#
# Usage:
#   bash scripts/batch_extract_phase4_features.sh [--force] [workspace_dir]
#
# Options:
#   --force: Overwrite existing features_used.txt and imputer.joblib
#   workspace_dir: Path to experiment workspace (default: experiment_workspace_v4)

set -euo pipefail

# Parse arguments
FORCE_FLAG=""
WORKSPACE_DIR="experiment_workspace_v4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        *)
            WORKSPACE_DIR="$1"
            shift
            ;;
    esac
done

# Validate workspace directory
if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo "❌ ERROR: Workspace directory not found: $WORKSPACE_DIR"
    exit 1
fi

PHASE4_DIR="$WORKSPACE_DIR/phase4/cross_target"

if [[ ! -d "$PHASE4_DIR" ]]; then
    echo "❌ ERROR: Phase 4 directory not found: $PHASE4_DIR"
    exit 1
fi

# Find all Phase 4 run directories
echo "Searching for Phase 4 runs in: $PHASE4_DIR"
PHASE4_RUNS=($(find "$PHASE4_DIR" -maxdepth 1 -type d -name "umap_features_*" | sort))

if [[ ${#PHASE4_RUNS[@]} -eq 0 ]]; then
    echo "❌ ERROR: No Phase 4 runs found matching pattern: umap_features_*"
    exit 1
fi

echo "Found ${#PHASE4_RUNS[@]} Phase 4 runs"
echo ""

# Process each run
SUCCESS_COUNT=0
SKIP_COUNT=0
ERROR_COUNT=0

for run_dir in "${PHASE4_RUNS[@]}"; do
    run_name=$(basename "$run_dir")
    
    # Check if already processed (unless --force)
    if [[ -z "$FORCE_FLAG" ]] && [[ -f "$run_dir/artifacts/features_used.txt" ]] && [[ -f "$run_dir/artifacts/imputer.joblib" ]]; then
        echo "⏭  SKIP: $run_name (already processed, use --force to reprocess)"
        ((SKIP_COUNT++))
        continue
    fi
    
    echo "▶ Processing: $run_name"
    
    # Run extraction script
    if python -m scripts.extract_phase4_features "$run_dir" $FORCE_FLAG; then
        ((SUCCESS_COUNT++))
    else
        echo "❌ FAILED: $run_name"
        ((ERROR_COUNT++))
    fi
done

# Summary
echo ""
echo "========================================"
echo "Batch Extraction Summary"
echo "========================================"
echo "Total runs found:    ${#PHASE4_RUNS[@]}"
echo "Successfully processed: $SUCCESS_COUNT"
echo "Skipped (existing): $SKIP_COUNT"
echo "Failed:             $ERROR_COUNT"
echo "========================================"

if [[ $ERROR_COUNT -gt 0 ]]; then
    exit 1
fi
