#!/bin/bash
# Script to find Phase 1 baseline runs for Phase 5 comparison
# Run this on the HPC to locate the baseline results

WORKSPACE_DIR="${1:-/home/ahagg2s/experiment_workspace_v4}"

echo "=========================================="
echo "Finding Phase 1 Baseline Runs"
echo "=========================================="
echo "Workspace: $WORKSPACE_DIR"
echo ""

# Check if Phase 1 directory exists
if [ ! -d "$WORKSPACE_DIR/phase1" ]; then
    echo "ERROR: $WORKSPACE_DIR/phase1 does not exist"
    echo ""
    echo "Searching for alternate locations..."
    find /home/ahagg2s -maxdepth 3 -type d -name "phase1" 2>/dev/null | head -10
    exit 1
fi

echo "Phase 1 directory found: $WORKSPACE_DIR/phase1"
echo ""

# Look for fingerprint+UMAP runs (for tanimoto baseline)
echo "1. FINGERPRINT+UMAP RUNS (for tanimoto baseline):"
echo "   Looking for: *UMAP*fingerprints* or *fingerprints*UMAP*"
echo ""
find "$WORKSPACE_DIR/phase1" -maxdepth 1 -type d -name "*UMAP*fingerprints*" -o -name "*fingerprints*UMAP*" | while read -r dir; do
    basename "$dir"
    if [ -f "$dir/logs/phase1_summary.json" ]; then
        echo "   ✓ Has phase1_summary.json"
    else
        echo "   ✗ Missing phase1_summary.json"
    fi
done

echo ""
echo "   Specifically looking for ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep[1-5]:"
for rep in {1..5}; do
    dir="$WORKSPACE_DIR/phase1/ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep${rep}"
    if [ -d "$dir" ]; then
        echo "   ✓ Found rep${rep}: $dir"
        if [ -f "$dir/logs/phase1_summary.json" ]; then
            # Extract EF@1% from summary
            ef1=$(grep -oP '"ef_1%":\s*\K[0-9.]+' "$dir/logs/phase1_summary.json" | head -1)
            echo "      EF@1% = $ef1"
        fi
    else
        echo "   ✗ Missing rep${rep}"
    fi
done

echo ""
echo "2. FEATURES+UMAP RUNS (for raw_descriptors baseline):"
echo "   Looking for: *UMAP*features* or *features*UMAP*"
echo ""

# Check Phase 1 first
if [ -d "$WORKSPACE_DIR/phase1" ]; then
    echo "   In Phase 1:"
    find "$WORKSPACE_DIR/phase1" -maxdepth 1 -type d -name "*UMAP*features*" -o -name "*features*UMAP*" | head -5 | while read -r dir; do
        basename "$dir"
        if [ -f "$dir/logs/phase1_summary.json" ]; then
            ef1=$(grep -oP '"ef_1%":\s*\K[0-9.]+' "$dir/logs/phase1_summary.json" | head -1)
            echo "      EF@1% = $ef1"
        fi
    done
fi

# Check Phase 4 (if exists)
if [ -d "$WORKSPACE_DIR/phase4" ]; then
    echo ""
    echo "   In Phase 4:"
    find "$WORKSPACE_DIR/phase4" -maxdepth 1 -type d -name "*UMAP*features*" -o -name "*features*UMAP*" | head -5 | while read -r dir; do
        basename "$dir"
        if [ -f "$dir/logs/phase4_summary.json" ]; then
            ef1=$(grep -oP '"ef_1%":\s*\K[0-9.]+' "$dir/logs/phase4_summary.json" | head -1)
            echo "      EF@1% = $ef1"
        fi
    done
fi

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total Phase 1 directories:"
ls -1 "$WORKSPACE_DIR/phase1" 2>/dev/null | wc -l

echo ""
echo "Fingerprint+UMAP runs found:"
find "$WORKSPACE_DIR/phase1" -maxdepth 1 -type d \( -name "*UMAP*fingerprints*" -o -name "*fingerprints*UMAP*" \) 2>/dev/null | wc -l

echo ""
echo "Features+UMAP runs found (Phase 1):"
find "$WORKSPACE_DIR/phase1" -maxdepth 1 -type d \( -name "*UMAP*features*" -o -name "*features*UMAP*" \) 2>/dev/null | wc -l

if [ -d "$WORKSPACE_DIR/phase4" ]; then
    echo ""
    echo "Features+UMAP runs found (Phase 4):"
    find "$WORKSPACE_DIR/phase4" -maxdepth 1 -type d \( -name "*UMAP*features*" -o -name "*features*UMAP*" \) 2>/dev/null | wc -l
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
