#!/bin/bash
# Quick test of potency-stratified analysis on HPC
# Run this to verify everything works before full analysis

set -e

echo "================================================================"
echo "Potency-Stratified Analysis - Quick Test"
echo "================================================================"
echo ""

# Configuration
WORKSPACE=${1:-"experiment_workspace_v3_phase1"}
TEST_OUTPUT="test_potency_output_$(date +%Y%m%d_%H%M%S)"

echo "Workspace: $WORKSPACE"
echo "Test output: $TEST_OUTPUT"
echo ""

# Step 1: Debug first run
echo "Step 1: Inspecting first run directory..."
echo "----------------------------------------------------------------"
python scripts/debug_run_structure.py "$WORKSPACE" || {
    echo "⚠️  Debug script had issues, but continuing..."
}
echo ""

# Step 2: Validate
echo "Step 2: Validating workspace data..."
echo "----------------------------------------------------------------"
python scripts/validate_potency_analysis.py --workspace_dir "$WORKSPACE" || {
    echo "⚠️  Validation warnings, but continuing..."
}
echo ""

# Step 3: Test on first seed only (faster)
echo "Step 3: Testing analysis on seed 42 only..."
echo "----------------------------------------------------------------"
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir "$WORKSPACE" \
    --seed 42 \
    --output_dir "$TEST_OUTPUT" \
    --summary_only

echo ""
echo "================================================================"
echo "Test Results"
echo "================================================================"

if [ -f "$TEST_OUTPUT/potency_stratified_report.txt" ]; then
    echo "✅ SUCCESS! Analysis completed."
    echo ""
    echo "Summary:"
    head -n 30 "$TEST_OUTPUT/potency_stratified_report.txt"
    echo ""
    echo "Full results in: $TEST_OUTPUT/"
    echo ""
    echo "Next step: Run full analysis with:"
    echo "  sbatch hpc/submit_potency_analysis.sh"
    echo "  or"
    echo "  python scripts/analyze_potency_stratified_enrichment.py \\"
    echo "    --workspace_dir $WORKSPACE \\"
    echo "    --output_dir potency_full_results"
else
    echo "❌ Analysis did not complete successfully"
    echo ""
    echo "Check the log:"
    cat "$TEST_OUTPUT/analysis.log" 2>/dev/null || echo "No log file found"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check debug output above for missing files"
    echo "  2. Verify affinity data is available"
    echo "  3. Check one of your ranked CSV files manually:"
    echo "     head $WORKSPACE/run_seed42_*/*/results/*/dim_*/*/*-RANKED.csv"
fi

echo ""
echo "================================================================"
