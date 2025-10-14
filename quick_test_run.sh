#!/bin/bash

# ============================================================================
# Quick Test Run - UMMBAS v2.0
# ============================================================================
# This script runs a complete test of the pipeline with a small dataset
# to validate everything works before submitting large HPC jobs.
#
# Usage: bash quick_test_run.sh
# ============================================================================

set -e  # Exit on any error

echo "========================================================================"
echo "UMMBAS v2.0 - Quick Test Run"
echo "========================================================================"
echo "This will:"
echo "  1. Generate test configurations (small dataset)"
echo "  2. Run ONE config locally with seed 42"
echo "  3. Validate the output"
echo ""
echo "Expected runtime: ~30-60 minutes (depends on system)"
echo "========================================================================"
echo ""

# Step 1: Generate test configurations
echo "Step 1/3: Generating test configurations..."
python config_generators/generate_generalization_configs.py --test-run

if [ $? -ne 0 ]; then
    echo "ERROR: Config generation failed!"
    exit 1
fi

echo ""
echo "✓ Test configurations created"
echo ""

# Step 2: Find a test config to run
TEST_CONFIG=$(find generalization_configs/ -name "*_TEST.json" | head -1)

if [ -z "$TEST_CONFIG" ]; then
    echo "ERROR: No test configuration found!"
    exit 1
fi

echo "Step 2/3: Running test experiment..."
echo "  Config: $(basename $TEST_CONFIG)"
echo "  Seed: 42"
echo "  Log: test_run.log"
echo ""

# Run the orchestrator locally
python main_orchestrator.py --config "$TEST_CONFIG" --random_seed 42 2>&1 | tee test_run.log

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Test run failed! Check test_run.log for details."
    exit 1
fi

echo ""
echo "✓ Test run completed successfully"
echo ""

# Step 3: Validate outputs
echo "Step 3/3: Validating outputs..."

# Check for workspace directory
if [ ! -d "test_experiment_workspace" ]; then
    echo "ERROR: Workspace directory not created!"
    exit 1
fi

# Check for key output files
WORKSPACE_DIR=$(find test_experiment_workspace/ -maxdepth 1 -type d -name "*seed42" | head -1)

if [ -z "$WORKSPACE_DIR" ]; then
    echo "ERROR: Workspace run directory not found!"
    exit 1
fi

echo "  Workspace: $WORKSPACE_DIR"

# Check for critical files
REQUIRED_FILES=(
    "rankings/held_out_actives_and_decoys_ranking_metrics.csv"
    "similarity_spaces/mf_cloud_coords.csv"
)

ALL_FOUND=true
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ -f "$WORKSPACE_DIR/$FILE" ]; then
        echo "  ✓ Found: $FILE"
    else
        echo "  ✗ Missing: $FILE"
        ALL_FOUND=false
    fi
done

echo ""

if [ "$ALL_FOUND" = true ]; then
    echo "========================================================================"
    echo "✓ TEST RUN SUCCESSFUL!"
    echo "========================================================================"
    echo ""
    echo "Outputs located in:"
    echo "  - Workspace: test_experiment_workspace/"
    echo "  - Log: test_run.log"
    echo ""
    echo "Check key metrics:"
    echo "  cat $WORKSPACE_DIR/rankings/held_out_actives_and_decoys_ranking_metrics.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Review test results to ensure they look reasonable"
    echo "  2. RECOMMENDED: Run hyperparameter optimization on HPC"
    echo "     python config_generators/generate_hyperparam_configs.py"
    echo "     bash hpc/submit_all_replicates_hyperparameterization.sh"
    echo "  3. After hyperparameter sweep, update optimal parameters in:"
    echo "     config_generators/generate_generalization_configs.py"
    echo "  4. Generate full configs: python config_generators/generate_generalization_configs.py"
    echo "  5. Submit HPC jobs: bash hpc/submit_generalization_jobs.sh"
    echo ""
    echo "See docs/HPC_EXECUTION_GUIDE.md for complete workflow"
    echo "========================================================================"
    exit 0
else
    echo "========================================================================"
    echo "✗ TEST RUN INCOMPLETE"
    echo "========================================================================"
    echo "Some expected output files are missing."
    echo "Check test_run.log for errors."
    echo "========================================================================"
    exit 1
fi
