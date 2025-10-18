#!/bin/bash

# ============================================================================
# Quick Test Run - UMMBAS v2.0
# ============================================================================
# This script runs a fast validation of the complete pipeline:
#   - Hyperparameter optimization (minimal grid)
#   - Generalization experiment (small dataset)
#
# Usage: bash quick_test_run.sh [--hyperparam-only]
#        --hyperparam-only: Only test hyperparameter sweep (faster)
# ============================================================================

set -e  # Exit on any error

HYPERPARAM_ONLY=false
if [[ "$1" == "--hyperparam-only" ]]; then
    HYPERPARAM_ONLY=true
fi

echo "========================================================================"
echo "UMMBAS v2.0 - Quick Test Run"
echo "========================================================================"

if [ "$HYPERPARAM_ONLY" = true ]; then
    echo "Mode: Hyperparameter test only"
    echo "This will:"
    echo "  1. Generate minimal hyperparameter configs (2 UMAP configs)"
    echo "  2. Run hyperparameter sweep with seed 42"
    echo "  3. Validate output"
    echo ""
    echo "Expected runtime: ~15-30 minutes"
else
    echo "Mode: Full pipeline test"
    echo "This will:"
    echo "  1. Generate minimal hyperparameter configs (2 UMAP configs)"
    echo "  2. Run hyperparameter sweep with seed 42"
    echo "  3. Generate test generalization config (small dataset)"
    echo "  4. Run generalization experiment with seed 42"
    echo "  5. Validate all outputs"
    echo ""
    echo "Expected runtime: ~30-45 minutes"
fi
echo "========================================================================"
echo ""

# ============================================================================
# PART 1: Hyperparameter Test
# ============================================================================
echo "========================================================================"
echo "PART 1: Hyperparameter Optimization Test"
echo "========================================================================"
echo ""

# Step 1a: Create minimal hyperparameter config generator
echo "Step 1a: Creating minimal hyperparameter test configs..."

# Create a temporary test hyperparameter config generator
cat > /tmp/generate_hyperparam_test.py << 'EOF'
import json
import os
import copy

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "hyperparam_configs_test"
SWEEP_WORKSPACE_DIR = "test_hyperparam_workspace/"
SWEEP_REPORT_DIR = "test_hyperparam_report/"
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

# Minimal test grid - just 2 configs per representation
FIXED_SIMSPACE_DIM = 2
UMAP_N_NEIGHBORS_VALUES = [50]  # Single value for speed
UMAP_MIN_DIST_VALUES = [0.01]  # Single value for speed

with open(BASE_CONFIG_FILE, 'r') as f:
    base_config = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

target_config = next((t for t in base_config.get("targets", []) if t.get("id_name") == TARGET_FOR_SWEEP), None)
if target_config:
    # Ensure v2.0 corrections
    target_config["molecular_function_canonical_name"] = "Transferase"
    target_config["molecular_function_filename_segment"] = "Transferase"
    target_config["molecular_function_display_name"] = "Transferase"
    target_config["molecular_function_kw_code"] = "KW-0808"

config_count = 0

for repr_type in ["features", "fingerprints"]:
    umap_key = "umap_euclidean" if repr_type == "features" else "umap_jaccard"
    
    for n_neighbors in UMAP_N_NEIGHBORS_VALUES:
        for min_dist in UMAP_MIN_DIST_VALUES:
            new_config = copy.deepcopy(base_config)
            new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
            new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
            new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
            new_config["representations"] = [repr_type]
            new_config["targets"] = [target_config]
            
            umap_cfg = copy.deepcopy(base_config["dimensionality_reduction_methods"][umap_key])
            umap_cfg["n_neighbors"] = n_neighbors
            umap_cfg["min_dist"] = min_dist
            new_config["dimensionality_reduction_methods"] = {umap_key: umap_cfg}

            filename = f"config_{repr_type}_{umap_key}_projection_nn{n_neighbors}_md{min_dist}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f: 
                json.dump(new_config, f, indent=2)
            config_count += 1

print(f"Created {config_count} minimal test configs in {OUTPUT_DIR}/")
EOF

python /tmp/generate_hyperparam_test.py

if [ $? -ne 0 ]; then
    echo "ERROR: Minimal config generation failed!"
    exit 1
fi

echo ""
echo "✓ Minimal hyperparameter configs created"
echo ""

# Step 1b: Run hyperparameter test
echo "Step 1b: Running hyperparameter test (features + fingerprints)..."

HYPERPARAM_CONFIGS=($(ls hyperparam_configs_test/*.json 2>/dev/null))
if [ ${#HYPERPARAM_CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: No hyperparameter test configs found!"
    exit 1
fi

echo "  Found ${#HYPERPARAM_CONFIGS[@]} configs to test"
echo ""

HYPERPARAM_SUCCESS=0
for CONFIG in "${HYPERPARAM_CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG")
    echo "  Testing: $CONFIG_NAME"
    
    python main_orchestrator.py --config "$CONFIG" --random_seed 42 > "test_hyperparam_${CONFIG_NAME%.json}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Success"
        ((HYPERPARAM_SUCCESS++))
    else
        echo "    ✗ Failed (check test_hyperparam_${CONFIG_NAME%.json}.log)"
    fi
done

echo ""
echo "Hyperparameter test results: $HYPERPARAM_SUCCESS/${#HYPERPARAM_CONFIGS[@]} configs succeeded"
echo ""

if [ $HYPERPARAM_SUCCESS -eq 0 ]; then
    echo "ERROR: All hyperparameter tests failed!"
    exit 1
fi

# Step 1c: Validate hyperparameter outputs
echo "Step 1c: Validating hyperparameter outputs..."

if [ ! -d "test_hyperparam_workspace" ]; then
    echo "ERROR: Hyperparameter workspace not created!"
    exit 1
fi

HYPERPARAM_RUNS=$(find test_hyperparam_workspace/ -maxdepth 1 -type d -name "*seed42" | wc -l)
echo "  ✓ Found $HYPERPARAM_RUNS hyperparameter run directories"

if [ $HYPERPARAM_SUCCESS -gt 0 ]; then
    FIRST_RUN=$(find test_hyperparam_workspace/ -maxdepth 1 -type d -name "*seed42" | head -1)
    if [ -f "$FIRST_RUN/rankings/held_out_actives_and_decoys_ranking_metrics.csv" ]; then
        echo "  ✓ Hyperparameter results validated"
    else
        echo "  ⚠️  Warning: Some output files may be missing"
    fi
fi

echo ""
echo "✓ HYPERPARAMETER TEST COMPLETED"
echo ""

# Exit here if hyperparam-only mode
if [ "$HYPERPARAM_ONLY" = true ]; then
    echo "========================================================================"
    echo "✓ HYPERPARAMETER TEST SUCCESSFUL!"
    echo "========================================================================"
    echo ""
    echo "Outputs located in:"
    echo "  - Workspace: test_hyperparam_workspace/"
    echo "  - Logs: test_hyperparam_*.log"
    echo ""
    echo "Next steps:"
    echo "  1. Generate full hyperparameter configs:"
    echo "     python config_generators/generate_hyperparam_configs.py"
    echo "  2. Submit to HPC:"
    echo "     bash hpc/submit_all_replicates_hyperparameterization.sh"
    echo "  3. See docs/HPC_EXECUTION_GUIDE.md for complete workflow"
    echo "========================================================================"
    exit 0
fi

# ============================================================================
# PART 2: Generalization Test (if not hyperparam-only)
# ============================================================================
echo "========================================================================"
echo "PART 2: Generalization Experiment Test"
echo "========================================================================"
echo ""

# Step 2a: Generate test configurations
echo "Step 2a: Generating test generalization config..."
python config_generators/generate_generalization_configs.py --test-run

if [ $? -ne 0 ]; then
    echo "ERROR: Generalization config generation failed!"
    exit 1
fi

echo "✓ Test generalization config created"
echo ""

# Step 2b: Find a test config to run
TEST_CONFIG=$(find generalization_configs/ -name "*_TEST.json" | head -1)

if [ -z "$TEST_CONFIG" ]; then
    echo "ERROR: No test generalization config found!"
    exit 1
fi

echo "Step 2b: Running generalization test..."
echo "  Config: $(basename $TEST_CONFIG)"
echo "  Seed: 42"
echo ""

# Run the orchestrator locally
python main_orchestrator.py --config "$TEST_CONFIG" --random_seed 42 2>&1 | tee test_generalization.log

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Generalization test failed! Check test_generalization.log"
    exit 1
fi

echo ""
echo "✓ Generalization test completed"
echo ""

# Step 2c: Validate generalization outputs
echo "Step 2c: Validating generalization outputs..."

if [ ! -d "test_experiment_workspace" ]; then
    echo "ERROR: Generalization workspace not created!"
    exit 1
fi

GEN_WORKSPACE=$(find test_experiment_workspace/ -maxdepth 1 -type d -name "*seed42" | head -1)

if [ -z "$GEN_WORKSPACE" ]; then
    echo "ERROR: Generalization workspace run directory not found!"
    exit 1
fi

echo "  Workspace: $GEN_WORKSPACE"

REQUIRED_FILES=(
    "rankings/held_out_actives_and_decoys_ranking_metrics.csv"
    "similarity_spaces/mf_cloud_coords.csv"
)

ALL_FOUND=true
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ -f "$GEN_WORKSPACE/$FILE" ]; then
        echo "  ✓ Found: $FILE"
    else
        echo "  ✗ Missing: $FILE"
        ALL_FOUND=false
    fi
done

echo ""

if [ "$ALL_FOUND" = true ]; then
    echo "========================================================================"
    echo "✓ COMPLETE PIPELINE TEST SUCCESSFUL!"
    echo "========================================================================"
    echo ""
    echo "Part 1: Hyperparameter Optimization"
    echo "  ✓ Tested $HYPERPARAM_SUCCESS configs"
    echo "  ✓ Outputs in: test_hyperparam_workspace/"
    echo ""
    echo "Part 2: Generalization Experiment"
    echo "  ✓ Pipeline validated"
    echo "  ✓ Outputs in: test_experiment_workspace/"
    echo ""
    echo "Logs:"
    echo "  - Hyperparameter: test_hyperparam_*.log"
    echo "  - Generalization: test_generalization.log"
    echo ""
    echo "Next steps:"
    echo "  1. Generate full hyperparameter configs:"
    echo "     python config_generators/generate_hyperparam_configs.py"
    echo "  2. Submit hyperparameter jobs to HPC:"
    echo "     bash hpc/submit_all_replicates_hyperparameterization.sh"
    echo "  3. After hyperparameter analysis, update optimal parameters"
    echo "  4. Generate generalization configs:"
    echo "     python config_generators/generate_generalization_configs.py"
    echo "  5. Submit generalization jobs to HPC:"
    echo "     bash hpc/submit_generalization_jobs.sh"
    echo ""
    echo "See docs/HPC_EXECUTION_GUIDE.md for complete workflow"
    echo "========================================================================"
    exit 0
else
    echo "========================================================================"
    echo "✗ GENERALIZATION TEST INCOMPLETE"
    echo "========================================================================"
    echo "Some expected output files are missing."
    echo "Check test_generalization.log for errors."
    echo "========================================================================"
    exit 1
fi
