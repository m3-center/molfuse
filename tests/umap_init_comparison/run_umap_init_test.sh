#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=umap_init_test
#SBATCH --output=tests/umap_init_comparison/slurm_logs/umap_init_test_%j.out
#SBATCH --error=tests/umap_init_comparison/slurm_logs/umap_init_test_%j.err

# UMAP Initialization Method Comparison Test
# 
# Tests different UMAP initialization strategies vs PCA baseline:
# - PCA 20D (baseline)
# - UMAP 20D with init='spectral', 'random', 'pca', 'tswspectral'
#
# Usage:
#   sbatch tests/umap_init_comparison/run_umap_init_test.sh

# Ensure directories exist
mkdir -p tests/umap_init_comparison/slurm_logs
mkdir -p tests/umap_init_comparison/workspace
mkdir -p tests/umap_init_comparison/results

# Log startup information
echo "=========================================="
echo "UMAP Initialization Comparison Test"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 200G"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Verify Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Generate configs if they don't exist
if [ ! -d "tests/umap_init_comparison/configs" ]; then
    echo "Generating test configurations..."
    python tests/umap_init_comparison/generate_configs.py
    echo ""
fi

# List configs
CONFIGS=(tests/umap_init_comparison/configs/*.json)
NUM_CONFIGS=${#CONFIGS[@]}

echo "=========================================="
echo "Found ${NUM_CONFIGS} configurations to run"
echo "=========================================="
echo ""

# Run each config sequentially
SUCCESS_COUNT=0
FAIL_COUNT=0

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .json)
    
    echo "=========================================="
    echo "Running: ${CONFIG_NAME}"
    echo "Config: ${CONFIG}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    
    # Run the test
    python tests/umap_init_comparison/umap_init_test.py \
        --config "${CONFIG}" \
        --workspace tests/umap_init_comparison/workspace \
        --output tests/umap_init_comparison/results
    
    EXIT_CODE=$?
    
    echo ""
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ ${CONFIG_NAME} completed successfully"
        ((SUCCESS_COUNT++))
    else
        echo "✗ ${CONFIG_NAME} failed with exit code ${EXIT_CODE}"
        ((FAIL_COUNT++))
    fi
    echo "End Time: $(date)"
    echo ""
done

# Summary
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="
echo "Total configs: ${NUM_CONFIGS}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo "Job End Time: $(date)"
echo "=========================================="

# Aggregate results if all successful
if [ ${FAIL_COUNT} -eq 0 ]; then
    echo ""
    echo "Generating comparison summary..."
    python tests/umap_init_comparison/aggregate_results.py \
        --results_dir tests/umap_init_comparison/results/results \
        --output tests/umap_init_comparison/results/comparison_summary.json
    
    if [ $? -eq 0 ]; then
        echo "✓ Comparison summary generated"
        cat tests/umap_init_comparison/results/comparison_summary.json
    fi
fi

# Exit with failure if any test failed
if [ ${FAIL_COUNT} -gt 0 ]; then
    exit 1
fi

exit 0
