#!/bin/bash

# ============================================================================
# Submit Dimensionality Experiment Jobs to HPC
# ============================================================================
# This script submits SLURM jobs for the dimensionality experiment.
# It runs each configuration with 5 random seeds (42-46) to get statistical data.
#
# Usage: bash hpc/submit_dimensionality_jobs.sh
# ============================================================================

echo "========================================================================"
echo "UMMBAS Dimensionality Experiment - Job Submission Script"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Random seeds for statistical analysis (same as all other experiments)
SEEDS=(42 43 44 45 46)

# Configuration directory
CONFIG_DIR="dimensionality_configs"

# Check if config directory exists
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found!"
    echo "Please run: python config_generators/generate_dimensionality_configs.py"
    exit 1
fi

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Count total configs
TOTAL_CONFIGS=$(find "${CONFIG_DIR}" -name "*.json" | wc -l)
TOTAL_JOBS=$((TOTAL_CONFIGS * ${#SEEDS[@]}))

echo "Configuration directory: ${CONFIG_DIR}"
echo "Number of configs found: ${TOTAL_CONFIGS}"
echo "Number of seeds per config: ${#SEEDS[@]}"
echo "Total jobs to submit: ${TOTAL_JOBS}"
echo ""
echo "Random seeds: ${SEEDS[*]}"
echo ""
echo "Note: Each config tests multiple dimensions [2, 3, 5, 10, 20]"
echo "      These are processed within each job automatically."
echo ""
echo "========================================================================"
echo ""

# Counter for submitted jobs
SUBMITTED=0
FAILED=0

# Loop through all JSON config files
for CONFIG_FILE in "${CONFIG_DIR}"/*.json; do
    if [ ! -f "${CONFIG_FILE}" ]; then
        continue
    fi
    
    CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
    echo "Config: ${CONFIG_BASENAME}"
    
    # Submit job for each random seed
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="DIM_${CONFIG_BASENAME%.json}_seed${SEED}"
        
        # Submit the job
        SUBMIT_OUTPUT=$(sbatch \
            --job-name="${JOB_NAME}" \
            --output="slurm_logs/${JOB_NAME}_%j.out" \
            --error="slurm_logs/${JOB_NAME}_%j.err" \
            hpc/ummbas_dimensionality_cpu.sh "${SEED}" "${CONFIG_FILE}" 2>&1)
        
        if [ $? -eq 0 ]; then
            JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'Submitted batch job \K\d+')
            echo "  ✓ Seed ${SEED}: Job ${JOB_ID} submitted"
            ((SUBMITTED++))
        else
            echo "  ✗ Seed ${SEED}: FAILED - ${SUBMIT_OUTPUT}"
            ((FAILED++))
        fi
    done
    echo ""
done

echo "========================================================================"
echo "Job Submission Summary"
echo "========================================================================"
echo "Successfully submitted: ${SUBMITTED}"
echo "Failed: ${FAILED}"
echo "Total: $((SUBMITTED + FAILED))"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: slurm_logs/"
echo ""
echo "After jobs complete, run:"
echo "  python analysis_scripts/aggregate_dimensionality_analysis.py"
echo ""
echo "End time: $(date)"
echo "========================================================================"
