#!/bin/bash

# ============================================================================
# Submit Generalization Experiment Jobs to HPC
# ============================================================================
# This script submits SLURM jobs for the generalization experiment.
# It runs each configuration with 5 random seeds (42-46) to get statistical data.
#
# Usage: 
#   bash hpc/submit_generalization_jobs.sh              # Full run (5 seeds)
#   bash hpc/submit_generalization_jobs.sh --test-run   # Test run (3 seeds)
# ============================================================================

# Check for test run flag
TEST_RUN=false
if [[ "$1" == "--test-run" ]]; then
    TEST_RUN=true
fi

echo "========================================================================"
if [ "$TEST_RUN" = true ]; then
    echo "UMMBAS Generalization Experiment - TEST RUN Job Submission"
else
    echo "UMMBAS Generalization Experiment - FULL RUN Job Submission"
fi
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Random seeds for statistical analysis
if [ "$TEST_RUN" = true ]; then
    SEEDS=(42 43 44)  # Reduced seeds for test run
    CONFIG_SUFFIX="_TEST.json"
else
    SEEDS=(42 43 44 45 46)  # Full seeds for production run
    CONFIG_SUFFIX=".json"
fi

# Configuration directory
CONFIG_DIR="generalization_configs"

# Check if config directory exists
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found!"
    if [ "$TEST_RUN" = true ]; then
        echo "Please run: python config_generators/generate_generalization_configs.py --test-run"
    else
        echo "Please run: python config_generators/generate_generalization_configs.py"
    fi
    exit 1
fi

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Count total configs based on run type
if [ "$TEST_RUN" = true ]; then
    TOTAL_CONFIGS=$(find "${CONFIG_DIR}" -name "*${CONFIG_SUFFIX}" | wc -l)
else
    TOTAL_CONFIGS=$(find "${CONFIG_DIR}" -name "*.json" ! -name "*_TEST.json" | wc -l)
fi
TOTAL_JOBS=$((TOTAL_CONFIGS * ${#SEEDS[@]}))

echo "Configuration directory: ${CONFIG_DIR}"
echo "Run mode: $([ "$TEST_RUN" = true ] && echo "TEST RUN" || echo "FULL RUN")"
echo "Number of configs found: ${TOTAL_CONFIGS}"
echo "Number of seeds per config: ${#SEEDS[@]}"
echo "Total jobs to submit: ${TOTAL_JOBS}"
echo ""
echo "Random seeds: ${SEEDS[*]}"
if [ "$TEST_RUN" = true ]; then
    echo "Config suffix: ${CONFIG_SUFFIX}"
fi
echo ""
echo "========================================================================"
echo ""

# Counter for submitted jobs
SUBMITTED=0
FAILED=0

# Loop through JSON config files based on run type
if [ "$TEST_RUN" = true ]; then
    # Test run: only process *_TEST.json files
    CONFIG_PATTERN="${CONFIG_DIR}/*${CONFIG_SUFFIX}"
else
    # Full run: process all .json files except *_TEST.json
    CONFIG_PATTERN="${CONFIG_DIR}/*.json"
fi

for CONFIG_FILE in ${CONFIG_PATTERN}; do
    if [ ! -f "${CONFIG_FILE}" ]; then
        continue
    fi
    
    # Skip test configs in full run
    if [ "$TEST_RUN" = false ] && [[ "${CONFIG_FILE}" == *"_TEST.json" ]]; then
        continue
    fi
    
    CONFIG_BASENAME=$(basename "${CONFIG_FILE}")
    echo "Config: ${CONFIG_BASENAME}"
    
    # Submit job for each random seed
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="GEN_${CONFIG_BASENAME%.json}_seed${SEED}"
        
        # Submit the job
        SUBMIT_OUTPUT=$(sbatch \
            --job-name="${JOB_NAME}" \
            --output="slurm_logs/${JOB_NAME}_%j.out" \
            --error="slurm_logs/${JOB_NAME}_%j.err" \
            hpc/ummbas_generalization_cpu.sh "${SEED}" "${CONFIG_FILE}" 2>&1)
        
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
echo "End time: $(date)"
echo "========================================================================"
