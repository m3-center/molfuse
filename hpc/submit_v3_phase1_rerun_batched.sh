#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 1 RERUN Batched Array Submission
# =============================================================================
# Submits configs in batches of 50 tasks per array job
# This works around HPC limits on job array size
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase1_rerun"
SLURM_SCRIPT="hpc/ummbas_v3_cpu_array.sh"
BATCH_SIZE=50  # Maximum tasks per array job

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 1 RERUN Batched Array Submission"
echo "============================================================"

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM array script '${SLURM_SCRIPT}' not found."
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    exit 1
fi

# --- Create Config File List ---
mkdir -p slurm_logs
CONFIG_LIST_FILE="slurm_logs/config_list_$(date +%Y%m%d_%H%M%S).txt"

# Sort configs to ensure consistent ordering
ls -1 "${CONFIG_DIR}"/*.json | sort > "${CONFIG_LIST_FILE}"

NUM_CONFIGS=$(wc -l < "${CONFIG_LIST_FILE}")
NUM_BATCHES=$(( (NUM_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Total Configs: ${NUM_CONFIGS}"
echo "Batch Size: ${BATCH_SIZE} tasks per array"
echo "Number of Batches: ${NUM_BATCHES}"
echo "Config List File: ${CONFIG_LIST_FILE}"
echo ""
echo "This will submit ${NUM_BATCHES} separate job arrays"
echo "============================================================"

read -p "Proceed with batched submission? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

# --- Submit Batched Arrays ---
echo ""
echo "Starting batched submission..."
echo "============================================================"

SUBMISSION_LOG="slurm_logs/batched_submission_$(date +%Y%m%d_%H%M%S).log"
echo "Batched submission started at $(date)" > "${SUBMISSION_LOG}"
echo "Total configs: ${NUM_CONFIGS}" >> "${SUBMISSION_LOG}"
echo "Batch size: ${BATCH_SIZE}" >> "${SUBMISSION_LOG}"
echo "Number of batches: ${NUM_BATCHES}" >> "${SUBMISSION_LOG}"
echo "" >> "${SUBMISSION_LOG}"

SUBMITTED_ARRAYS=()
FAILED_BATCHES=0

for ((batch=0; batch<NUM_BATCHES; batch++)); do
    START_IDX=$((batch * BATCH_SIZE))
    END_IDX=$((START_IDX + BATCH_SIZE - 1))
    
    # Don't exceed total config count
    if [ $END_IDX -ge $NUM_CONFIGS ]; then
        END_IDX=$((NUM_CONFIGS - 1))
    fi
    
    BATCH_NAME="UMMBAS_v3_phase1_rerun_batch${batch}"
    
    echo "Submitting batch ${batch}: tasks ${START_IDX}-${END_IDX} ($(($END_IDX - $START_IDX + 1)) tasks)"
    
    SUBMIT_OUTPUT=$(sbatch \
        --array=${START_IDX}-${END_IDX} \
        --output="slurm_logs/batch${batch}_task_%a.out" \
        --error="slurm_logs/batch${batch}_task_%a.err" \
        --job-name="${BATCH_NAME}" \
        "${SLURM_SCRIPT}" "${CONFIG_LIST_FILE}" 2>&1)
    
    if [ $? -eq 0 ]; then
        ARRAY_JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
        echo "  ✓ SUCCESS: Job ${ARRAY_JOB_ID}"
        echo "Batch ${batch}: Job ${ARRAY_JOB_ID} (tasks ${START_IDX}-${END_IDX})" >> "${SUBMISSION_LOG}"
        SUBMITTED_ARRAYS+=("${ARRAY_JOB_ID}")
    else
        echo "  ✗ FAILED: ${SUBMIT_OUTPUT}"
        echo "Batch ${batch}: FAILED - ${SUBMIT_OUTPUT}" >> "${SUBMISSION_LOG}"
        FAILED_BATCHES=$((FAILED_BATCHES + 1))
    fi
    
    # Small delay to avoid overwhelming scheduler
    sleep 1
done

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Batches submitted: $((NUM_BATCHES - FAILED_BATCHES))"
echo "Batches failed: ${FAILED_BATCHES}"
echo "Submission log: ${SUBMISSION_LOG}"
echo ""
echo "Array Job IDs:"
printf '  %s\n' "${SUBMITTED_ARRAYS[@]}"
echo ""
echo "============================================================"
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER | wc -l"
echo ""
echo "Check all batches:"
for job_id in "${SUBMITTED_ARRAYS[@]}"; do
    echo "  sacct -j ${job_id} -X --format=State --noheader | sort | uniq -c"
done
echo ""
echo "Cancel all batches if needed:"
echo "  scancel ${SUBMITTED_ARRAYS[*]}"
echo ""
echo "Check completion:"
echo "  ls slurm_logs/batch*_task_*.out | wc -l  # Should reach ${NUM_CONFIGS}"
echo "============================================================"
