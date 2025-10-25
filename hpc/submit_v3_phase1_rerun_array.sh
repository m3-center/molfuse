#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 1 RERUN Job Array Submission Script
# =============================================================================
# Submits all Phase 1 rerun configs as a SINGLE job array
# This bypasses HPC's "bf_max_job_user=10" limitation
# 
# Job arrays count as 1 job with N tasks, much more efficient than N individual jobs
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase1_rerun"
SLURM_SCRIPT="hpc/ummbas_v3_cpu_array.sh"
MAX_CONCURRENT=500  # Run max 500 tasks simultaneously

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 1 RERUN Array Submission"
echo "============================================================"

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM array script '${SLURM_SCRIPT}' not found."
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    echo "Run: python generate_phase1_configs.py"
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    echo "Run: python generate_phase1_configs.py"
    exit 1
fi

# --- Create Config File List ---
mkdir -p slurm_logs
CONFIG_LIST_FILE="slurm_logs/config_list_$(date +%Y%m%d_%H%M%S).txt"

# Sort configs to ensure consistent ordering
ls -1 "${CONFIG_DIR}"/*.json | sort > "${CONFIG_LIST_FILE}"

NUM_CONFIGS=$(wc -l < "${CONFIG_LIST_FILE}")

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Number of Configs: ${NUM_CONFIGS} (expected: 455)"
echo "Config List File: ${CONFIG_LIST_FILE}"
echo "Max Concurrent Tasks: ${MAX_CONCURRENT}"
echo ""
echo "CRITICAL CHANGES FROM ORIGINAL:"
echo "  - MF cloud deduplicated (2.23× → 1.0×)"
echo "  - Features UMAP nn: [10, 20, 50, 100, 500] (was [3, 5, 10, 20])"
echo "  - Fixed seed DISABLED (10× UMAP speedup)"
echo "  - Fingerprints also rerun (same duplication issue)"
echo "============================================================"

read -p "Proceed with array submission? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

# --- Submit Job Array ---
echo "Submitting job array..."
echo "------------------------------------------------------------"

# Array indices: 0 to (NUM_CONFIGS-1)
# %MAX_CONCURRENT limits concurrent execution
SUBMIT_OUTPUT=$(sbatch \
    --array=0-$((NUM_CONFIGS-1))%${MAX_CONCURRENT} \
    --output="slurm_logs/array_%A_task_%a.out" \
    --error="slurm_logs/array_%A_task_%a.err" \
    --job-name="UMMBAS_v3_phase1_rerun_array" \
    "${SLURM_SCRIPT}" "${CONFIG_LIST_FILE}" 2>&1)

if [ $? -eq 0 ]; then
    ARRAY_JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "SUCCESS: Job array submitted"
    echo "Array Job ID: ${ARRAY_JOB_ID}"
    echo "Number of tasks: ${NUM_CONFIGS}"
    echo "Max concurrent: ${MAX_CONCURRENT}"
else
    echo "ERROR: Array submission failed"
    echo "Error output: ${SUBMIT_OUTPUT}"
    exit 1
fi

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${ARRAY_JOB_ID} | head -20"
echo "  sacct -j ${ARRAY_JOB_ID} --format=JobID,State,ExitCode | head -20"
echo ""
echo "Check task status:"
echo "  # Running tasks"
echo "  squeue -j ${ARRAY_JOB_ID} -t RUNNING | wc -l"
echo "  # Completed tasks"
echo "  sacct -j ${ARRAY_JOB_ID} -s COMPLETED | wc -l"
echo "  # Failed tasks"
echo "  sacct -j ${ARRAY_JOB_ID} -s FAILED,CANCELLED,TIMEOUT | wc -l"
echo ""
echo "Check logs:"
echo "  ls -lht slurm_logs/array_${ARRAY_JOB_ID}_task_*.out | head -10"
echo "  tail -f slurm_logs/array_${ARRAY_JOB_ID}_task_0.out"
echo ""
echo "Cancel if needed:"
echo "  scancel ${ARRAY_JOB_ID}"
echo "============================================================"
