#!/bin/bash
# Quick diagnostic for array job status

echo "==================================================================="
echo "UMMBAS Phase 1 Rerun Array Job Diagnostics"
echo "==================================================================="
echo ""

# Find the array job ID
ARRAY_JOB_ID=$(squeue -u $USER -h -o "%A" | head -1)

if [ -z "${ARRAY_JOB_ID}" ]; then
    echo "No running jobs found. Checking recent history..."
    ARRAY_JOB_ID=$(sacct -u $USER -S $(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID --name=UMMBAS_v3_phase1_rerun_array -n | head -1 | cut -d'_' -f1)
fi

if [ -z "${ARRAY_JOB_ID}" ]; then
    echo "ERROR: Could not find array job ID"
    echo "Recent jobs:"
    sacct -u $USER -S $(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State | head -20
    exit 1
fi

echo "Array Job ID: ${ARRAY_JOB_ID}"
echo ""

# Job array configuration
echo "--- Job Array Configuration ---"
scontrol show job ${ARRAY_JOB_ID} | grep -E "ArrayTaskId|JobName|JobState|Reason"
echo ""

# Task state summary
echo "--- Task State Summary ---"
sacct -j ${ARRAY_JOB_ID} -X --format=State --noheader | sort | uniq -c | sort -rn
echo ""

# Detailed task states (first 50)
echo "--- Detailed Task States (first 50) ---"
sacct -j ${ARRAY_JOB_ID} --format=JobID%20,State%15,ExitCode%10,Elapsed%12 -X | head -51
echo ""

# Running tasks
echo "--- Currently Running Tasks ---"
RUNNING_COUNT=$(squeue -j ${ARRAY_JOB_ID} -t RUNNING -h | wc -l)
echo "Running: ${RUNNING_COUNT}"
squeue -j ${ARRAY_JOB_ID} -t RUNNING -o "%.18i %.8T %.10M %.6D %R" | head -20
echo ""

# Pending tasks
echo "--- Pending Tasks ---"
PENDING_COUNT=$(squeue -j ${ARRAY_JOB_ID} -t PENDING -h | wc -l)
echo "Pending: ${PENDING_COUNT}"
if [ ${PENDING_COUNT} -gt 0 ]; then
    squeue -j ${ARRAY_JOB_ID} -t PENDING -o "%.18i %.8T %50R" | head -10
fi
echo ""

# Failed tasks
echo "--- Failed/Cancelled Tasks (first 20) ---"
sacct -j ${ARRAY_JOB_ID} --format=JobID%20,State%15,ExitCode%10,Reason%30 -X | grep -E "FAILED|CANCELLED|TIMEOUT" | head -20
echo ""

# Check log files
echo "--- Log Files ---"
LOG_COUNT=$(ls slurm_logs/array_${ARRAY_JOB_ID}_task_*.out 2>/dev/null | wc -l)
echo "Log files found: ${LOG_COUNT}"

if [ ${LOG_COUNT} -gt 0 ]; then
    echo ""
    echo "Sample log (task 0):"
    if [ -f "slurm_logs/array_${ARRAY_JOB_ID}_task_0.out" ]; then
        head -30 "slurm_logs/array_${ARRAY_JOB_ID}_task_0.out"
    fi
    
    echo ""
    echo "Checking for common errors in first 10 logs:"
    for i in {0..9}; do
        if [ -f "slurm_logs/array_${ARRAY_JOB_ID}_task_${i}.out" ]; then
            ERROR=$(grep -i "ERROR\|FAILED\|Traceback" "slurm_logs/array_${ARRAY_JOB_ID}_task_${i}.out" | head -1)
            if [ -n "$ERROR" ]; then
                echo "  Task ${i}: ${ERROR}"
            fi
        fi
    done
fi

echo ""
echo "==================================================================="
echo "To cancel this job array:"
echo "  scancel ${ARRAY_JOB_ID}"
echo ""
echo "To monitor continuously:"
echo "  watch -n 10 'sacct -j ${ARRAY_JOB_ID} -X --format=State --noheader | sort | uniq -c'"
echo "==================================================================="
