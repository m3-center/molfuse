#!/bin/bash

# --- Configuration ---
REPRESENTATION_MODES=("fingerprints" "features")
# Define your 5 random seeds
RANDOM_SEEDS=(42 43 44 45 46) 

SLURM_SCRIPT_TEMPLATE="hpc/ummbas_leaveoneout.sh" # Name of your SLURM template file

# --- Ensure SLURM template exists ---
if [ ! -f "${SLURM_SCRIPT_TEMPLATE}" ]; then
    echo "ERROR: SLURM script template '${SLURM_SCRIPT_TEMPLATE}' not found."
    exit 1
fi

# --- Loop and Submit ---
echo "Starting submission of UMMBAS replicate jobs..."

for repr_mode in "${REPRESENTATION_MODES[@]}"; do
    for seed in "${RANDOM_SEEDS[@]}"; do
        echo "--------------------------------------------------"
        echo "Submitting job for: Representation=${repr_mode}, Seed=${seed}"
        
        # Construct unique job name for sbatch command (optional, but good for tracking)
        # This is different from SLURM_JOB_NAME inside the script, this is for the `sbatch` command itself.
        sbatch_job_name="UMMBAS_${repr_mode}_s${seed}"
        
        # Submit the SLURM job, passing representation mode and seed as arguments
        sbatch --job-name="${sbatch_job_name}" "${SLURM_SCRIPT_TEMPLATE}" "${repr_mode}" "${seed}"
        
        if [ $? -eq 0 ]; then
            echo "Job submitted successfully."
        else
            echo "ERROR: Failed to submit job for Representation=${repr_mode}, Seed=${seed}"
        fi
        
        # Optional: Add a small delay between submissions if your scheduler prefers it
        sleep 1 
        echo "--------------------------------------------------"
    done
done

echo "All UMMBAS replicate jobs submitted."
