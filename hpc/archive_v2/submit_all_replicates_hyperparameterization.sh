#!/bin/bash

# --- Configuration ---
# Directory where the hyperparameter sweep config files are located.
# This script will submit a job for every .json file in this directory.
CONFIG_DIR="hyperparam_configs"
# "generalization_configs" # "hyperparam_configs"

# Define your 5 random seeds for the replicates.
RANDOM_SEEDS=(42 43 44 45 46)

# The name of your SLURM script template.
# SLURM_SCRIPT_TEMPLATE="hpc/ummbas_hyperparameterization.sh" # GPU version
SLURM_SCRIPT_TEMPLATE="hpc/ummbas_hyperparameterization_cpu.sh" # CPU version for larger memory

# --- Pre-submission Checks ---
if [ ! -f "${SLURM_SCRIPT_TEMPLATE}" ]; then
    echo "ERROR: SLURM script template '${SLURM_SCRIPT_TEMPLATE}' not found."
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


# --- Loop and Submit All Jobs ---
echo "============================================================"
echo "Starting submission of UMMBAS hyperparameter sweep jobs..."
echo "============================================================"

JOB_COUNT=0

for config_file in "${CONFIG_DIR}"/*.json; do
    for seed in "${RANDOM_SEEDS[@]}"; do
        echo "--------------------------------------------------"
        echo "Submitting job for:"
        echo "  Config File: ${config_file}"
        echo "  Seed: ${seed}"
        
        config_basename=$(basename "${config_file}" .json)
        sbatch_job_name="UMMBAS_sweep_${config_basename}_s${seed}"

        # Submit the SLURM job, passing only the seed and the config file path
        sbatch --job-name="${sbatch_job_name}" "${SLURM_SCRIPT_TEMPLATE}" "${seed}" "${config_file}"
        
        if [ $? -eq 0 ]; then
            echo "Job submitted successfully."
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            echo "ERROR: Failed to submit job for Config=${config_file}, Seed=${seed}"
        fi
        
        # sleep 1
    done
done

echo "============================================================"
echo "Finished submission process."
echo "Total jobs submitted: ${JOB_COUNT}"
echo "============================================================"
