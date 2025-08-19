#!/bin/bash

# --- Configuration ---
# Directory where the hyperparameter sweep config files are located.
# This script will submit a job for every .json file in this directory.
CONFIG_DIR="hyperparam_configs"

# Define your 5 random seeds for the replicates.
RANDOM_SEEDS=(42 43 44 45 46) # 43 44 45 46) 

# The name of your SLURM script template.
# SLURM_SCRIPT_TEMPLATE="hpc/ummbas_hyperparameterization.sh"
SLURM_SCRIPT_TEMPLATE="hpc/ummbas_hyperparameterization_cpu.sh"

# --- Pre-submission Checks ---

# Ensure the SLURM template script exists
if [ ! -f "${SLURM_SCRIPT_TEMPLATE}" ]; then
    echo "ERROR: SLURM script template '${SLURM_SCRIPT_TEMPLATE}' not found in the current directory."
    exit 1
fi

# Ensure the config directory exists and is not empty
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    echo "Please run 'python generate_hyperparam_configs.py' first."
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

# Loop through every .json file in the specified config directory
for config_file in "${CONFIG_DIR}"/*.json; do
    
    # Extract the representation mode from the config filename for logging and job naming
    # This makes assumptions about the filename format produced by the generator script.
    if [[ $config_file == *"_features_"* ]]; then
        repr_mode="features"
    elif [[ $config_file == *"_fingerprints_"* ]]; then
        repr_mode="fingerprints"
    else
        echo "WARNING: Could not determine representation mode from filename: ${config_file}"
        echo "         Skipping this configuration file."
        continue
    fi

    # Loop through each replicate seed
    for seed in "${RANDOM_SEEDS[@]}"; do
        echo "--------------------------------------------------"
        echo "Submitting job for:"
        echo "  Config File: ${config_file}"
        echo "  Representation: ${repr_mode}"
        echo "  Seed: ${seed}"
        
        # Construct a unique job name for `sbatch` to make `squeue` output clearer
        # Example: UMMBAS_fp_umap_n5_s42
        config_basename=$(basename "${config_file}" .json)
        # A simplified name for the job scheduler
        sbatch_job_name="UMMBAS_sweep_${config_basename}_s${seed}"
        sbatch_job_name=${sbatch_job_name//config_target_/} # Shorten it

        # Submit the SLURM job, passing representation mode, seed, and the config file path as arguments
        sbatch --job-name="${sbatch_job_name}" "${SLURM_SCRIPT_TEMPLATE}" "${repr_mode}" "${seed}" "${config_file}"
        
        # Check if the submission was successful
        if [ $? -eq 0 ]; then
            echo "Job submitted successfully."
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            echo "ERROR: Failed to submit job for Config=${config_file}, Seed=${seed}"
        fi
        
        # Optional: Add a small delay between submissions if your scheduler is sensitive
        sleep 2
    done
done

echo "============================================================"
echo "Finished submission process."
echo "Total jobs submitted: ${JOB_COUNT}"
echo "============================================================"
