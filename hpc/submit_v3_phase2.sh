#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Phase 2 Job Submission Script
# =============================================================================
# Phase 2: Affinity Cutoff Sensitivity Analysis [REORDERED - was Phase 4]
# - Target: TyrosineProteinKinaseABL1 (Tyro)
# - Cutoffs: 100 nM, 1 μM, 10 μM, 100 μM (4 cutoffs)
# - Configurations: PCA/features/5D + 2×UMAP/features/5D (3 configs)
# - Seeds: 42, 43, 44, 45, 46 (5 seeds)
# - Total: 60 runs (4 cutoffs × 3 configs × 5 seeds)
# 
# COMPUTATIONAL STRATEGY:
# - REUSES Phase 1 similarity spaces (features/fingerprints)
# - REUSES Phase 1 DR models (fitted PCA/UMAP)
# - ONLY reruns ranking with different affinity cutoffs
# - Expected runtime: ~10-15 min/run (vs hours for full pipeline)
# =============================================================================

# --- Configuration ---
CONFIG_DIR="hyperparam_configs_v3_phase2_cutoff"
PHASE1_WORKSPACE="experiment_workspace_v3_phase1"
PHASE2_WORKSPACE="experiment_workspace_v3_phase2"
ORCHESTRATOR_SCRIPT="scripts/run_phase2_cutoff_analysis.py"

# SLURM settings for Phase 2 (shorter runtime than Phase 1)
PARTITION="hpc"        # Use short partition for quick jobs
TIME_LIMIT="02:00:00"    # 30 minutes should be plenty
MEMORY="32G"              # Less memory needed than full pipeline
CPUS=32                   # Single-threaded

# NOTE: Each config file already contains a specific seed.
# We do NOT loop over seeds here - that would create duplicate jobs!

# --- Pre-submission Checks ---
echo "============================================================"
echo "UMMBAS v3.0 - Phase 2 Cutoff Sensitivity Submission"
echo "============================================================"

if [ ! -f "${ORCHESTRATOR_SCRIPT}" ]; then
    echo "ERROR: Orchestrator script '${ORCHESTRATOR_SCRIPT}' not found."
    exit 1
fi

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "ERROR: Configuration directory '${CONFIG_DIR}' not found."
    echo "Run: python generate_phase2_configs.py"
    exit 1
fi

if [ ! -d "${PHASE1_WORKSPACE}" ]; then
    echo "ERROR: Phase 1 workspace '${PHASE1_WORKSPACE}' not found."
    echo "Phase 2 requires Phase 1 data to reuse!"
    exit 1
fi

if [ -z "$(ls -A ${CONFIG_DIR}/*.json 2>/dev/null)" ]; then
    echo "ERROR: No .json configuration files found in '${CONFIG_DIR}'."
    echo "Run: python generate_phase2_configs.py"
    exit 1
fi

# Count config files
NUM_CONFIGS=$(ls -1 ${CONFIG_DIR}/*.json | wc -l)

echo "Configuration Directory: ${CONFIG_DIR}"
echo "Number of Configs: ${NUM_CONFIGS}"
echo "Expected Jobs: 60 (4 cutoffs × 3 configs × 5 seeds)"
echo "Phase 1 Workspace: ${PHASE1_WORKSPACE}"
echo "Phase 2 Workspace: ${PHASE2_WORKSPACE}"
echo "Expected Runtime: ~10-15 min per job"
echo "============================================================"

read -p "Proceed with submission? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

# --- Create directories ---
mkdir -p slurm_logs
mkdir -p "${PHASE2_WORKSPACE}"

# --- Submit All Jobs ---
echo "Starting job submission..."
echo "------------------------------------------------------------"

JOB_COUNT=0
FAILED_COUNT=0

# Submit each config as a separate job calling the Phase 2 orchestrator
for config_file in "${CONFIG_DIR}"/*.json; do
    config_basename=$(basename "${config_file}" .json)
    job_name="UMMBAS_v3_phase2_${config_basename}"
    
    # Extract seed from config filename (format: config_seed42_...)
    seed=$(echo "${config_basename}" | grep -oP 'seed\K\d+' || echo "unknown")
    
    # Create inline SLURM submission script
    cat > /tmp/phase2_job_${config_basename}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=slurm_logs/${job_name}_%j.out
#SBATCH --error=slurm_logs/${job_name}_%j.err

echo "============================================================"
echo "UMMBAS v3.0 - Phase 2 Cutoff Analysis"
echo "============================================================"
echo "Job: ${job_name}"
echo "Config: ${config_file}"
echo "Started: \$(date)"
echo "Node: \$(hostname)"
echo "============================================================"

# Load environment (adapt to your cluster)
# module load python/3.9
# source activate ummbas_env

# Run Phase 2 cutoff analysis (reuses Phase 1 data)
python ${ORCHESTRATOR_SCRIPT} \\
    --config "${config_file}" \\
    --phase1_workspace "${PHASE1_WORKSPACE}" \\
    --phase2_workspace "${PHASE2_WORKSPACE}"

exit_code=\$?

echo "============================================================"
echo "Job finished: \$(date)"
echo "Exit code: \${exit_code}"
echo "============================================================"

exit \${exit_code}
EOF
    
    # Submit the job
    sbatch /tmp/phase2_job_${config_basename}.sh
    
    if [ $? -eq 0 ]; then
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "ERROR: Failed to submit job for ${config_file}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # Clean up temporary script
    rm -f /tmp/phase2_job_${config_basename}.sh
done

echo "============================================================"
echo "Submission Complete"
echo "============================================================"
echo "Jobs submitted: ${JOB_COUNT}"
echo "Jobs failed: ${FAILED_COUNT}"
echo "============================================================"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName,State,ExitCode"
echo ""
echo "Check Phase 2 results:"
echo "  ls -lh ${PHASE2_WORKSPACE}"
echo ""
echo "After completion, analyze Phase 2 results to determine optimal cutoff,"
echo "then proceed to Phase 3 (MF cloud ablation)."
echo "============================================================"
