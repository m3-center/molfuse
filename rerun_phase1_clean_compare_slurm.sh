#!/bin/bash
#SBATCH --job-name=phase1_clean_compare
#SBATCH --output=logs/phase1_clean_compare_%j.out
#SBATCH --error=logs/phase1_clean_compare_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=160GB
#SBATCH --cpus-per-task=64
#SBATCH --partition=hpc

# Rerun Phase 1 Configuration with Deduplicated Data and Compare
# Tests whether MF cloud duplicates significantly affect DR model training and enrichment

echo "========================================================================"
echo "RERUN PHASE 1 WITH DEDUPLICATED DATA - COMPARE TO ORIGINAL"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "Start time: $(date)"
echo ""

# Configuration
ORIGINAL_RUN="experiment_workspace_v3_phase1/run_seed44_config_tyro_features_umap_euclidean_dim5_nn10_md0.1_seed44"
OUTPUT_WORKSPACE="experiment_workspace_v3_phase1_clean_test"
EXPERIMENT_CONFIG="experiment_config.json"

echo "Configuration:"
echo "  Original run: ${ORIGINAL_RUN}"
echo "  Output workspace: ${OUTPUT_WORKSPACE}"
echo "  Experiment config: ${EXPERIMENT_CONFIG}"
echo ""

# Verify original run exists
if [ ! -d "${ORIGINAL_RUN}" ]; then
    echo "ERROR: Original run directory not found: ${ORIGINAL_RUN}"
    exit 1
fi

# Verify experiment config exists
if [ ! -f "${EXPERIMENT_CONFIG}" ]; then
    echo "ERROR: Experiment config not found: ${EXPERIMENT_CONFIG}"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_WORKSPACE}"
mkdir -p logs

# Activate conda environment if needed
# Uncomment and adjust if you use conda:
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate your_env_name
mamba activate ummbas-screening

echo "========================================================================"
echo "PIPELINE FIXES APPLIED:"
echo "========================================================================"
echo "✓ prepare_data.py - Deduplicates MF cloud using minimum affinity"
echo "✓ calculate_similarityspaces_exp.py - Safety check for duplicates"
echo ""
echo "This will:"
echo "1. Run prepare_data.py to create deduplicated MF cloud"
echo "2. Train DR model (UMAP-Euclidean 5D) on clean data"
echo "3. Project target ligands and calculate enrichment metrics"
echo "4. Compare to original results from contaminated data"
echo "5. Determine if full Phase 1 rerun is necessary"
echo ""

# Run comparison script
echo "========================================================================"
echo "RUNNING COMPARISON SCRIPT"
echo "========================================================================"
echo ""

python rerun_phase1_config_compare.py \
    --original_run "${ORIGINAL_RUN}" \
    --output_workspace "${OUTPUT_WORKSPACE}" \
    --experiment_config "${EXPERIMENT_CONFIG}"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "COMPARISON COMPLETE"
echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"
echo ""

case ${EXIT_CODE} in
    0)
        echo "✓ RESULT: MINIMAL IMPACT"
        echo ""
        echo "  EF@1% change < 5%"
        echo "  DR models appear robust to duplicate data in training"
        echo ""
        echo "INTERPRETATION:"
        echo "  - Despite duplicates in MF cloud, UMAP learned similar manifold structure"
        echo "  - Enrichment metrics are not significantly different"
        echo "  - Existing Phase 1 results are likely valid"
        echo ""
        echo "RECOMMENDATION:"
        echo "  - Can proceed with current Phase 1 results"
        echo "  - Document this as a limitation in methods section"
        echo "  - Optional: Test other configs (PCA, different dims) to confirm robustness"
        echo "  - Use deduplicated pipeline for future runs"
        echo ""
        echo "NEXT STEPS:"
        echo "  1. Review comparison_seed44.json for detailed metrics"
        echo "  2. Optionally test 1-2 more configurations (different DR methods)"
        echo "  3. Update methods documentation to note this issue and validation"
        echo "  4. Continue with Phase 2/3/4 analyses as planned"
        ;;
    1)
        echo "✗ RESULT: SIGNIFICANT IMPACT"
        echo ""
        echo "  EF@1% change ≥ 5%"
        echo "  DR models learned distorted manifolds from duplicate data"
        echo "  Enrichment metrics are significantly different"
        echo ""
        echo "INTERPRETATION:"
        echo "  - Duplicates in MF cloud caused UMAP to learn wrong manifold structure"
        echo "  - Distance calculations are biased toward duplicate-dense regions"
        echo "  - Published Phase 1 results are INVALID"
        echo ""
        echo "RECOMMENDATION:"
        echo "  - MUST rerun ALL Phase 1 experiments with deduplicated data"
        echo "  - MUST update ALL downstream analyses (Phase 2, 3, 4)"
        echo "  - MUST rerun stratified enrichment analysis"
        echo ""
        echo "CRITICAL NEXT STEPS:"
        echo "  1. Review comparison_seed44.json to understand magnitude of impact"
        echo "  2. Test 1-2 other configs (PCA, different seeds) to confirm"
        echo "  3. If confirmed: Schedule full Phase 1 rerun on HPC"
        echo "  4. Update all analysis scripts to use new Phase 1 data"
        echo "  5. Rerun Phase 2, 3, 4 experiments"
        echo "  6. Update all figures and tables in manuscript"
        ;;
    2)
        echo "? RESULT: PIPELINE FAILED"
        echo ""
        echo "  Script encountered an error during execution"
        echo "  Check logs for details"
        echo ""
        echo "TROUBLESHOOTING:"
        echo "  1. Review error messages above"
        echo "  2. Check that original run directory exists and is complete"
        echo "  3. Verify experiment_config.json is valid"
        echo "  4. Check file permissions and disk space"
        echo "  5. Try running prepare_data.py manually to isolate issue"
        ;;
    *)
        echo "ERROR: Unexpected exit code ${EXIT_CODE}"
        ;;
esac

echo ""
echo "OUTPUT LOCATION:"
echo "  Clean rerun data: ${OUTPUT_WORKSPACE}"
echo "  Comparison JSON: ${OUTPUT_WORKSPACE}/comparison_seed44.json"
echo "  Job log: logs/phase1_clean_compare_${SLURM_JOB_ID}.out"
echo ""
echo "End time: $(date)"
echo "========================================================================"

exit ${EXIT_CODE}
