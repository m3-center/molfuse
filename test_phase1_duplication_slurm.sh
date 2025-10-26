#!/bin/bash
#SBATCH --job-name=test_phase1_dup
#SBATCH --output=logs/test_phase1_duplication_%j.out
#SBATCH --error=logs/test_phase1_duplication_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Test Impact of MF Cloud Duplicates on Phase 1 Results
# Tests whether deduplicating coordinates is sufficient or if DR models need retraining

echo "========================================================================"
echo "TESTING PHASE 1 DUPLICATION IMPACT"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Configuration
PHASE1_WORKSPACE="experiment_workspace_v3_phase1"
RUN_DIR="run_seed44_config_tyro_features_umap_euclidean_dim5_nn10_md0.1_seed44"
TARGET="TyrosineProteinKinaseABL1_P00519"
REPRESENTATION="features"
DR_METHOD="UMAP-Euclidean"
DIMENSION=5

echo "Configuration:"
echo "  Phase 1 workspace: ${PHASE1_WORKSPACE}"
echo "  Run directory: ${RUN_DIR}"
echo "  Target: ${TARGET}"
echo "  Representation: ${REPRESENTATION}"
echo "  DR method: ${DR_METHOD}"
echo "  Dimension: ${DIMENSION}"
echo ""

# Activate conda environment if needed
# Uncomment and adjust if you use conda:
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate your_env_name

# Run test script
echo "========================================================================"
echo "RUNNING TEST SCRIPT"
echo "========================================================================"
echo ""

python test_phase1_duplication_impact.py \
    --phase1_run "${PHASE1_WORKSPACE}/${RUN_DIR}" \
    --target "${TARGET}" \
    --representation "${REPRESENTATION}" \
    --dr_method "${DR_METHOD}" \
    --dimension ${DIMENSION}

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "========================================================================"
echo "Exit code: ${EXIT_CODE}"
echo ""

case ${EXIT_CODE} in
    0)
        echo "✓ RESULT: MINIMAL FIX ACCEPTABLE"
        echo "  - EF@1% change < 5%"
        echo "  - Can deduplicate existing coordinates without retraining DR models"
        echo "  - Saves ~80% of Phase 1 computation time"
        echo ""
        echo "NEXT STEPS:"
        echo "  1. Create script to deduplicate all Phase 1 similarity space coordinates"
        echo "  2. Recalculate distances and enrichment metrics for all configurations"
        echo "  3. Update analysis results"
        ;;
    1)
        echo "✗ RESULT: FULL RETRAIN REQUIRED"
        echo "  - EF@1% change ≥ 5%"
        echo "  - DR models learned distorted manifolds from duplicate data"
        echo "  - Must rerun entire Phase 1 with deduplicated MF cloud"
        echo ""
        echo "NEXT STEPS:"
        echo "  1. Fix prepare_data.py to deduplicate MF cloud before Phase 1"
        echo "  2. Fix calculate_similarityspaces_exp.py to deduplicate during loading"
        echo "  3. Rerun all Phase 1 experiments with clean data"
        echo "  4. Rerun all downstream analyses (Phase 2, 3, 4)"
        ;;
    2)
        echo "? RESULT: INCONCLUSIVE"
        echo "  - Unable to determine impact from this test"
        echo "  - Check log output for errors or missing data"
        echo ""
        echo "NEXT STEPS:"
        echo "  1. Review test output above for errors"
        echo "  2. Try different Phase 1 configuration"
        echo "  3. Check if files exist and have correct format"
        ;;
    *)
        echo "ERROR: Unexpected exit code ${EXIT_CODE}"
        ;;
esac

echo ""
echo "End time: $(date)"
echo "========================================================================"

exit ${EXIT_CODE}
