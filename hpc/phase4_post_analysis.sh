#!/bin/bash
#SBATCH --job-name=phase4_post_analysis
#SBATCH --output=logs/phase4_post_analysis_%j.out
#SBATCH --error=logs/phase4_post_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=hpc

# Phase 4 Post-Analysis HPC Script
# Aggregates results, computes metrics, and generates publication plots
#
# Usage:
#   sbatch hpc/phase4_post_analysis.sh
#   sbatch hpc/phase4_post_analysis.sh --stratify           # With stratified analysis
#   sbatch hpc/phase4_post_analysis.sh --debug              # Quick debug mode
#   sbatch hpc/phase4_post_analysis.sh --stratify --debug   # Both options

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-experiment_workspace_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-reporting/phase4_post_analysis}"

# Parse command-line arguments for flags
STRATIFY_FLAG=""
DEBUG_FLAG=""

for arg in "$@"; do
    case $arg in
        --stratify)
            STRATIFY_FLAG="--stratify"
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=================================================="
echo "Phase 4 Post-Analysis Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Workspace: $WORKSPACE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Stratified analysis: ${STRATIFY_FLAG:-disabled}"
echo "Debug mode: ${DEBUG_FLAG:-disabled}"
echo "=================================================="
echo ""

# Load required modules (adjust for your HPC environment)
# module load python/3.11
# module load anaconda3

# Activate conda environment if needed
# conda activate molfuse_env

# Run the post-analysis script
echo "Starting Phase 4 post-analysis..."
python scripts/phase4_post_analysis.py \
    --workspace_dir "$WORKSPACE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $STRATIFY_FLAG \
    $DEBUG_FLAG

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Post-analysis completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"/*.{png,pdf,csv,json,md} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "✗ Post-analysis failed with exit code $EXIT_CODE"
    echo "Check error log: logs/phase4_post_analysis_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
