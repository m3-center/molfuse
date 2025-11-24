#!/bin/bash
#SBATCH --job-name=molfuse_benchmark
#SBATCH --output=slurm_logs/benchmark_computational_%j.out
#SBATCH --error=slurm_logs/benchmark_computational_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --partition=any

# =============================================================================
# Computational Performance Benchmark for MolFuSE
# =============================================================================
# 
# Purpose: Benchmark wall-clock time for scoring candidates using:
#   1. Raw ECFP4 (Tanimoto, no UMAP)
#   2. Raw Features (Euclidean, no UMAP)
#   3. ECFP4 + UMAP (20D, pretrained)
#   4. Features + UMAP (2D, pretrained)
#
# Tests: 1, 10, 100, 1K, 10K, 100K candidate molecules
#
# Usage:
#   sbatch hpc/benchmark_computational_performance.sh
#
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Load conda environment
source ~/.bashrc
conda activate ummbas-screening-mordredcommunity

echo "=========================================="
echo "MolFuSE Computational Performance Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Paths
WORKSPACE="experiment_workspace_v4"
ZINC_FP_CSV="output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv"
ZINC_FEAT_CSV="output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_features_2D.csv"
PHASE1_RUN="ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep1"
PHASE4_RUN="umap_features_Transferase_rep1"
OUTPUT_DIR="reporting/computational_benchmark"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

# Check if required files exist
echo "Checking input files..."
if [ ! -f "$ZINC_FP_CSV" ]; then
    echo "ERROR: ZINC fingerprints CSV not found: $ZINC_FP_CSV"
    exit 1
fi

if [ ! -f "$ZINC_FEAT_CSV" ]; then
    echo "ERROR: ZINC features CSV not found: $ZINC_FEAT_CSV"
    exit 1
fi

if [ ! -d "$WORKSPACE/phase1/$PHASE1_RUN" ]; then
    echo "ERROR: Phase 1 run not found: $WORKSPACE/phase1/$PHASE1_RUN"
    exit 1
fi

if [ ! -d "$WORKSPACE/phase4/cross_target/$PHASE4_RUN" ]; then
    echo "ERROR: Phase 4 run not found: $WORKSPACE/phase4/cross_target/$PHASE4_RUN"
    exit 1
fi

echo "  ✓ All input files found"
echo ""

# Run benchmark
echo "Running computational benchmark..."
echo "Sample sizes: 1, 10, 100, 1000, 10000, 100000"
echo ""

python scripts/benchmark_computational_performance.py \
    --workspace "$WORKSPACE" \
    --zinc-fp-csv "$ZINC_FP_CSV" \
    --zinc-feat-csv "$ZINC_FEAT_CSV" \
    --phase1-run "$PHASE1_RUN" \
    --phase4-run "$PHASE4_RUN" \
    --sample-sizes 1 10 100 1000 10000 100000 \
    --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - benchmark_results.csv (detailed timing data)"
echo "  - benchmark_table.tex (LaTeX table for paper)"
echo ""
echo "Job finished: $(date)"
