#!/bin/bash
# Local test script for computational benchmark (small sample sizes only)

set -e

echo "========================================"
echo "Testing Computational Benchmark (Local)"
echo "========================================"

# Test with small sample sizes only
python scripts/benchmark_computational_performance.py \
    --workspace experiment_workspace_v4 \
    --zinc-fp-csv output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv \
    --zinc-feat-csv output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_features.csv \
    --phase1-run ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep1 \
    --phase4-run umap_features_Transferase_rep1 \
    --sample-sizes 1 10 100 \
    --output reporting/computational_benchmark_test

echo ""
echo "Test complete! Check: reporting/computational_benchmark_test/"
