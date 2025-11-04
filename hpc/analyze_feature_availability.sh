#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --job-name=feature_availability_analysis
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#
# SLURM submission script for feature availability analysis
# Analyzes feature missingness patterns across three Mordred representations
#
# This script:
#   1. Loads actives, MF, ZINC for each representation (40-feature, full2d, full2d3d)
#   2. Computes per-feature missingness rates
#   3. Computes per-molecule NaN patterns (actives only)
#   4. Compares four feature selection strategies:
#      - training_only: Select based on MF+ZINC variance (current behavior)
#      - all_molecules: Only features with 100% coverage
#      - actives_first: Only features computable for ALL actives
#      - threshold_95: Features with ≥95% coverage + imputation
#   5. Generates comparison reports
#
# Resource requirements:
#   - 4 CPUs sufficient (no parallelism needed)
#   - 64 GB RAM sufficient for full dataset
#   - ~30-60 minutes runtime for full dataset
#   - <5 minutes with --test flag
#
# Usage examples:
#   # Quick test run (<5 min)
#   sbatch --export=ALL,TEST_MODE=1 hpc/analyze_feature_availability.sh
#
#   # Default: P00519 from KW-0808, all molecules
#   sbatch hpc/analyze_feature_availability.sh
#
#   # Subsample for faster testing
#   sbatch --export=ALL,N_MF=1000,N_ZINC=1000 hpc/analyze_feature_availability.sh
#
#   # Different target
#   sbatch --export=ALL,TARGET_ACCESSION=P00533 hpc/analyze_feature_availability.sh
#
# To override variables, pass them via --export=ALL,VAR=VALUE,...

# Resolve repo root
WORKDIR="${WORKDIR:-/home/ahagg2s/UMMBAS_screening_experiments}"
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults
TEST_MODE=${TEST_MODE:-0}

# Required: KW file and target accession
KW_FILE="${KW_FILE:-KW-0808_Transferase_affinity_extracted_features.csv}"
TARGET_ACCESSION="${TARGET_ACCESSION:-P00519}"

# Sampling parameters
N_MF=${N_MF:-}
N_TARGET=${N_TARGET:-}
N_ZINC=${N_ZINC:-}

# Data directories
BASE_DIR="${BASE_DIR:-.}"
FULL_2D_DIR="${FULL_2D_DIR:-/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d_all}"
FULL_2D3D_DIR="${FULL_2D3D_DIR:-/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d3d_all}"

# Output
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/feature_availability_${KW_FILE%.csv}_${TARGET_ACCESSION}}

# Misc
SEED=${SEED:-42}

# Test mode override
if [ "${TEST_MODE}" = "1" ]; then
  echo "[SLURM] TEST MODE ENABLED - Using minimal sample sizes"
  N_MF=50
  N_TARGET=20
  N_ZINC=50
  OUTPUT_DIR="tests/mordred_full_feature_eval/feature_availability_test_${TARGET_ACCESSION}"
fi

# Environment setup
echo "[SLURM] Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "[SLURM] Configuration:"
echo "  TEST_MODE=${TEST_MODE}"
echo "  KW_FILE=${KW_FILE}"
echo "  TARGET_ACCESSION=${TARGET_ACCESSION}"
echo "  N_MF=${N_MF:-all}"
echo "  N_TARGET=${N_TARGET:-all}"
echo "  N_ZINC=${N_ZINC:-all}"
echo "  BASE_DIR=${BASE_DIR}"
echo "  FULL_2D_DIR=${FULL_2D_DIR}"
echo "  FULL_2D3D_DIR=${FULL_2D3D_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SEED=${SEED}"
echo "  CPUS_ALLOCATED=${SLURM_CPUS_PER_TASK:-unknown}"

# Build command
CMD=(
  python tests/mordred_full_feature_eval/analyze_feature_availability.py
    --kw_file "${KW_FILE}"
    --target_accession "${TARGET_ACCESSION}"
    --base_dir "${BASE_DIR}"
    --full_2d_dir "${FULL_2D_DIR}"
    --full_2d3d_dir "${FULL_2D3D_DIR}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
)

# Add optional sampling parameters
if [ -n "${N_MF}" ]; then
  CMD+=( --n_mf "${N_MF}" )
fi

if [ -n "${N_TARGET}" ]; then
  CMD+=( --n_target "${N_TARGET}" )
fi

if [ -n "${N_ZINC}" ]; then
  CMD+=( --n_zinc "${N_ZINC}" )
fi

echo "[SLURM] Running in $(pwd)"
echo "[SLURM] Command: ${CMD[*]}"
echo "[SLURM] Start time: $(date)"

"${CMD[@]}"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)"
echo "[SLURM] Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SLURM] ✓ Feature availability analysis completed successfully"
  echo "[SLURM] Results saved to: ${OUTPUT_DIR}"
  echo "[SLURM]"
  echo "[SLURM] Output files:"
  echo "[SLURM]   - summary.json (high-level comparison)"
  echo "[SLURM]   - feature_missingness_*.csv (per-feature stats)"
  echo "[SLURM]   - molecule_nan_patterns_*.csv (per-molecule patterns)"
  echo "[SLURM]   - strategy_comparison_*.csv (strategy comparison)"
else
  echo "[SLURM] ✗ Feature availability analysis failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
