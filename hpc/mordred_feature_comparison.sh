#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=mordred_feature_comp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#
# SLURM submission script for Mordred full-feature comparison (PARALLELIZED for 64 CPUs)
# Runs tests/mordred_full_feature_eval/feature_comparison.py on HPC with parallel processing.
#
# This script compares three descriptor sets:
#   - Current 40-feature subset
#   - Full 2D Mordred descriptors (1613 features)
#   - Full 2D+3D Mordred descriptors (1826 features)
#
# Parallelization: Uses all 64 CPUs for:
#   - SMILES parsing and 3D embedding (multiprocessing Pool)
#   - UMAP neighbor search and optimization
#
# Usage examples:
#   sbatch hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,N_TARGET=300,N_MF=600,N_ZINC=600 hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,N_TARGET=500,N_MF=2000,N_ZINC=50000 hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,N_JOBS=32 hpc/mordred_feature_comparison.sh  # Use fewer cores if needed
#
# To override variables, pass them via --export=ALL,VAR=VALUE,...
# Common overrides: OUTPUT_DIR, N_TARGET, N_MF, N_ZINC, CACHE_DIR, N_JOBS
#

# Resolve repo root and move there so relative paths work
WORKDIR="${WORKDIR:-/home/ahagg2s/UMMBAS_screening_experiments}"
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults (override via SBATCH --export=ALL,VAR=value)
N_TARGET=${N_TARGET:-500}
N_MF=${N_MF:-2000}
N_ZINC=${N_ZINC:-50000}
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/output_comparison}
CACHE_DIR=${CACHE_DIR:-tests/mordred_full_feature_eval/cache}
UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS:-10}
UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.1}
ENABLE_SWEEP=${ENABLE_SWEEP:-1}
N_JOBS=${N_JOBS:--1}  # -1 = use all available CPUs (64)

# Coverage-aware selection thresholds
PF_TARGET=${PF_TARGET:-0.95}
PF_MF=${PF_MF:-0.7}
PR_TARGET_GUARD=${PR_TARGET_GUARD:-0.2}
PR_MF=${PR_MF:-0.6}
PR_ZINC=${PR_ZINC:-0.8}

# --- Environment Setup ---
echo "[SLURM] Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

# Log configuration
echo "[SLURM] Configuration:"
echo "  N_TARGET=${N_TARGET}"
echo "  N_MF=${N_MF}"
echo "  N_ZINC=${N_ZINC}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  CACHE_DIR=${CACHE_DIR}"
echo "  UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS}"
echo "  UMAP_MIN_DIST=${UMAP_MIN_DIST}"
echo "  ENABLE_SWEEP=${ENABLE_SWEEP}"
echo "  N_JOBS=${N_JOBS}"
echo "  CPUS_ALLOCATED=${SLURM_CPUS_PER_TASK:-unknown}"

# Build command
CMD=(
  python tests/mordred_full_feature_eval/feature_comparison.py
    --output_dir "${OUTPUT_DIR}"
    --n_target "${N_TARGET}"
    --n_mf "${N_MF}"
    --n_zinc "${N_ZINC}"
    --umap_n_neighbors "${UMAP_N_NEIGHBORS}"
    --umap_min_dist "${UMAP_MIN_DIST}"
    --cache_dir "${CACHE_DIR}"
    --n_jobs "${N_JOBS}"
    --pf_target "${PF_TARGET}"
    --pf_mf "${PF_MF}"
    --pr_target_guard "${PR_TARGET_GUARD}"
    --pr_mf "${PR_MF}"
    --pr_zinc "${PR_ZINC}"
)

if [ "${ENABLE_SWEEP}" = "1" ]; then
  CMD+=( --enable_sweep )
fi

echo "[SLURM] Running in $(pwd): ${CMD[*]}"
echo "[SLURM] Start time: $(date)"

"${CMD[@]}"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)"
echo "[SLURM] Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SLURM] ✓ Feature comparison completed successfully"
  echo "[SLURM] Results saved to: ${OUTPUT_DIR}"
  echo "[SLURM] Cache saved to: ${CACHE_DIR}"
else
  echo "[SLURM] ✗ Feature comparison failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
