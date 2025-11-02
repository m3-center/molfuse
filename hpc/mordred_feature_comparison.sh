#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=mordred_feature_comp_v2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#
# SLURM submission script for Mordred full-feature comparison v2 (pre-computed features)
# Runs tests/mordred_full_feature_eval/feature_comparison_v2.py on HPC.
#
# This script compares three descriptor sets using PRE-COMPUTED features:
#   - Current 40-feature subset (from datasets/molecular_function_features_fingerprints/)
#   - Full 2D Mordred descriptors (1613 features, from datasets_2d_all/)
#   - Full 2D+3D Mordred descriptors (1826 features, from datasets_2d3d_all/)
#
# Key differences from v1:
#   - NO descriptor computation (much faster)
#   - Loads from pre-computed CSV files
#   - Uses Phase 1 evaluation protocol (within-file accession split)
#   - Requires completed dataset recreation (recreate_datasets.py)
#
# Resource requirements:
#   - 16 CPUs sufficient (UMAP parallelism only)
#   - 128 GB RAM sufficient (no descriptor computation)
#   - ~2-4 hours runtime (vs 24+ hours for v1)
#
# Usage examples:
#   sbatch hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,KW_FILE=KW-0808_Transferase_affinity_extracted_features.csv,TARGET_ACCESSION=P00519 hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,N_MF=600,N_ZINC=600 hpc/mordred_feature_comparison.sh
#   sbatch --export=ALL,DRY_RUN=1 hpc/mordred_feature_comparison.sh  # Dry run (print command only)
#
# To override variables, pass them via --export=ALL,VAR=VALUE,...
# Common overrides: KW_FILE, TARGET_ACCESSION, N_MF, N_TARGET, N_ZINC, OUTPUT_DIR, DRY_RUN
#

# Resolve repo root and move there so relative paths work
WORKDIR="${WORKDIR:-/home/ahagg2s/UMMBAS_screening_experiments}"
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults (override via SBATCH --export=ALL,VAR=value)
# Required: KW file and target accession for Phase 1 style split
KW_FILE="${KW_FILE:-KW-0808_Transferase_affinity_extracted_features.csv}"
TARGET_ACCESSION="${TARGET_ACCESSION:-P00519}"

# Sampling parameters (None = use all)
N_MF=${N_MF:-}          # MF cloud sample size (blank = all)
N_TARGET=${N_TARGET:-}  # Target actives sample size (blank = all)
N_ZINC=${N_ZINC:-600}   # ZINC decoys sample size

# Data directories (pre-computed features)
BASE_DIR="${BASE_DIR:-.}"
FULL_2D_DIR="${FULL_2D_DIR:-/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d_all}"
FULL_2D3D_DIR="${FULL_2D3D_DIR:-/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d3d_all}"

# Evaluation parameters
AFFINITY_CUTOFF_NM=${AFFINITY_CUTOFF_NM:-100}
SEED=${SEED:-42}

# UMAP parameters
UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS:-15}
UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.1}

# Output
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/output_v2_${KW_FILE%.csv}_${TARGET_ACCESSION}}

# Dry run flag (set to 1 to print command only)
DRY_RUN=${DRY_RUN:-0}

# --- Environment Setup ---
echo "[SLURM] Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "[SLURM] Configuration:"
echo "  KW_FILE=${KW_FILE}"
echo "  TARGET_ACCESSION=${TARGET_ACCESSION}"
echo "  N_MF=${N_MF:-all}"
echo "  N_TARGET=${N_TARGET:-all}"
echo "  N_ZINC=${N_ZINC}"
echo "  AFFINITY_CUTOFF_NM=${AFFINITY_CUTOFF_NM}"
echo "  SEED=${SEED}"
echo "  BASE_DIR=${BASE_DIR}"
echo "  FULL_2D_DIR=${FULL_2D_DIR}"
echo "  FULL_2D3D_DIR=${FULL_2D3D_DIR}"
echo "  UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS}"
echo "  UMAP_MIN_DIST=${UMAP_MIN_DIST}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  DRY_RUN=${DRY_RUN}"
echo "  CPUS_ALLOCATED=${SLURM_CPUS_PER_TASK:-unknown}"

# Build command
CMD=(
  python tests/mordred_full_feature_eval/feature_comparison_v2.py
    --kw_file "${KW_FILE}"
    --target_accession "${TARGET_ACCESSION}"
    --n_zinc "${N_ZINC}"
    --base_dir "${BASE_DIR}"
    --full_2d_dir "${FULL_2D_DIR}"
    --full_2d3d_dir "${FULL_2D3D_DIR}"
    --affinity_cutoff_nM "${AFFINITY_CUTOFF_NM}"
    --seed "${SEED}"
    --umap_n_neighbors "${UMAP_N_NEIGHBORS}"
    --umap_min_dist "${UMAP_MIN_DIST}"
    --output_dir "${OUTPUT_DIR}"
)

# Add optional sampling parameters if set
if [ -n "${N_MF}" ]; then
  CMD+=( --n_mf "${N_MF}" )
fi

if [ -n "${N_TARGET}" ]; then
  CMD+=( --n_target "${N_TARGET}" )
fi

echo "[SLURM] Running in $(pwd)"
echo "[SLURM] Command: ${CMD[*]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[SLURM] DRY RUN MODE - Command NOT executed"
  echo "[SLURM]"
  echo "[SLURM] Full command:"
  printf '%s \\\n' "${CMD[@]}"
  echo ""
  echo "[SLURM] To execute, run without DRY_RUN=1"
  exit 0
fi

echo "[SLURM] Start time: $(date)"

"${CMD[@]}"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)"
echo "[SLURM] Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SLURM] ✓ Feature comparison v2 completed successfully"
  echo "[SLURM] Results saved to: ${OUTPUT_DIR}"
  echo "[SLURM]"
  echo "[SLURM] Output files:"
  echo "[SLURM]   - summary.json (metrics and config)"
  echo "[SLURM]   - metrics_comparison.csv (side-by-side table)"
  echo "[SLURM]   - umap_*.png (visualizations)"
  echo "[SLURM]   - scored_molecules_*.csv (ranked results)"
  echo "[SLURM]   - movement_*.png (Procrustes alignment plots)"
else
  echo "[SLURM] ✗ Feature comparison v2 failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
