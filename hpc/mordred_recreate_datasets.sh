#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=mordred_recreate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#
# SLURM submission script for Mordred dataset recreation
# Runs tests/mordred_full_feature_eval/recreate_datasets.py on HPC.
#
# This script recreates the entire dataset structure with full Mordred descriptors:
#   - datasets_2d_all/: Full 2D Mordred descriptors (1613 features)
#   - datasets_2d3d_all/: Full 2D+3D Mordred descriptors (1826 features)
#   - Also filters and recreates fingerprint files to match feature files
#
# Expected runtime: 2-4 hours for full dataset (1.3M ZINC + 58 KW files)
# Memory usage: ~10-20GB peak (uses chunked processing)
#
# Usage examples:
#   # Full production run (all molecules)
#   sbatch hpc/mordred_recreate_datasets.sh
#
#   # Test run with limited molecules
#   sbatch --export=ALL,LIMIT_ZINC=10000,LIMIT_KW=1000 hpc/mordred_recreate_datasets.sh
#
#   # Custom output directory and seed
#   sbatch --export=ALL,OUTPUT_DIR=output_full_descriptors,SEED=123 hpc/mordred_recreate_datasets.sh
#
# To override variables, pass them via --export=ALL,VAR=VALUE,...
#

# Resolve repo root and move there so relative paths work
WORKDIR="${WORKDIR:-/home/ahagg2s/UMMBAS_screening_experiments}"
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults (override via SBATCH --export=ALL,VAR=value)
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/output_full_datasets}
BASE_DIR=${BASE_DIR:-.}
SEED=${SEED:-42}
LIMIT_ZINC=${LIMIT_ZINC:-}
LIMIT_KW=${LIMIT_KW:-}

# --- Environment Setup ---
echo "[SLURM] Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log configuration
echo "[SLURM] Configuration:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SEED=${SEED}"
if [ -n "${LIMIT_ZINC}" ]; then
  echo "  LIMIT_ZINC=${LIMIT_ZINC} (testing mode)"
else
  echo "  LIMIT_ZINC=none (processing all ZINC molecules)"
fi
if [ -n "${LIMIT_KW}" ]; then
  echo "  LIMIT_KW=${LIMIT_KW} (testing mode)"
else
  echo "  LIMIT_KW=none (processing all KW molecules)"
fi

# Build command
CMD=(
  python tests/mordred_full_feature_eval/recreate_datasets.py
    --base_dir "${BASE_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED}"
)

# Add optional limit parameters if specified
if [ -n "${LIMIT_ZINC}" ]; then
  CMD+=( --limit-zinc "${LIMIT_ZINC}" )
fi

if [ -n "${LIMIT_KW}" ]; then
  CMD+=( --limit-kw "${LIMIT_KW}" )
fi

echo "[SLURM] Running in $(pwd): ${CMD[*]}"
echo "[SLURM] Start time: $(date)"

"${CMD[@]}"

EXIT_CODE=$?
echo "[SLURM] End time: $(date)"
echo "[SLURM] Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SLURM] ✓ Dataset recreation completed successfully"
  echo "[SLURM] Output directories:"
  echo "  2D descriptors:    ${OUTPUT_DIR}/datasets_2d_all/"
  echo "  2D+3D descriptors: ${OUTPUT_DIR}/datasets_2d3d_all/"
  echo "[SLURM] Verification report: ${OUTPUT_DIR}/verification_report.log"
  
  # Show summary statistics if verification report exists
  if [ -f "${OUTPUT_DIR}/verification_report.log" ]; then
    echo ""
    echo "[SLURM] Summary statistics:"
    grep -A 10 "OVERALL SUCCESS RATES" "${OUTPUT_DIR}/verification_report.log" || true
  fi
else
  echo "[SLURM] ✗ Dataset recreation failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
