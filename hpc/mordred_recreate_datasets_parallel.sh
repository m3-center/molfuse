#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=mordred_recreate_parallel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM submission script for the parallel Mordred dataset recreation
# Runs tests/mordred_full_feature_eval/recreate_datasets_parallel.py on HPC.
# Produces two output trees under OUTPUT_DIR: datasets_2d_all/ and datasets_2d3d_all/

WORKDIR="${WORKDIR:-/home/ahagg2s/UMMBAS_screening_experiments}"
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults (override via --export=ALL,VAR=value)
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/output_full_datasets}
BASE_DIR=${BASE_DIR:-.}
SEED=${SEED:-42}
WORKERS=${WORKERS:-64}
CHUNK_SIZE=${CHUNK_SIZE:-50000}
BATCH_2D=${BATCH_2D:-1000}
BATCH_3D=${BATCH_3D:-250}
LIMIT_ZINC=${LIMIT_ZINC:-}
LIMIT_KW=${LIMIT_KW:-}

# Environment
echo "[SLURM] Activating conda environment: ummbas-screening-mordredcommunity"
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

# Create output dir
mkdir -p "${OUTPUT_DIR}"

echo "[SLURM] Configuration:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SEED=${SEED}"
echo "  WORKERS=${WORKERS}"
echo "  CHUNK_SIZE=${CHUNK_SIZE}"
echo "  BATCH_2D=${BATCH_2D}"
echo "  BATCH_3D=${BATCH_3D}"
if [ -n "${LIMIT_ZINC}" ]; then echo "  LIMIT_ZINC=${LIMIT_ZINC}"; else echo "  LIMIT_ZINC=none"; fi
if [ -n "${LIMIT_KW}" ]; then echo "  LIMIT_KW=${LIMIT_KW}"; else echo "  LIMIT_KW=none"; fi

CMD=(
  python tests/mordred_full_feature_eval/recreate_datasets_parallel.py \
    --base_dir "${BASE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --chunk-size "${CHUNK_SIZE}" \
    --batch-2d "${BATCH_2D}" \
    --batch-3d "${BATCH_3D}"
)

if [ -n "${LIMIT_ZINC}" ]; then CMD+=( --limit-zinc "${LIMIT_ZINC}" ); fi
if [ -n "${LIMIT_KW}" ]; then CMD+=( --limit-kw "${LIMIT_KW}" ); fi

echo "[SLURM] Running in $(pwd): ${CMD[*]}"
echo "[SLURM] Start time: $(date)"
"${CMD[@]}"
EXIT_CODE=$?
echo "[SLURM] End time: $(date)"
echo "[SLURM] Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SLURM] ✓ Parallel dataset recreation completed successfully"
  echo "[SLURM] Output directories:"
  echo "  2D descriptors:    ${OUTPUT_DIR}/datasets_2d_all/"
  echo "  2D+3D descriptors: ${OUTPUT_DIR}/datasets_2d3d_all/"
else
  echo "[SLURM] ✗ Dataset recreation failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
