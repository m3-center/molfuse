#!/usr/bin/env bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=mordred_eval
#SBATCH --chdir=/home/%u/UMMBAS_screening_experiments
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#
# SLURM submission script for the standalone Mordred full-feature evaluator
# Runs tests/mordred_full_feature_eval/mordred_full_feature_compare.py on HPC.
#
# Usage examples:
#   sbatch hpc/mordred_eval_cpu.sh                                     # use defaults
#   sbatch --export=ALL,N_TARGET=300,N_MF=600,N_ZINC=600 \
#          --export=ALL,OUTPUT_DIR=tests/mordred_full_feature_eval/output_hpc hpc/mordred_eval_cpu.sh
#   sbatch --export=ALL,N_TARGET=200,N_MF=500,N_ZINC=2000,ENABLE_SWEEP=1 hpc/mordred_eval_cpu.sh
#
# To override variables, pass them via --export=ALL,VAR=VALUE,...
# Common overrides: OUTPUT_DIR, N_TARGET, N_MF, N_ZINC, ENABLE_SWEEP (0/1)
#

# Resolve repo root and move there so relative paths work (fallback if --chdir unsupported)
WORKDIR=${WORKDIR:-"/home/ahagg2s/UMMBAS_screening_experiments"}
echo "[SLURM] WORKDIR=${WORKDIR}"
cd "${WORKDIR}" || { echo "[SLURM][ERROR] Repo not found at ${WORKDIR}"; exit 1; }

# Defaults (override via SBATCH --export=ALL,VAR=value)
N_TARGET=${N_TARGET:-200}
N_MF=${N_MF:-500}
N_ZINC=${N_ZINC:-2000}
OUTPUT_DIR=${OUTPUT_DIR:-tests/mordred_full_feature_eval/output_hpc}
CACHE_DIR=${CACHE_DIR:-tests/mordred_full_feature_eval/cache}
UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS:-1}
UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.1}
ENABLE_SWEEP=${ENABLE_SWEEP:-1}

# Coverage-aware selection thresholds
PF_TARGET=${PF_TARGET:-0.95}
PF_MF=${PF_MF:-0.7}
PR_TARGET_GUARD=${PR_TARGET_GUARD:-0.2}
PR_MF=${PR_MF:-0.6}
PR_ZINC=${PR_ZINC:-0.8}

# --- Environment Setup ---
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening-mordredcommunity

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python tests/mordred_full_feature_eval/mordred_full_feature_compare.py
    --output_dir "${OUTPUT_DIR}"
    --n_target "${N_TARGET}"
    --n_mf "${N_MF}"
    --n_zinc "${N_ZINC}"
    --umap_n_neighbors "${UMAP_N_NEIGHBORS}"
    --umap_min_dist "${UMAP_MIN_DIST}"
    --cache_dir "${CACHE_DIR}"
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

"${CMD[@]}"
