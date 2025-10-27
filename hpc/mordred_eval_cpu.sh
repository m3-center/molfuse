#!/bin/bash
#SBATCH --partition=any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=mordred_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

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

set -euo pipefail

# Resolve repo root and move there so relative paths work
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm_logs

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

# Try to activate the requested environment
# Prefer mamba; fallback to conda if mamba not available
if command -v mamba >/dev/null 2>&1; then
  # On many clusters, a bashrc contains the mamba hook
  source "$HOME/.bashrc" 2>/dev/null || true
  mamba activate ummbas-screening-mordredcommunity 2>/dev/null || conda activate ummbas-screening-mordredcommunity
else
  # Fallback for conda without mamba
  if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate ummbas-screening-mordredcommunity
fi

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

echo "[SLURM] Running: ${CMD[*]}"
# Use srun if available for better resource accounting
if command -v srun >/dev/null 2>&1; then
  srun "${CMD[@]}"
else
  "${CMD[@]}"
fi
