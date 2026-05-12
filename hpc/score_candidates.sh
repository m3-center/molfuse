#!/bin/bash
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Score candidate molecules against a trained MolFuSE model (headless / HPC).
#
# Usage:
#   sbatch hpc/score_candidates.sh <RUN_NAME> <CANDIDATES_FILE> [WORKSPACE_DIR] [OUTPUT_CSV]
#
# Example:
#   sbatch hpc/score_candidates.sh ABL1_UMAP_features_10d candidates.parquet \
#       experiment_workspace_v4 scored_candidates.csv
#
# Requirements:
#   - Conda environment 'molfuse' must be available on the compute node.
#   - 48 GB RAM is sufficient for most UMAP models; increase --mem for very
#     large MF clouds (KW-0808 Transferase, 425 K molecules).
#   - For PCA models 16 GB is sufficient; adjust --mem accordingly.

RUN_NAME="${1:-}"
CANDIDATES="${2:-}"
WORKSPACE_DIR="${3:-experiment_workspace_v4}"
OUTPUT_CSV="${4:-scored_candidates.csv}"

if [[ -z "$RUN_NAME" || -z "$CANDIDATES" ]]; then
  echo "Usage: sbatch hpc/score_candidates.sh <RUN_NAME> <CANDIDATES_FILE> [WORKSPACE_DIR] [OUTPUT_CSV]" >&2
  exit 1
fi

if [[ ! -f "$CANDIDATES" ]]; then
  echo "Candidates file not found: $CANDIDATES" >&2
  exit 1
fi

mkdir -p slurm_logs || true

echo "==== MolFuSE CLI Scorer ===="
echo "Run name   : $RUN_NAME"
echo "Candidates : $CANDIDATES"
echo "Workspace  : $WORKSPACE_DIR"
echo "Output     : $OUTPUT_CSV"
echo "Node       : $(hostname)"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memory     : ${SLURM_MEM_PER_NODE:-48}G"
echo "============================"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate molfuse

python scripts/score_candidates.py \
    --workspace "$WORKSPACE_DIR" \
    --run-name  "$RUN_NAME" \
    --candidates "$CANDIDATES" \
    --output    "$OUTPUT_CSV"

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Scoring completed successfully → $OUTPUT_CSV"
else
  echo "Scoring failed with exit code $EXIT_CODE" >&2
fi
exit $EXIT_CODE
