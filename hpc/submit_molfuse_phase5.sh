#!/bin/bash
# Submit all Phase 5 v4 configs to SLURM
#
# Phase 5: Validation & Baseline Experiments
#   1. Database Bias Negative Control (5 replicates)
#   2. Raw Descriptor Baseline (5 replicates)
#   3. Tanimoto Baseline (5 replicates)
#
# Usage: bash hpc/submit_molfuse_phase5.sh [--dry-run] [CONFIG_DIR] [WORKSPACE_DIR]
# Defaults: CONFIG_DIR=configs/molfuse_phase5_grid, WORKSPACE_DIR=experiment_workspace_v4
# 
# Examples:
#   bash hpc/submit_molfuse_phase5.sh --dry-run
#   bash hpc/submit_molfuse_phase5.sh
#   bash hpc/submit_molfuse_phase5.sh configs/molfuse_phase5_grid experiment_workspace_v4

# Parse dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

CONFIG_DIR="${1:-configs/molfuse_phase5_grid}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "ERROR: Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

mkdir -p slurm_logs || true

# Check if Python is available for JSON parsing
if ! command -v python3 &> /dev/null; then
  echo "ERROR: python3 is required to parse config files" >&2
  exit 1
fi

COUNT_SUBMITTED=0
COUNT_SKIPPED=0
COUNT_TOTAL=0

if [[ "$DRY_RUN" == "true" ]]; then
  echo "=========================================="
  echo "DRY RUN MODE - No jobs will be submitted"
  echo "=========================================="
  echo ""
fi

echo "Scanning configs in: $CONFIG_DIR"
echo ""

for cfg in "$CONFIG_DIR"/*.json; do
  if [[ ! -f "$cfg" ]]; then continue; fi
  COUNT_TOTAL=$((COUNT_TOTAL+1))
  
  # Extract config basename for job name
  CONFIG_BASENAME=$(basename "$cfg" .json)
  
  # Extract run_name and experiment_type from JSON config
  RUN_INFO=$(python3 -c "
import json, sys
try:
    with open('$cfg') as f:
        cfg = json.load(f)
    run_name = cfg.get('run_name', 'unknown')
    exp_type = cfg.get('experiment_type', 'unknown')
    print(f'{run_name}|{exp_type}')
except Exception as e:
    print('ERROR|ERROR', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)
  
  if [[ -z "$RUN_INFO" ]] || [[ "$RUN_INFO" == "ERROR|ERROR" ]]; then
    echo "WARNING: Could not parse $cfg, skipping"
    COUNT_SKIPPED=$((COUNT_SKIPPED+1))
    continue
  fi
  
  RUN_NAME=$(echo "$RUN_INFO" | cut -d'|' -f1)
  EXP_TYPE=$(echo "$RUN_INFO" | cut -d'|' -f2)
  
  # Phase 5 uses phase5/validation/{run_name} directory structure
  PHASE5_RUN_NAME="validation"
  COMPLETION_MARKER="${WORKSPACE_DIR}/phase5/${PHASE5_RUN_NAME}/${RUN_NAME}/logs/phase5_summary.json"
  
  # Check if already completed
  if [[ -f "$COMPLETION_MARKER" ]]; then
    # Verify it's valid JSON and contains expected keys
    IS_VALID=$(python3 -c "
import json, sys
try:
    with open('$COMPLETION_MARKER') as f:
        summary = json.load(f)
    if 'config' in summary and 'experiment_type' in summary:
        print('valid')
    else:
        print('invalid')
except:
    print('invalid')
" 2>/dev/null)
    
    if [[ "$IS_VALID" == "valid" ]]; then
      echo "SKIP: $RUN_NAME [$EXP_TYPE] (already completed)"
      COUNT_SKIPPED=$((COUNT_SKIPPED+1))
      continue
    else
      echo "WARNING: Completion marker exists but is invalid, will rerun: $RUN_NAME"
    fi
  fi
  
  # Submit job
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "WOULD SUBMIT: $RUN_NAME [$EXP_TYPE] (config: $(basename $cfg))"
    COUNT_SUBMITTED=$((COUNT_SUBMITTED+1))
  else
    JOB_ID=$(sbatch --job-name="phase5_${RUN_NAME}" hpc/molfuse_phase5_cpu.sh "$cfg" "$WORKSPACE_DIR" 2>&1 | grep -oP 'Submitted batch job \K\d+')
    if [[ -n "$JOB_ID" ]]; then
      echo "SUBMITTED: $RUN_NAME [$EXP_TYPE] (Job ID: $JOB_ID)"
      COUNT_SUBMITTED=$((COUNT_SUBMITTED+1))
    else
      echo "FAILED: Could not submit $RUN_NAME" >&2
    fi
  fi
done

echo ""
echo "=========================================="
echo "SUBMISSION SUMMARY"
echo "=========================================="
echo "Total configs:     $COUNT_TOTAL"
echo "Submitted:         $COUNT_SUBMITTED"
echo "Skipped (done):    $COUNT_SKIPPED"
echo "=========================================="

if [[ "$DRY_RUN" == "true" ]]; then
  echo ""
  echo "DRY RUN MODE: No jobs were actually submitted"
  echo "Run without --dry-run to submit jobs"
fi

echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER | grep phase5"
echo ""
echo "Check results:"
echo "  ls ${WORKSPACE_DIR}/phase5/validation/*/logs/phase5_summary.json"
