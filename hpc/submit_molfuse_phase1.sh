#!/bin/bash
# Submit all Phase 1 v4 configs to SLURM
# Usage: bash hpc/submit_molfuse_phase1.sh [--dry-run] [CONFIG_DIR] [WORKSPACE_DIR]
# Defaults: CONFIG_DIR=configs/molfuse_phase1_grid, WORKSPACE_DIR=experiment_workspace_v4
# 
# Examples:
#   bash hpc/submit_molfuse_phase1.sh --dry-run
#   bash hpc/submit_molfuse_phase1.sh
#   bash hpc/submit_molfuse_phase1.sh configs/molfuse_phase1_grid experiment_workspace_v4

# Parse dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

CONFIG_DIR="${1:-configs/molfuse_phase1_grid}"
WORKSPACE_DIR="${2:-experiment_workspace_v4}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
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

for cfg in "$CONFIG_DIR"/*.json; do
  if [[ ! -f "$cfg" ]]; then continue; fi
  COUNT_TOTAL=$((COUNT_TOTAL+1))
  
  # Extract config basename for job name
  CONFIG_BASENAME=$(basename "$cfg" .json)
  
  # Extract run_name from JSON config
  RUN_NAME=$(python3 -c "
import json, sys
try:
    with open('$cfg') as f:
        cfg = json.load(f)
    run_name = cfg.get('run_name')
    if not run_name:
        target = cfg.get('target', 'target')
        method = cfg.get('method', 'pca')
        dim = cfg.get('dim', 2)
        run_name = f'{target}_{method}_{dim}d'
    print(run_name)
except Exception as e:
    print(f'ERROR_PARSING_{CONFIG_BASENAME}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)
  
  if [[ -z "$RUN_NAME" ]]; then
    echo "WARNING: Could not parse run_name from $cfg, skipping"
    COUNT_SKIPPED=$((COUNT_SKIPPED+1))
    continue
  fi
  
  # Check if completion marker exists
  COMPLETION_MARKER="${WORKSPACE_DIR}/phase1/${RUN_NAME}/logs/phase1_summary.json"
  
  if [[ -f "$COMPLETION_MARKER" ]]; then
    # Verify it's valid JSON and contains expected keys
    IS_VALID=$(python3 -c "
import json, sys
try:
    with open('$COMPLETION_MARKER') as f:
        summary = json.load(f)
    if 'config' in summary and 'metrics_path' in summary:
        print('valid')
    else:
        print('invalid')
except:
    print('invalid')
" 2>/dev/null)
    
    if [[ "$IS_VALID" == "valid" ]]; then
      echo "SKIP: $RUN_NAME (already completed: $COMPLETION_MARKER)"
      COUNT_SKIPPED=$((COUNT_SKIPPED+1))
      continue
    else
      echo "WARNING: Completion marker exists but is invalid, will rerun: $RUN_NAME"
    fi
  fi
  
  # Submit job or show what would be submitted
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "WOULD SUBMIT: $RUN_NAME"
    echo "  Config: $cfg"
    echo "  Job name: phase1_${CONFIG_BASENAME}"
    echo ""
  else
    sbatch --job-name="phase1_${CONFIG_BASENAME}" hpc/molfuse_phase1_cpu.sh "$cfg" "$WORKSPACE_DIR"
    echo "SUBMITTED: $RUN_NAME"
  fi
  
  COUNT_SUBMITTED=$((COUNT_SUBMITTED+1))
  sleep 0.1
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total configs found: $COUNT_TOTAL"
echo "Already completed (skipped): $COUNT_SKIPPED"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Would submit: $COUNT_SUBMITTED"
else
  echo "Submitted: $COUNT_SUBMITTED"
fi
echo "=========================================="
