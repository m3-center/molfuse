#!/bin/bash
# Submit Phase 5 Expansion (Cross-Target Validation) to SLURM
# Usage: bash hpc/submit_molfuse_phase5_expansion.sh [--dry-run] [CONFIG_DIR] [WORKSPACE_DIR]
# Defaults: CONFIG_DIR=configs/molfuse_phase5_expansion, WORKSPACE_DIR=experiment_workspace_v4

# Parse dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

CONFIG_DIR="${1:-configs/molfuse_phase5_expansion}"
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
  
  # Extract run_name and phase5_run_name from JSON config
  RUN_INFO=$(python3 -c "
import json, sys
try:
    with open('$cfg') as f:
        cfg = json.load(f)
    run_name = cfg.get('run_name', 'unknown')
    phase5_run_name = cfg.get('phase5_run_name', 'validation')
    print(f'{phase5_run_name}|{run_name}')
except Exception as e:
    print(f'ERROR|ERROR', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)
  
  PHASE5_SUBDIR=$(echo "$RUN_INFO" | cut -d'|' -f1)
  RUN_NAME=$(echo "$RUN_INFO" | cut -d'|' -f2)
  
  if [[ "$RUN_NAME" == "unknown" || "$RUN_NAME" == "ERROR" ]]; then
    echo "WARNING: Could not parse run_name from $cfg, skipping"
    COUNT_SKIPPED=$((COUNT_SKIPPED+1))
    continue
  fi
  
  # Check if completion marker exists
  # Path: workspace/phase5/{phase5_run_name}/{run_name}/logs/phase5_summary.json
  COMPLETION_MARKER="${WORKSPACE_DIR}/phase5/${PHASE5_SUBDIR}/${RUN_NAME}/logs/phase5_summary.json"
  
  if [[ -f "$COMPLETION_MARKER" ]]; then
    # Verify it's valid JSON
    IS_VALID=$(python3 -c "
import json
try:
    with open('$COMPLETION_MARKER') as f:
        json.load(f)
    print('valid')
except:
    print('invalid')
" 2>/dev/null)
    
    if [[ "$IS_VALID" == "valid" ]]; then
      echo "SKIP: $RUN_NAME (already completed)"
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
    echo "  Job name: p5exp_${CONFIG_BASENAME}"
    echo "  Log dir: ${WORKSPACE_DIR}/phase5/${PHASE5_SUBDIR}/${RUN_NAME}/logs"
    echo ""
  else
    sbatch --job-name="p5exp_${CONFIG_BASENAME}" hpc/molfuse_phase5_cpu.sh "$cfg" "$WORKSPACE_DIR"
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
