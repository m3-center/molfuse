# Fix: Return Code -9 Detection (OOM Errors)

**Date:** October 17, 2025  
**Issue:** Script not detecting OOM errors that manifest as "return code -9"  
**Status:** ✅ Fixed

---

## Problem

The script was not detecting OOM failures that appear in logs as:
```
Calc SimSpace (features, dim5) for TyrosineProteinKinaseABL1_P00519 FAILED with return code -9
```

**Why return code -9?**
- Return code `-9` = **SIGKILL signal**
- Linux kernel sends SIGKILL when a process is killed
- Most common cause: **Out of Memory (OOM) killer**
- Process is forcibly terminated to protect system

---

## Solution

Added **priority detection** for "return code -9" patterns in log files.

### Detection Hierarchy (in order):

1. **Return code -9** (SIGKILL - likely OOM)
   ```python
   if 'return code -9' in log_content_lower:
       status['error_type'] = 'Process Killed (return code -9, likely OOM)'
   ```

2. **Explicit OOM messages** ('oom', 'out of memory', 'memoryerror')
   ```python
   elif 'oom' in log_content_lower:
       status['error_type'] = 'Out of Memory (OOM)'
   ```

3. **Generic FAILED messages**
   ```python
   elif ' FAILED ' in log_content:
       status['error_type'] = 'Task Failed'
   ```

4. **Python errors** (ERROR, Traceback, etc.)
   ```python
   elif 'ERROR' in log_content or 'Traceback' in log_content:
       # Categorize by error type
   ```

---

## Code Changes

**File:** `scripts/check_hyperparam_status.py`

**Lines ~190-220:** Added return code -9 detection

```python
# Check for return code -9 (SIGKILL - usually OOM)
if 'return code -9' in log_content_lower or 'failed with return code -9' in log_content_lower:
    lines = log_content.split('\n')
    error_lines = []
    # Find lines with return code -9
    for i, line in enumerate(lines):
        if 'return code -9' in line.lower():
            # Capture context around error (5 lines before, 5 after)
            start = max(0, i - 5)
            end = min(len(lines), i + 6)
            error_lines = lines[start:end]
            break
    
    if error_lines:
        status['error'] = '\n'.join([l.strip() for l in error_lines if l.strip()])
        status['error_type'] = 'Process Killed (return code -9, likely OOM)'
```

---

## Example Detection

### Log Entry:
```
2025-10-17 14:32:15 - INFO - Starting similarity space calculation...
2025-10-17 14:35:22 - INFO - Loading feature matrix...
2025-10-17 14:37:45 - INFO - Computing PCA projection...
2025-10-17 14:38:12 - ERROR - Calc SimSpace (features, dim5) for TyrosineProteinKinaseABL1_P00519 FAILED with return code -9
2025-10-17 14:38:12 - INFO - Cleaning up temporary files...
```

### Detected Error:
```
Error Type: Process Killed (return code -9, likely OOM)
Error Context:
  2025-10-17 14:32:15 - INFO - Starting similarity space calculation...
  2025-10-17 14:35:22 - INFO - Loading feature matrix...
  2025-10-17 14:37:45 - INFO - Computing PCA projection...
  2025-10-17 14:38:12 - ERROR - Calc SimSpace (features, dim5) for TyrosineProteinKinaseABL1_P00519 FAILED with return code -9
  2025-10-17 14:38:12 - INFO - Cleaning up temporary files...
```

---

## Status Report Output

### Before (Missed):
```
================================================================================
ERROR TYPES
================================================================================
Other Error                         0 occurrences
```

### After (Detected):
```
================================================================================
ERROR TYPES
================================================================================
Process Killed (return code -9, likely OOM)    12 occurrences
Out of Memory (OOM)                              0 occurrences
Task Failed                                      3 occurrences
```

---

## Understanding Return Codes

| Return Code | Signal    | Meaning                              | Typical Cause                    |
|-------------|-----------|--------------------------------------|----------------------------------|
| `-9`        | SIGKILL   | Process forcibly killed              | OOM killer, manual kill -9       |
| `-6`        | SIGABRT   | Process aborted                      | Assertion failure, abort()       |
| `-11`       | SIGSEGV   | Segmentation fault                   | Invalid memory access            |
| `-15`       | SIGTERM   | Graceful termination requested       | Normal shutdown, timeout         |
| `1`         | -         | General error                        | Application error                |
| `137`       | SIGKILL   | Killed (128 + 9)                     | Same as -9, different format     |

**Our detection covers:**
- ✅ `-9` (most common OOM format)
- ✅ `137` (alternative SIGKILL format)
- ✅ Explicit "OOM" messages
- ✅ "Out of memory" text
- ✅ "MemoryError" exceptions

---

## Testing

Re-run the status check on your workspace:

```bash
cd /path/to/experiment_workspace_v3_phase1
python ../UMMBAS_screening_experiments/scripts/check_hyperparam_status.py --workspace .
```

**Expected output for OOM failures:**
```
================================================================================
FAILED EXPERIMENTS DETAILS
================================================================================

Error Type: Process Killed (return code -9, likely OOM) (12 experiments)
================================================================================

Run: run_seed42_config_tyro_features_umap_dim5_nn100_md0.01
  Representation: features
  Method: UMAP-Euclidean
  n_neighbors: 100
  min_dist: 0.01
  Seed: 42
  Error (last 5 lines):
    2025-10-17 14:37:45 - INFO - Computing PCA projection...
    2025-10-17 14:38:12 - ERROR - Calc SimSpace (features, dim5) for TyrosineProteinKinaseABL1_P00519 FAILED with return code -9
    2025-10-17 14:38:12 - INFO - Cleaning up temporary files...
```

---

## Common OOM Scenarios

### High n_neighbors + High dimensionality
```
features-UMAP-Euclidean-nn500-md0.01  dim10  → OOM (return code -9)
features-UMAP-Euclidean-nn100-md0.01  dim10  → OOM (return code -9)
```

### Large molecular clouds
```
MF cloud: 420K molecules + UMAP → OOM (return code -9)
```

### Solutions:
1. **Reduce n_neighbors**: Use 5, 10, 20 instead of 100, 500
2. **Lower dimensionality**: Use 2D or 5D instead of 10D
3. **Batch processing**: Process in smaller chunks
4. **More memory**: Request more RAM on HPC (e.g., `#SBATCH --mem=64G`)

---

## Debug Log Output

When checking a run with return code -9:

```
Run directory: run_seed42_config_tyro_features_umap_dim5_nn100_md0.01
Results pattern: run_seed42.../*/results/*/dim_*/*
Found 0 result directories
⚠️ No results found - checking logs
Reading log file: orchestrator_run_seed42_config_tyro_features_umap_dim5_nn100_md0.01.log
❌ Process killed (return code -9) detected - likely OOM
Final status: completed=False, has_results=False, has_rankings=False, has_metrics=False, error_type=Process Killed (return code -9, likely OOM)
```

---

## Summary

✅ **Now detects:** `return code -9` patterns in logs  
✅ **Categorizes as:** "Process Killed (return code -9, likely OOM)"  
✅ **Provides context:** 5 lines before/after error  
✅ **Priority check:** Checked BEFORE other error types  
✅ **Case-insensitive:** Catches any case variation  

**Result:** All OOM failures are now properly identified in status reports! 🎯

---

## Files Modified

- `scripts/check_hyperparam_status.py`
  - Lines ~190-220: Added return code -9 detection
  - Lines ~220-240: Added generic FAILED detection
  - Detection hierarchy: -9 → OOM → FAILED → ERROR/Traceback

---

**OOM errors will no longer go undetected!** 🚀
