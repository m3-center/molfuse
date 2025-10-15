# UMMBAS v3.0 - Critical Fixes Applied (October 15, 2025)

## Issues Fixed

### 1. ✅ Duplicate Seed Loop (CRITICAL)
**Problem:** Submission scripts were looping over seeds even though config files already contain seeds.
- Phase 1: 260 configs → 1,300 jobs submitted (5× multiplication)
- Each phase affected: 260, 60, 80, 40 → became 1,300, 300, 400, 200

**Fix:** Removed `RANDOM_SEEDS` array and seed loop from all submission scripts.
- `hpc/submit_v3_phase1.sh` 
- `hpc/submit_v3_phase2.sh`
- `hpc/submit_v3_phase3.sh`
- `hpc/submit_v3_phase4.sh`

**Impact:** Now correctly submits 260, 60, 80, 40 jobs per phase (440 total).

---

### 2. ✅ Wrong Workspace Paths (CRITICAL)
**Problem:** Config generators weren't overriding `workspace_base_dir` from base config.
- All phases wrote to default `experiment_workspace/` instead of phase-specific workspaces
- Phase 1 data went to wrong location instead of `experiment_workspace_v3_phase1/`

**Fix:** Added `workspace_base_dir` override in all phase config generators:
- `generate_phase1_configs.py` → `experiment_workspace_v3_phase1/`
- `generate_phase2_configs.py` → `experiment_workspace_v3_phase2/`
- `generate_phase3_configs.py` → `experiment_workspace_v3_phase3/`
- `generate_phase4_configs.py` → `experiment_workspace_v3_phase4/`

**Impact:** Each phase now writes to its designated workspace (symlinked to `/work` partition).

---

### 3. ✅ Data Validation Tool
**Created:** `check_workspace_overlaps.py`

**Purpose:** Verify data integrity in experiment workspaces.

**What it checks:**
- Target ↔ MF overlap (expected: some, target ligands in MF)
- Target ↔ ZINC overlap (expected: ZERO)
- MF ↔ ZINC overlap (expected: ZERO)

**Usage:**
```bash
python check_workspace_overlaps.py --temp_data_dir \
    /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1/run_seed44_config_tyro_features_pca_dim5_seed44/TyrosineProteinKinaseABL1_P00519/temp_data
```

---

## Required Actions

### ⚠️ CRITICAL: Regenerate All Configs

Because workspace paths were fixed, **you MUST regenerate all Phase 1 configs**:

```bash
# Delete old configs with wrong workspace paths
rm -rf hyperparam_configs_v3_phase1/

# Regenerate with correct workspace paths
python generate_phase1_configs.py
```

**Verify correct workspace:**
```bash
# Check a config file
cat hyperparam_configs_v3_phase1/config_tyro_features_pca_dim2_seed42.json | grep workspace_base_dir

# Should show:
# "workspace_base_dir": "experiment_workspace_v3_phase1/"
```

### ⚠️ Clean Up Old Data

If Phase 1 already ran with wrong workspace:

```bash
# Old data went here (wrong location)
ls experiment_workspace/

# Phase 1 should go here (correct location)
ls experiment_workspace_v3_phase1/

# Optionally move data:
mv experiment_workspace/run_seed* experiment_workspace_v3_phase1/ 2>/dev/null || true
```

---

## Git Commits

```
f589440 - Fix critical bug: Remove duplicate seed loop in submission scripts
953d687 - Fix workspace paths in all phase config generators  
449b724 - Add workspace data overlap checker script
```

---

## Verification Steps

### 1. Verify Submission Scripts Fixed

```bash
# Check Phase 1 script - should NOT have seed loop
grep -A 5 "for config_file" hpc/submit_v3_phase1.sh | grep "for seed"
# Should return nothing (no seed loop)

# Count expected vs actual in script
grep "expected: 260" hpc/submit_v3_phase1.sh
# Should show: Total Jobs: ${TOTAL_JOBS} (expected: 260)
```

### 2. Verify Config Workspace Paths

```bash
# Generate fresh configs
python generate_phase1_configs.py

# Check a config
jq '.global_settings.workspace_base_dir' hyperparam_configs_v3_phase1/config_tyro_features_pca_dim2_seed42.json

# Should output: "experiment_workspace_v3_phase1/"
```

### 3. Test Data Overlap Checker

```bash
# After running an experiment, check temp_data
python check_workspace_overlaps.py --temp_data_dir \
    experiment_workspace_v3_phase1/run_seed42_*/TyrosineProteinKinaseABL1_P00519/temp_data

# Should show:
#   ✓ Target ↔ ZINC overlap: 0 molecules (CORRECT)
#   ✓ MF ↔ ZINC overlap: 0 molecules (CORRECT)
```

---

## Timeline Impact

**Before fixes:**
- Would have submitted 2,200 jobs instead of 440 (5× waste)
- All phases writing to same workspace (data collision risk)

**After fixes:**
- Correct job count: 440 total
- Proper workspace separation: 260+60+80+40 across 4 workspaces
- Data validation tool available

---

## Next Steps

1. ✅ Pull latest changes on HPC cluster
2. ✅ Regenerate Phase 1 configs with correct workspace paths
3. ✅ Submit Phase 1 with fixed submission script
4. ✅ Monitor that data goes to `experiment_workspace_v3_phase1/`
5. ✅ Run overlap checker on completed experiments

---

**Status:** All critical bugs fixed and committed to branch `3.0`  
**Ready for:** Production HPC execution  
**Date:** October 15, 2025
