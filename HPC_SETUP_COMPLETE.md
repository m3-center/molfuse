# UMMBAS v3.0 - HPC Setup Complete

**Date:** October 15, 2025  
**Branch:** 3.0  
**Status:** ✅ Ready for Execution

---

## Summary of Changes

This session completed the HPC infrastructure for UMMBAS v3.0, making the entire experimental pipeline ready to execute on the cluster.

### 1. Cleaned Up Orchestration

**Removed:**
- `orchestrate_v3_pipeline.py` - Unnecessary wrapper script

**Reason:** The script was a leftover that just called `main_orchestrator.py`. Direct use of submission scripts is cleaner and more explicit.

### 2. Created v3.0 HPC Scripts

**New Files:**

| File | Purpose | Lines |
|------|---------|-------|
| `hpc/ummbas_v3_cpu.sh` | SLURM job script for single experiments | 85 |
| `hpc/submit_v3_phase1.sh` | Submit 260 Phase 1 jobs (dimensionality sweep) | 101 |
| `hpc/submit_v3_phase2.sh` | Submit 60 Phase 2 jobs (MF ablation) | 99 |
| `hpc/submit_v3_phase3.sh` | Submit 80 Phase 3 jobs (generalization) | 99 |
| `hpc/submit_v3_phase4.sh` | Submit 40 Phase 4 jobs (cutoff analysis) | 101 |
| `hpc/README.md` | HPC scripts documentation | 98 |

**Features:**
- ✅ Automatic prerequisite checking (configs, best configs)
- ✅ Job count verification
- ✅ User confirmation prompts
- ✅ Clear monitoring instructions
- ✅ Resource configuration (64 cores, 350 GB, 24h)

### 3. Workspace Management

**New File:**
- `setup_hpc_workspaces.sh` - Creates workspaces on `/work` partition with symlinks

**What it does:**
```bash
# Creates on high-capacity /work partition
/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase1/
/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase2/
/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase3/
/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase4/

# Symlinks in repository root
experiment_workspace_v3_phase1 -> /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase1
experiment_workspace_v3_phase2 -> /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase2
experiment_workspace_v3_phase3 -> /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase3
experiment_workspace_v3_phase4 -> /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase4
```

**Why:**
- `/work` partition has much larger capacity than `/home`
- Symlinks provide convenient access from repository
- Prevents "disk quota exceeded" errors during experiments

### 4. Archived Old Scripts

**Moved to `hpc/archive_v2/`:**
- All 22 v2.0 HPC scripts (for reference only)
- Includes: hyperparameterization, generalization, cutoff analysis, ablation studies

**Current HPC folder structure:**
```
hpc/
├── README.md                     # Documentation
├── ummbas_v3_cpu.sh             # SLURM job script
├── submit_v3_phase1.sh          # Phase 1 submission
├── submit_v3_phase2.sh          # Phase 2 submission
├── submit_v3_phase3.sh          # Phase 3 submission
├── submit_v3_phase4.sh          # Phase 4 submission
└── archive_v2/                   # Old v2.0 scripts
```

### 5. Updated Documentation

**Files Modified:**
1. **`docs/QUICKSTART_V3.md`**
   - Added Step 1: Workspace setup
   - Updated all step numbers
   - Added HPC-specific commands (bash scripts, not python)
   - Corrected resource estimates (32/64 cores)

2. **`hpc/README.md`**
   - Added workspace setup as prerequisite
   - Documented all 5 v3.0 scripts
   - Added troubleshooting section
   - Updated resource configuration details

---

## Execution Checklist

### ✅ One-Time Setup (on HPC)

```bash
# 1. Clone/update repository
cd ~/path/to/UMMBAS_screening_experiments
git checkout 3.0
git pull

# 2. Setup workspaces (creates /work symlinks)
bash setup_hpc_workspaces.sh

# 3. Edit conda environment path
nano hpc/ummbas_v3_cpu.sh
# Line 55: source /home/ahagg2s/miniforge3/bin/activate ummbas-screening
```

### 🚀 Run Phase 1 (260 runs)

```bash
# 1. Generate configs
python generate_phase1_configs.py

# 2. Submit jobs
bash hpc/submit_v3_phase1.sh
# Confirm: y

# 3. Monitor
squeue -u $USER
python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1
```

### 📊 After Phase 1 Completes

```bash
# Extract best configs
python extract_phase1_best_configs.py \
    --workspace experiment_workspace_v3_phase1 \
    --output phase1_best_configs.json
```

### 🔄 Run Phases 2-4

```bash
# Phase 2 (MF Ablation)
python generate_phase2_configs.py
bash hpc/submit_v3_phase2.sh

# Phase 3 (Generalization)
python generate_phase3_configs.py
bash hpc/submit_v3_phase3.sh

# Phase 4 (Cutoff Analysis)
python generate_phase4_configs.py
bash hpc/submit_v3_phase4.sh
```

---

## What's Different from v2.0

| Aspect | v2.0 | v3.0 |
|--------|------|------|
| **HPC Scripts** | 22 scripts (mixed purposes) | 5 clean scripts (1 per phase + 1 job runner) |
| **Submission** | Manual config of multiple scripts | Single script per phase |
| **Workspaces** | Created in home directory | Symlinked to /work partition |
| **Validation** | None | Automatic prerequisite checks |
| **Confirmation** | None | User prompted before submission |
| **Documentation** | Scattered | Centralized in hpc/README.md |
| **Monitoring** | Generic squeue | Integrated status checker |

---

## Expected Timeline (64 cores)

| Phase | Jobs | Duration | Cumulative |
|-------|------|----------|------------|
| Phase 1 | 260 | ~3-5 days | 3-5 days |
| Phase 2 | 60 | ~0.5-1 day | 4-6 days |
| Phase 3 | 80 | ~1-1.5 days | 5-8 days |
| Phase 4 | 40 | ~0.5 day | 6-8 days |
| **Total** | **440** | | **~6-8 days** |

*Times depend on cluster load and specific hardware*

---

## Git Commits

```
913b3ba - Add v3.0 HPC submission scripts and clean up old v2.0 scripts
a09c312 - Add HPC workspace setup script with /work partition symlinks
```

**Total changes:**
- 5 new HPC scripts created
- 1 workspace setup script created
- 2 documentation files updated
- 22 old scripts archived
- 1 wrapper script removed

---

## Files Ready for Execution

### Core Pipeline
✅ `main_orchestrator.py` - Experiment runner  
✅ `generate_phase1_configs.py` - 260 configs  
✅ `generate_phase2_configs.py` - 60 configs  
✅ `generate_phase3_configs.py` - 80 configs  
✅ `generate_phase4_configs.py` - 40 configs  
✅ `extract_phase1_best_configs.py` - Best config extractor  

### HPC Infrastructure
✅ `setup_hpc_workspaces.sh` - Workspace creator  
✅ `hpc/ummbas_v3_cpu.sh` - SLURM job runner  
✅ `hpc/submit_v3_phase1.sh` - Phase 1 submission  
✅ `hpc/submit_v3_phase2.sh` - Phase 2 submission  
✅ `hpc/submit_v3_phase3.sh` - Phase 3 submission  
✅ `hpc/submit_v3_phase4.sh` - Phase 4 submission  

### Documentation
✅ `docs/QUICKSTART_V3.md` - Quick start guide  
✅ `docs/README_V3_PIPELINE.md` - Comprehensive docs  
✅ `hpc/README.md` - HPC-specific docs  
✅ `V3_REFACTORING_COMPLETE.md` - Refactoring summary  

### Monitoring
✅ `scripts/check_hyperparam_status.py` - Status checker  
✅ `test_data_integrity.py` - Data validation  

---

## Next Action

**You can now run the experiments on the HPC cluster:**

1. SSH to cluster
2. Run: `bash setup_hpc_workspaces.sh`
3. Edit conda path in `hpc/ummbas_v3_cpu.sh`
4. Generate Phase 1 configs: `python generate_phase1_configs.py`
5. Submit Phase 1: `bash hpc/submit_v3_phase1.sh`

**Everything is ready! 🚀**

---

**Version:** 3.0  
**Last Updated:** October 15, 2025  
**Status:** Production Ready
