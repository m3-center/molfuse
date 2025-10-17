# UMMBAS v3.0 - Quick Start Guide

## Immediate Next Steps

You're on branch `3.0` and ready to launch the new experimental pipeline!

### ✅ What's Ready

1. ✅ Base config updated (3 targets, 3 dimensions: 2/5/10)
2. ✅ Phase 1 config generator (260 runs)
3. ✅ Phase 2 config generator (60 runs) 
4. ✅ Phase 3 config generator (80 runs)
5. ✅ Phase 4 config generator (40 runs)
6. ✅ Best config extractor
7. ✅ Status checker with visualizations
8. ✅ Master orchestrator script

---

## Step-by-Step Launch

### 1. Setup HPC Workspaces (One-Time, on Cluster)

**On the HPC cluster, run:**

```bash
bash setup_hpc_workspaces.sh
```

**This script will:**
- Create workspace directories in `/work/ahagg2s/ummbas_results/`
- Create symlinks in repository root for easy access
- Verify all symlinks are correct

**Expected output:**
```
============================================================
UMMBAS v3.0 - HPC Workspace Setup
============================================================
✓ Created directory: /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase1
✓ Created symlink: experiment_workspace_v3_phase1 -> /work/ahagg2s/ummbas_results/experiment_workspace_v3_phase1
...
✓ All workspaces set up successfully!
```

---

### 2. Configure HPC Environment (One-Time Setup)

**Edit the conda environment path in `hpc/ummbas_v3_cpu.sh` (line 55):**

```bash
# Change this line to match your HPC setup:
source /home/YOUR_USERNAME/miniforge3/bin/activate ummbas-screening
```

---

### 3. Generate Phase 1 Configurations

```bash
python generate_phase1_configs.py
```

**Expected output:**
```
Total configs generated: 260

Breakdown:
  Features-PCA:               15
  Features-UMAP-Euclidean:    180
  Fingerprints-PCA:           5
  Fingerprints-UMAP-Jaccard:  60
```

---

### 4. Launch Phase 1 on HPC

```bash
# Submit all 260 jobs (52 configs × 5 seeds)
bash hpc/submit_v3_phase1.sh
```

**The script will:**
- Verify all prerequisites (configs exist, scripts present)
- Show summary (260 total jobs expected)
- Ask for confirmation
- Submit all jobs to SLURM queue

**Expected output:**
```
============================================================
UMMBAS v3.0 - Phase 1 Submission
============================================================
Configuration Directory: hyperparam_configs_v3_phase1
Number of Configs: 52
Number of Seeds: 5
Total Jobs: 260 (expected: 260)
============================================================
Proceed with submission? (y/n): y
```

**This will:**
- Test all dimensionalities (2D, 5D, 10D)
- Test all UMAP hyperparameters (nn=10,20,100,500 × md=0.01,0.1,0.5)
- Run 5 seeds each
- Take ~3-7 days depending on cluster resources

---

### 5. Monitor Progress

While Phase 1 runs:

```bash
# Check SLURM queue
squeue -u $USER

# Check experiment progress with status checker
python scripts/check_hyperparam_status.py \
  --workspace experiment_workspace_v3_phase1

# View live logs
tail -f slurm_logs/UMMBAS_v3_*.out
```

**Status checker output:**
- Status report CSV
- Debug logs
- Scatter plot visualizations (PCA vs UMAP per seed)

---

### 6. Extract Best Configs (After Phase 1 Completes)

```bash
python extract_phase1_best_configs.py \
  --workspace experiment_workspace_v3_phase1 \
  --output phase1_best_configs.json
```

**This identifies:**
- Best PCA-features per dimension (2D, 5D, 10D)
- Best UMAP-Euclidean-features per dimension
- Best PCA-fingerprints at 2D
- Best UMAP-Jaccard-fingerprints at 2D
- Overall best PCA and UMAP

---

### 7. Run Subsequent Phases

**Phase 2 (MF Cloud Ablation - 60 runs):**
```bash
# Generate configs based on Phase 1 best
python generate_phase2_configs.py

# Submit jobs
bash hpc/submit_v3_phase2.sh
```

**Phase 3 (Cross-Protein Generalization - 80 runs):**
```bash
# Generate configs for Pyru and Iso proteins
python generate_phase3_configs.py

# Submit jobs
bash hpc/submit_v3_phase3.sh
```

**Phase 4 (Affinity Cutoff Analysis - 40 runs):**
```bash
# Generate configs for cutoff sensitivity
python generate_phase4_configs.py

# Submit jobs
bash hpc/submit_v3_phase4.sh
```

Each submission script will:
- Verify prerequisites exist
- Show job count summary
- Ask for confirmation before submitting

---

## Key Files to Check

| File | Purpose |
|------|---------|
| `experiment_config.json` | Base configuration (updated for v3.0) |
| `README_V3_PIPELINE.md` | Complete pipeline documentation |
| `phase1_best_configs.json` | Best configs from Phase 1 (created after Phase 1) |
| `check_hyperparam_status.py` | Monitor progress + visualizations |

---

## Expected Timeline

| Phase | Runs | Time (32 cores) | Time (64 cores) |
|-------|------|-----------------|-----------------|
| Phase 1 | 260 | ~6-10 days | ~3-5 days |
| Phase 2 | 60 | ~1-2 days | ~0.5-1 day |
| Phase 3 | 80 | ~2-3 days | ~1-1.5 days |
| Phase 4 | 40 | ~1 day | ~0.5 day |
| **Total** | **440** | **~10-16 days** | **~5-8 days** |

Times depend on cluster load and hardware.

---

## Troubleshooting

### Config Generator Fails
```bash
# Check that base config is valid
python -c "import json; json.load(open('experiment_config.json'))"
```

### Phase 1 Best Configs Not Found
```bash
# Verify Phase 1 workspace exists and has results
ls experiment_workspace_v3_phase1/run_*/*/results/*/dim_*/*/*_ranking_metrics.csv | wc -l
# Should show ~260 files when complete
```

### Status Checker Shows No Figures
- Ensure matplotlib backend is set (already fixed in check_hyperparam_status.py)
- Check that similarity_space CSV files exist
- Run with: `python check_hyperparam_status.py --workspace <workspace> 2>&1 | tee status_log.txt`

---

## What's Different from v2.0?

| Aspect | v2.0 | v3.0 |
|--------|------|------|
| Dimensions | 2D only | 2D, 5D, 10D |
| UMAP hyperparams | nn=5,10,15,20,30 × md=0.01,0.05,0.1,0.5,1.0 | nn=10,20,100,500 × md=0.01,0.1,0.5 |
| Targets | 1 (Tyro) | 3 (Tyro, Pyru, Iso) |
| Phases | 1 (hyperparam sweep) | 4 (sweep, ablation, generalization, cutoff) |
| Total runs | 650 | 440 (more efficient!) |
| MF ablation | No | Yes (6 sizes) |
| Analysis | Basic | Comprehensive (PCA dominance module) |

---

## Ready to Launch! 🚀

```bash
# Step 1: Setup workspaces (one-time, on HPC)
bash setup_hpc_workspaces.sh

# Step 2: Configure HPC (one-time)
# Edit: hpc/ummbas_v3_cpu.sh line 55
# Set your conda environment path

# Step 3: Generate Phase 1 configs
python generate_phase1_configs.py

# Step 4: Launch on HPC
bash hpc/submit_v3_phase1.sh

# Step 5: Monitor progress
squeue -u $USER
python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1
```

**Good luck with the experiments!**

---

**Questions?** 
- Quick reference: This document
- Full details: `README_V3_PIPELINE.md`
- HPC scripts: `hpc/README.md`

**Version:** 3.0  
**Last Updated:** October 15, 2025
