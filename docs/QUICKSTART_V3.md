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

### 1. Generate Phase 1 Configurations

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

### 2. Launch Phase 1 on HPC

```bash
# On HPC login node
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase1 \
  --workspace experiment_workspace_v3_phase1 \
  --n_jobs 20
```

**This will:**
- Test all dimensionalities (2D, 5D, 10D)
- Test all UMAP hyperparameters (nn=10,20,100,500 × md=0.01,0.1,0.5)
- Run 5 seeds each
- Take ~2-7 days depending on parallelization

---

### 3. Monitor Progress

While Phase 1 runs:

```bash
python check_hyperparam_status.py \
  --workspace experiment_workspace_v3_phase1
```

**Output:**
- Status report CSV
- Debug logs
- Scatter plot visualizations (PCA vs UMAP per seed)

---

### 4. Extract Best Configs (After Phase 1 Completes)

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

### 5. Generate and Run Subsequent Phases

**Phase 2 (Ablation):**
```bash
python generate_phase2_configs.py
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase2_ablation \
  --workspace experiment_workspace_v3_phase2 \
  --n_jobs 20
```

**Phase 3 (Generalization):**
```bash
python generate_phase3_configs.py
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase3_generalization \
  --workspace experiment_workspace_v3_phase3 \
  --n_jobs 20
```

**Phase 4 (Cutoff):**
```bash
python generate_phase4_configs.py
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase4_cutoff \
  --workspace experiment_workspace_v3_phase4 \
  --n_jobs 20
```

---

## Alternative: Use Master Orchestrator

```bash
python orchestrate_v3_pipeline.py
```

This will:
1. Detect which phase you're on
2. Generate configs automatically
3. Provide instructions for running each phase
4. Extract best configs between phases

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

| Phase | Runs | Time (10 cores) | Time (20 cores) |
|-------|------|-----------------|-----------------|
| Phase 1 | 260 | ~5-10 days | ~3-5 days |
| Phase 2 | 60 | ~1-2 days | ~0.5-1 day |
| Phase 3 | 80 | ~2-3 days | ~1-2 days |
| Phase 4 | 40 | ~1 day | ~0.5 day |
| **Total** | **440** | **~9-16 days** | **~5-8 days** |

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
# Step 1: Generate Phase 1 configs
python generate_phase1_configs.py

# Step 2: Launch on HPC
python main_orchestrator.py \
  --config_dir hyperparam_configs_v3_phase1 \
  --workspace experiment_workspace_v3_phase1 \
  --n_jobs 20
```

**Good luck with the experiments!**

---

**Questions?** Check `README_V3_PIPELINE.md` for full documentation.

**Version:** 3.0  
**Last Updated:** October 15, 2025
